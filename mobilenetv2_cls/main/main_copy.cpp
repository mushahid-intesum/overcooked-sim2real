/*
 * app_main.cpp
 * ─────────────────────────────────────────────────────────────────────────────
 * On-board camera inference for the MAPPO Actor network on ESP32-S3.
 *
 * Architecture overview
 * ─────────────────────
 *   Camera (OV2640/OV7670) → JPEG frame → RGB decode
 *       │
 *       ▼
 *   Perception pipeline  ← THIS is what the camera feeds, NOT the Actor
 *       │  detect objects in egocentric view (ArUco / color blob)
 *       │  quantise positions to grid cells
 *       ▼
 *   Grid tensor  (C × D × D, int8)   ← Actor input 0  "grid"
 *   Scalar tensor (7, int8)           ← Actor input 1  "scalars"
 *   Hidden tensor (128, int8)         ← Actor input 2  "hidden_in"
 *       │
 *       ▼
 *   Actor model  (player_0_actor.espdl)
 *       │
 *       ├─ logits  (6, int8)  → argmax → action index
 *       └─ hidden_out (128, int8) → carry forward as next hidden_in
 *
 * Input tensor layout (matches ONNX export order in quantize_actor.py)
 * ─────────────────────────────────────────────────────────────────────
 *   "grid"      : shape [1, 11, 11, 11]  = [batch, C, D, D]
 *                 C=11 channels (one per object type), D=11 (2*radius+1)
 *                 Values: 0 or 1 (binary one-hot), quantised to int8
 *
 *   "scalars"   : shape [1, 7]
 *                 [delta_row, delta_col, held_none, held_onion,
 *                  held_tomato, held_dish, held_soup]
 *                 delta_row/col in grid-cell units, quantised to int8
 *
 *   "hidden_in" : shape [1, 128]
 *                 GRU hidden state, zero-initialised at episode start
 *                 Carry hidden_out → hidden_in every step within an episode
 *
 * Quantisation exponents
 * ──────────────────────
 *   INT8 value = round(float_value / 2^exponent)
 *   The correct exponent for each tensor is embedded in the .espdl file and
 *   readable from TensorBase::exponent after model->get_inputs().
 *   For reference, esp-ppq typically assigns:
 *       grid    : exponent = 0   (binary {0,1} → int8 {0, 1})
 *       scalars : exponent = -4  (range ±1.0 → ±16 in int8, fits [-128,127])
 *       hidden  : exponent = -7  (tanh range → common GRU exponent)
 *   These are READ from the model at runtime, not hardcoded here.
 *
 * IMU (MPU-6050)
 * ──────────────
 *   The IMU is used to compute delta_row/delta_col (ego-velocity).
 *   Integration runs in a background task. Heading is used to align the
 *   camera FoV cone with the grid's ego-centric orientation.
 *
 * Action index → motor command mapping
 * ─────────────────────────────────────
 *   Matches Action.INDEX_TO_ACTION order in Overcooked-AI:
 *       0: NORTH    → drive forward
 *       1: SOUTH    → drive backward
 *       2: EAST     → turn right
 *       3: WEST     → turn left
 *       4: STAY     → stop motors
 *       5: INTERACT → trigger servo / grabber mechanism
 *
 * Dependencies (idf_component.yml)
 * ─────────────────────────────────
 *   espressif/esp-dl     >= 3.1.0
 *   espressif/esp_camera >= 2.0.0
 *
 * Flash partition (partitions.csv)
 * ─────────────────────────────────
 *   model, data, 0x10000, 0x100000   ← store player_0_actor.espdl here
 */

#include <cstring>
#include <cmath>
#include <algorithm>

#include "esp_camera.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

// ESP-DL v3 headers
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"
#include "fbs_loader.hpp"

static const char *TAG = "actor_inference";

// ─────────────────────────────────────────────────────────────────────────────
// Constants — must match training configuration in mappo_train.py / wrapper
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int NUM_CHANNELS  = 11;    // grid channels (object types)
static constexpr int FOV_RADIUS    = 5;
static constexpr int FOV_D         = 2 * FOV_RADIUS + 1;   // 11
static constexpr int SCALAR_DIM    = 7;                     // [drow,dcol,held×5]
static constexpr int HIDDEN_DIM    = 128;
static constexpr int ACTION_DIM    = 6;

// Grid channel indices (mirrors CHANNELS dict in overcooked_partial_obs_wrapper.py)
static constexpr int CH_WALL       = 0;
static constexpr int CH_FLOOR      = 1;
static constexpr int CH_SELF       = 2;
static constexpr int CH_TEAMMATE   = 3;
static constexpr int CH_ONION      = 4;
static constexpr int CH_TOMATO     = 5;
static constexpr int CH_DISH       = 6;
static constexpr int CH_SOUP       = 7;
static constexpr int CH_POT        = 8;
static constexpr int CH_SERVE      = 9;
static constexpr int CH_DELIVERY   = 10;

// Scalar held-object indices
static constexpr int HELD_NONE     = 0;
static constexpr int HELD_ONION    = 1;
static constexpr int HELD_TOMATO   = 2;
static constexpr int HELD_DISH     = 3;
static constexpr int HELD_SOUP     = 4;

// Control loop timing
static constexpr int CONTROL_LOOP_MS  = 100;   // 10 Hz — limited by camera

// ─────────────────────────────────────────────────────────────────────────────
// Camera pin config — adjust to your board
// ─────────────────────────────────────────────────────────────────────────────

#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5
#define CAM_PIN_D0      11
#define CAM_PIN_D1      9
#define CAM_PIN_D2      8
#define CAM_PIN_D3      10
#define CAM_PIN_D4      12
#define CAM_PIN_D5      18
#define CAM_PIN_D6      17
#define CAM_PIN_D7      16
#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK    13

// ─────────────────────────────────────────────────────────────────────────────
// Grid tensor — allocated once in PSRAM, written each step by perception
// ─────────────────────────────────────────────────────────────────────────────

// Raw float grid (C × D × D), written by perception, quantised before inference
static float   s_grid_float[NUM_CHANNELS][FOV_D][FOV_D];
static float   s_scalars_float[SCALAR_DIM];

// GRU hidden state — float, carried step-to-step within an episode
// Zero-initialised at episode start. Written from hidden_out each step.
static float   s_hidden_float[HIDDEN_DIM];

// ─────────────────────────────────────────────────────────────────────────────
// IMU state (updated by background task or inline)
// ─────────────────────────────────────────────────────────────────────────────

static volatile float s_delta_row = 0.0f;  // ego-velocity from IMU integration
static volatile float s_delta_col = 0.0f;
static volatile int   s_held_obj  = HELD_NONE;  // set by motor/interact logic

// ─────────────────────────────────────────────────────────────────────────────
// Camera initialisation
// ─────────────────────────────────────────────────────────────────────────────

static esp_err_t camera_init(void)
{
    camera_config_t config = {};

    config.pin_pwdn       = CAM_PIN_PWDN;
    config.pin_reset      = CAM_PIN_RESET;
    config.pin_xclk       = CAM_PIN_XCLK;
    config.pin_sccb_sda   = CAM_PIN_SIOD;
    config.pin_sccb_scl   = CAM_PIN_SIOC;
    config.pin_d7         = CAM_PIN_D7;
    config.pin_d6         = CAM_PIN_D6;
    config.pin_d5         = CAM_PIN_D5;
    config.pin_d4         = CAM_PIN_D4;
    config.pin_d3         = CAM_PIN_D3;
    config.pin_d2         = CAM_PIN_D2;
    config.pin_d1         = CAM_PIN_D1;
    config.pin_d0         = CAM_PIN_D0;
    config.pin_vsync      = CAM_PIN_VSYNC;
    config.pin_href       = CAM_PIN_HREF;
    config.pin_pclk       = CAM_PIN_PCLK;

    config.xclk_freq_hz   = 20000000;
    config.ledc_timer     = LEDC_TIMER_0;
    config.ledc_channel   = LEDC_CHANNEL_0;

    // RGB565 instead of JPEG — avoids decode step and gives direct pixel access
    // for the colour-blob / ArUco perception pipeline.
    // Use FRAMESIZE_QVGA (320×240) — enough resolution for marker detection
    // while fitting comfortably in PSRAM.
    config.pixel_format   = PIXFORMAT_RGB565;
    config.frame_size     = FRAMESIZE_QVGA;
    config.fb_count       = 2;
    config.grab_mode      = CAMERA_GRAB_LATEST;
    config.fb_location    = CAMERA_FB_IN_PSRAM;
    config.sccb_i2c_port  = 0;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
    }
    return err;
}

// ─────────────────────────────────────────────────────────────────────────────
// Perception pipeline
// ─────────────────────────────────────────────────────────────────────────────
//
// This is the MAIN engineering task for Approach A (on-board camera).
// The function receives an RGB565 frame and must fill s_grid_float with
// a (NUM_CHANNELS, FOV_D, FOV_D) binary tensor in ego-centric coordinates.
//
// Recommended implementation for a controlled arena:
//   1. Detect ArUco markers (one per object type, distinct IDs per class)
//      using a lightweight marker detector (e.g. esp-who's ArUco detector,
//      or a simplified 4-corner homography-based approach).
//   2. Convert each detected marker's pixel position to a relative grid-cell
//      offset from the robot's current position (use IMU heading for rotation).
//   3. Set the corresponding channel at that (row, col) offset to 1.0f.
//   4. Apply the forward-cone visibility mask — cells behind the robot
//      (row > FOV_RADIUS in ego-centric frame) stay zero regardless.
//
// For now this stub populates an example grid with just the self-channel set
// at the ego-centric centre. Replace the body with your actual detector.
// ─────────────────────────────────────────────────────────────────────────────

static void perception_pipeline(const uint8_t *rgb565_buf,
                                int width, int height)
{
    // Clear all channels
    memset(s_grid_float, 0, sizeof(s_grid_float));

    // ── Set self at ego-centric centre ────────────────────────────────────
    // Agent is always at (FOV_RADIUS, FOV_RADIUS) in the ego-centric crop.
    s_grid_float[CH_SELF][FOV_RADIUS][FOV_RADIUS] = 1.0f;

    // ── Apply static terrain (walls / floor) from known map ──────────────
    // In a controlled arena with a known layout, the static terrain can be
    // hardcoded based on the grid map rather than detected from the camera.
    // This avoids the need to detect featureless walls optically.
    //
    // Example: mark cells that are known walls in the FoV crop.
    // Replace with your actual arena layout or dynamic detection.
    //
    // for (int r = 0; r < FOV_D; r++) {
    //     for (int c = 0; c < FOV_D; c++) {
    //         if (is_wall_at_ego_offset(r - FOV_RADIUS, c - FOV_RADIUS)) {
    //             s_grid_float[CH_WALL][r][c] = 1.0f;
    //         } else {
    //             s_grid_float[CH_FLOOR][r][c] = 1.0f;
    //         }
    //     }
    // }

    // ── Detect objects via colour blobs or ArUco ─────────────────────────
    // For each detected object:
    //   1. Compute ego-centric grid offset (row_off, col_off) from pixel pos
    //   2. Apply forward-cone mask: row_off < 0 means in front of agent
    //   3. Clamp to FoV bounds: abs(row_off) <= FOV_RADIUS etc.
    //   4. Set s_grid_float[channel][FOV_RADIUS + row_off][FOV_RADIUS + col_off] = 1.f
    //
    // Example skeleton (fill in your detection results):
    //
    // for (auto &det : detected_objects) {
    //     int row_off = det.grid_row_offset;  // negative = in front
    //     int col_off = det.grid_col_offset;
    //     int r = FOV_RADIUS + row_off;
    //     int c = FOV_RADIUS + col_off;
    //     if (r >= 0 && r < FOV_D && c >= 0 && c < FOV_D) {
    //         // Forward-cone mask: cells behind agent (r > FOV_RADIUS) are blind
    //         if (r <= FOV_RADIUS) {
    //             s_grid_float[det.channel][r][c] = 1.0f;
    //         }
    //     }
    // }

    (void)rgb565_buf;   // suppress unused-parameter warning until implemented
    (void)width;
    (void)height;
}

// ─────────────────────────────────────────────────────────────────────────────
// Build scalar feature vector
// ─────────────────────────────────────────────────────────────────────────────

static void build_scalars(void)
{
    // [0] delta_row  — IMU-integrated ego-velocity, row component
    // [1] delta_col  — IMU-integrated ego-velocity, col component
    s_scalars_float[0] = s_delta_row;
    s_scalars_float[1] = s_delta_col;

    // [2..6] held object one-hot
    for (int i = 0; i < 5; i++) {
        s_scalars_float[2 + i] = 0.0f;
    }
    s_scalars_float[2 + s_held_obj] = 1.0f;

    // Reset velocity after consuming — IMU background task will update again
    s_delta_row = 0.0f;
    s_delta_col = 0.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Quantise float → int8 using the exponent stored in the TensorBase
// ─────────────────────────────────────────────────────────────────────────────
//
// ESP-DL uses fixed-point: int8_val = round(float_val / 2^exponent)
// The exponent for each tensor is set by esp-ppq during calibration and
// stored in the .espdl file. Read it from tensor->exponent at runtime.
//
// For binary grid values {0.0, 1.0}:
//   exponent = 0  → int8 = {0, 1}     (no scaling needed)
// For scalar velocities in [-1.0, 1.0]:
//   exponent = -4 → int8 = {-16..16}  (typical)
// For GRU hidden in tanh range [-1.0, 1.0]:
//   exponent = -7 → int8 = {-128..127}
// ─────────────────────────────────────────────────────────────────────────────

static void float_to_int8(const float *src, int8_t *dst, int n, int exponent)
{
    // scale = 2^(-exponent) = 1.0 / 2^exponent
    float scale = ldexpf(1.0f, -exponent);
    for (int i = 0; i < n; i++) {
        float v = src[i] * scale;
        v = fmaxf(-128.0f, fminf(127.0f, roundf(v)));
        dst[i] = static_cast<int8_t>(v);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dequantise int8 → float  (for reading hidden_out back to s_hidden_float)
// ─────────────────────────────────────────────────────────────────────────────

static void int8_to_float(const int8_t *src, float *dst, int n, int exponent)
{
    float scale = ldexpf(1.0f, exponent);   // 2^exponent
    for (int i = 0; i < n; i++) {
        dst[i] = static_cast<float>(src[i]) * scale;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Action → motor command mapping
// ─────────────────────────────────────────────────────────────────────────────
//
// Replace motor_forward(), motor_backward() etc. with your actual motor driver
// calls (L298N / DRV8833 PWM via ledc or mcpwm).
// ─────────────────────────────────────────────────────────────────────────────

static const char *ACTION_NAMES[] = {
    "NORTH (fwd)", "SOUTH (back)", "EAST (right)",
    "WEST (left)",  "STAY",         "INTERACT"
};

static void execute_action(int action_idx)
{
    ESP_LOGI(TAG, "Action: %d  (%s)", action_idx, ACTION_NAMES[action_idx]);

    switch (action_idx) {
        case 0: /* NORTH → forward  */ /* motor_forward();   */ break;
        case 1: /* SOUTH → backward */ /* motor_backward();  */ break;
        case 2: /* EAST  → right    */ /* motor_turn_right(); */ break;
        case 3: /* WEST  → left     */ /* motor_turn_left();  */ break;
        case 4: /* STAY             */ /* motor_stop();       */ break;
        case 5: /* INTERACT         */ /* trigger_interact(); */ break;
        default: break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

extern "C" void app_main(void)
{
    // ── Camera init ───────────────────────────────────────────────────────
    ESP_ERROR_CHECK(camera_init());

    // ── Load Actor model from flash partition ─────────────────────────────
    // The .espdl file is flashed to the "model" partition defined in
    // partitions.csv. The model name string must match the name used in
    // pack_espdl_models.py (or the single-model default "model").
    dl::Model *model = new dl::Model(
        "model",                             // partition label / model name
        fbs::MODEL_LOCATION_IN_FLASH_PARTITION
    );
    if (!model) {
        ESP_LOGE(TAG, "Failed to load actor model from flash");
        return;
    }
    ESP_LOGI(TAG, "Actor model loaded");

    // ── Get named input/output tensors from the model ─────────────────────
    // Input names match the input_names set in quantize_actor.py:
    //   "grid", "scalars", "hidden_in"
    // Output names: "logits", "hidden_out"
    //
    // The model owns the memory for these tensors — do NOT free them.
    auto model_inputs  = model->get_inputs();
    auto model_outputs = model->get_outputs();

    dl::TensorBase *t_grid    = model_inputs.at("grid");
    dl::TensorBase *t_scalars = model_inputs.at("scalars");
    dl::TensorBase *t_hidden  = model_inputs.at("hidden_in");
    dl::TensorBase *t_logits  = model_outputs.at("logits");
    dl::TensorBase *t_hid_out = model_outputs.at("hidden_out");

    ESP_LOGI(TAG, "Inputs : grid%s  scalars%s  hidden%s",
             t_grid->shape_str().c_str(),
             t_scalars->shape_str().c_str(),
             t_hidden->shape_str().c_str());
    ESP_LOGI(TAG, "Outputs: logits%s  hidden_out%s",
             t_logits->shape_str().c_str(),
             t_hid_out->shape_str().c_str());
    ESP_LOGI(TAG, "Exponents: grid=%d  scalars=%d  hidden=%d  logits=%d  hid_out=%d",
             t_grid->exponent, t_scalars->exponent, t_hidden->exponent,
             t_logits->exponent, t_hid_out->exponent);

    // ── Zero-initialise GRU hidden state (episode start) ─────────────────
    memset(s_hidden_float, 0, sizeof(s_hidden_float));

    // ── Control loop ──────────────────────────────────────────────────────
    int64_t next_step_us = esp_timer_get_time();
    bool episode_done    = false;

    while (true) {
        // Rate-limit to CONTROL_LOOP_MS
        int64_t now    = esp_timer_get_time();
        int64_t wait   = next_step_us - now;
        if (wait > 1000) {
            vTaskDelay(pdMS_TO_TICKS(wait / 1000));
        }
        next_step_us += (int64_t)CONTROL_LOOP_MS * 1000LL;

        int64_t t0 = esp_timer_get_time();

        // ── 1. Capture camera frame ───────────────────────────────────────
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "Camera capture failed");
            continue;
        }

        // ── 2. Perception: camera frame → binary grid tensor ──────────────
        // fb->buf is RGB565, fb->width × fb->height
        // perception_pipeline fills s_grid_float[C][D][D]
        perception_pipeline(fb->buf, fb->width, fb->height);
        esp_camera_fb_return(fb);   // return frame buffer immediately

        // ── 3. Build scalar features from IMU + held-object state ─────────
        build_scalars();   // fills s_scalars_float[7]

        // ── 4. Quantise floats → int8 and write into model input tensors ──
        // Grid: (1, C, D, D) = 1 × 11 × 11 × 11 = 1331 elements
        float_to_int8(
            &s_grid_float[0][0][0],
            static_cast<int8_t *>(t_grid->get_element_ptr()),
            NUM_CHANNELS * FOV_D * FOV_D,
            t_grid->exponent
        );

        // Scalars: (1, 7)
        float_to_int8(
            s_scalars_float,
            static_cast<int8_t *>(t_scalars->get_element_ptr()),
            SCALAR_DIM,
            t_scalars->exponent
        );

        // Hidden state: (1, 128)
        float_to_int8(
            s_hidden_float,
            static_cast<int8_t *>(t_hidden->get_element_ptr()),
            HIDDEN_DIM,
            t_hidden->exponent
        );

        // ── 5. Run actor inference ─────────────────────────────────────────
        // Model::run() with no arguments uses the pre-filled input tensors
        // and writes results into the output tensors.
        model->run(RUNTIME_MODE_AUTO);

        int64_t t_infer = esp_timer_get_time();

        // ── 6. Read logits → argmax → action ──────────────────────────────
        const int8_t *logits_raw =
            static_cast<const int8_t *>(t_logits->get_element_ptr());

        int best_action = 0;
        int8_t best_val = logits_raw[0];
        for (int i = 1; i < ACTION_DIM; i++) {
            if (logits_raw[i] > best_val) {
                best_val   = logits_raw[i];
                best_action = i;
            }
        }

        // ── 7. Read hidden_out → dequantise → store for next step ─────────
        // hidden_out shares memory with intermediate tensors in esp-dl's
        // static memory plan — copy it OUT before the next model->run() call
        // or before any subsequent tensor operation overwrites it.
        int8_to_float(
            static_cast<const int8_t *>(t_hid_out->get_element_ptr()),
            s_hidden_float,
            HIDDEN_DIM,
            t_hid_out->exponent
        );

        // ── 8. Execute action ─────────────────────────────────────────────
        execute_action(best_action);

        // ── 9. Timing log ─────────────────────────────────────────────────
        ESP_LOGD(TAG, "inference=%lld ms  total_step=%lld ms",
                 (t_infer - t0) / 1000,
                 (esp_timer_get_time() - t0) / 1000);

        // ── 10. Episode reset (triggered externally, e.g. via button/signal)
        // When an episode ends (horizon reached or task complete), zero the
        // GRU hidden state before the next reset() call.
        if (episode_done) {
            memset(s_hidden_float, 0, sizeof(s_hidden_float));
            episode_done = false;
            ESP_LOGI(TAG, "Episode reset — hidden state cleared");
        }

        vTaskDelay(1);   // yield to FreeRTOS scheduler
    }

    delete model;
}