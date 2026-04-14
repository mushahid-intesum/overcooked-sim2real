#include <cstring>
#include <cmath>
#include <map>
#include <string>
#include <chrono>
#include <thread>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_heap_caps.h"

#include "dl_model_base.hpp"
#include "camera.h"
#include "driver/gpio.h"
#include "driver/ledc.h"

// ── Motor pins (adjust to your wiring) ───────────────────────────────────────
#define IN1  GPIO_NUM_35
#define IN2  GPIO_NUM_36
#define IN3  GPIO_NUM_37
#define IN4  GPIO_NUM_38
#define ENA  GPIO_NUM_39
#define ENB  GPIO_NUM_40

#define MOTOR_SPEED     100   // 0–255
#define LEDC_FREQ_HZ    1000
#define LEDC_RESOLUTION LEDC_TIMER_8_BIT   // 0–255

// ── Polling / stability ───────────────────────────────────────────────────────
#define POLL_WINDOW     2     // consecutive identical actions required
#define STEP_MS         50    // ms between inference calls

// ── Model dimensions ──────────────────────────────────────────────────────────
static constexpr int GRID_C     = 11;
static constexpr int GRID_H     = 11;
static constexpr int GRID_W     = 11;
static constexpr int SCALAR_DIM = 7;
static constexpr int HIDDEN_DIM = 128;
static constexpr int ACTION_DIM = 6;
static constexpr int GRID_SIZE  = GRID_C * GRID_H * GRID_W;

static constexpr int GRID_EXPONENT   = -7;
static constexpr int SCALAR_EXPONENT = -7;
static constexpr int HIDDEN_EXPONENT = -27;
static constexpr int LOGIT_EXPONENT  = -5;

// ── Quantisation helpers ──────────────────────────────────────────────────────
static inline int8_t f32_to_q8(float f, int exponent) {
    float scale = 1.0f;
    for (int i = 0; i < -exponent; i++) scale *= 2.0f;
    int v = static_cast<int>(f * scale + 0.5f);
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return static_cast<int8_t>(v);
}

static inline float q8_to_f32(int8_t q, int exponent) {
    float scale = 1.0f;
    for (int i = 0; i < -exponent; i++) scale *= 2.0f;
    return static_cast<float>(q) / scale;
}

static void chw_to_hwc(const int8_t *src, int8_t *dst, int C, int H, int W) {
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                dst[h * W * C + w * C + c] = src[c * H * W + h * W + w];
}

// ── PWM / Motor driver ────────────────────────────────────────────────────────
static const char *TAG_MOT = "motor";

static void pwm_init() {
    ledc_timer_config_t timer = {
        .speed_mode      = LEDC_LOW_SPEED_MODE,
        .duty_resolution = LEDC_RESOLUTION,
        .timer_num       = LEDC_TIMER_0,
        .freq_hz         = LEDC_FREQ_HZ,
        .clk_cfg         = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&timer));

    ledc_channel_config_t ch_a = {
        .gpio_num   = ENA,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel    = LEDC_CHANNEL_0,
        .timer_sel  = LEDC_TIMER_0,
        .duty       = 0,
        .hpoint     = 0,
    };
    ledc_channel_config_t ch_b = {
        .gpio_num   = ENB,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel    = LEDC_CHANNEL_1,
        .timer_sel  = LEDC_TIMER_0,
        .duty       = 0,
        .hpoint     = 0,
    };
    ESP_ERROR_CHECK(ledc_channel_config(&ch_a));
    ESP_ERROR_CHECK(ledc_channel_config(&ch_b));
}

static void set_speed(uint32_t speed) {
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, speed);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_1, speed);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_1);
}

static void motor_gpio_init() {
    gpio_config_t io = {
        .pin_bit_mask = (1ULL << IN1) | (1ULL << IN2) |
                        (1ULL << IN3) | (1ULL << IN4),
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&io));
}

static void motor_stop() {
    set_speed(0);
    ESP_LOGI(TAG_MOT, "STOP");
}

static void motor_forward() {
    gpio_set_level(IN1, 1); gpio_set_level(IN2, 0);
    gpio_set_level(IN3, 1); gpio_set_level(IN4, 0);
    set_speed(MOTOR_SPEED);
    ESP_LOGI(TAG_MOT, "FORWARD");
}

static void motor_reverse() {
    gpio_set_level(IN1, 0); gpio_set_level(IN2, 1);
    gpio_set_level(IN3, 0); gpio_set_level(IN4, 1);
    set_speed(MOTOR_SPEED);
    ESP_LOGI(TAG_MOT, "REVERSE");
}

static void motor_turn_right() {
    gpio_set_level(IN1, 0); gpio_set_level(IN2, 1);
    gpio_set_level(IN3, 1); gpio_set_level(IN4, 0);
    set_speed(MOTOR_SPEED);
    ESP_LOGI(TAG_MOT, "RIGHT");
}

static void motor_turn_left() {
    gpio_set_level(IN1, 1); gpio_set_level(IN2, 0);
    gpio_set_level(IN3, 0); gpio_set_level(IN4, 1);
    set_speed(MOTOR_SPEED);
    ESP_LOGI(TAG_MOT, "LEFT");
}

static void execute_action(uint8_t action) {
    switch (action) {
        case 0: motor_forward();    break;   // NORTH
        case 1: motor_reverse();    break;   // SOUTH
        case 2: motor_turn_right(); break;   // EAST
        case 3: motor_turn_left();  break;   // WEST
        case 4: motor_stop();       break;   // STAY
        case 5: motor_stop();       break;   // INTERACT
        default:
            ESP_LOGW(TAG_MOT, "Unknown action %d", action);
            motor_stop();
            break;
    }
}

// ── Actor inference ───────────────────────────────────────────────────────────
static const char *TAG_INF = "inference";

extern const uint8_t _player_1_actor_espdl_start[] asm("_binary_player_1_actor_espdl_start");
extern const uint8_t _player_1_actor_espdl_end[]   asm("_binary_player_1_actor_espdl_end");

static dl::Model *s_model      = nullptr;
static int8_t    *s_hidden_buf = nullptr;

static esp_err_t actor_init() {
    s_hidden_buf = static_cast<int8_t *>(
        heap_caps_calloc(HIDDEN_DIM, sizeof(int8_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (!s_hidden_buf) {
        ESP_LOGE(TAG_INF, "Failed to allocate hidden state buffer");
        return ESP_ERR_NO_MEM;
    }

    s_model = new dl::Model(
        reinterpret_cast<const char *>(_player_1_actor_espdl_start),
        fbs::MODEL_LOCATION_IN_FLASH_RODATA,
        0,
        dl::MEMORY_MANAGER_GREEDY
    );

    if (!s_model) {
        ESP_LOGE(TAG_INF, "dl::Model allocation failed");
        return ESP_FAIL;
    }

    if (!s_model->get_input("input.1") ||
        !s_model->get_input("onnx::Gemm_1") ||
        !s_model->get_input("onnx::Gemm_2")) {
        ESP_LOGE(TAG_INF, "Expected input tensors not found");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG_INF, "Actor model loaded (%d bytes)",
             static_cast<int>(_player_1_actor_espdl_end - _player_1_actor_espdl_start));
    return ESP_OK;
}

static void actor_reset_hidden() {
    if (s_hidden_buf) memset(s_hidden_buf, 0, HIDDEN_DIM);
}

static int actor_run(const int8_t *grid_q8, const int8_t *scalars_q8) {
    dl::TensorBase *grid_t   = s_model->get_input("input.1");
    dl::TensorBase *scalar_t = s_model->get_input("onnx::Gemm_1");
    dl::TensorBase *hidden_t = s_model->get_input("onnx::Gemm_2");

    static int8_t grid_hwc[GRID_SIZE];
    chw_to_hwc(grid_q8, grid_hwc, GRID_C, GRID_H, GRID_W);
    memcpy(grid_t->data,   grid_hwc,     GRID_SIZE);
    memcpy(scalar_t->data, scalars_q8,   SCALAR_DIM);
    memcpy(hidden_t->data, s_hidden_buf, HIDDEN_DIM);

    s_model->run();

    int8_t *logits_data     = nullptr;
    int8_t *new_hidden_data = nullptr;

    for (auto &kv : s_model->get_outputs()) {
        dl::TensorBase *t = kv.second;
        int n = 1;
        for (int d : t->shape) n *= d;
        if      (n == ACTION_DIM && !logits_data)     logits_data     = static_cast<int8_t *>(t->data);
        else if (n == HIDDEN_DIM && !new_hidden_data) new_hidden_data = static_cast<int8_t *>(t->data);
    }

    if (!logits_data) {
        ESP_LOGE(TAG_INF, "Logits output tensor not found");
        return -1;
    }

    if (new_hidden_data)
        memcpy(s_hidden_buf, new_hidden_data, HIDDEN_DIM);
    else
        ESP_LOGW(TAG_INF, "Hidden state output not found");

    int best = 0;
    for (int i = 1; i < ACTION_DIM; i++)
        if (logits_data[i] > logits_data[best]) best = i;

    ESP_LOGI(TAG_INF, "logits=[%d,%d,%d,%d,%d,%d] action=%d",
             logits_data[0], logits_data[1], logits_data[2],
             logits_data[3], logits_data[4], logits_data[5], best);

    return best;
}

// ── Main task ─────────────────────────────────────────────────────────────────
static const char *TAG = "main";

static int8_t g_grid_q8   [GRID_SIZE]  __attribute__((aligned(16)));
static int8_t g_scalars_q8[SCALAR_DIM] __attribute__((aligned(16)));

static void inference_task(void *) {
    ESP_LOGI(TAG, "Inference task started");

    // ── Polling state ─────────────────────────────────────────────────────────
    int     poll_action    = -1;   // candidate action being confirmed
    int     poll_count     = 0;    // how many times seen consecutively
    int     current_action = -1;   // last committed action

    while (true) {
        // 1. Capture frame
        camera_fb_t *fb = camera_capture();
        if (!fb) {
            ESP_LOGW(TAG, "camera_capture failed — retrying");
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        memcpy(g_grid_q8, fb->buf, GRID_SIZE);
        camera_return_fb(fb);

        // 2. [TEST MODE] Override: cycle actions 0–3 regardless of model output
        //    Remove this block and use `raw_action = actor_run(...)` for real inference
        static int test_action = 0;
        int raw_action = test_action;          // always 0–3
        test_action = (test_action + 1) % 4;  // cycle N→S→E→W for wiring test
        // int raw_action = actor_run(g_grid_q8, g_scalars_q8);  // ← real inference

        if (raw_action < 0 || raw_action > 3) {
            // Clamp to 0–3 for test mode safety
            raw_action = 0;
        }

        // 3. Polling filter — require POLL_WINDOW identical results before acting
        if (raw_action == poll_action) {
            poll_count++;
        } else {
            poll_action = raw_action;
            poll_count  = 1;
        }

        // if (poll_count >= POLL_WINDOW && raw_action != current_action) {
            current_action = raw_action;
            ESP_LOGI(TAG, "Committed action=%d (confirmed %d×)", current_action, POLL_WINDOW);
            execute_action(static_cast<uint8_t>(current_action));
            poll_count = 0;  
            vTaskDelay(pdMS_TO_TICKS(1000));
        // }

        vTaskDelay(pdMS_TO_TICKS(STEP_MS));
    }
}

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "Booting…");

    motor_gpio_init();
    pwm_init();
    motor_stop();
    ESP_LOGI(TAG, "Motor driver ready");

    ESP_ERROR_CHECK(camera_init());
    ESP_LOGI(TAG, "Camera ready");

    ESP_ERROR_CHECK(actor_init());
    actor_reset_hidden();

    xTaskCreatePinnedToCore(
        inference_task,
        "infer",
        8192,
        nullptr,
        5,
        nullptr,
        1
    );

    ESP_LOGI(TAG, "Startup complete");
}