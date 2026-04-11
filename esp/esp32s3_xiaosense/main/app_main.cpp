#include <cstring>
#include <cmath>
#include <map>
#include <string>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_heap_caps.h"

#include "dl_model_base.hpp"

#include "camera.h"
#include "spi_comm.h"

// ── Model dimensions ─────────────────────────────────────────────────────────

static constexpr int GRID_C     = 11;
static constexpr int GRID_H     = 11;
static constexpr int GRID_W     = 11;
static constexpr int SCALAR_DIM = 7;
static constexpr int HIDDEN_DIM = 128;
static constexpr int ACTION_DIM = 6;
static constexpr int GRID_SIZE  = GRID_C * GRID_H * GRID_W;  // 1331

static constexpr int GRID_EXPONENT   = -7;
static constexpr int SCALAR_EXPONENT = -7;
static constexpr int HIDDEN_EXPONENT = -27;
static constexpr int LOGIT_EXPONENT  = -5;

// ── Quantisation helpers ─────────────────────────────────────────────────────

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

static void chw_to_hwc(const int8_t *src, int8_t *dst,
                        int C, int H, int W) {
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                dst[h * W * C + w * C + c] = src[c * H * W + h * W + w];
}

// ── Actor inference ──────────────────────────────────────────────────────────

static const char *TAG_INF = "inference";

extern const uint8_t _player_1_actor_espdl_start[] asm("_binary_player_1_actor_espdl_start");
extern const uint8_t _player_1_actor_espdl_end[]   asm("_binary_player_1_actor_espdl_end");

static dl::Model *s_model      = nullptr;
static int8_t    *s_hidden_buf = nullptr;

static esp_err_t actor_init() {
    s_hidden_buf = static_cast<int8_t *>(
        heap_caps_calloc(HIDDEN_DIM, sizeof(int8_t), MALLOC_CAP_SPIRAM));
    if (!s_hidden_buf) {
        ESP_LOGE(TAG_INF, "Failed to allocate hidden state buffer in PSRAM");
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
        ESP_LOGE(TAG_INF, "Expected input tensors not found — check player_1_actor.info");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG_INF, "Actor model loaded (%d bytes)",
             static_cast<int>(_player_1_actor_espdl_end - _player_1_actor_espdl_start));
    return ESP_OK;
}

static void actor_reset_hidden() {
    if (s_hidden_buf) memset(s_hidden_buf, 0, HIDDEN_DIM);
}

// Returns selected action index (0–5), or -1 on error.
static int actor_run(const int8_t *grid_q8, const int8_t *scalars_q8) {
    dl::TensorBase *grid_t   = s_model->get_input("input.1");
    dl::TensorBase *scalar_t = s_model->get_input("onnx::Gemm_1");
    dl::TensorBase *hidden_t = s_model->get_input("onnx::Gemm_2");

    static int8_t grid_hwc[GRID_SIZE];
    chw_to_hwc(grid_q8, grid_hwc, GRID_C, GRID_H, GRID_W);
    memcpy(grid_t->data, grid_hwc, GRID_SIZE);
    memcpy(scalar_t->data, scalars_q8,  SCALAR_DIM);
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

    if (new_hidden_data) {
        memcpy(s_hidden_buf, new_hidden_data, HIDDEN_DIM);
    } else {
        ESP_LOGW(TAG_INF, "Hidden state output not found — GRU state not updated");
    }

    int best = 0;
    for (int i = 1; i < ACTION_DIM; i++) {
        if (logits_data[i] > logits_data[best]) best = i;
    }

    ESP_LOGD(TAG_INF, "logits=[%d,%d,%d,%d,%d,%d] action=%d",
             logits_data[0], logits_data[1], logits_data[2],
             logits_data[3], logits_data[4], logits_data[5], best);

    return best;
}

// ── Main task ────────────────────────────────────────────────────────────────

static const char *TAG = "main";

// Zero-initialised (static storage).
// TODO: replace with values extracted from the camera frame once the
//       vision processing pipeline (camera → occupancy grid + scalars) is built.
static int8_t g_grid_q8   [GRID_SIZE]  __attribute__((aligned(16)));
static int8_t g_scalars_q8[SCALAR_DIM] __attribute__((aligned(16)));

static void inference_task(void *) {
    ESP_LOGI(TAG, "Inference task started");

    while (true) {
        // 1. Capture frame from onboard camera
        camera_fb_t *fb = camera_capture();
        if (!fb) {
            ESP_LOGW(TAG, "camera_capture failed — retrying");
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        // 2. Feed raw camera bytes into the grid input
        // The frame is 128×128 = 16 384 grayscale bytes; GRID_SIZE is 1 331.
        // Copying the first GRID_SIZE bytes is semantically wrong but lets the
        // full pipeline run end-to-end until proper observation extraction is added.
        memcpy(g_grid_q8, fb->buf, GRID_SIZE);
        camera_return_fb(fb);

        // 3. Run actor inference
        int action = actor_run(g_grid_q8, g_scalars_q8);
        if (action < 0) {
            ESP_LOGE(TAG, "Inference error — skipping step");
            continue;
        }

        ESP_LOGI(TAG, "action=%d", action);

        // 4. Send action to Arduino Nano BLE 33 Rev2 over SPI
        spi_comm_send_action(static_cast<uint8_t>(action));
    }
}

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "=== Overcooked Actor Inference — XIAO ESP32-S3 Sense ===");

    ESP_ERROR_CHECK(camera_init());
    ESP_LOGI(TAG, "Camera ready (128x128 grayscale)");

    ESP_ERROR_CHECK(spi_comm_init());

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

    ESP_LOGI(TAG, "Startup complete — inference task running");
}
