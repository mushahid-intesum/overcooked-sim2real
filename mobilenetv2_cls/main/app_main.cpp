#include "esp_camera.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "dl_image_jpeg.hpp"
#include "imagenet_cls.hpp"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "mobilenetv2_cls";

// ─── Camera pin config (ESP32-S3-EYE) ────────────────────────────────────────
// If using a different board, replace these with your board's GPIO numbers.
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

#define CAPTURE_INTERVAL_MS  100   // 10 fps = 100 ms between frames

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

    config.pixel_format   = PIXFORMAT_JPEG;
    config.frame_size     = FRAMESIZE_240X240;
    config.jpeg_quality   = 12;
    config.fb_count       = 2;
    config.grab_mode      = CAMERA_GRAB_LATEST;

    // 🔴 REQUIRED NEW FIELDS
    config.fb_location    = CAMERA_FB_IN_PSRAM;
    config.sccb_i2c_port  = 0;  // usually 0 (I2C_NUM_0)

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
    }
    return err;
}

extern "C" void app_main(void)
{
    ESP_ERROR_CHECK(camera_init());

    // Instantiate model once — avoids repeated heap alloc/free overhead
    ImageNetCls *cls = new ImageNetCls();

    int64_t next_capture = esp_timer_get_time();  // microseconds

    while (true) {
        // ── Rate limiting: wait until next 100 ms slot ────────────────────
        int64_t now = esp_timer_get_time();
        int64_t wait_us = next_capture - now;
        if (wait_us > 0) {
            vTaskDelay(pdMS_TO_TICKS(wait_us / 1000));
        }
        next_capture += CAPTURE_INTERVAL_MS * 1000LL;  // schedule next slot

        // ── Capture frame ─────────────────────────────────────────────────
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "Camera capture failed");
            continue;
        }

        int64_t t_capture = esp_timer_get_time();

        // ── Decode JPEG → RGB888 ──────────────────────────────────────────
        dl::image::jpeg_img_t jpeg_img = {
            .data     = (void *)fb->buf,
            .data_len = fb->len,
        };
        auto img = dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

        // Return frame buffer to driver immediately after decode
        esp_camera_fb_return(fb);

        if (!img.data) {
            ESP_LOGE(TAG, "JPEG decode failed");
            continue;
        }

        // ── Run inference ─────────────────────────────────────────────────
        auto &results = cls->run(img);

        int64_t t_infer = esp_timer_get_time();

        vTaskDelay(1);

        // ── Log top results ───────────────────────────────────────────────
        ESP_LOGI(TAG, "── Inference (decode+infer: %lld ms) ──",
                 (t_infer - t_capture) / 1000);
        for (const auto &res : results) {
            ESP_LOGI(TAG, "  category: %-20s score: %.4f",
                     res.cat_name, res.score);
        }

        // ── Free decoded image buffer ─────────────────────────────────────
        heap_caps_free(img.data);
    }

    // Unreachable, but good practice
    delete cls;
}