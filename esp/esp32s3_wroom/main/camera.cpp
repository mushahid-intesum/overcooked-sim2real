#include "camera.h"
#include "esp_log.h"

static const char *TAG = "camera";

esp_err_t camera_init() {
    camera_config_t config = {};
    config.pin_pwdn     = CAM_PIN_PWDN;
    config.pin_reset    = CAM_PIN_RESET;
    config.pin_xclk     = CAM_PIN_XCLK;
    config.pin_sccb_sda = CAM_PIN_SIOD;
    config.pin_sccb_scl = CAM_PIN_SIOC;
    config.pin_d7       = CAM_PIN_D7;
    config.pin_d6       = CAM_PIN_D6;
    config.pin_d5       = CAM_PIN_D5;
    config.pin_d4       = CAM_PIN_D4;
    config.pin_d3       = CAM_PIN_D3;
    config.pin_d2       = CAM_PIN_D2;
    config.pin_d1       = CAM_PIN_D1;
    config.pin_d0       = CAM_PIN_D0;
    config.pin_vsync    = CAM_PIN_VSYNC;
    config.pin_href     = CAM_PIN_HREF;
    config.pin_pclk     = CAM_PIN_PCLK;
    config.xclk_freq_hz = 10000000;
    config.ledc_timer   = LEDC_TIMER_0;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.pixel_format = PIXFORMAT_GRAYSCALE;
    config.frame_size   = FRAMESIZE_QQVGA;
    config.jpeg_quality = 12;
    config.fb_count     = 2;
    config.fb_location  = CAMERA_FB_IN_DRAM;
    config.grab_mode    = CAMERA_GRAB_LATEST;
    config.sccb_i2c_port = -1;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK)
        ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
    else
        ESP_LOGI(TAG, "Camera init OK (128x128 grayscale, OV5640)");
    return err;
}

camera_fb_t *camera_capture()                  { return esp_camera_fb_get(); }
void         camera_return_fb(camera_fb_t *fb) { esp_camera_fb_return(fb); }