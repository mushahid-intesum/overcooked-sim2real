#pragma once
#include "esp_camera.h"
#include "esp_err.h"

// ESP32-S3-WROOM CAM board — OV5640
#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD    4     // SDA
#define CAM_PIN_SIOC    5     // SCL
#define CAM_PIN_D7      16    // Y9
#define CAM_PIN_D6      17    // Y8
#define CAM_PIN_D5      18    // Y7
#define CAM_PIN_D4      12    // Y6
#define CAM_PIN_D3      10    // Y5
#define CAM_PIN_D2      8     // Y4
#define CAM_PIN_D1      9     // Y3
#define CAM_PIN_D0      11    // Y2
#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK    13

esp_err_t    camera_init();
camera_fb_t *camera_capture();
void         camera_return_fb(camera_fb_t *fb);