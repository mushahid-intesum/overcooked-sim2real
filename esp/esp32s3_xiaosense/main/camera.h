#pragma once
#include "esp_camera.h"
#include "esp_err.h"

// ── XIAO ESP32-S3 Sense camera pins

#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK     10
#define CAM_PIN_SIOD     40    // SDA
#define CAM_PIN_SIOC     39    // SCL
#define CAM_PIN_D7       48    // Y9
#define CAM_PIN_D6       11    // Y8
#define CAM_PIN_D5       12    // Y7
#define CAM_PIN_D4       14    // Y6
#define CAM_PIN_D3       16    // Y5
#define CAM_PIN_D2       18    // Y4
#define CAM_PIN_D1       17    // Y3
#define CAM_PIN_D0       15    // Y2
#define CAM_PIN_VSYNC    38
#define CAM_PIN_HREF     47
#define CAM_PIN_PCLK     13

esp_err_t camera_init();
camera_fb_t* camera_capture();
void        camera_return_fb(camera_fb_t* fb);
