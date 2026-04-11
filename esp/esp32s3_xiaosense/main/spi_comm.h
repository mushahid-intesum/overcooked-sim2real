#pragma once
#include <cstdint>
#include "esp_err.h"

// ── ESP32-S3 → Arduino Nano BLE 33 Rev2 (SPI master) ────────────────────────
// The ESP32 is the master; the Arduino is the slave.
// Action byte is sent as a single 8-bit SPI transfer.
//
// XIAO ESP32-S3 pin mapping (all free from camera DVP interface):
//   MOSI : GPIO9  (D10)
//   MISO : GPIO8  (D9)   — not used, wired for completeness
//   SCK  : GPIO7  (D8)
//   CS   : GPIO4  (D3)

#define SPI_PIN_MOSI  9
#define SPI_PIN_MISO  8
#define SPI_PIN_SCK   7
#define SPI_PIN_CS    4

esp_err_t spi_comm_init();
esp_err_t spi_comm_send_action(uint8_t action);
