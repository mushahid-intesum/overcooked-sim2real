#include "spi_comm.h"

#include <cstring>
#include "esp_log.h"
#include "esp_check.h" 
#include "driver/spi_master.h"
#include "driver/gpio.h"

static const char *TAG = "spi_comm";

static spi_device_handle_t s_spi = nullptr;

esp_err_t spi_comm_init() {
    spi_bus_config_t bus_cfg = {
        .mosi_io_num     = SPI_PIN_MOSI,
        .miso_io_num     = SPI_PIN_MISO,
        .sclk_io_num     = SPI_PIN_SCK,
        .quadwp_io_num   = -1,
        .quadhd_io_num   = -1,
        .max_transfer_sz = 16,
    };
    ESP_RETURN_ON_ERROR(
        spi_bus_initialize(SPI2_HOST, &bus_cfg, SPI_DMA_CH_AUTO),
        TAG, "spi_bus_initialize failed");

    spi_device_interface_config_t dev_cfg = {
        .mode           = 0,
        .clock_speed_hz = 1000000,
        .spics_io_num   = SPI_PIN_CS,
        .queue_size     = 4,
    };
    ESP_RETURN_ON_ERROR(
        spi_bus_add_device(SPI2_HOST, &dev_cfg, &s_spi),
        TAG, "spi_bus_add_device failed");

    ESP_LOGI(TAG, "SPI master ready (MOSI=%d SCK=%d CS=%d @ 1 MHz)",
             SPI_PIN_MOSI, SPI_PIN_SCK, SPI_PIN_CS);
    return ESP_OK;
}

esp_err_t spi_comm_send_action(uint8_t action) {
    spi_transaction_t t = {};
    t.length    = 8;          // bits
    t.tx_buffer = &action;
    esp_err_t err = spi_device_transmit(s_spi, &t);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "spi_device_transmit failed: %s", esp_err_to_name(err));
    }
    return err;
}
