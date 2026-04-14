#include "spi_comm.h"
#include "driver/uart.h"
#include "esp_log.h"

#define UART_PORT   UART_NUM_1
#define UART_TX_PIN 3
#define UART_RX_PIN 4
#define UART_BAUD   115200

static const char *TAG = "uart_comm";

esp_err_t spi_comm_init() {
    uart_config_t cfg = {};
    cfg.baud_rate  = UART_BAUD;
    cfg.data_bits  = UART_DATA_8_BITS;
    cfg.parity     = UART_PARITY_DISABLE;
    cfg.stop_bits  = UART_STOP_BITS_1;
    cfg.flow_ctrl  = UART_HW_FLOWCTRL_DISABLE;

    ESP_ERROR_CHECK(uart_param_config(UART_PORT, &cfg));
    ESP_ERROR_CHECK(uart_set_pin(UART_PORT, UART_TX_PIN, UART_RX_PIN,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    ESP_ERROR_CHECK(uart_driver_install(UART_PORT, 256, 0, 0, NULL, 0));

    ESP_LOGI(TAG, "UART ready (TX=%d @ %d baud)", UART_TX_PIN, UART_BAUD);
    return ESP_OK;
}

esp_err_t spi_comm_send_action(uint8_t action) {
    int sent = uart_write_bytes(UART_PORT, (const char *)&action, 1);
    if (sent != 1) {
        ESP_LOGE(TAG, "uart_write_bytes failed");
        return ESP_FAIL;
    }
    return ESP_OK;
}