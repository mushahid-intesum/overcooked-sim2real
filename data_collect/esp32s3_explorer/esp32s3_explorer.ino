/*
 * ESP32-S3 Overcooked Data Collector
 * =====================================
 * Board  : ESP32-S3-EYE  (no PSRAM)
 * Server : FastAPI  (overcooked_server.py)
 *
 * Key fixes vs previous version:
 *  - httpPostImage() streams directly over WiFiClient — zero ps_malloc/malloc copy
 *  - Frame size set to FRAMESIZE_CIF (400×296, ~10-15 KB JPEG) so frames fit in DRAM
 *    alongside WiFi buffers. Bump to FRAMESIZE_VGA if you add PSRAM later.
 *
 * Flow:
 *  1. GET /layout         → fetch active layout name
 *  2. Scan N/E/S/W × 5   → POST /upload per image (streamed, no copy)
 *  3. POST /endscan       → get next move: N|E|S|W|DONE
 *  4. Drive, increment step, repeat from 2
 */

#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ═══════════════════════════════════════════════════════════════════════════════
//  CAMERA PINS  —  ESP32S3_EYE
// ═══════════════════════════════════════════════════════════════════════════════
#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    15
#define SIOD_GPIO_NUM     4
#define SIOC_GPIO_NUM     5
#define Y2_GPIO_NUM      11
#define Y3_GPIO_NUM       9
#define Y4_GPIO_NUM       8
#define Y5_GPIO_NUM      10
#define Y6_GPIO_NUM      12
#define Y7_GPIO_NUM      18
#define Y8_GPIO_NUM      17
#define Y9_GPIO_NUM      16
#define VSYNC_GPIO_NUM    6
#define HREF_GPIO_NUM     7
#define PCLK_GPIO_NUM    13

// ─── WiFi ────────────────────────────────────────────────────────────────────
const char* WIFI_SSID     = "Network";
const char* WIFI_PASSWORD = "Excels!or";

// ─── Server ──────────────────────────────────────────────────────────────────
const char* SERVER_HOST = "192.168.68.107";
const int   SERVER_PORT = 8000;

// ─── Motor pins ───────────────────────────────────────────────────────────────
#define MOTOR_L_IN1  35
#define MOTOR_L_IN2  36
#define MOTOR_L_EN   40
#define MOTOR_R_IN3  37
#define MOTOR_R_IN4  38
#define MOTOR_R_EN   39
#define MOTOR_SPEED  150

// ─── LEDC ────────────────────────────────────────────────────────────────────
#define LEDC_FREQ   5000
#define LEDC_RES    8

// ─── Timing ──────────────────────────────────────────────────────────────────
#define IMAGES_PER_DIRECTION  5
#define CAPTURE_INTERVAL_MS   600   // gap between shots (camera AEC settle)
#define STEP_DRIVE_MS         600   // forward drive duration
#define ROTATE_90_MS          250   // time for one 90° spin
#define POST_ROTATE_MS        800   // settle time after each rotation before shooting
#define POST_DRIVE_MS        1000   // settle time after driving before next scan

// ─── State ───────────────────────────────────────────────────────────────────
enum Direction { DIR_N = 0, DIR_E = 1, DIR_S = 2, DIR_W = 3 };
const char DIR_CHARS[] = {'N', 'E', 'S', 'W'};

Direction currentHeading = DIR_N;
int       currentStep    = 0;
String    currentLayout  = "unknown";
bool      missionDone    = false;

// ═══════════════════════════════════════════════════════════════════════════════
//  MOTORS
// ═══════════════════════════════════════════════════════════════════════════════
void motorsInit() {
    pinMode(MOTOR_L_IN1, OUTPUT); pinMode(MOTOR_L_IN2, OUTPUT);
    pinMode(MOTOR_R_IN3, OUTPUT); pinMode(MOTOR_R_IN4, OUTPUT);
    ledcAttach(MOTOR_L_EN, LEDC_FREQ, LEDC_RES);
    ledcAttach(MOTOR_R_EN, LEDC_FREQ, LEDC_RES);
}
static inline void setPWM(uint8_t spd) {
    ledcWrite(MOTOR_L_EN, spd);
    ledcWrite(MOTOR_R_EN, spd);
}
void motorsStop() {
    digitalWrite(MOTOR_L_IN1, LOW); digitalWrite(MOTOR_L_IN2, LOW);
    digitalWrite(MOTOR_R_IN3, LOW); digitalWrite(MOTOR_R_IN4, LOW);
    setPWM(0);
}
void motorsForward(int spd = MOTOR_SPEED) {
    digitalWrite(MOTOR_L_IN1, HIGH); digitalWrite(MOTOR_L_IN2, LOW);
    digitalWrite(MOTOR_R_IN3, HIGH); digitalWrite(MOTOR_R_IN4, LOW);
    setPWM(spd);
}
void motorsSpinRight(int spd = MOTOR_SPEED) {
    digitalWrite(MOTOR_L_IN1, HIGH); digitalWrite(MOTOR_L_IN2, LOW);
    digitalWrite(MOTOR_R_IN3, LOW);  digitalWrite(MOTOR_R_IN4, HIGH);
    setPWM(spd);
}
void motorsSpinLeft(int spd = MOTOR_SPEED) {
    digitalWrite(MOTOR_L_IN1, LOW);  digitalWrite(MOTOR_L_IN2, HIGH);
    digitalWrite(MOTOR_R_IN3, HIGH); digitalWrite(MOTOR_R_IN4, LOW);
    setPWM(spd);
}
void rotateToHeading(Direction target) {
    int delta = ((int)target - (int)currentHeading + 4) % 4;
    if (delta == 0) return;
    if (delta <= 2) {
        for (int i = 0; i < delta; i++) {
            motorsSpinRight(); delay(ROTATE_90_MS);
            motorsStop();      delay(100);
        }
    } else {
        motorsSpinLeft(); delay(ROTATE_90_MS);
        motorsStop();     delay(100);
    }
    currentHeading = target;
}
void driveForward() {
    motorsForward(); delay(STEP_DRIVE_MS); motorsStop();
    delay(POST_DRIVE_MS);   // let robot fully stop before next scan begins
}

// ═══════════════════════════════════════════════════════════════════════════════
//  CAMERA
//  No PSRAM → FRAMESIZE_CIF (400×296).
//  JPEG quality 10 produces ~10-15 KB frames — fits comfortably in DRAM heap
//  (the ESP32-S3 has ~320 KB usable DRAM; WiFi + camera driver use ~200 KB,
//   leaving ~120 KB free — more than enough for a 15 KB frame buffer).
// ═══════════════════════════════════════════════════════════════════════════════
bool initCamera() {
    camera_config_t cfg;
    cfg.ledc_channel = LEDC_CHANNEL_0;
    cfg.ledc_timer   = LEDC_TIMER_0;
    cfg.pin_d0 = Y2_GPIO_NUM; cfg.pin_d1 = Y3_GPIO_NUM;
    cfg.pin_d2 = Y4_GPIO_NUM; cfg.pin_d3 = Y5_GPIO_NUM;
    cfg.pin_d4 = Y6_GPIO_NUM; cfg.pin_d5 = Y7_GPIO_NUM;
    cfg.pin_d6 = Y8_GPIO_NUM; cfg.pin_d7 = Y9_GPIO_NUM;
    cfg.pin_xclk     = XCLK_GPIO_NUM;
    cfg.pin_pclk     = PCLK_GPIO_NUM;
    cfg.pin_vsync    = VSYNC_GPIO_NUM;
    cfg.pin_href     = HREF_GPIO_NUM;
    cfg.pin_sccb_sda = SIOD_GPIO_NUM;
    cfg.pin_sccb_scl = SIOC_GPIO_NUM;
    cfg.pin_pwdn     = PWDN_GPIO_NUM;
    cfg.pin_reset    = RESET_GPIO_NUM;
    cfg.xclk_freq_hz = 20000000;
    cfg.pixel_format = PIXFORMAT_JPEG;
    cfg.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
    cfg.fb_location  = CAMERA_FB_IN_DRAM;
    cfg.frame_size   = FRAMESIZE_CIF;   // 400×296 — ~10-15 KB per JPEG
    cfg.jpeg_quality = 10;              // 0=best quality, 63=worst
    cfg.fb_count     = 1;

    esp_err_t err = esp_camera_init(&cfg);
    if (err != ESP_OK) {
        Serial.printf("[CAM] Init failed: 0x%x\n", err);
        return false;
    }
    sensor_t* s = esp_camera_sensor_get();
    if (!s) { Serial.println("[CAM] sensor_get NULL"); return false; }
    s->set_vflip(s, 1);

    // Warm up AEC/AWB at QVGA, then restore CIF
    s->set_framesize(s, FRAMESIZE_QVGA);
    delay(500);
    s->set_framesize(s, FRAMESIZE_CIF);
    Serial.println("[CAM] Ready (CIF, DRAM-only).");
    return true;
}

camera_fb_t* captureFrame() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) Serial.println("[CAM] Capture failed");
    return fb;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  HTTP — GET /layout
// ═══════════════════════════════════════════════════════════════════════════════
String httpGetLayout() {
    HTTPClient http;
    http.begin(String("http://") + SERVER_HOST + ":" + SERVER_PORT + "/layout");
    http.setTimeout(10000);
    int code = http.GET();
    if (code != 200) {
        Serial.printf("[HTTP] /layout → %d\n", code);
        http.end();
        return "unknown";
    }
    String resp = http.getString();
    http.end();
    JsonDocument doc;
    if (deserializeJson(doc, resp) || !doc["layout"].is<const char*>()) {
        Serial.println("[HTTP] /layout bad JSON");
        return "unknown";
    }
    return String(doc["layout"].as<const char*>());
}

bool httpPostImage(char dirChar, uint8_t shotIdx,
                   const uint8_t* buf, size_t len) {
    if (!buf || len == 0) return false;

    const char* BOUNDARY = "ESP32B";

    // Build preamble string (all text fields + file part header)
    String pre;
    pre.reserve(400);
    pre += "--"; pre += BOUNDARY;
    pre += "\r\nContent-Disposition: form-data; name=\"layout\"\r\n\r\n";
    pre += currentLayout; pre += "\r\n";

    pre += "--"; pre += BOUNDARY;
    pre += "\r\nContent-Disposition: form-data; name=\"step\"\r\n\r\n";
    pre += currentStep; pre += "\r\n";

    pre += "--"; pre += BOUNDARY;
    pre += "\r\nContent-Disposition: form-data; name=\"dir\"\r\n\r\n";
    pre += dirChar; pre += "\r\n";

    pre += "--"; pre += BOUNDARY;
    pre += "\r\nContent-Disposition: form-data; name=\"shot\"\r\n\r\n";
    pre += shotIdx; pre += "\r\n";

    pre += "--"; pre += BOUNDARY; pre += "\r\n";
    pre += "Content-Disposition: form-data; name=\"file\"; filename=\"shot.jpg\"\r\n";
    pre += "Content-Type: image/jpeg\r\n\r\n";

    const char* tail    = "\r\n--ESP32B--\r\n";
    size_t      tailLen = strlen(tail);
    size_t      bodyLen = pre.length() + len + tailLen;

    // Open TCP connection
    WiFiClient client;
    client.setTimeout(20);
    if (!client.connect(SERVER_HOST, SERVER_PORT)) {
        Serial.println("[HTTP] TCP connect failed");
        return false;
    }

    // Write HTTP request headers
    client.printf("POST /upload HTTP/1.1\r\n");
    client.printf("Host: %s:%d\r\n", SERVER_HOST, SERVER_PORT);
    client.printf("Content-Type: multipart/form-data; boundary=%s\r\n", BOUNDARY);
    client.printf("Content-Length: %u\r\n", bodyLen);
    client.printf("Connection: close\r\n\r\n");

    // Write body: preamble → JPEG (direct, no copy) → tail
    client.write((const uint8_t*)pre.c_str(), pre.length());
    client.write(buf, len);
    client.write((const uint8_t*)tail, tailLen);
    client.flush();

    // Read status line only
    String statusLine = client.readStringUntil('\n');
    client.stop();

    int sp   = statusLine.indexOf(' ');
    int code = (sp >= 0) ? statusLine.substring(sp + 1, sp + 4).toInt() : 0;
    if (code == 200) return true;
    Serial.printf("[HTTP] /upload → %d\n", code);
    return false;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  HTTP — POST /endscan
// ═══════════════════════════════════════════════════════════════════════════════
String httpEndScan() {
    HTTPClient http;
    http.begin(String("http://") + SERVER_HOST + ":" + SERVER_PORT + "/endscan");
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(30000);
    String body = "{\"layout\":\"" + currentLayout + "\",\"step\":" + String(currentStep) + "}";
    int code = http.POST(body);
    if (code != 200) {
        Serial.printf("[HTTP] /endscan → %d\n", code);
        http.end();
        return "N";
    }
    String resp = http.getString();
    http.end();
    JsonDocument doc;
    if (deserializeJson(doc, resp) || !doc["move"].is<const char*>()) {
        Serial.println("[HTTP] /endscan bad JSON");
        return "N";
    }
    String cmd = String(doc["move"].as<const char*>());
    Serial.printf("[HTTP] /endscan → move=%s\n", cmd.c_str());
    return cmd;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SCAN
// ═══════════════════════════════════════════════════════════════════════════════
void scanAllDirections() {
    Direction scanOrder[] = {DIR_N, DIR_E, DIR_S, DIR_W};
    for (int d = 0; d < 4; d++) {
        Direction dir = scanOrder[d];
        char      dc  = DIR_CHARS[dir];
        Serial.printf("[SCAN] Facing %c\n", dc);
        rotateToHeading(dir);
        delay(POST_ROTATE_MS);   // wait for vibration to die & AEC/AWB to re-settle
        for (int i = 0; i < IMAGES_PER_DIRECTION; i++) {
            camera_fb_t* fb = captureFrame();
            if (fb) {
                Serial.printf("[SCAN] step=%d dir=%c shot=%d  %u B\n",
                              currentStep, dc, i, fb->len);
                bool ok = httpPostImage(dc, i, fb->buf, fb->len);
                esp_camera_fb_return(fb);
            }
            delay(CAPTURE_INTERVAL_MS);
        }
    }
    rotateToHeading(DIR_N);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SETUP & LOOP
// ═══════════════════════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    Serial.setDebugOutput(false);
    Serial.println("\n[BOOT] Overcooked Data Collector");

    motorsInit();
    motorsStop();

    if (!initCamera()) {
        Serial.println("[FATAL] Camera init failed.");
        while (true) delay(1000);
    }

    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    WiFi.setSleep(false);
    Serial.print("[WiFi] Connecting");
    while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
    Serial.printf("\n[WiFi] IP: %s\n", WiFi.localIP().toString().c_str());

    currentLayout = httpGetLayout();
    Serial.printf("[LAYOUT] %s\n", currentLayout.c_str());

    currentHeading = DIR_N;
    currentStep    = 0;
    missionDone    = false;
    Serial.println("[READY]");
}

void loop() {
    if (missionDone) { delay(5000); return; }

    Serial.printf("\n── STEP %d ──\n", currentStep);
    scanAllDirections();

    String cmd = httpEndScan();

    if (cmd == "DONE") {
        Serial.println("[DONE] Mission complete!");
        motorsStop();
        missionDone = true;
        return;
    }

    Direction moveDir = DIR_N;
    if      (cmd == "N") moveDir = DIR_N;
    else if (cmd == "E") moveDir = DIR_E;
    else if (cmd == "S") moveDir = DIR_S;
    else if (cmd == "W") moveDir = DIR_W;
    else {
        Serial.printf("[WARN] Unknown cmd '%s'\n", cmd.c_str());
        currentStep++;
        return;
    }

    Serial.printf("[MOVE] %s\n", cmd.c_str());
    rotateToHeading(moveDir);
    driveForward();
    rotateToHeading(DIR_N);
    currentStep++;
    delay(500);
}
