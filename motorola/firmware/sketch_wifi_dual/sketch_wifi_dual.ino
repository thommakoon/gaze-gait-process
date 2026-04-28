// WiFi UDP dual IMU: two identical ICM-20948 sensors on Qwiic mux (LF + RF).

#define USE_BNO08X 0

#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <SparkFun_I2C_Mux_Arduino_Library.h>

#if USE_BNO08X
#include "Adafruit_BNO08x.h"
#else
#include <Adafruit_ICM20X.h>
#include <Adafruit_ICM20948.h>
#include <Adafruit_Sensor.h>
#endif

// ── WiFi config ─────────────────────────────────────────────────────
const char* WIFI_SSID = "YourPhoneHotspot";
const char* WIFI_PASS = "YourPassword";
const uint16_t UDP_PORT = 9999;

// ── Hardware config ─────────────────────────────────────────────────
#if USE_BNO08X
#define BNO08X_RESET       -1
#define IMU_I2C_ADDR       0x4B
#else
#define IMU_I2C_ADDR       0x69
#endif
#define BTN_PIN            A0
#define DEBOUNCE_MS        50
#define CAL_SAMPLES        500
#define SAMPLE_INTERVAL_US 5000

// ── State machine ───────────────────────────────────────────────────
enum State { WAITING, CALIBRATING, READY, STREAMING };
State state = WAITING;

// ── Peripherals ─────────────────────────────────────────────────────
QWIICMUX myMux;

#if USE_BNO08X
Adafruit_BNO08x bnoLf(BNO08X_RESET);
Adafruit_BNO08x bnoRf(BNO08X_RESET);
sh2_SensorValue_t sensorValueLf;
sh2_SensorValue_t sensorValueRf;

struct BnoCache {
  float qw, qx, qy, qz;
  float ax, ay, az;
  float gx, gy, gz;
  float mx, my, mz;
  bool haveQ, haveA, haveG, haveM;
};

BnoCache cacheLf;
BnoCache cacheRf;

#else
Adafruit_ICM20948 icm1;
Adafruit_ICM20948 icm2;

float bias1Ax = 0, bias1Ay = 0, bias1Az = 0;
float bias1Gx = 0, bias1Gy = 0, bias1Gz = 0;
float sum1Ax = 0, sum1Ay = 0, sum1Az = 0;
float sum1Gx = 0, sum1Gy = 0, sum1Gz = 0;

float bias2Ax = 0, bias2Ay = 0, bias2Az = 0;
float bias2Gx = 0, bias2Gy = 0, bias2Gz = 0;
float sum2Ax = 0, sum2Ay = 0, sum2Az = 0;
float sum2Gx = 0, sum2Gy = 0, sum2Gz = 0;
#endif

WiFiUDP udp;
IPAddress phoneIP;
bool wifiConnected = false;

static unsigned long lastSampleTime = 0;
static uint32_t seq = 0;
static uint32_t calCount = 0;

bool btnPressed = false;
bool btnLastReading = HIGH;
unsigned long btnStableSince = 0;

char buf[640];

void sendLine(const char* line) {
  Serial.println(line);
  if (wifiConnected) {
    udp.beginPacket(phoneIP, UDP_PORT);
    udp.print(line);
    udp.endPacket();
  }
}

bool buttonJustPressed() {
  bool reading = digitalRead(BTN_PIN);
  if (reading != btnLastReading) {
    btnStableSince = millis();
  }
  btnLastReading = reading;

  if ((millis() - btnStableSince) >= DEBOUNCE_MS) {
    if (reading == LOW && !btnPressed) {
      btnPressed = true;
      return true;
    }
    if (reading == HIGH) {
      btnPressed = false;
    }
  }
  return false;
}

#if USE_BNO08X

void clearBnoCache(BnoCache& c) {
  c.haveQ = c.haveA = c.haveG = c.haveM = false;
}

void drainBno(Adafruit_BNO08x& bno, sh2_SensorValue_t& sv, BnoCache& c) {
  while (bno.getSensorEvent(&sv)) {
    switch (sv.sensorId) {
      case SH2_ROTATION_VECTOR:
        c.qw = sv.un.rotationVector.real;
        c.qx = sv.un.rotationVector.i;
        c.qy = sv.un.rotationVector.j;
        c.qz = sv.un.rotationVector.k;
        c.haveQ = true;
        break;
      case SH2_ACCELEROMETER:
        c.ax = sv.un.accelerometer.x;
        c.ay = sv.un.accelerometer.y;
        c.az = sv.un.accelerometer.z;
        c.haveA = true;
        break;
      case SH2_GYROSCOPE_CALIBRATED:
        c.gx = sv.un.gyroscope.x;
        c.gy = sv.un.gyroscope.y;
        c.gz = sv.un.gyroscope.z;
        c.haveG = true;
        break;
      case SH2_MAGNETIC_FIELD_CALIBRATED:
        c.mx = sv.un.magneticField.x;
        c.my = sv.un.magneticField.y;
        c.mz = sv.un.magneticField.z;
        c.haveM = true;
        break;
      default:
        break;
    }
  }
}

bool configureBno(Adafruit_BNO08x& bno) {
  if (!bno.enableReport(SH2_ROTATION_VECTOR, SAMPLE_INTERVAL_US)) return false;
  if (!bno.enableReport(SH2_ACCELEROMETER, SAMPLE_INTERVAL_US)) return false;
  if (!bno.enableReport(SH2_GYROSCOPE_CALIBRATED, SAMPLE_INTERVAL_US)) return false;
  if (!bno.enableReport(SH2_MAGNETIC_FIELD_CALIBRATED, SAMPLE_INTERVAL_US)) return false;
  return true;
}

void appendCacheCsv(char* dest, size_t destSize, size_t& off, const BnoCache& c) {
  float qw = c.haveQ ? c.qw : 1.f;
  float qx = c.haveQ ? c.qx : 0.f;
  float qy = c.haveQ ? c.qy : 0.f;
  float qz = c.haveQ ? c.qz : 0.f;
  float ax = c.haveA ? c.ax : 0.f;
  float ay = c.haveA ? c.ay : 0.f;
  float az = c.haveA ? c.az : 0.f;
  float gx = c.haveG ? c.gx : 0.f;
  float gy = c.haveG ? c.gy : 0.f;
  float gz = c.haveG ? c.gz : 0.f;
  float mx = c.haveM ? c.mx : 0.f;
  float my = c.haveM ? c.my : 0.f;
  float mz = c.haveM ? c.mz : 0.f;
  off += snprintf(dest + off, destSize - off,
                  "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                  qw, qx, qy, qz,
                  ax, ay, az,
                  gx, gy, gz,
                  mx, my, mz);
}

void formatDualLineBno(uint32_t s, uint32_t t_us, const BnoCache& lf, const BnoCache& rf) {
  size_t n = snprintf(buf, sizeof(buf), "%lu,%lu,", (unsigned long)s, (unsigned long)t_us);
  appendCacheCsv(buf, sizeof(buf), n, lf);
  if (n < sizeof(buf) - 1) {
    buf[n++] = ',';
  }
  appendCacheCsv(buf, sizeof(buf), n, rf);
}

#else  // ICM20948

void configureICM(Adafruit_ICM20948& icm) {
  icm.setAccelRange(ICM20948_ACCEL_RANGE_16_G);
  icm.setGyroRange(ICM20948_GYRO_RANGE_2000_DPS);
  icm.setAccelRateDivisor(0);
  icm.setGyroRateDivisor(0);
}

#endif

// ── State transitions ───────────────────────────────────────────────

void enterWaiting() {
  state = WAITING;
  sendLine("STATE:WAITING");
}

void enterCalibrating() {
  state = CALIBRATING;
  calCount = 0;
  seq = 0;
  lastSampleTime = micros();
#if !USE_BNO08X
  sum1Ax = sum1Ay = sum1Az = 0;
  sum1Gx = sum1Gy = sum1Gz = 0;
  sum2Ax = sum2Ay = sum2Az = 0;
  sum2Gx = sum2Gy = sum2Gz = 0;
#endif
  sendLine("STATE:CALIBRATING");
}

void enterReady() {
#if !USE_BNO08X
  bias1Ax = sum1Ax / calCount;
  bias1Ay = sum1Ay / calCount;
  bias1Az = sum1Az / calCount;
  bias1Gx = sum1Gx / calCount;
  bias1Gy = sum1Gy / calCount;
  bias1Gz = sum1Gz / calCount;

  bias2Ax = sum2Ax / calCount;
  bias2Ay = sum2Ay / calCount;
  bias2Az = sum2Az / calCount;
  bias2Gx = sum2Gx / calCount;
  bias2Gy = sum2Gy / calCount;
  bias2Gz = sum2Gz / calCount;

  int n = snprintf(buf, sizeof(buf), "GRAVITY_LF:%.6f,%.6f,%.6f", bias1Ax, bias1Ay, bias1Az);
  sendLine(buf);
  snprintf(buf, sizeof(buf), "GYRO_BIAS_LF:%.6f,%.6f,%.6f", bias1Gx, bias1Gy, bias1Gz);
  sendLine(buf);
  snprintf(buf, sizeof(buf), "GRAVITY_RF:%.6f,%.6f,%.6f", bias2Ax, bias2Ay, bias2Az);
  sendLine(buf);
  snprintf(buf, sizeof(buf), "GYRO_BIAS_RF:%.6f,%.6f,%.6f", bias2Gx, bias2Gy, bias2Gz);
  sendLine(buf);
#endif
  state = READY;
  sendLine("STATE:CAL_DONE");
  sendLine("STATE:READY");
}

void enterStreaming() {
  seq = 0;
  state = STREAMING;
  sendLine("STATE:STREAMING");
}

// ── Setup ───────────────────────────────────────────────────────────

void setup() {
  Serial.begin(230400);
  while (!Serial) delay(10);

  pinMode(BTN_PIN, INPUT_PULLUP);

#if USE_BNO08X
  Serial.println("Dual IMU WiFi (sketch_wifi_dual) — BNO08x");
#else
  Serial.println("Dual IMU WiFi (sketch_wifi_dual) — ICM-20948");
#endif
  Serial.println("  LF = mux port 0, RF = mux port 1");

  Serial.printf("Connecting to WiFi: %s\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  unsigned long wifiStart = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - wifiStart) < 15000) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    wifiConnected = true;
    phoneIP = WiFi.gatewayIP();
    Serial.printf("WiFi connected! IP: %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("Phone (gateway) IP: %s\n", phoneIP.toString().c_str());
    Serial.printf("UDP target: %s:%d\n", phoneIP.toString().c_str(), UDP_PORT);
  } else {
    Serial.println("WiFi FAILED — running serial-only mode");
  }

  Wire1.begin();

  if (!myMux.begin(0x70, Wire1)) {
    Serial.println("Mux not detected. Freezing...");
    while (1) delay(10);
  }
  Serial.println("Mux detected");

#if USE_BNO08X
  myMux.setPort(0);
  if (!bnoLf.begin_I2C(IMU_I2C_ADDR, &Wire1)) {
    Serial.println("Failed to find BNO08x on port 0 (LF)");
    while (1) delay(10);
  }
  if (!configureBno(bnoLf)) {
    Serial.println("BNO LF report setup failed");
    while (1) delay(10);
  }
  Serial.println("Port 0 BNO08x (LF) initialized");

  myMux.setPort(1);
  if (!bnoRf.begin_I2C(IMU_I2C_ADDR, &Wire1)) {
    Serial.println("Failed to find BNO08x on port 1 (RF)");
    while (1) delay(10);
  }
  if (!configureBno(bnoRf)) {
    Serial.println("BNO RF report setup failed");
    while (1) delay(10);
  }
  Serial.println("Port 1 BNO08x (RF) initialized");

  clearBnoCache(cacheLf);
  clearBnoCache(cacheRf);
#else
  myMux.setPort(0);
  if (!icm1.begin_I2C(IMU_I2C_ADDR, &Wire1)) {
    Serial.println("Failed to find ICM20948 on port 0");
    while (1) delay(10);
  }
  configureICM(icm1);
  Serial.println("Port 0 ICM20948 (LF) initialized");

  myMux.setPort(1);
  if (!icm2.begin_I2C(IMU_I2C_ADDR, &Wire1)) {
    Serial.println("Failed to find ICM20948 on port 1");
    while (1) delay(10);
  }
  configureICM(icm2);
  Serial.println("Port 1 ICM20948 (RF) initialized");
#endif

  Serial.println();
  enterWaiting();
}

// ── Main loop ───────────────────────────────────────────────────────

void loop() {
  bool btn = buttonJustPressed();
  unsigned long now = micros();

#if USE_BNO08X
  myMux.setPort(0);
  if (bnoLf.wasReset()) {
    Serial.println("BNO LF reset — re-enabling reports");
    configureBno(bnoLf);
  }
  drainBno(bnoLf, sensorValueLf, cacheLf);

  myMux.setPort(1);
  if (bnoRf.wasReset()) {
    Serial.println("BNO RF reset — re-enabling reports");
    configureBno(bnoRf);
  }
  drainBno(bnoRf, sensorValueRf, cacheRf);
#endif

  switch (state) {

    case WAITING:
      if (btn) enterCalibrating();
      break;

    case CALIBRATING:
      if (now - lastSampleTime >= SAMPLE_INTERVAL_US) {
        lastSampleTime = now;

#if USE_BNO08X
        formatDualLineBno(seq, now, cacheLf, cacheRf);
        sendLine(buf);
#else
        sensors_event_t a1, g1, t1, m1;
        sensors_event_t a2, g2, t2, m2;
        myMux.setPort(0);
        icm1.getEvent(&a1, &g1, &t1, &m1);
        myMux.setPort(1);
        icm2.getEvent(&a2, &g2, &t2, &m2);

        sum1Ax += a1.acceleration.x;
        sum1Ay += a1.acceleration.y;
        sum1Az += a1.acceleration.z;
        sum1Gx += g1.gyro.x;
        sum1Gy += g1.gyro.y;
        sum1Gz += g1.gyro.z;

        sum2Ax += a2.acceleration.x;
        sum2Ay += a2.acceleration.y;
        sum2Az += a2.acceleration.z;
        sum2Gx += g2.gyro.x;
        sum2Gy += g2.gyro.y;
        sum2Gz += g2.gyro.z;

        int n = snprintf(buf, sizeof(buf), "%lu,%lu,", (unsigned long)seq, now);
        n += snprintf(buf + n, sizeof(buf) - n,
                      "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,",
                      a1.acceleration.x, a1.acceleration.y, a1.acceleration.z,
                      g1.gyro.x, g1.gyro.y, g1.gyro.z,
                      m1.magnetic.x, m1.magnetic.y, m1.magnetic.z);
        snprintf(buf + n, sizeof(buf) - n,
                 "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
                 a2.acceleration.x, a2.acceleration.y, a2.acceleration.z,
                 g2.gyro.x, g2.gyro.y, g2.gyro.z,
                 m2.magnetic.x, m2.magnetic.y, m2.magnetic.z);
        sendLine(buf);
#endif
        seq++;
        calCount++;

        if (calCount >= CAL_SAMPLES) {
          enterReady();
        }
      }
      break;

    case READY:
      if (btn) enterStreaming();
      break;

    case STREAMING:
      if (btn) {
        enterWaiting();
        break;
      }
      if (now - lastSampleTime >= SAMPLE_INTERVAL_US) {
        lastSampleTime = now;

#if USE_BNO08X
        formatDualLineBno(seq, now, cacheLf, cacheRf);
        sendLine(buf);
#else
        sensors_event_t a1, g1, t1, m1;
        sensors_event_t a2, g2, t2, m2;
        myMux.setPort(0);
        icm1.getEvent(&a1, &g1, &t1, &m1);
        myMux.setPort(1);
        icm2.getEvent(&a2, &g2, &t2, &m2);

        int n = snprintf(buf, sizeof(buf), "%lu,%lu,", (unsigned long)seq, now);
        n += snprintf(buf + n, sizeof(buf) - n,
                      "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,",
                      a1.acceleration.x, a1.acceleration.y, a1.acceleration.z,
                      g1.gyro.x - bias1Gx, g1.gyro.y - bias1Gy, g1.gyro.z - bias1Gz,
                      m1.magnetic.x, m1.magnetic.y, m1.magnetic.z);
        snprintf(buf + n, sizeof(buf) - n,
                 "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
                 a2.acceleration.x, a2.acceleration.y, a2.acceleration.z,
                 g2.gyro.x - bias2Gx, g2.gyro.y - bias2Gy, g2.gyro.z - bias2Gz,
                 m2.magnetic.x, m2.magnetic.y, m2.magnetic.z);
        sendLine(buf);
#endif
        seq++;
      }
      break;
  }
}
