// WiFi UDP IMU — single sensor (ICM-20948).
//
// Edit this line for your hardware:
//   USE_QWIIC_MUX  1 = SparkFun Qwiic mux at 0x70, sensor on port 0
//                  0 = no mux — IMU wired directly to Wire1

#define USE_BNO08X 0
#define USE_QWIIC_MUX 1

#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>

#if USE_QWIIC_MUX
#include <SparkFun_I2C_Mux_Arduino_Library.h>
#endif

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

enum State { WAITING, CALIBRATING, READY, STREAMING };
State state = WAITING;

#if USE_QWIIC_MUX
QWIICMUX myMux;
#endif

#if USE_BNO08X
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

struct BnoCache {
  float qw, qx, qy, qz;
  float ax, ay, az;
  float gx, gy, gz;
  float mx, my, mz;
  bool haveQ, haveA, haveG, haveM;
} bnoCache;

#else
Adafruit_ICM20948 icm1;

float biasAx = 0, biasAy = 0, biasAz = 0;
float biasGx = 0, biasGy = 0, biasGz = 0;
float sumAx = 0, sumAy = 0, sumAz = 0;
float sumGx = 0, sumGy = 0, sumGz = 0;
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

char buf[384];

#if USE_QWIIC_MUX
static void imuBusSelect() {
  myMux.setPort(0);
}
#else
static void imuBusSelect() {}
#endif

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

void formatSampleLineBno(uint32_t s, uint32_t t_us, const BnoCache& c) {
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
  snprintf(buf, sizeof(buf),
           "%lu,%lu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
           (unsigned long)s, (unsigned long)t_us,
           qw, qx, qy, qz,
           ax, ay, az,
           gx, gy, gz,
           mx, my, mz);
}

#else

void configureICM(Adafruit_ICM20948& icm) {
  icm.setAccelRange(ICM20948_ACCEL_RANGE_16_G);
  icm.setGyroRange(ICM20948_GYRO_RANGE_2000_DPS);
  icm.setAccelRateDivisor(0);
  icm.setGyroRateDivisor(0);
}

#endif

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
  sumAx = sumAy = sumAz = 0;
  sumGx = sumGy = sumGz = 0;
#endif
  sendLine("STATE:CALIBRATING");
}

void enterReady() {
#if !USE_BNO08X
  biasAx = sumAx / calCount;
  biasAy = sumAy / calCount;
  biasAz = sumAz / calCount;
  biasGx = sumGx / calCount;
  biasGy = sumGy / calCount;
  biasGz = sumGz / calCount;

  snprintf(buf, sizeof(buf), "GRAVITY:%.6f,%.6f,%.6f", biasAx, biasAy, biasAz);
  sendLine(buf);
  snprintf(buf, sizeof(buf), "GYRO_BIAS:%.6f,%.6f,%.6f", biasGx, biasGy, biasGz);
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

void setup() {
  Serial.begin(230400);
  while (!Serial) delay(10);

  pinMode(BTN_PIN, INPUT_PULLUP);

#if USE_BNO08X
  Serial.print("IMU WiFi (sketch_wifi) — BNO08x");
#else
  Serial.print("IMU WiFi (sketch_wifi) — ICM-20948");
#endif
#if USE_QWIIC_MUX
  Serial.println(", Qwiic mux port 0");
#else
  Serial.println(", direct Wire1 (no mux)");
#endif

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

#if USE_QWIIC_MUX
  if (!myMux.begin(0x70, Wire1)) {
    Serial.println("Mux not detected. Freezing...");
    while (1) delay(10);
  }
  Serial.println("Mux detected");
#endif

  imuBusSelect();
#if USE_BNO08X
  if (!bno08x.begin_I2C(IMU_I2C_ADDR, &Wire1)) {
    Serial.println("Failed to find BNO08x on port 0");
    while (1) delay(10);
  }
  if (!configureBno(bno08x)) {
    Serial.println("BNO report setup failed");
    while (1) delay(10);
  }
  clearBnoCache(bnoCache);
  Serial.println("Port 0 BNO08x initialized");
#else
  if (!icm1.begin_I2C(IMU_I2C_ADDR, &Wire1)) {
    Serial.println("Failed to find ICM20948 on port 0");
    while (1) delay(10);
  }
  configureICM(icm1);
  Serial.println("Port 0 ICM20948 initialized");
#endif

  Serial.println();
  enterWaiting();
}

void loop() {
  bool btn = buttonJustPressed();
  unsigned long now = micros();

#if USE_BNO08X
  imuBusSelect();
  if (bno08x.wasReset()) {
    Serial.println("BNO08x reset — re-enabling reports");
    configureBno(bno08x);
  }
  drainBno(bno08x, sensorValue, bnoCache);
#endif

  switch (state) {

    case WAITING:
      if (btn) enterCalibrating();
      break;

    case CALIBRATING:
      if (now - lastSampleTime >= SAMPLE_INTERVAL_US) {
        lastSampleTime = now;

#if USE_BNO08X
        formatSampleLineBno(seq, now, bnoCache);
        sendLine(buf);
#else
        sensors_event_t a, g, temp, m;
        imuBusSelect();
        icm1.getEvent(&a, &g, &temp, &m);

        sumAx += a.acceleration.x;
        sumAy += a.acceleration.y;
        sumAz += a.acceleration.z;
        sumGx += g.gyro.x;
        sumGy += g.gyro.y;
        sumGz += g.gyro.z;

        snprintf(buf, sizeof(buf),
                 "%lu,%lu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
                 (unsigned long)seq, now,
                 a.acceleration.x, a.acceleration.y, a.acceleration.z,
                 g.gyro.x, g.gyro.y, g.gyro.z,
                 m.magnetic.x, m.magnetic.y, m.magnetic.z);
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
        formatSampleLineBno(seq, now, bnoCache);
        sendLine(buf);
#else
        sensors_event_t a, g, temp, m;
        imuBusSelect();
        icm1.getEvent(&a, &g, &temp, &m);

        snprintf(buf, sizeof(buf),
                 "%lu,%lu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
                 (unsigned long)seq, now,
                 a.acceleration.x, a.acceleration.y, a.acceleration.z,
                 g.gyro.x - biasGx, g.gyro.y - biasGy, g.gyro.z - biasGz,
                 m.magnetic.x, m.magnetic.y, m.magnetic.z);
        sendLine(buf);
#endif
        seq++;
      }
      break;
  }
}
