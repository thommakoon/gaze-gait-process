// USB Serial dual IMU: two identical ICM-20948 sensors on Qwiic mux (LF + RF).
// Output protocol is intentionally kept identical to sketch_wifi_dual ICM mode:
// - STATE:* lines
// - GRAVITY_LF/RF and GYRO_BIAS_LF/RF lines
// - 20-field data lines: seq,time_us,LF(9),RF(9)

#include <Wire.h>
#include <SparkFun_I2C_Mux_Arduino_Library.h>
#include <Adafruit_ICM20X.h>
#include <Adafruit_ICM20948.h>
#include <Adafruit_Sensor.h>

// Hardware / timing config
#define IMU_I2C_ADDR       0x69
#define BTN_PIN            A0
#define DEBOUNCE_MS        50
#define CAL_SAMPLES        500
#define SAMPLE_INTERVAL_US 5000

enum State { WAITING, CALIBRATING, READY, STREAMING };
State state = WAITING;

QWIICMUX myMux;
Adafruit_ICM20948 icm1;  // LF on mux port 0
Adafruit_ICM20948 icm2;  // RF on mux port 1

float bias1Ax = 0, bias1Ay = 0, bias1Az = 0;
float bias1Gx = 0, bias1Gy = 0, bias1Gz = 0;
float sum1Ax = 0, sum1Ay = 0, sum1Az = 0;
float sum1Gx = 0, sum1Gy = 0, sum1Gz = 0;

float bias2Ax = 0, bias2Ay = 0, bias2Az = 0;
float bias2Gx = 0, bias2Gy = 0, bias2Gz = 0;
float sum2Ax = 0, sum2Ay = 0, sum2Az = 0;
float sum2Gx = 0, sum2Gy = 0, sum2Gz = 0;

static unsigned long lastSampleTime = 0;
static uint32_t seq = 0;
static uint32_t calCount = 0;

bool btnPressed = false;
bool btnLastReading = HIGH;
unsigned long btnStableSince = 0;

char buf[640];

void sendLine(const char* line) {
  Serial.println(line);
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

void configureICM(Adafruit_ICM20948& icm) {
  icm.setAccelRange(ICM20948_ACCEL_RANGE_16_G);
  icm.setGyroRange(ICM20948_GYRO_RANGE_2000_DPS);
  icm.setAccelRateDivisor(0);
  icm.setGyroRateDivisor(0);
}

void enterWaiting() {
  state = WAITING;
  sendLine("STATE:WAITING");
}

void enterCalibrating() {
  state = CALIBRATING;
  calCount = 0;
  seq = 0;
  lastSampleTime = micros();
  sum1Ax = sum1Ay = sum1Az = 0;
  sum1Gx = sum1Gy = sum1Gz = 0;
  sum2Ax = sum2Ay = sum2Az = 0;
  sum2Gx = sum2Gy = sum2Gz = 0;
  sendLine("STATE:CALIBRATING");
}

void enterReady() {
  if (calCount > 0) {
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
  }

  snprintf(buf, sizeof(buf), "GRAVITY_LF:%.6f,%.6f,%.6f", bias1Ax, bias1Ay, bias1Az);
  sendLine(buf);
  snprintf(buf, sizeof(buf), "GYRO_BIAS_LF:%.6f,%.6f,%.6f", bias1Gx, bias1Gy, bias1Gz);
  sendLine(buf);
  snprintf(buf, sizeof(buf), "GRAVITY_RF:%.6f,%.6f,%.6f", bias2Ax, bias2Ay, bias2Az);
  sendLine(buf);
  snprintf(buf, sizeof(buf), "GYRO_BIAS_RF:%.6f,%.6f,%.6f", bias2Gx, bias2Gy, bias2Gz);
  sendLine(buf);

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
  // USB-first firmware: avoid blocking forever on `while(!Serial)`.
  delay(50);

  pinMode(BTN_PIN, INPUT_PULLUP);
  Wire1.begin();

  Serial.println("Dual IMU USB (sketch_usb_dual) - ICM-20948");
  Serial.println("  LF = mux port 0, RF = mux port 1");

  if (!myMux.begin(0x70, Wire1)) {
    Serial.println("Mux not detected. Freezing...");
    while (1) delay(10);
  }
  Serial.println("Mux detected");

  myMux.setPort(0);
  if (!icm1.begin_I2C(IMU_I2C_ADDR, &Wire1)) {
    Serial.println("Failed to find ICM20948 on port 0 (LF)");
    while (1) delay(10);
  }
  configureICM(icm1);
  Serial.println("Port 0 ICM20948 (LF) initialized");

  myMux.setPort(1);
  if (!icm2.begin_I2C(IMU_I2C_ADDR, &Wire1)) {
    Serial.println("Failed to find ICM20948 on port 1 (RF)");
    while (1) delay(10);
  }
  configureICM(icm2);
  Serial.println("Port 1 ICM20948 (RF) initialized");

  Serial.println();
  enterWaiting();
}

void loop() {
  bool btn = buttonJustPressed();
  unsigned long now = micros();

  switch (state) {
    case WAITING:
      if (btn) enterCalibrating();
      break;

    case CALIBRATING:
      if (now - lastSampleTime >= SAMPLE_INTERVAL_US) {
        lastSampleTime = now;

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

        seq++;
      }
      break;
  }
}
