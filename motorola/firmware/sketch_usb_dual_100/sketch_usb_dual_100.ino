// USB Serial dual IMU — same as sketch_usb_dual_timer but HARDWARE TIMER @ 100 Hz
// (SAMPLE_PERIOD_US = 10000). Use when 200 Hz risks timer/USB overrun; bench with
// sketch_usb_dual_i2c_bench_100 for margin checks.
//
// Same behavior as sketch_usb_dual_timer except cadence:
//   * 100 Hz (10 ms) hardware timer ISR; loop() only services sampleDue.
//   * ISR captures micros() -> CSV time_us (SampleTimeFine on Android).
//   * 14-field CSV, no mag: seq,time_us,LF(ax,ay,az,gx,gy,gz),RF(ax,ay,az,gx,gy,gz)
//   * IMU ODR divisor 0 (~1125 Hz) so each 10 ms read still sees a fresh sample.
//
// Protocol: STATE:*, GRAVITY_*, GYRO_BIAS_*, same button flow on A0.
//
// Note: CAL_SAMPLES=500 @ 100 Hz -> ~5 s calibration (vs ~2.5 s @ 200 Hz).
//
// Target: QT Py ESP32-S2 (Arduino-ESP32 v3.x timer API).

#include <Wire.h>
#include <SparkFun_I2C_Mux_Arduino_Library.h>
#include <Adafruit_ICM20X.h>
#include <Adafruit_ICM20948.h>
#include <Adafruit_Sensor.h>

#define IMU_I2C_ADDR       0x69
#define BTN_PIN            A0
#define DEBOUNCE_MS        50
#define CAL_SAMPLES        500
#define LF_MUX_PORT        3
#define RF_MUX_PORT        2
#define I2C_CLOCK_HZ       100000UL

// 100 Hz (10000 us per sample).
#define SAMPLE_PERIOD_US   10000UL
#define TIMER_TICK_HZ      1000000UL

// Adafruit begin_I2C() fails if magnetometer setup fails; we stream accel+gyro only.
class Adafruit_ICM20948_AgGyroOnly : public Adafruit_ICM20948 {
public:
  bool begin_I2C_AgGyroOnly(uint8_t i2c_address, TwoWire* wire, int32_t sensor_id = 0) {
    if (i2c_dev) {
      delete i2c_dev;
    }
    i2c_dev = new Adafruit_I2CDevice(i2c_address, wire);
    if (!i2c_dev->begin()) {
      return false;
    }
    return _init(sensor_id);
  }
};

enum State { WAITING, CALIBRATING, READY, STREAMING };
State state = WAITING;

QWIICMUX myMux;
Adafruit_ICM20948_AgGyroOnly icm1;
Adafruit_ICM20948_AgGyroOnly icm2;

float bias1Ax = 0, bias1Ay = 0, bias1Az = 0;
float bias1Gx = 0, bias1Gy = 0, bias1Gz = 0;
float sum1Ax = 0, sum1Ay = 0, sum1Az = 0;
float sum1Gx = 0, sum1Gy = 0, sum1Gz = 0;

float bias2Ax = 0, bias2Ay = 0, bias2Az = 0;
float bias2Gx = 0, bias2Gy = 0, bias2Gz = 0;
float sum2Ax = 0, sum2Ay = 0, sum2Az = 0;
float sum2Gx = 0, sum2Gy = 0, sum2Gz = 0;

static uint32_t seq = 0;
static uint32_t calCount = 0;

bool btnPressed = false;
bool btnLastReading = HIGH;
unsigned long btnStableSince = 0;

char buf[320];

hw_timer_t* sampleTimer = nullptr;
volatile bool     sampleDue = false;
volatile uint32_t sampleDueUs = 0;
volatile uint32_t timerOverrunCount = 0;

void IRAM_ATTR onSampleTimer() {
  if (sampleDue) {
    timerOverrunCount++;
    return;
  }
  sampleDueUs = micros();
  sampleDue = true;
}

void sendLine(const char* line) {
  Serial.println(line);
  Serial.flush();
}

void waitForUsbSerial() {
  unsigned long t0 = millis();
  while (!Serial && (millis() - t0) < 5000) {
    delay(10);
  }
}

// Hot-path mux switch: no ms delays. Old delay(2)+delay(10) ran twice per
// sample (LF+RF) and capped streaming at ~33 Hz instead of 100 Hz.
void selectMuxPort(uint8_t port) {
  myMux.setPort(port);
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

void clearSampleFlag() {
  noInterrupts();
  sampleDue = false;
  timerOverrunCount = 0;
  interrupts();
}

void enterWaiting() {
  state = WAITING;
  sendLine("STATE:WAITING");
}

void enterCalibrating() {
  state = CALIBRATING;
  calCount = 0;
  seq = 0;
  sum1Ax = sum1Ay = sum1Az = 0;
  sum1Gx = sum1Gy = sum1Gz = 0;
  sum2Ax = sum2Ay = sum2Az = 0;
  sum2Gx = sum2Gy = sum2Gz = 0;
  clearSampleFlag();
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
  clearSampleFlag();
  state = STREAMING;
  sendLine("STATE:STREAMING");
}

bool consumeImuSample(uint32_t& tUsOut,
                      sensors_event_t& a1, sensors_event_t& g1,
                      sensors_event_t& a2, sensors_event_t& g2) {
  if (!sampleDue) return false;

  noInterrupts();
  uint32_t tUs = sampleDueUs;
  sampleDue = false;
  interrupts();

  sensors_event_t tEvt, mEvt;
  selectMuxPort(LF_MUX_PORT);
  icm1.getEvent(&a1, &g1, &tEvt, &mEvt);
  selectMuxPort(RF_MUX_PORT);
  icm2.getEvent(&a2, &g2, &tEvt, &mEvt);

  tUsOut = tUs;
  return true;
}

void setup() {
  Serial.begin(230400);
  waitForUsbSerial();
  Serial.println();
  Serial.println("BOOT sketch_usb_dual_100");
  Serial.flush();

  pinMode(BTN_PIN, INPUT_PULLUP);

  Wire1.begin();
  Wire1.setClock(I2C_CLOCK_HZ);

  Serial.print("Dual IMU USB (sketch_usb_dual_100) - ICM-20948, hw-timer ");
  Serial.print(1000000UL / SAMPLE_PERIOD_US);
  Serial.println(" Hz, no mag");
  Serial.println("  LF = mux port 3, RF = mux port 2");

  if (!myMux.begin(0x70, Wire1)) {
    Serial.println("Mux not detected. Freezing...");
    while (1) delay(10);
  }
  Serial.println("Mux detected");

  selectMuxPort(LF_MUX_PORT);
  if (!icm1.begin_I2C_AgGyroOnly(IMU_I2C_ADDR, &Wire1)) {
    Serial.println("Failed to find ICM20948 on port 3 (LF)");
    while (1) delay(10);
  }
  configureICM(icm1);
  Serial.println("Port 3 ICM20948 (LF) initialized");

  selectMuxPort(RF_MUX_PORT);
  if (!icm2.begin_I2C_AgGyroOnly(IMU_I2C_ADDR, &Wire1)) {
    Serial.println("Failed to find ICM20948 on port 2 (RF)");
    while (1) delay(10);
  }
  configureICM(icm2);
  Serial.println("Port 2 ICM20948 (RF) initialized");

  sampleTimer = timerBegin(TIMER_TICK_HZ);
  if (sampleTimer == nullptr) {
    Serial.println("timerBegin failed. Freezing...");
    while (1) delay(10);
  }
  timerAttachInterrupt(sampleTimer, &onSampleTimer);
  timerAlarm(sampleTimer, SAMPLE_PERIOD_US, true, 0);

  Serial.println();
  enterWaiting();
  Serial.println("Press button on A0: 1x=calibrate, 2x=stream, 3x=stop");
  Serial.println("(No CSV until calibrating/streaming — unlike dual_plain)");
  Serial.flush();
}

void loop() {
  bool btn = buttonJustPressed();

  switch (state) {
    case WAITING:
      if (btn) enterCalibrating();
      break;

    case CALIBRATING: {
      uint32_t tUs;
      sensors_event_t a1, g1, a2, g2;
      if (!consumeImuSample(tUs, a1, g1, a2, g2)) break;

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

      int n = snprintf(buf, sizeof(buf), "%lu,%lu,",
                       (unsigned long)seq, (unsigned long)tUs);
      n += snprintf(buf + n, sizeof(buf) - n,
                    "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,",
                    a1.acceleration.x, a1.acceleration.y, a1.acceleration.z,
                    g1.gyro.x, g1.gyro.y, g1.gyro.z);
      snprintf(buf + n, sizeof(buf) - n,
               "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
               a2.acceleration.x, a2.acceleration.y, a2.acceleration.z,
               g2.gyro.x, g2.gyro.y, g2.gyro.z);
      sendLine(buf);

      seq++;
      calCount++;
      if (calCount >= CAL_SAMPLES) {
        enterReady();
      }
    } break;

    case READY:
      if (btn) enterStreaming();
      break;

    case STREAMING: {
      if (btn) {
        enterWaiting();
        break;
      }

      uint32_t tUs;
      sensors_event_t a1, g1, a2, g2;
      if (!consumeImuSample(tUs, a1, g1, a2, g2)) break;

      int n = snprintf(buf, sizeof(buf), "%lu,%lu,",
                       (unsigned long)seq, (unsigned long)tUs);
      n += snprintf(buf + n, sizeof(buf) - n,
                    "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,",
                    a1.acceleration.x, a1.acceleration.y, a1.acceleration.z,
                    g1.gyro.x - bias1Gx, g1.gyro.y - bias1Gy, g1.gyro.z - bias1Gz);
      snprintf(buf + n, sizeof(buf) - n,
               "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
               a2.acceleration.x, a2.acceleration.y, a2.acceleration.z,
               g2.gyro.x - bias2Gx, g2.gyro.y - bias2Gy, g2.gyro.z - bias2Gz);
      sendLine(buf);

      seq++;
    } break;
  }
}
