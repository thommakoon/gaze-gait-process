// USB Serial dual IMU, HARDWARE-TIMER-DRIVEN variant of sketch_usb_dual.
//
// Differences vs sketch_usb_dual (the polled version):
//   * Sampling cadence comes from an ESP32-S2 hardware timer ISR at exactly
//     200 Hz (SAMPLE_PERIOD_US = 5000). loop() no longer schedules anything;
//     it only services the ISR's "sample due" flag.
//   * The ISR captures `micros()` at the timer instant, and that value is what
//     the CSV `time_us` column reports -- timestamps are timer-precise even
//     if the I2C/USB work in loop() is a millisecond or two behind.
//   * Magnetometer columns are dropped from the output (AK09916 caps at ~100
//     Hz and lin2025's IMU branch does not use mag). Row shape is 14 fields:
//       seq,time_us,LFax,LFay,LFaz,LFgx,LFgy,LFgz,RFax,RFay,RFaz,RFgx,RFgy,RFgz
//   * IMU internal ODR stays at ~1125 Hz (divisor 0). The timer reads at 200
//     Hz, so every read sees a sample that is at most ~0.9 ms old -- no
//     aliasing between the MCU schedule and the sensor schedule.
//
// No hardware wiring change vs sketch_usb_dual: same Qwiic mux + button on A0.
// In particular, no DRDY pin needs to be wired -- this version is pure
// software scheduling.
//
// Protocol shape (preserved for the host parser, only data-row width changes):
//   STATE:WAITING / STATE:CALIBRATING / STATE:CAL_DONE / STATE:READY / STATE:STREAMING
//   GRAVITY_LF:ax,ay,az    GYRO_BIAS_LF:gx,gy,gz
//   GRAVITY_RF:ax,ay,az    GYRO_BIAS_RF:gx,gy,gz
//
// Target: QT Py ESP32-S2 (Arduino-ESP32 v3.x). The v3.x timer API is used:
//   timerBegin(uint32_t freq) / timerAttachInterrupt(t, fn) /
//   timerAlarm(t, alarm_ticks, autoreload, reload_count).

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
#define LF_MUX_PORT        0
#define RF_MUX_PORT        1

// 200 Hz exact (5000 us per sample). Change this if you want a different rate.
#define SAMPLE_PERIOD_US   5000UL
// Timer base frequency: 1 MHz -> 1 tick = 1 us. Keeps alarm value == us value.
#define TIMER_TICK_HZ      1000000UL

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

static uint32_t seq = 0;
static uint32_t calCount = 0;

bool btnPressed = false;
bool btnLastReading = HIGH;
unsigned long btnStableSince = 0;

char buf[320];

// Timer ISR state.
hw_timer_t* sampleTimer = nullptr;
volatile bool     sampleDue        = false;
volatile uint32_t sampleDueUs      = 0;
volatile uint32_t timerOverrunCount = 0;  // ISR fired while previous flag still pending.

void IRAM_ATTR onSampleTimer() {
  if (sampleDue) {
    // Main loop did not consume the previous tick in time. Count it and
    // keep the older timestamp so seq/time stay in lockstep with elapsed
    // ticks (the next consumed row will reflect the dropped interval).
    timerOverrunCount++;
    return;
  }
  sampleDueUs = micros();
  sampleDue = true;
}

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
  // Oversample relative to the 200 Hz timer so every read gets a fresh sample
  // (internal ODR ~1125 Hz / ~1100 Hz with DLPF on, divisor 0).
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

// Consume one timer tick: copy the ISR timestamp, then read both IMUs.
// Returns true if a fresh tick was waiting; false if not.
bool consumeImuSample(uint32_t& tUsOut,
                      sensors_event_t& a1, sensors_event_t& g1,
                      sensors_event_t& a2, sensors_event_t& g2) {
  if (!sampleDue) return false;

  noInterrupts();
  uint32_t tUs = sampleDueUs;
  sampleDue = false;
  interrupts();

  sensors_event_t tEvt, mEvt;  // discarded (mag/temp not emitted)
  myMux.setPort(LF_MUX_PORT);
  icm1.getEvent(&a1, &g1, &tEvt, &mEvt);
  myMux.setPort(RF_MUX_PORT);
  icm2.getEvent(&a2, &g2, &tEvt, &mEvt);

  tUsOut = tUs;
  return true;
}

void setup() {
  Serial.begin(230400);
  // USB-first firmware: avoid blocking forever on `while(!Serial)`.
  delay(50);

  pinMode(BTN_PIN, INPUT_PULLUP);

  Wire1.begin();
  Wire1.setClock(400000);

  Serial.print("Dual IMU USB (sketch_usb_dual_timer) - ICM-20948, hw-timer ");
  Serial.print(1000000UL / SAMPLE_PERIOD_US);
  Serial.println(" Hz, no mag");
  Serial.println("  LF = mux port 0, RF = mux port 1");

  if (!myMux.begin(0x70, Wire1)) {
    Serial.println("Mux not detected. Freezing...");
    while (1) delay(10);
  }
  Serial.println("Mux detected");

  myMux.setPort(LF_MUX_PORT);
  if (!icm1.begin_I2C(IMU_I2C_ADDR, &Wire1)) {
    Serial.println("Failed to find ICM20948 on port 0 (LF)");
    while (1) delay(10);
  }
  configureICM(icm1);
  Serial.println("Port 0 ICM20948 (LF) initialized");

  myMux.setPort(RF_MUX_PORT);
  if (!icm2.begin_I2C(IMU_I2C_ADDR, &Wire1)) {
    Serial.println("Failed to find ICM20948 on port 1 (RF)");
    while (1) delay(10);
  }
  configureICM(icm2);
  Serial.println("Port 1 ICM20948 (RF) initialized");

  // Hardware timer: 1 us tick, autoreload at SAMPLE_PERIOD_US, runs forever.
  // The flag-consumer in loop() simply ignores ticks outside CALIBRATING /
  // STREAMING, so we leave the timer running across all states.
  sampleTimer = timerBegin(TIMER_TICK_HZ);
  if (sampleTimer == nullptr) {
    Serial.println("timerBegin failed. Freezing...");
    while (1) delay(10);
  }
  timerAttachInterrupt(sampleTimer, &onSampleTimer);
  // autoreload = true, reload_count = 0 -> reload forever
  timerAlarm(sampleTimer, SAMPLE_PERIOD_US, true, 0);

  Serial.println();
  enterWaiting();
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
