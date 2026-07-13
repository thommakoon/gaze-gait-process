// USB Serial dual IMU — sketch_usb_dual_100 + remote serial commands.
//
// Same as sketch_usb_dual_100:
//   * 100 Hz (10 ms) hardware timer ISR; loop() only services sampleDue.
//   * ISR captures micros() -> CSV time_us (SampleTimeFine on Android).
//   * 14-field CSV, no mag: seq,time_us,LF(6),RF(6)
//   * Button on A0 still works (same transitions as before).
//
// NEW: newline-framed serial commands (host = phone URP2026 or PC serial):
//   CMD:CALIBRATE   WAITING    -> CALIBRATING
//   CMD:START       READY      -> STREAMING
//   CMD:STOP        STREAMING  -> WAITING
//                   READY      -> WAITING   (abort without streaming)
//   CMD:STATUS      reprint STATE:* (and biases if past calib)
//   CMD:NEXT        same as one button press (WAITING->cal, READY->stream, STREAMING->wait)
//
// Replies (also newline-framed):
//   CMD:OK:<name>
//   CMD:ERR:<reason>
//   plus existing STATE:*, GRAVITY_*, GYRO_BIAS_* lines.
//
// Note: CAL_SAMPLES=500 @ 100 Hz -> ~5 s calibration.
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
#define LF_MUX_PORT        0
#define RF_MUX_PORT        1

#define SAMPLE_PERIOD_US   10000UL
#define TIMER_TICK_HZ      1000000UL

#define CMD_BUF_LEN        48

enum State { WAITING, CALIBRATING, READY, STREAMING };
State state = WAITING;

QWIICMUX myMux;
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

static uint32_t seq = 0;
static uint32_t calCount = 0;
static bool     haveBias = false;

bool btnPressed = false;
bool btnLastReading = HIGH;
unsigned long btnStableSince = 0;

char buf[320];
char cmdBuf[CMD_BUF_LEN];
uint8_t cmdLen = 0;

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

void emitBiases() {
  snprintf(buf, sizeof(buf), "GRAVITY_LF:%.6f,%.6f,%.6f", bias1Ax, bias1Ay, bias1Az);
  sendLine(buf);
  snprintf(buf, sizeof(buf), "GYRO_BIAS_LF:%.6f,%.6f,%.6f", bias1Gx, bias1Gy, bias1Gz);
  sendLine(buf);
  snprintf(buf, sizeof(buf), "GRAVITY_RF:%.6f,%.6f,%.6f", bias2Ax, bias2Ay, bias2Az);
  sendLine(buf);
  snprintf(buf, sizeof(buf), "GYRO_BIAS_RF:%.6f,%.6f,%.6f", bias2Gx, bias2Gy, bias2Gz);
  sendLine(buf);
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
    haveBias = true;
  }

  emitBiases();
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

void sendCurrentState() {
  switch (state) {
    case WAITING:     sendLine("STATE:WAITING"); break;
    case CALIBRATING: sendLine("STATE:CALIBRATING"); break;
    case READY:       sendLine("STATE:READY"); break;
    case STREAMING:   sendLine("STATE:STREAMING"); break;
  }
  if (haveBias && (state == READY || state == STREAMING)) {
    emitBiases();
  }
}

// Button / CMD:NEXT advance (same semantics as sketch_usb_dual_100).
bool advanceLikeButton() {
  switch (state) {
    case WAITING:
      enterCalibrating();
      return true;
    case READY:
      enterStreaming();
      return true;
    case STREAMING:
      enterWaiting();
      return true;
    case CALIBRATING:
      return false;  // ignore during calib (~5 s)
  }
  return false;
}

void handleCommand(char* line) {
  // Trim CR and leading/trailing spaces.
  while (*line == ' ' || *line == '\t') line++;
  size_t n = strlen(line);
  while (n > 0 && (line[n - 1] == '\r' || line[n - 1] == ' ' || line[n - 1] == '\t')) {
    line[--n] = '\0';
  }
  if (n == 0) return;

  // Accept "CMD:FOO" or bare "FOO" for convenience in serial monitors.
  const char* cmd = line;
  if (strncmp(line, "CMD:", 4) == 0) {
    cmd = line + 4;
  }

  if (strcmp(cmd, "CALIBRATE") == 0) {
    if (state != WAITING) {
      sendLine("CMD:ERR:need_WAITING");
      return;
    }
    enterCalibrating();
    sendLine("CMD:OK:CALIBRATE");
    return;
  }

  if (strcmp(cmd, "START") == 0) {
    if (state != READY) {
      sendLine("CMD:ERR:need_READY");
      return;
    }
    enterStreaming();
    sendLine("CMD:OK:START");
    return;
  }

  if (strcmp(cmd, "STOP") == 0) {
    if (state == STREAMING || state == READY) {
      enterWaiting();
      sendLine("CMD:OK:STOP");
      return;
    }
    if (state == WAITING) {
      sendLine("CMD:OK:STOP");  // already stopped
      return;
    }
    sendLine("CMD:ERR:busy_CALIBRATING");
    return;
  }

  if (strcmp(cmd, "STATUS") == 0) {
    sendCurrentState();
    sendLine("CMD:OK:STATUS");
    return;
  }

  if (strcmp(cmd, "NEXT") == 0) {
    if (!advanceLikeButton()) {
      sendLine("CMD:ERR:busy_CALIBRATING");
      return;
    }
    sendLine("CMD:OK:NEXT");
    return;
  }

  sendLine("CMD:ERR:unknown");
}

void pollSerialCommands() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      cmdBuf[cmdLen] = '\0';
      if (cmdLen > 0) {
        handleCommand(cmdBuf);
      }
      cmdLen = 0;
      continue;
    }
    if (c == '\r') continue;
    if (cmdLen < CMD_BUF_LEN - 1) {
      cmdBuf[cmdLen++] = c;
    } else {
      // Overflow: drop until newline.
      cmdLen = 0;
    }
  }
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
  myMux.setPort(LF_MUX_PORT);
  icm1.getEvent(&a1, &g1, &tEvt, &mEvt);
  myMux.setPort(RF_MUX_PORT);
  icm2.getEvent(&a2, &g2, &tEvt, &mEvt);

  tUsOut = tUs;
  return true;
}

void setup() {
  Serial.begin(230400);
  delay(50);

  pinMode(BTN_PIN, INPUT_PULLUP);

  Wire1.begin();
  Wire1.setClock(400000);

  Serial.print("Dual IMU USB (sketch_usb_dual_100_cmd) - ICM-20948, hw-timer ");
  Serial.print(1000000UL / SAMPLE_PERIOD_US);
  Serial.println(" Hz, no mag + CMD:*");
  Serial.println("  LF = mux port 0, RF = mux port 1");
  Serial.println("  CMD:CALIBRATE|START|STOP|STATUS|NEXT");

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

  sampleTimer = timerBegin(TIMER_TICK_HZ);
  if (sampleTimer == nullptr) {
    Serial.println("timerBegin failed. Freezing...");
    while (1) delay(10);
  }
  timerAttachInterrupt(sampleTimer, &onSampleTimer);
  timerAlarm(sampleTimer, SAMPLE_PERIOD_US, true, 0);

  Serial.println();
  enterWaiting();
}

void loop() {
  pollSerialCommands();

  bool btn = buttonJustPressed();
  if (btn) {
    advanceLikeButton();
  }

  switch (state) {
    case WAITING:
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
      break;

    case STREAMING: {
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
