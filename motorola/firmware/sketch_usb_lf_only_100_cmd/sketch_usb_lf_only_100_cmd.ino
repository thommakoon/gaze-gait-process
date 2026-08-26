// USB Serial LF-only IMU — same protocol as sketch_usb_dual_100_cmd.
//
// Use this when the RF Qwiic wire is broken / unplugged.
//   * Never selects mux port 2, never begin()/getEvent() on RF.
//   * Still emits 14-field CSV: seq,time_us,LF(6),RF(6) with RF = 0.
//   * GRAVITY_RF / GYRO_BIAS_RF are 0 so the phone/PC recorder stays happy.
//
// Flash instead of sketch_usb_dual_100_cmd. Switch back when RF is rewired.
//
// Hardware: QT Py ESP32-S2, Wire1, SparkFun mux 0x70, LF ICM-20948 on port 3.
// Commands: CMD:CALIBRATE|START|STOP|STATUS|NEXT  (same as dual).

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
#define I2C_CLOCK_HZ       100000UL

#define SAMPLE_PERIOD_US   10000UL
#define TIMER_TICK_HZ      1000000UL

#define CMD_BUF_LEN        48

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
Adafruit_ICM20948_AgGyroOnly icmLf;

float bias1Ax = 0, bias1Ay = 0, bias1Az = 0;
float bias1Gx = 0, bias1Gy = 0, bias1Gz = 0;
float sum1Ax = 0, sum1Ay = 0, sum1Az = 0;
float sum1Gx = 0, sum1Gy = 0, sum1Gz = 0;

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
  Serial.flush();
}

void waitForUsbSerial() {
  unsigned long t0 = millis();
  while (!Serial && (millis() - t0) < 5000) {
    delay(10);
  }
}

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
  clearSampleFlag();
  sendLine("STATE:CALIBRATING");
}

void emitBiases() {
  snprintf(buf, sizeof(buf), "GRAVITY_LF:%.6f,%.6f,%.6f", bias1Ax, bias1Ay, bias1Az);
  sendLine(buf);
  snprintf(buf, sizeof(buf), "GYRO_BIAS_LF:%.6f,%.6f,%.6f", bias1Gx, bias1Gy, bias1Gz);
  sendLine(buf);
  sendLine("GRAVITY_RF:0.000000,0.000000,0.000000");
  sendLine("GYRO_BIAS_RF:0.000000,0.000000,0.000000");
}

void enterReady() {
  if (calCount > 0) {
    bias1Ax = sum1Ax / calCount;
    bias1Ay = sum1Ay / calCount;
    bias1Az = sum1Az / calCount;
    bias1Gx = sum1Gx / calCount;
    bias1Gy = sum1Gy / calCount;
    bias1Gz = sum1Gz / calCount;
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
      return false;
  }
  return false;
}

void handleCommand(char* line) {
  while (*line == ' ' || *line == '\t') line++;
  size_t n = strlen(line);
  while (n > 0 && (line[n - 1] == '\r' || line[n - 1] == ' ' || line[n - 1] == '\t')) {
    line[--n] = '\0';
  }
  if (n == 0) return;

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
      sendLine("CMD:OK:STOP");
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
      cmdLen = 0;
    }
  }
}

bool consumeLfSample(uint32_t& tUsOut, sensors_event_t& a1, sensors_event_t& g1) {
  if (!sampleDue) return false;

  noInterrupts();
  uint32_t tUs = sampleDueUs;
  sampleDue = false;
  interrupts();

  sensors_event_t tEvt, mEvt;
  selectMuxPort(LF_MUX_PORT);
  icmLf.getEvent(&a1, &g1, &tEvt, &mEvt);

  tUsOut = tUs;
  return true;
}

void setup() {
  Serial.begin(230400);
  waitForUsbSerial();
  Serial.println();
  Serial.println("BOOT sketch_usb_lf_only_100_cmd");
  Serial.flush();

  pinMode(BTN_PIN, INPUT_PULLUP);

  Wire1.begin();
  Wire1.setClock(I2C_CLOCK_HZ);

  Serial.print("LF-only USB (sketch_usb_lf_only_100_cmd) - ICM-20948, hw-timer ");
  Serial.print(1000000UL / SAMPLE_PERIOD_US);
  Serial.println(" Hz, RF stubbed to 0 + CMD:*");
  Serial.println("  LF = mux port 3; RF not accessed (wire problem)");
  Serial.println("  CMD:CALIBRATE|START|STOP|STATUS|NEXT");

  if (!myMux.begin(0x70, Wire1)) {
    Serial.println("Mux not detected. Freezing...");
    while (1) delay(10);
  }
  Serial.println("Mux detected");

  selectMuxPort(LF_MUX_PORT);
  if (!icmLf.begin_I2C_AgGyroOnly(IMU_I2C_ADDR, &Wire1)) {
    Serial.println("Failed to find ICM20948 on port 3 (LF)");
    while (1) delay(10);
  }
  configureICM(icmLf);
  Serial.println("Port 3 ICM20948 (LF) initialized");
  Serial.println("RF skipped (CSV RF fields = 0)");

  sampleTimer = timerBegin(TIMER_TICK_HZ);
  if (sampleTimer == nullptr) {
    Serial.println("timerBegin failed. Freezing...");
    while (1) delay(10);
  }
  timerAttachInterrupt(sampleTimer, &onSampleTimer);
  timerAlarm(sampleTimer, SAMPLE_PERIOD_US, true, 0);

  Serial.println();
  enterWaiting();
  Serial.println("Send CMD:STATUS or CMD:CALIBRATE (no CSV until calibrating/streaming)");
  Serial.flush();
}

void loop() {
  pollSerialCommands();

  if (buttonJustPressed()) {
    advanceLikeButton();
  }

  switch (state) {
    case WAITING:
      break;

    case CALIBRATING: {
      uint32_t tUs;
      sensors_event_t a1, g1;
      if (!consumeLfSample(tUs, a1, g1)) break;

      sum1Ax += a1.acceleration.x;
      sum1Ay += a1.acceleration.y;
      sum1Az += a1.acceleration.z;
      sum1Gx += g1.gyro.x;
      sum1Gy += g1.gyro.y;
      sum1Gz += g1.gyro.z;

      int n = snprintf(buf, sizeof(buf), "%lu,%lu,",
                       (unsigned long)seq, (unsigned long)tUs);
      n += snprintf(buf + n, sizeof(buf) - n,
                    "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,",
                    a1.acceleration.x, a1.acceleration.y, a1.acceleration.z,
                    g1.gyro.x, g1.gyro.y, g1.gyro.z);
      snprintf(buf + n, sizeof(buf) - n, "0.0000,0.0000,0.0000,0.0000,0.0000,0.0000");
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
      sensors_event_t a1, g1;
      if (!consumeLfSample(tUs, a1, g1)) break;

      int n = snprintf(buf, sizeof(buf), "%lu,%lu,",
                       (unsigned long)seq, (unsigned long)tUs);
      n += snprintf(buf + n, sizeof(buf) - n,
                    "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,",
                    a1.acceleration.x, a1.acceleration.y, a1.acceleration.z,
                    g1.gyro.x - bias1Gx, g1.gyro.y - bias1Gy, g1.gyro.z - bias1Gz);
      snprintf(buf + n, sizeof(buf) - n, "0.0000,0.0000,0.0000,0.0000,0.0000,0.0000");
      sendLine(buf);

      seq++;
    } break;
  }
}
