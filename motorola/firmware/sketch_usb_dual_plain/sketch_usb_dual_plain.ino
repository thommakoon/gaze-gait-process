// USB Serial — dual ICM-20948, continuous raw stream.
// No state machine, no button, no hardware timer / ISR.
//
// Wiring (SparkFun Qwiic Mux):
//   QT Py Qwiic  ->  mux MAIN (upstream) connector
//   LF IMU       ->  mux port 3
//   RF IMU       ->  mux port 2
//   Do NOT daisy-chain both IMUs on one port.
//
// Set USE_MUX 0 to test one IMU plugged directly into QT Py (no mux).
//
// Serial: 230400
// CSV: seq,time_us, LF(6), RF(6)

#include <Wire.h>
#include <string.h>
#include <SparkFun_I2C_Mux_Arduino_Library.h>
#include <Adafruit_ICM20X.h>
#include <Adafruit_ICM20948.h>
#include <Adafruit_Sensor.h>

#define USE_MUX                1     // 0 = one IMU direct on QT Py Qwiic
#define I2C_CLOCK_HZ           100000UL
#define IMU_I2C_ADDR           0x69
#define IMU_I2C_ADDR_ALT       0x68
#define LF_MUX_PORT            3
#define RF_MUX_PORT            2
#define SAMPLE_INTERVAL_US     10000UL
#define ICM20948_WHOAMI_EXPECT 0xEA
#define ICM20948_REG_BANK_SEL  0x7F
#define ICM20948_REG_WHOAMI    0x00

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

QWIICMUX myMux;
Adafruit_ICM20948_AgGyroOnly icmLf;
Adafruit_ICM20948_AgGyroOnly icmRf;

static bool lfReady = false;
static bool rfReady = false;
static uint32_t seq = 0;
static unsigned long lastSampleUs = 0;
char buf[320];

void configureICM(Adafruit_ICM20948& icm) {
  icm.setAccelRange(ICM20948_ACCEL_RANGE_16_G);
  icm.setGyroRange(ICM20948_GYRO_RANGE_2000_DPS);
  icm.setAccelRateDivisor(0);
  icm.setGyroRateDivisor(0);
}

void waitForUsbSerial() {
  unsigned long t0 = millis();
  while (!Serial && (millis() - t0) < 5000) {
    delay(10);
  }
}

bool i2cAck(uint8_t addr) {
  Wire1.beginTransmission(addr);
  return Wire1.endTransmission() == 0;
}

bool readWhoAmI(uint8_t addr, uint8_t& whoamiOut) {
  Wire1.beginTransmission(addr);
  Wire1.write(ICM20948_REG_BANK_SEL);
  Wire1.write(0x00);
  if (Wire1.endTransmission() != 0) {
    return false;
  }
  delay(1);

  Wire1.beginTransmission(addr);
  Wire1.write(ICM20948_REG_WHOAMI);
  if (Wire1.endTransmission(false) != 0) {
    return false;
  }
  if (Wire1.requestFrom(addr, (uint8_t)1) != 1) {
    return false;
  }
  whoamiOut = Wire1.read();
  return true;
}

void printWhoAmI(const char* label, uint8_t addr) {
  Serial.print(label);
  Serial.print(F(" 0x"));
  if (addr < 16) Serial.print('0');
  Serial.print(addr, HEX);
  Serial.print(F(": ACK="));
  if (!i2cAck(addr)) {
    Serial.println(F("no"));
    return;
  }
  Serial.print(F("yes"));
  uint8_t whoami = 0xFF;
  if (!readWhoAmI(addr, whoami)) {
    Serial.println(F(", WHOAMI read failed"));
    return;
  }
  Serial.print(F(", WHOAMI=0x"));
  if (whoami < 16) Serial.print('0');
  Serial.print(whoami, HEX);
  if (whoami == ICM20948_WHOAMI_EXPECT) {
    Serial.println(F(" (ICM20948 OK)"));
  } else {
    Serial.println(F(" (NOT ICM20948 — expect 0xEA)"));
  }
}

#if USE_MUX
bool selectMuxPort(uint8_t port) {
  myMux.setPortState(0);
  delay(2);
  if (!myMux.setPort(port)) {
    Serial.print(F("ERROR: mux setPort("));
    Serial.print(port);
    Serial.println(F(") failed"));
    return false;
  }
  delay(10);

  uint8_t active = myMux.getPort();
  Serial.print(F("Mux channel set: want port "));
  Serial.print(port);
  Serial.print(F(", getPort()="));
  if (active == 254) {
    Serial.println(F("I2C error"));
    return false;
  }
  if (active == 255) {
    Serial.println(F("none enabled"));
    return false;
  }
  Serial.println(active);
  return active == port;
}

void scanMuxPort(uint8_t port) {
  if (!selectMuxPort(port)) {
    return;
  }
  Serial.print(F("I2C scan behind mux port "));
  Serial.print(port);
  Serial.print(F(": "));
  bool any = false;
  for (uint8_t addr = 1; addr < 127; addr++) {
    if (addr == 0x70) {
      continue;  // upstream mux — ignore in downstream scan
    }
    Wire1.beginTransmission(addr);
    if (Wire1.endTransmission() == 0) {
      if (any) Serial.print(F(", "));
      Serial.print(F("0x"));
      if (addr < 16) Serial.print('0');
      Serial.print(addr, HEX);
      any = true;
    }
  }
  if (!any) Serial.print(F("(none)"));
  Serial.println();
}

bool tryBeginImu(Adafruit_ICM20948_AgGyroOnly& icm,
                 uint8_t port,
                 const char* label,
                 uint8_t& usedAddrOut) {
  const uint8_t addrs[] = {IMU_I2C_ADDR, IMU_I2C_ADDR_ALT};
  for (uint8_t i = 0; i < 2; i++) {
    uint8_t addr = addrs[i];
    if (!selectMuxPort(port)) {
      return false;
    }
    if (icm.begin_I2C_AgGyroOnly(addr, &Wire1)) {
      configureICM(icm);
      usedAddrOut = addr;
      Serial.print(label);
      Serial.print(F(" init OK @ 0x"));
      if (addr < 16) Serial.print('0');
      Serial.print(addr, HEX);
      Serial.print(F(" mux port "));
      Serial.println(port);
      return true;
    }
  }

  Serial.print(label);
  Serial.print(F(" init FAIL on mux port "));
  Serial.println(port);
  selectMuxPort(port);
  printWhoAmI("  probe", IMU_I2C_ADDR);
  return false;
}
#endif

bool tryBeginDirect(Adafruit_ICM20948_AgGyroOnly& icm, const char* label) {
  const uint8_t addrs[] = {IMU_I2C_ADDR, IMU_I2C_ADDR_ALT};
  for (uint8_t i = 0; i < 2; i++) {
    uint8_t addr = addrs[i];
    if (icm.begin_I2C_AgGyroOnly(addr, &Wire1)) {
      configureICM(icm);
      Serial.print(label);
      Serial.print(F(" direct init OK @ 0x"));
      if (addr < 16) Serial.print('0');
      Serial.println(addr, HEX);
      return true;
    }
  }
  Serial.print(label);
  Serial.println(F(" direct init FAIL"));
  printWhoAmI("  probe", IMU_I2C_ADDR);
  printWhoAmI("  probe", IMU_I2C_ADDR_ALT);
  return false;
}

void setup() {
  Serial.begin(230400);
  waitForUsbSerial();
  Serial.println();
  Serial.println(F("BOOT sketch_usb_dual_plain"));
  Serial.flush();

  Wire1.begin();
  Wire1.setClock(I2C_CLOCK_HZ);
  Serial.print(F("Wire1 clock "));
  Serial.print(I2C_CLOCK_HZ);
  Serial.println(F(" Hz"));

  Serial.println(F("seq,time_us,lf_ax,lf_ay,lf_az,lf_gx,lf_gy,lf_gz,rf_ax,rf_ay,rf_az,rf_gx,rf_gy,rf_gz"));

#if !USE_MUX
  Serial.println(F("MODE: direct (USE_MUX=0) — one IMU on QT Py Qwiic"));
  printWhoAmI("Direct", IMU_I2C_ADDR);
  printWhoAmI("Direct", IMU_I2C_ADDR_ALT);
  lfReady = tryBeginDirect(icmLf, "LF");
  rfReady = false;
#else
  Serial.print(F("MODE: mux — LF=port "));
  Serial.print(LF_MUX_PORT);
  Serial.print(F(", RF=port "));
  Serial.println(RF_MUX_PORT);
  printWhoAmI("Direct bus (before mux)", IMU_I2C_ADDR);

  if (!myMux.begin(0x70, Wire1)) {
    Serial.println(F("ERROR: mux not found @ 0x70"));
    while (1) delay(10);
  }
  Serial.println(F("Mux OK @ 0x70"));

  scanMuxPort(LF_MUX_PORT);
  scanMuxPort(RF_MUX_PORT);
  selectMuxPort(LF_MUX_PORT);
  printWhoAmI("LF", IMU_I2C_ADDR);
  selectMuxPort(RF_MUX_PORT);
  printWhoAmI("RF", IMU_I2C_ADDR);

  uint8_t lfAddr = 0;
  uint8_t rfAddr = 0;
  lfReady = tryBeginImu(icmLf, LF_MUX_PORT, "LF", lfAddr);
  rfReady = tryBeginImu(icmRf, RF_MUX_PORT, "RF", rfAddr);
#endif

  Serial.println(F("=== SUMMARY ==="));
  Serial.print(F("LF: "));
  Serial.println(lfReady ? F("OK") : F("FAIL"));
  Serial.print(F("RF: "));
  Serial.println(rfReady ? F("OK") : F("FAIL"));

  if (!lfReady && !rfReady) {
    Serial.println(F("HINT: WHOAMI must be 0xEA. 0xB6 = bad mux wiring or ghost ACK."));
    Serial.println(F("      Try USE_MUX 0 with IMU direct on QT Py to confirm IMU."));
    while (1) delay(10);
  }

  if (!lfReady || !rfReady) {
    Serial.println(F("WARN: missing side streams as 0.0"));
  } else {
    Serial.println(F("OK — streaming"));
  }

  lastSampleUs = micros();
}

void loop() {
  unsigned long now = micros();
  if (now - lastSampleUs < SAMPLE_INTERVAL_US) {
    return;
  }
  lastSampleUs = now;

  sensors_event_t a1, g1, t1, m1;
  sensors_event_t a2, g2, t2, m2;

  if (lfReady) {
#if USE_MUX
    selectMuxPort(LF_MUX_PORT);
#endif
    icmLf.getEvent(&a1, &g1, &t1, &m1);
  } else {
    memset(&a1, 0, sizeof(a1));
    memset(&g1, 0, sizeof(g1));
  }

  if (rfReady) {
#if USE_MUX
    selectMuxPort(RF_MUX_PORT);
#endif
    icmRf.getEvent(&a2, &g2, &t2, &m2);
  } else {
    memset(&a2, 0, sizeof(a2));
    memset(&g2, 0, sizeof(g2));
  }

  int n = snprintf(buf, sizeof(buf), "%lu,%lu,",
                   (unsigned long)seq, (unsigned long)now);
  n += snprintf(buf + n, sizeof(buf) - n,
                "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,",
                a1.acceleration.x, a1.acceleration.y, a1.acceleration.z,
                g1.gyro.x, g1.gyro.y, g1.gyro.z);
  snprintf(buf + n, sizeof(buf) - n,
           "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
           a2.acceleration.x, a2.acceleration.y, a2.acceleration.z,
           g2.gyro.x, g2.gyro.y, g2.gyro.z);
  Serial.println(buf);
  seq++;
}
