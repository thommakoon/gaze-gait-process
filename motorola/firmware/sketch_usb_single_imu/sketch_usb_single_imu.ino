// USB Serial — single ICM-20948 (raw accel / gyro / mag).
//
// Hardware (same as sketch_wifi / sketch_usb_dual):
//   USE_QWIIC_MUX  1 = SparkFun Qwiic mux @ 0x70, sensor on port 0
//                  0 = IMU on Wire1 directly (no mux)
// I2C bus: Wire1 (QT Py secondary I2C for Qwiic).

#define USE_QWIIC_MUX 1

#include <Wire.h>
#if USE_QWIIC_MUX
#include <SparkFun_I2C_Mux_Arduino_Library.h>
#endif
#include <Adafruit_ICM20X.h>
#include <Adafruit_ICM20948.h>
#include <Adafruit_Sensor.h>

#define IMU_I2C_ADDR       0x69
#define SAMPLE_INTERVAL_US 5000

#if USE_QWIIC_MUX
QWIICMUX myMux;
#endif
Adafruit_ICM20948 icm;

static unsigned long lastSampleTime = 0;
static uint32_t seq = 0;

char buf[192];

static void imuBusSelect() {
#if USE_QWIIC_MUX
  myMux.setPort(0);
#endif
}

void configureICM(Adafruit_ICM20948& icmSensor) {
  icmSensor.setAccelRange(ICM20948_ACCEL_RANGE_16_G);
  icmSensor.setGyroRange(ICM20948_GYRO_RANGE_2000_DPS);
  icmSensor.setAccelRateDivisor(0);
  icmSensor.setGyroRateDivisor(0);
}

void setup() {
  Serial.begin(230400);
  while (!Serial) {
    delay(10);
  }

  Serial.println(F("sketch_usb_single_imu — ICM-20948"));
#if USE_QWIIC_MUX
  Serial.println(F("Qwiic mux port 0"));
#else
  Serial.println(F("Wire1 direct (no mux)"));
#endif

  Wire1.begin();

#if USE_QWIIC_MUX
  if (!myMux.begin(0x70, Wire1)) {
    Serial.println(F("Mux not detected."));
    while (1) {
      delay(10);
    }
  }
  Serial.println(F("Mux OK"));
#endif

  imuBusSelect();
  if (!icm.begin_I2C(IMU_I2C_ADDR, &Wire1)) {
    Serial.println(F("ICM20948 not found."));
    while (1) {
      delay(10);
    }
  }
  configureICM(icm);
  Serial.println(F("ICM20948 OK — streaming CSV:"));
  Serial.println(F("seq,time_us,ax,ay,az,gx,gy,gz,mx,my,mz"));

  lastSampleTime = micros();
}

void loop() {
  unsigned long now = micros();
  if (now - lastSampleTime < SAMPLE_INTERVAL_US) {
    return;
  }
  lastSampleTime = now;

  sensors_event_t a, g, temp, m;
  imuBusSelect();
  icm.getEvent(&a, &g, &temp, &m);

  snprintf(buf, sizeof(buf),
           "%lu,%lu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
           (unsigned long)seq, (unsigned long)now,
           a.acceleration.x, a.acceleration.y, a.acceleration.z,
           g.gyro.x, g.gyro.y, g.gyro.z,
           m.magnetic.x, m.magnetic.y, m.magnetic.z);
  Serial.println(buf);
  seq++;
}
