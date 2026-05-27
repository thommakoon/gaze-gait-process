// Bench / diagnostics — closest load to sketch_usb_dual_timer when BENCH_WITH_CSV_STREAM=1
// (I2C + snprintf + Serial.println @ 200 Hz). Use a PC Serial Monitor / terminal, not the
// URP2026 recorder (no STATE:/button flow; Android parser may not match 14 fields).
//
// Measures IMU ↔ QT Py cost on the wire:
//   * LF and RF `getEvent()` duration (includes mux select + Adafruit I2C + mag read inside getEvent)
//   * Total dual-IMU block time
//
// Optional: same 200 Hz hardware timer as sketch_usb_dual_timer to report
//   * `pipe_i2c_us` = micros() after both reads minus ISR `sampleDueUs` (dispatch + I2C)
//   * `pipe_full_us` = ISR → after `Serial.println` of the CSV line (when BENCH_WITH_CSV_STREAM)
//   * `timerOverrunCount` if loop() missed a tick
//
// Optional BENCH_WITH_CSV_STREAM: each tick emits the same 14-field CSV line as
//   sketch_usb_dual_timer STREAMING (gyro bias terms = 0) so USB + snprintf load
//   matches production for overrun / full-pipeline sanity checks.
//
// Hardware: QT Py ESP32-S2, Wire1 @ 400 kHz, SparkFun Qwiic mux 0x70,
//   ICM-20948 on port 0 (LF) and port 1 (RF), same as production dual sketches.
//
// Serial: 230400. When CSV stream is on, expect ~200 lines/s of data + rare BENCH summaries.

#include <Wire.h>
#include <SparkFun_I2C_Mux_Arduino_Library.h>
#include <Adafruit_ICM20X.h>
#include <Adafruit_ICM20948.h>
#include <Adafruit_Sensor.h>

#define IMU_I2C_ADDR     0x69
#define LF_MUX_PORT      0
#define RF_MUX_PORT      1

// 1 = 200 Hz timer + pipeline stats (matches dual_timer cadence).
// 0 = tight loop, max-rate I2C — shows raw read cost without timer dispatch.
#define BENCH_USE_TIMER  1

// 1 = default: same per-tick CSV + USB as dual_timer (~200 lines/s) + pipe_full_us in summary.
// 0 = I2C-only + BENCH summaries (no serial flood; no pipe_full line).
#define BENCH_WITH_CSV_STREAM  1

#define SAMPLE_PERIOD_US 5000UL
#define TIMER_TICK_HZ    1000000UL

#if BENCH_USE_TIMER && BENCH_WITH_CSV_STREAM
// 5 s @ 200 Hz — fewer BENCH lines on top of full CSV flood.
#define SUMMARY_EVERY_N  1000
#else
#define SUMMARY_EVERY_N  200
#endif

#if BENCH_WITH_CSV_STREAM && !BENCH_USE_TIMER
#error "BENCH_WITH_CSV_STREAM requires BENCH_USE_TIMER (set BENCH_USE_TIMER to 1)."
#endif

QWIICMUX myMux;
Adafruit_ICM20948 icm1;
Adafruit_ICM20948 icm2;

static char csvBuf[320];

// Zero biases → same string length / code path as dual_timer STREAMING.
static const float bias1Gx = 0, bias1Gy = 0, bias1Gz = 0;
static const float bias2Gx = 0, bias2Gy = 0, bias2Gz = 0;
static uint32_t csvSeq = 0;

#if BENCH_USE_TIMER
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
#endif

void configureICM(Adafruit_ICM20948& icm) {
  icm.setAccelRange(ICM20948_ACCEL_RANGE_16_G);
  icm.setGyroRange(ICM20948_GYRO_RANGE_2000_DPS);
  icm.setAccelRateDivisor(0);
  icm.setGyroRateDivisor(0);
}

static void emitDualTimerCsvLine(
    uint32_t tUs,
    const sensors_event_t& a1, const sensors_event_t& g1,
    const sensors_event_t& a2, const sensors_event_t& g2
) {
  int nb = snprintf(csvBuf, sizeof(csvBuf), "%lu,%lu,",
                    (unsigned long)csvSeq, (unsigned long)tUs);
  nb += snprintf(csvBuf + nb, sizeof(csvBuf) - nb,
                 "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,",
                 a1.acceleration.x, a1.acceleration.y, a1.acceleration.z,
                 g1.gyro.x - bias1Gx, g1.gyro.y - bias1Gy, g1.gyro.z - bias1Gz);
  snprintf(csvBuf + nb, sizeof(csvBuf) - nb,
           "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
           a2.acceleration.x, a2.acceleration.y, a2.acceleration.z,
           g2.gyro.x - bias2Gx, g2.gyro.y - bias2Gy, g2.gyro.z - bias2Gz);
  Serial.println(csvBuf);
  csvSeq++;
}

static void printBenchLine(
    const char* mode,
    uint32_t n,
    uint32_t lfMin, uint32_t lfMax, uint64_t lfSum,
    uint32_t rfMin, uint32_t rfMax, uint64_t rfSum,
    uint32_t totMin, uint32_t totMax, uint64_t totSum,
    uint32_t pipeI2cMin, uint32_t pipeI2cMax, uint64_t pipeI2cSum,
#if BENCH_WITH_CSV_STREAM
    uint32_t pipeFullMin, uint32_t pipeFullMax, uint64_t pipeFullSum,
#endif
    uint32_t overruns
) {
  if (n == 0) return;
  char line[224];
  snprintf(line, sizeof(line),
           "BENCH mode=%s n=%lu LF_us[min=%lu max=%lu avg=%.1f] "
           "RF_us[min=%lu max=%lu avg=%.1f] "
           "TOT_us[min=%lu max=%lu avg=%.1f]",
           mode, (unsigned long)n,
           (unsigned long)lfMin, (unsigned long)lfMax, (double)lfSum / (double)n,
           (unsigned long)rfMin, (unsigned long)rfMax, (double)rfSum / (double)n,
           (unsigned long)totMin, (unsigned long)totMax, (double)totSum / (double)n);
  Serial.println(line);
#if BENCH_USE_TIMER
  snprintf(line, sizeof(line),
           "BENCH pipe_i2c_us[min=%lu max=%lu avg=%.1f] (ISR→after RF getEvent) overruns=%lu",
           (unsigned long)pipeI2cMin, (unsigned long)pipeI2cMax,
           (double)pipeI2cSum / (double)n, (unsigned long)overruns);
  Serial.println(line);
#if BENCH_WITH_CSV_STREAM
  snprintf(line, sizeof(line),
           "BENCH pipe_full_us[min=%lu max=%lu avg=%.1f] (ISR→after Serial.println CSV)",
           (unsigned long)pipeFullMin, (unsigned long)pipeFullMax,
           (double)pipeFullSum / (double)n);
  Serial.println(line);
#endif
#else
  (void)pipeI2cMin;
  (void)pipeI2cMax;
  (void)pipeI2cSum;
  (void)overruns;
#endif
}

void setup() {
  Serial.begin(230400);
  delay(50);

  Wire1.begin();
  Wire1.setClock(400000);

  Serial.println(F("sketch_usb_dual_i2c_bench — I2C timing (bench only)"));
#if BENCH_USE_TIMER
  Serial.println(F("  Timer: 200 Hz"));
#if BENCH_WITH_CSV_STREAM
  Serial.println(F("  CSV stream: ON (14-field, same snprintf path as dual_timer STREAMING)"));
#else
  Serial.println(F("  CSV stream: OFF (I2C-only load)"));
#endif
#else
  Serial.println(F("  Mode: max-rate loop (no timer)"));
#endif

  if (!myMux.begin(0x70, Wire1)) {
    Serial.println(F("Mux missing. Halt."));
    while (1) delay(100);
  }
  myMux.setPort(LF_MUX_PORT);
  if (!icm1.begin_I2C(IMU_I2C_ADDR, &Wire1)) {
    Serial.println(F("ICM LF missing. Halt."));
    while (1) delay(100);
  }
  configureICM(icm1);
  myMux.setPort(RF_MUX_PORT);
  if (!icm2.begin_I2C(IMU_I2C_ADDR, &Wire1)) {
    Serial.println(F("ICM RF missing. Halt."));
    while (1) delay(100);
  }
  configureICM(icm2);
  Serial.print(F("  BENCH summary every "));
  Serial.print(SUMMARY_EVERY_N);
  Serial.println(F(" ticks"));
  Serial.println(F("Init OK. Starting bench...\n"));

#if BENCH_USE_TIMER
  sampleTimer = timerBegin(TIMER_TICK_HZ);
  if (sampleTimer == nullptr) {
    Serial.println(F("timerBegin failed."));
    while (1) delay(100);
  }
  timerAttachInterrupt(sampleTimer, &onSampleTimer);
  timerAlarm(sampleTimer, SAMPLE_PERIOD_US, true, 0);
#endif
}

void loop() {
#if BENCH_USE_TIMER
  if (!sampleDue) return;

  noInterrupts();
  const uint32_t tIsr = sampleDueUs;
  sampleDue = false;
  interrupts();

  uint32_t t0 = micros();
  sensors_event_t a1, g1, t1, m1;
  myMux.setPort(LF_MUX_PORT);
  uint32_t tLf0 = micros();
  icm1.getEvent(&a1, &g1, &t1, &m1);
  uint32_t tLf1 = micros();

  sensors_event_t a2, g2, t2, m2;
  myMux.setPort(RF_MUX_PORT);
  uint32_t tRf0 = micros();
  icm2.getEvent(&a2, &g2, &t2, &m2);
  uint32_t tRf1 = micros();

  const uint32_t tEnd = tRf1;
  const uint32_t lfUs = tLf1 - tLf0;
  const uint32_t rfUs = tRf1 - tRf0;
  const uint32_t totUs = tEnd - t0;
  const uint32_t pipeI2cUs = tEnd - tIsr;

#if BENCH_WITH_CSV_STREAM
  emitDualTimerCsvLine(tIsr, a1, g1, a2, g2);
  const uint32_t tAfterSerial = micros();
  const uint32_t pipeFullUs = tAfterSerial - tIsr;
#endif

  static uint32_t lfMin = UINT32_MAX, lfMax = 0, rfMin = UINT32_MAX, rfMax = 0;
  static uint32_t totMin = UINT32_MAX, totMax = 0, pipeI2cMin = UINT32_MAX, pipeI2cMax = 0;
  static uint64_t lfSum = 0, rfSum = 0, totSum = 0, pipeI2cSum = 0;
#if BENCH_WITH_CSV_STREAM
  static uint32_t pipeFullMin = UINT32_MAX, pipeFullMax = 0;
  static uint64_t pipeFullSum = 0;
#endif
  static uint32_t n = 0;

  lfMin = min(lfMin, lfUs);
  lfMax = max(lfMax, lfUs);
  rfMin = min(rfMin, rfUs);
  rfMax = max(rfMax, rfUs);
  totMin = min(totMin, totUs);
  totMax = max(totMax, totUs);
  pipeI2cMin = min(pipeI2cMin, pipeI2cUs);
  pipeI2cMax = max(pipeI2cMax, pipeI2cUs);
  lfSum += lfUs;
  rfSum += rfUs;
  totSum += totUs;
  pipeI2cSum += pipeI2cUs;
#if BENCH_WITH_CSV_STREAM
  pipeFullMin = min(pipeFullMin, pipeFullUs);
  pipeFullMax = max(pipeFullMax, pipeFullUs);
  pipeFullSum += pipeFullUs;
#endif
  n++;

  if (n % SUMMARY_EVERY_N == 0) {
    uint32_t overCumulative;
    noInterrupts();
    overCumulative = timerOverrunCount;
    interrupts();
    printBenchLine("timer200", n, lfMin, lfMax, lfSum, rfMin, rfMax, rfSum,
                   totMin, totMax, totSum, pipeI2cMin, pipeI2cMax, pipeI2cSum,
#if BENCH_WITH_CSV_STREAM
                   pipeFullMin, pipeFullMax, pipeFullSum,
#endif
                   overCumulative);
    Serial.println();
    lfMin = UINT32_MAX;
    lfMax = 0;
    rfMin = UINT32_MAX;
    rfMax = 0;
    totMin = UINT32_MAX;
    totMax = 0;
    pipeI2cMin = UINT32_MAX;
    pipeI2cMax = 0;
    lfSum = rfSum = totSum = pipeI2cSum = 0;
#if BENCH_WITH_CSV_STREAM
    pipeFullMin = UINT32_MAX;
    pipeFullMax = 0;
    pipeFullSum = 0;
#endif
    n = 0;
  }

#else
  // Max-rate: no timer; measures pure I2C + driver cost.
  uint32_t t0 = micros();
  sensors_event_t a1, g1, t1, m1;
  myMux.setPort(LF_MUX_PORT);
  uint32_t tLf0 = micros();
  icm1.getEvent(&a1, &g1, &t1, &m1);
  uint32_t tLf1 = micros();

  sensors_event_t a2, g2, t2, m2;
  myMux.setPort(RF_MUX_PORT);
  uint32_t tRf0 = micros();
  icm2.getEvent(&a2, &g2, &t2, &m2);
  uint32_t tRf1 = micros();

  const uint32_t lfUs = tLf1 - tLf0;
  const uint32_t rfUs = tRf1 - tRf0;
  const uint32_t totUs = tRf1 - t0;

  static uint32_t lfMin = UINT32_MAX, lfMax = 0, rfMin = UINT32_MAX, rfMax = 0;
  static uint32_t totMin = UINT32_MAX, totMax = 0;
  static uint64_t lfSum = 0, rfSum = 0, totSum = 0;
  static uint32_t n = 0;

  lfMin = min(lfMin, lfUs);
  lfMax = max(lfMax, lfUs);
  rfMin = min(rfMin, rfUs);
  rfMax = max(rfMax, rfUs);
  totMin = min(totMin, totUs);
  totMax = max(totMax, totUs);
  lfSum += lfUs;
  rfSum += rfUs;
  totSum += totUs;
  n++;

  if (n >= 2000) {
    printBenchLine("maxrate", n, lfMin, lfMax, lfSum, rfMin, rfMax, rfSum,
                   totMin, totMax, totSum, 0, 0, 0,
#if BENCH_WITH_CSV_STREAM
                   0, 0, 0,
#endif
                   0);
    Serial.println();
    lfMin = UINT32_MAX;
    lfMax = 0;
    rfMin = UINT32_MAX;
    rfMax = 0;
    totMin = UINT32_MAX;
    totMax = 0;
    lfSum = rfSum = totSum = 0;
    n = 0;
  }
#endif
}
