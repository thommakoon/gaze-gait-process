// USB serial smoke test — no IMU, no I2C, no libraries.
// Flash this FIRST when Serial Monitor shows nothing.
//
// Arduino IDE (Adafruit QT Py ESP32-S2):
//   Tools -> USB CDC On Boot -> Enabled
//   Tools -> Upload Mode -> USB
//   Serial Monitor -> 115200 baud, same COM port as Tools -> Port
//
// Open monitor, press RESET. Expect "HELLO" then millis() once per second.

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && (millis() - t0) < 5000) {
    delay(10);
  }
  Serial.println();
  Serial.println(F("HELLO from sketch_usb_serial_test"));
  Serial.println(F("If you see this, USB serial works. Reflash dual IMU sketch next."));
  Serial.flush();
}

void loop() {
  Serial.println(millis());
  Serial.flush();
  delay(1000);
}
