// Arduino: receive 1-byte commands from MATLAB over USB serial
// Commands:
//   0x00 = LED off
//   0x01 = LED on
//   0xA5 = ping -> replies 0x5A

const uint8_t LED_PIN = 10;

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.begin(115200);

  // Optional: wait for serial to be ready on boards that need it
  // while (!Serial) {}
}

void loop() {
  if (Serial.available() > 0) {
    uint8_t cmd = (uint8_t)Serial.read();

    if (cmd == 0x00) {
      digitalWrite(LED_PIN, LOW);
    } 
    else if (cmd == 0x01) {
      digitalWrite(LED_PIN, HIGH);
    } 
    else if (cmd == 0xA5) {
      Serial.write((uint8_t)0x5A);   // reply "pong"
    }
    // else: ignore unknown command
  }
}
