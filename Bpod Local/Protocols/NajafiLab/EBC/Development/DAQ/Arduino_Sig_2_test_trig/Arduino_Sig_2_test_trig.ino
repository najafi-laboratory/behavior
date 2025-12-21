// Arduino: receive 1-byte commands from MATLAB over USB serial
// Commands:
//   0x00 = LED off
//   0x01 = LED on
//   0xA5 = ping -> replies 0x5A
volatile uint8_t pulseCount = 0;
volatile bool running = false;

const uint8_t N_PULSES = 10;
const uint16_t DUTY = 125;   // example duty

void setup() {
  pinMode(10, OUTPUT);   // OC1B
  digitalWrite(10, LOW);  // ensure clean low
  
  Serial.begin(115200);

  // Stop Timer1
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1  = 0;

  // TOP value → 4 ms period
  ICR1 = 1000;
//ICR1 = 500;
  OCR1B = 0;   // start disabled

  // 3 ms HIGH (75% duty)
  //OCR1B = 750;
  //OCR1B = 250;
  //OCR1B = 125;


  // Fast PWM mode, TOP = ICR1
  //TCCR1A |= (1 << COM1B1);           // non-inverting on OC1B (pin 10)
  TCCR1A |= (1 << WGM11);
  TCCR1B |= (1 << WGM12) | (1 << WGM13);

  // Enable OC1B output
  TCCR1A |= (1 << COM1B1);

  // Enable Timer1 overflow interrupt
  TIMSK1 |= (1 << TOIE1);

  // Prescaler = 64
  TCCR1B |= (1 << CS11) | (1 << CS10);


  digitalWrite(10, LOW);  // ensure clean low
  // Optional: wait for serial to be ready on boards that need it
  // while (!Serial) {}
}

void startPulses() {
  pulseCount = 0;
  running = true;
  OCR1B = DUTY;   // enable PWM  
  TCCR1A |= (1 << COM1B1);   // connect OC1B at cycle boundary
}

ISR(TIMER1_OVF_vect) {
  if (running) {
    pulseCount++;
    if (pulseCount >= N_PULSES) {
      TCCR1A &= ~(1 << COM1B1);  // disconnect at cycle boundary
      OCR1B = 0;       // stop PWM cleanly
      running = false;
      digitalWrite(10, LOW);  // ensure clean low
    }
  }
}

void loop() {
  if (Serial.available() > 0) {
    uint8_t cmd = (uint8_t)Serial.read();

    if (cmd == 0x01) {
      // Enable PWM output on OC1B (pin 10)
      //TCCR1A |= (1 << COM1B1);
      startPulses();
    } 
    else if (cmd == 0x02) {
      // Disable PWM output (pin forced LOW)
      TCCR1A &= ~(1 << COM1B1);
      digitalWrite(10, LOW);  // ensure clean low
    } 
    // else: ignore unknown command
  }
}
