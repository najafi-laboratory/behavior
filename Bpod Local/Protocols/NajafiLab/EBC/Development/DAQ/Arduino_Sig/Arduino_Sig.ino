void setup() {
  pinMode(10, OUTPUT);   // OC1B

  // Stop Timer1
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1  = 0;

  // TOP value → 4 ms period
  ICR1 = 1000;
//ICR1 = 500;

  // 3 ms HIGH (75% duty)
  //OCR1B = 750;
  //OCR1B = 250;
  OCR1B = 125;


  // Fast PWM mode, TOP = ICR1
  TCCR1A |= (1 << COM1B1);           // non-inverting on OC1B (pin 10)
  TCCR1A |= (1 << WGM11);
  TCCR1B |= (1 << WGM12) | (1 << WGM13);

  // Prescaler = 64
  TCCR1B |= (1 << CS11) | (1 << CS10);
}

void loop() {
  // nothing needed — hardware runs PWM
}
