#include <avr/interrupt.h>
#include <avr/pgmspace.h>

#define PWM_PIN 9
#define TABLE_SIZE 100

// 8-bit sine lookup table (0–255)
const uint8_t sineTable[TABLE_SIZE] PROGMEM = {
  128,136,144,152,160,168,176,184,191,199,
  207,214,221,228,235,241,247,252,254,255,
  255,254,252,247,241,235,228,221,214,207,
  199,191,184,176,168,160,152,144,136,128,
  120,112,104,96,88,80,72,64,57,49,
  41,34,27,20,13,7,3,1,0,0,
  0,1,3,7,13,20,27,34,41,49,
  57,64,72,80,88,96,104,112,120
};

volatile uint8_t index = 0;

ISR(TIMER1_COMPA_vect) {
  OCR1A = pgm_read_byte(&sineTable[index]);
  index++;
  if (index >= TABLE_SIZE) index = 0;
}

void setup() {
  pinMode(PWM_PIN, OUTPUT);

  // Fast PWM, 8-bit, clear OC1A on compare
  TCCR1A = (1 << COM1A1) | (1 << WGM10);
  TCCR1B = (1 << WGM12) | (1 << CS10);  // no prescaler

  // Interrupt at 25 kHz
  OCR1A = 128;
  OCR1B = 0;

  // Set compare match for interrupt
  OCR1A = 0;
  TIMSK1 |= (1 << OCIE1A);

  sei();
}

void loop() {
  // Nothing here — waveform is hardware-driven
}
