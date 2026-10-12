// Pines analógicos del ADC1 (seguros con Serial)
const int pin1 = 34;  // ADC1_CH6
const int pin2 = 32;  // ADC1_CH4

// Intervalo de muestreo (en microsegundos)
const unsigned long intervalo = 100;  // 100 µs → 10 kHz

unsigned long ultimo_tiempo = 0;

void setup() {
  Serial.begin(921600);          // Alta velocidad por USB
  analogReadResolution(12);      // Resolución de 12 bits (0–4095)
  while (!Serial);               // Esperar conexión (opcional)
}

void loop() {
  unsigned long ahora = micros();
  if (ahora - ultimo_tiempo >= intervalo) {
    ultimo_tiempo = ahora;

    int valor1 = analogRead(pin1);  // Primer sensor
    int valor2 = analogRead(pin2);  // Segundo sensor

    // Enviar como "valor1,valor2\n"
    Serial.print(valor1);
    Serial.print(",");
    Serial.println(valor2);

  }
}
