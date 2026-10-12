#include "BluetoothSerial.h"

BluetoothSerial SerialBT;

// Pines analógicos del ADC1 (seguros con Bluetooth)
const int pin1 = 34;  // ADC1_CH6
const int pin2 = 32;  // ADC1_CH4

// Intervalo de muestreo (en microsegundos)
const unsigned long intervalo = 100;  // 100 µs → 10 kHz

unsigned long ultimo_tiempo = 0;

void setup() {
  SerialBT.begin("ESP32_TENG");  // Nombre Bluetooth
  analogReadResolution(12);      // Resolución de 12 bits (0–4095)
}

void loop() {
  unsigned long ahora = micros();
  if (ahora - ultimo_tiempo >= intervalo) {
    ultimo_tiempo = ahora;

    int valor1 = analogRead(pin1);  // Primer sensor
    int valor2 = analogRead(pin2);  // Segundo sensor

    // Enviar como "valor1,valor2\n"
    SerialBT.printf("%d,%d\n", valor1, valor2);
  }
}
