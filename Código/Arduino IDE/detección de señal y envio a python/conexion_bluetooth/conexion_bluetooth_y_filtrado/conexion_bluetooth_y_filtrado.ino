#include "BluetoothSerial.h"
BluetoothSerial SerialBT;

const int pinADC = 34;           // Pin conectado al TENG
const int intervalo_us = 100;    // 100 µs = 10 kHz

void setup() {
  SerialBT.begin("ESP32_TENG");  // Nombre del dispositivo Bluetooth
}

void loop() {
  int valor = analogRead(pinADC);                  // Lectura en bruto (0–4095)
  SerialBT.println(valor);                    // Enviar 4 decimales
  delayMicroseconds(intervalo_us);                 // 10 kHz
}
