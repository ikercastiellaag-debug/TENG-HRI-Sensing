#include "BluetoothSerial.h"

BluetoothSerial SerialBT;

const int pinADC = 34;  // Asegúrate de que este pin esté conectado al TENG
const int intervalo_us = 100;  // 100 µs = 10 kHz

void setup() {
  SerialBT.begin("ESP32_TENG");  // Nombre del dispositivo Bluetooth
}

void loop() {
  int valor = analogRead(pinADC);
  SerialBT.println(valor);
  delayMicroseconds(intervalo_us);
}
