void setup() {
  Serial.begin(921600);  // Alta velocidad para transmisión
}

void loop() {
  int valorADC = analogRead(34);
  if (Serial.availableForWrite() > 0) {
    Serial.println(valorADC);
  }
  delayMicroseconds(100);
}
