void setup() {
  Serial.begin(115200); // Inicia la comunicación serial a 115200 baudios
  delay(1000); // Espera un poco por si acaso
  Serial.println("Iniciando ESP32...");
}

void loop() {
  Serial.println("Hola desde ESP32");
  delay(1000); // Espera 1 segundo
}