#include <BluetoothSerial.h> // Librería para usar Bluetooth clásico (SPP)

// Instanciamos el objeto Bluetooth
BluetoothSerial SerialBT;

// Pin de entrada analógica (puedes cambiarlo según tu circuito)
const int ADC_PIN = 34;  // GPIO34 es solo entrada, ideal para ADC
const float ADC_RESOLUTION = 4095.0; // Resolución de 12 bits del ADC
const float VOLTAJE_REF = 3.3;       // Tensión de referencia del ESP32

void setup() {
  // Inicializamos la comunicación serie para depuración por USB
  Serial.begin(115200);
  delay(1000); // Espera para estabilizar

  // Inicializamos el Bluetooth con un nombre visible
  SerialBT.begin("ESP32_ADC_BT"); // Este es el nombre que verá la Raspberry Pi
  Serial.println("Bluetooth iniciado. Esperando conexión...");

  // Configuramos el pin ADC (aunque no es obligatorio)
  pinMode(ADC_PIN, INPUT);
}

void loop() {
  // 1. Leer el valor analógico desde el pin
  int adcValue = analogRead(ADC_PIN);

  // 2. Convertir el valor a voltaje (opcional, pero útil para entender la señal)
  float voltage = (adcValue / ADC_RESOLUTION) * VOLTAJE_REF;

  // 3. Aquí podrías aplicar tus filtros o procesamiento
  // TODO: Filtros digitales u otros tratamientos de señal
  float processedSignal = voltage; // Placeholder: ahora solo pasa el valor tal cual

  // 4. Enviar el valor procesado por Bluetooth a la Raspberry Pi
  // Puedes enviarlo como texto o como binario. Aquí va en texto para pruebas:
  SerialBT.println(processedSignal);

  // También lo mostramos por consola serial para depuración
  Serial.print("ADC: ");
  Serial.print(adcValue);
  Serial.print(" -> Voltaje: ");
  Serial.print(voltage, 3);
  Serial.println(" V");

  delay(100); // Frecuencia de muestreo: cada 100ms (~10Hz)
}