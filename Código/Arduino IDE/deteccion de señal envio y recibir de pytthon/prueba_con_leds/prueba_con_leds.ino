// --- Versión con temporizador hardware para muestreo a 10kHz (compatible con Core 2.x) ---
#include "BluetoothSerial.h"
BluetoothSerial SerialBT;

// --- Configuración del ADC y temporizador ---
const int pinADC = 34;  // Pin conectado al TENG
const int frecuencia_muestreo = 10000;  // 10kHz

// Pines LED para visualización de etiquetas
const int pinGolpe   = 16;
const int pinCaricia = 17;
const int pinNinguna = 18;

// Variables del temporizador
hw_timer_t* timer = NULL;
portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;

volatile bool adc_ready = false;
volatile int ultimo_valor_adc = 0;

// --- Interrupción del temporizador ---
void IRAM_ATTR onTimer() {
  portENTER_CRITICAL_ISR(&timerMux);
  ultimo_valor_adc = analogRead(pinADC);
  adc_ready = true;
  portEXIT_CRITICAL_ISR(&timerMux);
}

String entrada = "";

void setup() {
  SerialBT.begin("ESP32_TENG");

  pinMode(pinGolpe, OUTPUT);
  pinMode(pinCaricia, OUTPUT);
  pinMode(pinNinguna, OUTPUT);

  digitalWrite(pinGolpe, LOW);
  digitalWrite(pinCaricia, LOW);
  digitalWrite(pinNinguna, LOW);

  // --- Configurar temporizador hardware ---
  timer = timerBegin(0, 80, true);             // Timer 0, prescaler 80 → 1 µs por tick
  timerAttachInterrupt(timer, &onTimer, true); // Adjuntar interrupción
  timerAlarmWrite(timer, 100, true);           // Disparar cada 100 µs = 10 kHz
  timerAlarmEnable(timer);                     // Habilitar la alarma
}

void loop() {
  // --- 1. Enviar datos del ADC cuando estén listos ---
  if (adc_ready) {
    portENTER_CRITICAL(&timerMux);
    int valor = ultimo_valor_adc;
    adc_ready = false;
    portEXIT_CRITICAL(&timerMux);

    SerialBT.println(valor);
  }

  // --- 2. Leer etiquetas desde Python ---
  while (SerialBT.available()) {
    char c = SerialBT.read();
    if (c == '\n') {
      entrada.trim();  // Elimina espacios o saltos de línea

      // Apagar todos los LEDs
      digitalWrite(pinGolpe, LOW);
      digitalWrite(pinCaricia, LOW);
      digitalWrite(pinNinguna, LOW);

      // Encender el correspondiente
      if (entrada == "golpe_dedo") {
        digitalWrite(pinGolpe, HIGH);
      } else if (entrada == "caricia_dedo") {
        digitalWrite(pinCaricia, HIGH);
      } else {
        digitalWrite(pinNinguna, HIGH);
      }

      entrada = ""; // Limpiar buffer
    } else {
      entrada += c;
    }
  }
}
