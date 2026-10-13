import numpy as np
import serial
import time
import threading
from tensorflow.keras.models import load_model
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtCore import Qt
from pyqtgraph.Qt import QtCore
import sys


# -------------------- PARÁMETROS --------------------
ventana_tamaño = 1000
ventanas_por_secuencia = 3
total_muestras = ventana_tamaño * ventanas_por_secuencia  # 3000 muestras ≈ 0.75 s a ~4 kHz
puerto_bt = 'COM13'
baudrate = 921600
intervalo_us = 250                                        # Tiempo real estimado entre muestras ≈ 250 µs (4 kHz)
muestras_grafica = 3000
v_max = 3.3
OFFSET = 1.5
ESCALA = 1.65
NUM_PREDICCIONES_REQUERIDAS = 2

# -------------------- CARGA DE MODELO --------------------
modelo = load_model("C:\\Users\\dcast\\Desktop\\Iker C\\UNIVERSIDAD POLITÉCNICA DE MADRID\\5º AÑO\\2º CUATRIMESTRE\\TFG\\Código\\Python\\Red neuronal_definitivo\\modelos\\mejor_modelo.keras")
clases = np.load("C:\\Users\\dcast\\Desktop\\Iker C\\UNIVERSIDAD POLITÉCNICA DE MADRID\\5º AÑO\\2º CUATRIMESTRE\\TFG\\Código\\Python\\Red neuronal_definitivo\\datos\\etiquetas_secuencias.npy", allow_pickle=True)

# -------------------- SERIAL --------------------
try:
    ser = serial.Serial(puerto_bt, baudrate, timeout=0.01)
    time.sleep(2)
    print(f"🟢 Conectado a {puerto_bt} — leyendo datos...")
except serial.SerialException:
    print("❌ Error: no se pudo conectar al puerto Bluetooth.")
    sys.exit(1)

# -------------------- APLICACIÓN Y VENTANA PRINCIPAL --------------------
app = QApplication([])
win = pg.GraphicsLayoutWidget(title="Voltaje TENG + Estado emocional")
win.resize(1000, 600)
plot = win.addPlot(title="Voltaje del TENG (V)")
tiempo = np.linspace(-((muestras_grafica - 1) * intervalo_us / 1e6), 0, muestras_grafica)
curve_raw = plot.plot(tiempo, np.zeros(muestras_grafica), pen='y')
plot.setLabel('left', 'Voltaje', units='V')
plot.setLabel('bottom', 'Tiempo', units='s')
plot.setYRange(0, v_max)
win.show()

# -------------------- ETIQUETA DE EMOCIÓN SUPERPUESTA --------------------
label = QLabel(win)
label.setStyleSheet("color: white; font-size: 28px; background-color: rgba(0,0,0,150); padding: 8px;")
label.move(30, 30)
label.resize(400, 50)
label.setText("😐 Normal")
label.setAttribute(Qt.WA_TransparentForMouseEvents)
label.show()

# -------------------- BUFFERS Y BLOQUEO --------------------
buffer = []
grafica_buffer = np.zeros(muestras_grafica)
lock = threading.Lock()

# -------------------- FUNCIÓN DE PREDICCIÓN --------------------
def predecir(buffer_local):
    # Crear las ventanas de tamaño fijo
    ventanas = np.array([
        buffer_local[i:i + ventana_tamaño] for i in range(0, total_muestras, ventana_tamaño)
    ])
    
    # Aplicar la misma normalización que en el entrenamiento
    ventanas_normalizadas = (ventanas - OFFSET) / ESCALA
    
    # Preparar entrada para el modelo
    X = ventanas_normalizadas[np.newaxis, ..., np.newaxis]
    
    # Predicción
    pred = modelo.predict(X, verbose=0)
    clase_idx = np.argmax(pred)
    confianza = np.max(pred)
    
    return clases[clase_idx], confianza

# -------------------- HILO DE INFERENCIA --------------------

estado_actual = "😐 Normal"
estado_fijado = False
tiempo_estado = 3.0  # segundos
ultimo_cambio = time.time()

def hilo_inferencia():
    global estado_actual, estado_fijado, ultimo_cambio
    clase_anterior = None
    repeticiones = 0
    MUESTRAS_DESPLAZAMIENTO = 1000  # Desplazamiento ≈ 0.25 s (1/3 secuencia)


    while True:
        time.sleep(0.1)  # Espera entre intentos de predicción

        with lock:
            if len(buffer) >= total_muestras:
                buffer_local = buffer[:total_muestras]  # Tomar solo las muestras necesarias
                del buffer[:MUESTRAS_DESPLAZAMIENTO]    # Desplazar la ventana
            else:
                continue  # Esperar a tener suficientes muestras

        clase, confianza = predecir(buffer_local)
        print(f"[{time.strftime('%H:%M:%S')}] Clase: {clase} — Confianza: {confianza:.2f}")

        if clase == clase_anterior:
            repeticiones += 1
        else:
            clase_anterior = clase
            repeticiones = 1

        if repeticiones == NUM_PREDICCIONES_REQUERIDAS:
            if clase in ['caricia_dedo', 'caricia_mano']:
                emocion = "😄 Contento"
            elif clase in ['golpe_dedo', 'golpe_mano']:
                emocion = "😠 Enfadado"
            else:
                emocion = "😐 Normal"

            if emocion != "😐 Normal":
                label.setText(emocion)
                estado_actual = emocion
                estado_fijado = True
                ultimo_cambio = time.time()

            repeticiones = 0

        # Si ya hay un estado emocional fuerte, mantenerlo durante tiempo_estado
        if estado_fijado:
            if time.time() - ultimo_cambio > tiempo_estado:
                estado_fijado = False
                estado_actual = "😐 Normal"
                label.setText(estado_actual)

# -------------------- FUNCIÓN DE ACTUALIZACIÓN --------------------
def update():
    global buffer, grafica_buffer
    count = 0
    max_reads = 200
    while ser.in_waiting and count < max_reads:
        try:
            linea = ser.readline().decode().strip()
            valor = int(linea)
            muestra = (valor * v_max) / 4095.0

            with lock:
                buffer.append(muestra)
                if len(buffer) > total_muestras:
                    buffer.pop(0)

            grafica_buffer = np.roll(grafica_buffer, -1)
            grafica_buffer[-1] = muestra
            count += 1
        except:
            continue

    curve_raw.setData(tiempo, grafica_buffer)

# -------------------- TEMPORIZADOR Y HILOS --------------------
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)

threading.Thread(target=hilo_inferencia, daemon=True).start()

# -------------------- CIERRE LIMPIO --------------------
def cerrar():
    if ser.is_open:
        ser.close()
    print("✅ Puerto cerrado. Saliendo...")

app.aboutToQuit.connect(cerrar)

# -------------------- EJECUTAR --------------------
app.exec_()
