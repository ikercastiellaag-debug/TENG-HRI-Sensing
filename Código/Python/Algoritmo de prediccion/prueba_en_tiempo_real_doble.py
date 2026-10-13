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
ventana_tamano = 1000
ventanas_por_secuencia = 3
total_muestras = ventana_tamano * ventanas_por_secuencia  # 4000 muestras
puerto = 'COM8'     #   COM13 --> Bluetooth ||  COM8 --> Cable
baudrate = 921600
intervalo_us = 100
muestras_grafica = 3000
v_max = 3.3
NUM_PREDICCIONES_REQUERIDAS = 2
tiempo_estado = 3.0
bloqueo_suaves = False


OFFSETS = [0.5087, 0.4862]
ESCALAS = [1.65, 1.65]

# -------------------- CARGA DE MODELO --------------------
modelo = load_model("C:\\Users\\dcast\\Desktop\\Iker C\\UNIVERSIDAD POLITÉCNICA DE MADRID\\5º AÑO\\2º CUATRIMESTRE\\jari2-ros2\\tactile_system\\Código\\Python\\MODELOS\\Prototipo_bluetooth\\mejor_modelo.keras")
clases = np.load("C:\\Users\\dcast\\Desktop\\Iker C\\UNIVERSIDAD POLITÉCNICA DE MADRID\\5º AÑO\\2º CUATRIMESTRE\\jari2-ros2\\tactile_system\\Código\\Python\\MODELOS\\Prototipo_bluetooth\\etiquetas_secuencias.npy", allow_pickle=True)

# -------------------- SERIAL --------------------
try:
    ser = serial.Serial(puerto, baudrate, timeout=0.01)
    time.sleep(2)
    print(f"🟢 Conectado a {puerto} — leyendo datos...")
except serial.SerialException:
    print("❌ Error: no se pudo conectar al puerto Bluetooth.")
    sys.exit(1)

# -------------------- INTERFAZ --------------------
app = QApplication([])
win = pg.GraphicsLayoutWidget(title="Voltaje TENG (multicanal)")
win.resize(1000, 600)
plot = win.addPlot(title="Voltajes (Sensor 1: rojo, Sensor 2: amarillo)")
tiempo = np.linspace(-((muestras_grafica - 1) * intervalo_us / 1e6), 0, muestras_grafica)
curve1 = plot.plot(tiempo, np.zeros(muestras_grafica), pen='r')
curve2 = plot.plot(tiempo, np.zeros(muestras_grafica), pen='y')
plot.setLabel('left', 'Voltaje', units='V')
plot.setLabel('bottom', 'Tiempo', units='s')
plot.setYRange(0, v_max)
win.show()

label = QLabel(win)
label.setStyleSheet("color: white; font-size: 28px; background-color: rgba(0,0,0,150); padding: 8px;")
label.move(30, 30)
label.resize(400, 50)
label.setText("😐 Normal")
label.setAttribute(Qt.WA_TransparentForMouseEvents)
label.show()

# -------------------- BUFFERS --------------------
buffer1 = []
buffer2 = []
grafica1 = np.zeros(muestras_grafica)
grafica2 = np.zeros(muestras_grafica)
lock = threading.Lock()

# -------------------- NORMALIZACIÓN --------------------
def normalizar_ventanas(X):
    X_norm = np.empty_like(X)
    for canal in range(X.shape[-1]):
        X_norm[..., canal] = (X[..., canal] - OFFSETS[canal]) / ESCALAS[canal]
    return X_norm

# -------------------- PREDICCIÓN --------------------
def predecir(buffer_local):
    ventanas = np.array([
        np.stack([
            buffer_local[0][i:i + ventana_tamano],
            buffer_local[1][i:i + ventana_tamano]
        ], axis=-1)
        for i in range(0, total_muestras, ventana_tamano)
    ])

    ventanas = normalizar_ventanas(ventanas)
    X = ventanas[np.newaxis, ...]  # (1, 4, 1000, 2)
    pred = modelo.predict(X, verbose=0)
    idx = np.argmax(pred)
    return clases[idx], np.max(pred)

# -------------------- HILO DE INFERENCIA --------------------
estado_actual = "😐 Normal"
estado_fijado = False
ultimo_cambio = time.time()

def hilo_inferencia():
    global estado_actual, estado_fijado, ultimo_cambio, bloqueo_suaves
    clase_anterior = None
    repeticiones = 0
    desplazamiento = ventana_tamano
    bloqueo_suaves = False

    while True:
        time.sleep(0.15)

        with lock:
            if len(buffer1) >= total_muestras and len(buffer2) >= total_muestras:
                buf1 = buffer1[:total_muestras]
                buf2 = buffer2[:total_muestras]
                del buffer1[:desplazamiento]
                del buffer2[:desplazamiento]
            else:
                continue

        clase, confianza = predecir((buf1, buf2))
        print(f"[{time.strftime('%H:%M:%S')}] Clase: {clase} — Confianza: {confianza:.2f}")

        # --- Golpes tienen prioridad y se muestran inmediatamente ---
        if clase in ['golpe_dedo', 'golpe_mano']:
            if clase == 'golpe_dedo':
                emocion = "😮 Atención"
            else:
                # Solo mostrar 'golpe_mano' si no se acaba de mostrar 'golpe_dedo'
                if estado_actual == "😮 Atención":
                    continue
                emocion = "😠 Enfado"

            if emocion != estado_actual:
                label.setText(emocion)
                estado_actual = emocion
                estado_fijado = True
                ultimo_cambio = time.time()
                bloqueo_suaves = True
            continue

        # --- Clases suaves requieren repeticiones ---
        if clase == clase_anterior:
            repeticiones += 1
        else:
            clase_anterior = clase
            repeticiones = 1

        if repeticiones == NUM_PREDICCIONES_REQUERIDAS:
            repeticiones = 0

            if clase != 'ninguna':
                if bloqueo_suaves and clase in ['caricia_dedo', 'caricia_mano']:
                    continue  # Ignorar caricias si estamos en modo bloqueo

                emocion = "😐 Normal"
                if clase == 'caricia_dedo':
                    emocion = "😆 Cosquillas"
                elif clase == 'caricia_mano':
                    emocion = "🥰 Cariño"

                if emocion != estado_actual:
                    label.setText(emocion)
                    estado_actual = emocion
                    estado_fijado = True
                    ultimo_cambio = time.time()

        # Restaurar estado a "Normal" después de tiempo_estado segundos
        if estado_fijado and time.time() - ultimo_cambio > tiempo_estado:
            estado_fijado = False
            bloqueo_suaves = False
            estado_actual = "😐 Normal"
            label.setText(estado_actual)

# -------------------- ACTUALIZACIÓN --------------------
def update():
    global grafica1, grafica2
    lista1 = []
    lista2 = []
    count = 0
    max_reads = 300

    while ser.in_waiting and count < max_reads:
        try:
            linea = ser.readline().decode().strip()
            if not linea:
                break
            valor1_str, valor2_str = linea.split(',')
            valor1 = int(valor1_str)
            valor2 = int(valor2_str)
            muestra1 = (valor1 * v_max) / 4095.0
            muestra2 = (valor2 * v_max) / 4095.0

            lista1.append(muestra1)
            lista2.append(muestra2)
            count += 1
        except:
            continue

    n = len(lista1)
    if n == 0:
        return

    with lock:
        buffer1.extend(lista1)
        buffer2.extend(lista2)
        while len(buffer1) > total_muestras * 2:
            buffer1.pop(0)
        while len(buffer2) > total_muestras * 2:
            buffer2.pop(0)

    grafica1 = np.roll(grafica1, -n)
    grafica2 = np.roll(grafica2, -n)
    grafica1[-n:] = lista1
    grafica2[-n:] = lista2

    curve1.setData(tiempo, grafica1)
    curve2.setData(tiempo, grafica2)

# -------------------- TEMPORIZADOR Y CIERRE --------------------
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)

threading.Thread(target=hilo_inferencia, daemon=True).start()

def cerrar():
    if ser.is_open:
        ser.close()
    print("✅ Puerto cerrado. Saliendo...")

app.aboutToQuit.connect(cerrar)
app.exec_()
