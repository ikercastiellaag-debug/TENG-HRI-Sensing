import serial
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from pyqtgraph.Qt import QtCore
import numpy as np
import csv
import time

# --- Configuración ---
puerto = 'COM8'     # Puerto por cable USB
baudios = 921600
muestras = 3000
v_max = 3.3
intervalo_us = 100  # 100 µs = 10 kHz

# --- Serial ---
ser = serial.Serial(puerto, baudios, timeout=0.1)
ser.reset_input_buffer()
time.sleep(1)

# --- Gráfica ---
app = QApplication([])
win = pg.GraphicsLayoutWidget(title="Señales desde ESP32")
win.show()
plot = win.addPlot(title="ADC ESP32:")
tiempo = np.linspace(-((muestras - 1) * intervalo_us / 1e6), 0, muestras)
curve1 = plot.plot(tiempo, np.zeros(muestras), pen='w')  # Blanco Pin 34
curve2 = plot.plot(tiempo, np.zeros(muestras), pen='y')  # Amarillo Pin 32

plot.setLabel('left', 'Voltaje', units='V')
plot.setLabel('bottom', 'Tiempo', units='s')
plot.setYRange(0, v_max)

# --- Datos ---
data1 = np.zeros(muestras)
data2 = np.zeros(muestras)
raw_values1, raw_values2 = [], []
labels = []

# --- Frecuencia de muestreo ---
contador_muestras = 0
tiempo_ultima_medicion = time.time()

current_label = None  # Etiqueta actual

def update():
    global data1, data2, raw_values1, raw_values2, labels, current_label
    global contador_muestras, tiempo_ultima_medicion

    count = 0
    max_reads = 200
    muestras_recibidas = 0

    while ser.in_waiting and count < max_reads:
        try:
            line = ser.readline().decode().strip()
            if not line or ',' not in line:
                continue

            partes = line.split(',')
            if len(partes) != 2:
                continue

            try:
                val1 = int(partes[0])
                val2 = int(partes[1])
            except ValueError:
                continue

            volt1 = (val1 * v_max) / 4095.0
            volt2 = (val2 * v_max) / 4095.0

            data1 = np.roll(data1, -1)
            data2 = np.roll(data2, -1)
            data1[-1] = volt1
            data2[-1] = volt2

            raw_values1.append(volt1)
            raw_values2.append(volt2)
            labels.append(current_label if current_label is not None else "ninguna")

            count += 1
            muestras_recibidas += 1
        except Exception as e:
            print(f"Error de lectura: {line} → {e}")
            continue

    curve1.setData(tiempo, data1)
    curve2.setData(tiempo, data2)

    # --- Frecuencia de muestreo en consola ---
    contador_muestras += muestras_recibidas
    ahora = time.time()
    if ahora - tiempo_ultima_medicion >= 1.0:
        frecuencia = contador_muestras / (ahora - tiempo_ultima_medicion)
        print(f"⏱ Frecuencia de muestreo aproximada: {frecuencia:.1f} Hz")
        contador_muestras = 0
        tiempo_ultima_medicion = ahora

# --- Temporizador ---
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)

# --- Guardar datos al salir ---
def guardar_csv():
    with open("C:\\Users\\dcast\\Documents\\Muestras\\muestra_matriz_cable_prototipo.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Voltaje 1 (V)', 'Voltaje 2 (V)', 'Etiqueta'])
        for v1, v2, label in zip(raw_values1, raw_values2, labels):
            writer.writerow([v1, v2, label])
    print("✅ Datos guardados")

app.aboutToQuit.connect(guardar_csv)

# --- Manejo de teclado para etiquetas ---
def key_press(event):
    global current_label
    key = event.text()
    if key == '1':
        current_label = 'golpe_dedo'
    elif key == '2':
        current_label = 'caricia_dedo'
    elif key == '3':
        current_label = 'golpe_mano'
    elif key == '4':
        current_label = 'caricia_mano'
    elif key == 'q':
        current_label = None
    print(f"Etiqueta actual: {current_label}")

app.focusChanged.connect(lambda old, new: app.installEventFilter(win))
win.keyPressEvent = key_press

# --- Ejecutar ---
app.exec_()
