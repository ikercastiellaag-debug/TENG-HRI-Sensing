import serial
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from pyqtgraph.Qt import QtCore
from collections import deque
import numpy as np
import csv
import time
from datetime import datetime

# --- Configuración ---
puerto ='/dev/ttyUSB1' # '/dev/rfcomm0'  '/dev/ttyUSB0'
baudios = 921600
muestras = 1000
v_max = 3.3
intervalo_us = 100

# --- Serial ---
ser = serial.Serial(puerto, baudios, timeout=1)

# --- Datos con deque (más rápido que np.roll)
data1 = deque([0.0]*muestras, maxlen=muestras)
data2 = deque([0.0]*muestras, maxlen=muestras)
raw_values1, raw_values2, labels = [], [], []
current_label = None

# --- Interfaz gráfica optimizada ---
app = QApplication([])
win = pg.GraphicsLayoutWidget(title="Señales desde ESP32")
win.show()
plot = win.addPlot(title="ADC ESP32:")
tiempo = np.linspace(-((muestras - 1) * intervalo_us / 1e6), 0, muestras)
curve1 = plot.plot(tiempo, np.zeros(muestras), pen='w')
curve2 = plot.plot(tiempo, np.zeros(muestras), pen='y')
plot.setLabel('left', 'Voltaje', units='V')
plot.setLabel('bottom', 'Tiempo', units='s')
plot.setYRange(0, v_max)

# --- Frecuencia de muestreo ---
contador_muestras = 0
tiempo_ultima_medicion = time.time()
frame_count = 0

def update():
    global contador_muestras, tiempo_ultima_medicion, frame_count

    muestras_recibidas = 0
    try:
        while ser.in_waiting:
            line = ser.readline().decode(errors='ignore').strip()
            if ',' in line:
                val1_str, val2_str = line.split(',')
                val1, val2 = int(val1_str), int(val2_str)
                volt1 = (val1 * v_max) / 4095.0
                volt2 = (val2 * v_max) / 4095.0

                data1.append(volt1)
                data2.append(volt2)

                raw_values1.append(volt1)
                raw_values2.append(volt2)
                labels.append(current_label if current_label else "ninguna")
                muestras_recibidas += 1
    except:
        pass

    # Actualización de gráfica cada 3 frames si hay muestras
    frame_count += 1
    if muestras_recibidas > 0 and frame_count % 3 == 0:
        curve1.setData(tiempo, list(data1))
        curve2.setData(tiempo, list(data2))

    # Frecuencia de muestreo
    contador_muestras += muestras_recibidas
    ahora = time.time()
    if ahora - tiempo_ultima_medicion >= 1.0:
        frecuencia = contador_muestras / (ahora - tiempo_ultima_medicion)
        print(f" Frecuencia de muestreo: {frecuencia:.1f} Hz")
        contador_muestras = 0
        tiempo_ultima_medicion = ahora

# --- Guardar CSV al salir ---
def guardar_csv():
    filename = f"/home/jari/Documentos/Muestras/robotmov_sintela_cable.csv"
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Voltaje 1 (V)', 'Voltaje 2 (V)', 'Etiqueta'])
        for v1, v2, label in zip(raw_values1, raw_values2, labels):
            writer.writerow([v1, v2, label])
    print(f" CSV guardado en: {filename}")

app.aboutToQuit.connect(guardar_csv)

# --- Etiquetas desde teclado ---
def key_press(event):
    global current_label
    key = event.text()
    if key == '1': current_label = 'golpe_dedo'
    elif key == '2': current_label = 'caricia_dedo'
    elif key == '3': current_label = 'golpe_mano'
    elif key == '4': current_label = 'caricia_mano'
    elif key == 'q': current_label = None
    print(f"Etiqueta actual: {current_label}")

app.focusChanged.connect(lambda old, new: app.installEventFilter(win))
win.keyPressEvent = key_press

# --- Bucle principal optimizado ---
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)
app.exec_()
