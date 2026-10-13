import serial
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from pyqtgraph.Qt import QtCore
import numpy as np
import csv

# --- Configuración ---
puerto = 'COM13'
baudios = 921600
muestras = 3000
v_max = 3.3
intervalo_us = 100  # ESP32 → 100 µs

# --- Serial ---
ser = serial.Serial(puerto, baudios, timeout=1)

# --- Gráfica ---
app = QApplication([])
win = pg.GraphicsLayoutWidget(title="Comparación señal cruda vs etiquetas")
win.show()
plot = win.addPlot(title="ADC ESP32:")
tiempo = np.linspace(-((muestras - 1) * intervalo_us / 1e6), 0, muestras)
curve_raw = plot.plot(tiempo, np.zeros(muestras), pen='w')

plot.setLabel('left', 'Voltaje', units='V')
plot.setLabel('bottom', 'Tiempo', units='s')
plot.setYRange(0, v_max)

# --- Datos ---
data = np.zeros(muestras)
raw_values = []
labels = []

current_label = None  # Etiqueta actual

def update():
    global data, raw_values, labels, current_label
    count = 0
    max_reads = 200
    while ser.in_waiting and count < max_reads:
        try:
            line = ser.readline().decode().strip()
            value = int(line)
            voltaje = (value * v_max) / 4095.0
            data = np.roll(data, -1)
            data[-1] = voltaje
            raw_values.append(voltaje)
            labels.append(current_label if current_label is not None else "ninguna")
            count += 1
        except Exception as e:
            print(f"Error: {e}")
    curve_raw.setData(tiempo, data)

# --- Temporizador ---
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)

# --- Guardar datos al salir ---
def guardar_csv():
    with open('prueba_medicion.csv', 'w', newline='') as csvfile: #prueba_medicion
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Voltaje (V)', 'Etiqueta'])
        for val, label in zip(raw_values, labels):
            writer.writerow([val, label])
    print("Datos guardados en prueba_sin_ruido.csv")

app.aboutToQuit.connect(guardar_csv)

# --- Manejo de teclado para asignar etiquetas ---
def key_press(event):
    global current_label
    key = event.text()
    if key == '1':
        current_label = 'golpe_dedo'
        print("Marcando como: golpe con dedo")
    elif key == '3':
        current_label = 'golpe_mano'
        print("Marcando como: golpe con mano")
    elif key == '2':
        current_label = 'caricia_dedo'
        print("Marcando como: caricia con dedo")
    elif key == '4':
        current_label = 'caricia_mano'
        print("Marcando como: caricia con mano")
    elif key == 'q':
        current_label = None
        print("Etiqueta borrada")

app.focusChanged.connect(lambda old, new: app.installEventFilter(win))
win.keyPressEvent = key_press

# --- Ejecutar ---
app.exec_()
