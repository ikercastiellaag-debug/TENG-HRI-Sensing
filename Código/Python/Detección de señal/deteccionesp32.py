import serial
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from pyqtgraph.Qt import QtCore
import numpy as np
import csv

# --- Configuración ---
puerto = '/dev/comm0'  # COM8 --> windows || /dev/rfcomm0 --> ros2
baudios = 921600
muestras = 3000
v_max = 3.3
intervalo_us = 100  # ESP32 → 100 µs

# --- Serial ---
ser = serial.Serial(puerto, baudios, timeout=1)

# --- Gráfica ---
app = QApplication([])
win = pg.GraphicsLayoutWidget(title="Comparación señal cruda vs suavizada")
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

def update():
    global data, raw_values
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
            count += 1
        except Exception as e:
            print(f"Error: {e}")

    # Curva cruda
    curve_raw.setData(tiempo, data)

# --- Temporizador ---
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)

# --- Guardar datos al salir ---
def guardar_csv():
    with open('datos_teng.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Voltaje (V)'])
        for val in raw_values:
            writer.writerow([val])
    print("Datos guardados en datos_teng.csv")

app.aboutToQuit.connect(guardar_csv)

# --- Ejecutar ---
app.exec_()
