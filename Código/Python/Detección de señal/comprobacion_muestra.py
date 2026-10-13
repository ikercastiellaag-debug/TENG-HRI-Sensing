import csv
from collections import deque
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QSlider, QWidget
from PyQt5.QtCore import Qt
from pyqtgraph.Qt import QtCore

# --- Parámetros ---
csv_path = "C:\\Users\\dcast\\Documents\\Muestras\\robotmov_tela_matriz_blt.csv"
v_max = 3.3
muestras = 2000
intervalo_us = 50
paso_muestras = 5

# --- Cargar CSV ---
with open(csv_path, newline='') as f:
    reader = csv.reader(f, delimiter=';')
    next(reader)
    datos_csv = list(reader)

total_muestras = len(datos_csv)

# --- Buffers de datos ---
data1 = deque([0.0]*muestras, maxlen=muestras)
data2 = deque([0.0]*muestras, maxlen=muestras)
labels = deque(["ninguna"]*muestras, maxlen=muestras)

# --- Aplicación Qt ---
app = QApplication([])
main_widget = QWidget()
layout = QVBoxLayout()
main_widget.setLayout(layout)

# --- Interfaz gráfica PyQtGraph ---
win = pg.GraphicsLayoutWidget()
layout.addWidget(win)
main_widget.setWindowTitle("Reproducción de CSV en tiempo real")
main_widget.resize(1000, 600)
main_widget.show()

plot = win.addPlot()
tiempo = np.linspace(-((muestras - 1) * intervalo_us / 1e6), 0, muestras)
curve1 = plot.plot(tiempo, np.zeros(muestras), pen='w')
curve2 = plot.plot(tiempo, np.zeros(muestras), pen='y')
plot.setLabel('left', 'Voltaje', units='V')
plot.setLabel('bottom', 'Tiempo', units='s')
plot.setYRange(0, v_max)
plot.setTitle(f"<span style='font-size:16pt'>Etiqueta: <b>ninguna</b></span>", color='w')

# --- Deslizador de navegación ---
slider = QSlider(Qt.Horizontal)
slider.setMinimum(0)
slider.setMaximum(total_muestras - muestras)
slider.setSingleStep(1)
slider.setPageStep(100)
layout.addWidget(slider)

indice = 0
deslizador_usado = False

def actualizar_grafica_desde(indice_inicio):
    data1.clear()
    data2.clear()
    labels.clear()
    for i in range(indice_inicio, min(indice_inicio + muestras, total_muestras)):
        try:
            volt1 = float(datos_csv[i][0])
            volt2 = float(datos_csv[i][1])
            etiqueta = datos_csv[i][2] if len(datos_csv[i]) > 2 else "ninguna"
            data1.append(volt1)
            data2.append(volt2)
            labels.append(etiqueta)
        except:
            continue
    curve1.setData(tiempo, list(data1))
    curve2.setData(tiempo, list(data2))
    etiqueta_actual = labels[-1] if labels else "ninguna"
    plot.setTitle(f"<span style='font-size:16pt'>Etiqueta: <b>{etiqueta_actual}</b></span>", color='w')

# --- Evento al mover deslizador ---
def on_slider_moved(value):
    global indice, deslizador_usado
    indice = value
    deslizador_usado = True
    actualizar_grafica_desde(indice)

slider.valueChanged.connect(on_slider_moved)

# --- Actualización automática (reproducción) ---
def update():
    global indice, deslizador_usado
    if deslizador_usado:
        deslizador_usado = False
        return

    for _ in range(paso_muestras):
        if indice >= total_muestras:
            print("Fin del archivo.")
            timer.stop()
            return
        try:
            volt1 = float(datos_csv[indice][0])
            volt2 = float(datos_csv[indice][1])
            etiqueta = datos_csv[indice][2] if len(datos_csv[indice]) > 2 else "ninguna"
            data1.append(volt1)
            data2.append(volt2)
            labels.append(etiqueta)
        except:
            pass
        indice += 1

    slider.setValue(indice)
    curve1.setData(tiempo, list(data1))
    curve2.setData(tiempo, list(data2))
    etiqueta_actual = labels[-1] if labels else "ninguna"
    plot.setTitle(f"<span style='font-size:16pt'>Etiqueta: <b>{etiqueta_actual}</b></span>", color='w')

# --- Teclado para velocidad ---
def key_press(event):
    global paso_muestras
    tecla = event.text()
    if tecla == '+':
        paso_muestras += 1
        print(f"Velocidad aumentada: {paso_muestras} muestras/iteración")
    elif tecla == '-' and paso_muestras > 1:
        paso_muestras -= 1
        print(f"Velocidad reducida: {paso_muestras} muestras/iteración")

main_widget.keyPressEvent = key_press

# --- Timer principal ---
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1)

app.exec_()
