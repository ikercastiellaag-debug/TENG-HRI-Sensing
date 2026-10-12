import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import serial
import time
import threading
import numpy as np
from tensorflow.keras.models import load_model
import os

VENTANA_TAMANO = 1000
VENTANAS_POR_SECUENCIA = 3
TOTAL_MUESTRAS = VENTANA_TAMANO * VENTANAS_POR_SECUENCIA
V_MAX = 3.3
NUM_PREDICCIONES_REQUERIDAS = 2
DESPLAZAMIENTO = VENTANA_TAMANO
TIEMPO_ESTADO = 3.0

OFFSETS = [0.5087, 0.4862]
ESCALAS = [1.65, 1.65]

CLASE_TO_EMOCION = {
    "caricia_dedo": "very_happy",
    "caricia_mano": "happy",
    "golpe_dedo": "scared",
    "golpe_mano": "angry",
    "ninguna": "off"
}

class TactileDetector(Node):
    def __init__(self):
        super().__init__('tactile_detector_node')

        self.declare_parameter('modo', 'bluetooth')
        self.declare_parameter('puerto', '/dev/ttyUSB0')

        self.modo = self.get_parameter('modo').get_parameter_value().string_value
        self.puerto = self.get_parameter('puerto').get_parameter_value().string_value

        ruta_base = os.path.expanduser('~/Documentos/MODELOS/Robot_sintela_bluetooth')
        modelo_path = os.path.join(ruta_base, 'mejor_modelo.keras')
        etiquetas_path = os.path.join(ruta_base, 'etiquetas_secuencias.npy')


        self.get_logger().info(f'Cargando modelo desde: {modelo_path}')
        self.modelo = load_model(modelo_path)
        self.clases = np.load(etiquetas_path, allow_pickle=True)

        self.buffer1 = []
        self.buffer2 = []
        self.lock = threading.Lock()
        self.publisher_ = self.create_publisher(String, 'contacto_detectado', 10)

        self.estado_actual = "ninguna"
        self.estado_fijado = False
        self.ultimo_cambio = time.time()
        self.bloqueo_suaves = False

        try:
            self.ser = serial.Serial(self.puerto, 921600, timeout=0.01)
            time.sleep(2)
            self.get_logger().info(f'Conectado a {self.puerto}')
        except serial.SerialException:
            self.get_logger().error(f'No se pudo abrir el puerto {self.puerto}')
            rclpy.shutdown()
            return

        self.timer = self.create_timer(0.01, self.read_serial)
        threading.Thread(target=self.hilo_inferencia, daemon=True).start()

    def normalizar_ventanas(self, X):
        X_norm = np.empty_like(X)
        for canal in range(X.shape[-1]):
            X_norm[..., canal] = (X[..., canal] - OFFSETS[canal]) / ESCALAS[canal]
        return X_norm

    def predecir(self, buffer_local):
        ventanas = np.array([
            np.stack([
                buffer_local[0][i:i + VENTANA_TAMANO],
                buffer_local[1][i:i + VENTANA_TAMANO]
            ], axis=-1)
            for i in range(0, TOTAL_MUESTRAS, VENTANA_TAMANO)
        ])
        ventanas = self.normalizar_ventanas(ventanas)
        X = ventanas[np.newaxis, ...]
        pred = self.modelo.predict(X, verbose=0)
        idx = np.argmax(pred)
        return self.clases[idx], np.max(pred)

    def hilo_inferencia(self):
        clase_anterior = None
        repeticiones = 0

        while rclpy.ok():
            time.sleep(0.15)

            with self.lock:
                if len(self.buffer1) >= TOTAL_MUESTRAS and len(self.buffer2) >= TOTAL_MUESTRAS:
                    buf1 = self.buffer1[:TOTAL_MUESTRAS]
                    buf2 = self.buffer2[:TOTAL_MUESTRAS]
                    del self.buffer1[:DESPLAZAMIENTO]
                    del self.buffer2[:DESPLAZAMIENTO]
                else:
                    continue

            clase, confianza = self.predecir((buf1, buf2))
            self.get_logger().info(f'Clase: {clase} — Confianza: {confianza:.2f}')

            if clase in ['golpe_dedo', 'golpe_mano']:
                if clase == 'golpe_mano' and self.estado_actual == 'golpe_dedo':
                    continue
                self.publicar_estado(clase)
                self.bloqueo_suaves = True
                continue

            if clase == clase_anterior:
                repeticiones += 1
            else:
                clase_anterior = clase
                repeticiones = 1

            if repeticiones >= NUM_PREDICCIONES_REQUERIDAS:
                repeticiones = 0
                if clase != 'ninguna':
                    if self.bloqueo_suaves and clase in ['caricia_dedo', 'caricia_mano']:
                        continue
                    self.publicar_estado(clase)

            if self.estado_fijado and (time.time() - self.ultimo_cambio > TIEMPO_ESTADO):
                self.estado_fijado = False
                self.bloqueo_suaves = False
                self.publicar_estado("ninguna")

    def publicar_estado(self, clase):
        if clase != self.estado_actual:
            self.estado_actual = clase
            self.estado_fijado = True
            self.ultimo_cambio = time.time()
            emocion = CLASE_TO_EMOCION.get(clase, "off")
            msg = String()
            msg.data = emocion
            self.publisher_.publish(msg)
            self.get_logger().info(f'📤 Publicado: {emocion} (clase: {clase})')

    def read_serial(self):
        count = 0
        max_reads = 300
        lista1, lista2 = [], []

        while self.ser.in_waiting and count < max_reads:
            try:
                linea = self.ser.readline().decode().strip()
                if not linea:
                    break
                valor1_str, valor2_str = linea.split(',')
                valor1 = int(valor1_str)
                valor2 = int(valor2_str)
                muestra1 = (valor1 * V_MAX) / 4095.0
                muestra2 = (valor2 * V_MAX) / 4095.0
                lista1.append(muestra1)
                lista2.append(muestra2)
                count += 1
            except:
                continue

        with self.lock:
            self.buffer1.extend(lista1)
            self.buffer2.extend(lista2)
            while len(self.buffer1) > TOTAL_MUESTRAS * 2:
                self.buffer1.pop(0)
            while len(self.buffer2) > TOTAL_MUESTRAS * 2:
                self.buffer2.pop(0)

def main(args=None):
    rclpy.init(args=args)
    node = TactileDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
