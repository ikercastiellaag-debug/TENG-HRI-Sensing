import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import serial
import time
import threading
from tensorflow.keras.models import load_model
import os

# ---------------- PARAMETROS ----------------
VENTANA_TAMANO = 1000
VENTANAS_POR_SECUENCIA = 4
TOTAL_MUESTRAS = VENTANA_TAMANO * VENTANAS_POR_SECUENCIA
OFFSETS = [0.77, 0.76]
ESCALAS = [1.65, 1.65]
NUM_PREDICCIONES_REQUERIDAS = 2
PUERTO_BT = '/dev/rfcomm0'  #  Cambiar a 'COM13' si usas Windows
BAUDRATE = 921600

# ---------------- MODELO ----------------
base = os.path.dirname(__file__)
modelo = load_model(os.path.join(base, 'model', 'mejor_modelo.keras'))
clases = np.load(os.path.join(base, 'model', 'etiquetas.npy'), allow_pickle=True)

# ---------------- NODO ROS ----------------
class ContactPublisher(Node):
    def __init__(self):
        super().__init__('contact_publisher')
        self.publisher = self.create_publisher(String, 'contact_type', 10)

        # Conexión Bluetooth serial
        try:
            self.serial = serial.Serial(PUERTO_BT, BAUDRATE, timeout=0.01)
            time.sleep(2)
            self.get_logger().info(f'🟢 Conectado a {PUERTO_BT}')
        except Exception as e:
            self.get_logger().error(f'❌ No se pudo abrir {PUERTO_BT}: {e}')
            raise e

        # Buffers de datos
        self.buffer1 = []
        self.buffer2 = []
        self.lock = threading.Lock()

        # Inferencia en hilo paralelo
        self.repeticiones = 0
        self.clase_anterior = None
        threading.Thread(target=self.inferencia_loop, daemon=True).start()

        # Lector en temporizador ROS
        self.create_timer(0.03, self.leer_datos)

    def normalizar_ventanas(self, X):
        X_norm = np.empty_like(X)
        for canal in range(X.shape[-1]):
            X_norm[..., canal] = (X[..., canal] - OFFSETS[canal]) / ESCALAS[canal]
        return X_norm

    def predecir(self, buf1, buf2):
        ventanas = np.array([
            np.stack([
                buf1[i:i + VENTANA_TAMANO],
                buf2[i:i + VENTANA_TAMANO]
            ], axis=-1)
            for i in range(0, TOTAL_MUESTRAS, VENTANA_TAMANO)
        ])
        X = self.normalizar_ventanas(ventanas)
        X = X[np.newaxis, ...]
        pred = modelo.predict(X, verbose=0)
        idx = np.argmax(pred)
        return clases[idx], np.max(pred)

    def inferencia_loop(self):
        while rclpy.ok():
            time.sleep(0.15)
            with self.lock:
                if len(self.buffer1) < TOTAL_MUESTRAS:
                    continue
                buf1 = self.buffer1[:TOTAL_MUESTRAS]
                buf2 = self.buffer2[:TOTAL_MUESTRAS]
                del self.buffer1[:VENTANA_TAMANO]
                del self.buffer2[:VENTANA_TAMANO]

            clase, confianza = self.predecir(buf1, buf2)
            self.get_logger().info(f'Clase: {clase} — Confianza: {confianza:.2f}')

            if clase == self.clase_anterior:
                self.repeticiones += 1
            else:
                self.clase_anterior = clase
                self.repeticiones = 1

            if self.repeticiones == NUM_PREDICCIONES_REQUERIDAS:
                msg = String()
                msg.data = clase
                self.publisher.publish(msg)
                self.repeticiones = 0

    def leer_datos(self):
        try:
            while self.serial.in_waiting:
                linea = self.serial.readline().decode().strip()
                if not linea:
                    break
                v1, v2 = linea.split(',')
                muestra1 = (int(v1) * 3.3) / 4095.0
                muestra2 = (int(v2) * 3.3) / 4095.0

                with self.lock:
                    self.buffer1.append(muestra1)
                    self.buffer2.append(muestra2)

                    while len(self.buffer1) > TOTAL_MUESTRAS * 2:
                        self.buffer1.pop(0)
                        self.buffer2.pop(0)
        except Exception as e:
            self.get_logger().warn(f'Lectura fallida: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ContactPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.serial.is_open:
            node.serial.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
