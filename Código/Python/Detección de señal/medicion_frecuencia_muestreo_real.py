import time
import serial

# --- Configuración ---
puerto = 'COM13'
baudios = 921600
N = 1000000  # Número de muestras a capturar

# --- Conexión ---
try:
    ser = serial.Serial(puerto, baudios, timeout=1)
    input("Pulsa Enter para empezar a medir...")
except serial.SerialException as e:
    print(f"Error al abrir el puerto: {e}")
    exit()

# --- Medición ---
datos = []
start = time.time()

while len(datos) < N:
    try:
        linea = ser.readline().decode(errors='ignore').strip()
        if linea.isdigit():  # solo si es un valor numérico
            datos.append(int(linea))
    except Exception as e:
        print(f"Error leyendo: {e}")
        continue

end = time.time()
ser.close()

# --- Cálculo ---
duracion = end - start
frec_muestreo = len(datos) / duracion

# --- Resultado ---
print(f"\nCapturadas {len(datos)} muestras en {duracion:.2f} segundos")
print(f"Frecuencia de muestreo estimada: {frec_muestreo:.2f} Hz")
