import time
import serial

# --- Configuración ---
puerto = '/dev/rfcomm0'   # <- Usa el puerto Bluetooth
baudios = 921600
N = 100000  # Número de muestras a capturar (líneas)

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
        if ',' in linea:
            val1_str, val2_str = linea.split(',')
            if val1_str.isdigit() and val2_str.isdigit():
                datos.append((int(val1_str), int(val2_str)))
    except Exception as e:
        print(f"Error leyendo: {e}")
        continue

end = time.time()
ser.close()

# --- Cálculo ---
duracion = end - start
frec_muestreo = len(datos) / duracion

# --- Resultado ---
print(f"\nCapturadas {len(datos)} líneas (muestras dobles) en {duracion:.2f} segundos")
print(f"Frecuencia de muestreo estimada: {frec_muestreo:.2f} Hz")
