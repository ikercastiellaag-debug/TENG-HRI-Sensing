import numpy as np
import pandas as pd
import random
from collections import Counter
from scipy.signal import find_peaks

# --- Parámetros globales ---
ventana_tamano = 1000
solapamiento = 300
ventanas_por_secuencia = 3
paso_secuencia = 1
pico_en_ventana = 1
umbral_golpe = 3
umbral_caricia = 1.3
tiempo_entre_picos = 200
tiempo_entre_ventanas = 7000
OFFSETS = [0.5087, 0.4862]   # Uno para cada sensor
ESCALAS = [1.65, 1.65]   # Uno para cada sensor

def cargar_csv(ruta_csv):
    df = pd.read_csv(ruta_csv, sep=';')
    df = df.dropna()
    df['Voltaje 1 (V)'] = df['Voltaje 1 (V)'].astype(float)
    df['Voltaje 2 (V)'] = df['Voltaje 2 (V)'].astype(float)
    return df

def normalizar_ventanas(X):
    
    X_norm = np.empty_like(X)
    for canal in range(X.shape[-1]):
        X_norm[..., canal] = (X[..., canal] - OFFSETS[canal]) / ESCALAS[canal]
    return X_norm


def detectar_picos(señal, umbral):
    picos, _ = find_peaks(señal, height=umbral, distance=tiempo_entre_picos)
    return picos
'''
def generar_ninguna(df):
    voltajes1 = df['Voltaje 1 (V)'].values
    voltajes2 = df['Voltaje 2 (V)'].values
    etiquetas = df['Etiqueta'].values
    X, y = [], []
    paso = paso_secuencia * ventana_tamano
    secuencia_total = ventana_tamano * ventanas_por_secuencia

    for i in range(0, len(voltajes1) - secuencia_total, paso):
        seg1 = voltajes1[i:i + secuencia_total]
        seg2 = voltajes2[i:i + secuencia_total]
        etiquetas_segmento = etiquetas[i:i + secuencia_total]

        ventanas1 = np.array([seg1[j:j + ventana_tamano] for j in range(0, secuencia_total, ventana_tamano)])
        ventanas2 = np.array([seg2[j:j + ventana_tamano] for j in range(0, secuencia_total, ventana_tamano)])

        if ventanas1.shape != (ventanas_por_secuencia, ventana_tamano):
            continue

        p1 = detectar_picos(seg1, umbral_caricia)
        p2 = detectar_picos(seg2, umbral_caricia)

        if p1.size == 0 and p2.size == 0:
            X.append(np.stack([ventanas1, ventanas2], axis=-1))
            y.append("ninguna")

    return np.array(X), np.array(y)
'''

def generar_ninguna(df, max_muestras=None):
    voltajes1 = df['Voltaje 1 (V)'].values
    voltajes2 = df['Voltaje 2 (V)'].values
    etiquetas = df['Etiqueta'].values
    X, y = [], []

    secuencia_total = ventanas_por_secuencia * ventana_tamano
    offsets = np.arange(-ventanas_por_secuencia + ventanas_por_secuencia/48, 0.01, ventanas_por_secuencia/48)
    paso_base = int(ventana_tamano * 5)  # salta cada 5 ventanas aprox.
    muestras_generadas = 0

    for i in range(0, len(voltajes1) - secuencia_total, paso_base):
        for offset in offsets:
            inicio = int(i + offset * ventana_tamano)
            fin = inicio + secuencia_total

            if inicio < 0 or fin > len(voltajes1):
                continue

            seg1 = voltajes1[inicio:fin]
            seg2 = voltajes2[inicio:fin]

            p1 = detectar_picos(seg1, umbral_caricia)
            p2 = detectar_picos(seg2, umbral_caricia)

            if p1.size > 0 or p2.size > 0:
                continue  # solo queremos segmentos limpios

            ventanas1 = np.array([seg1[j:j + ventana_tamano] for j in range(0, secuencia_total, ventana_tamano)])
            ventanas2 = np.array([seg2[j:j + ventana_tamano] for j in range(0, secuencia_total, ventana_tamano)])

            if ventanas1.shape != (ventanas_por_secuencia, ventana_tamano):
                continue

            X.append(np.stack([ventanas1, ventanas2], axis=-1))
            y.append("ninguna")
            muestras_generadas += 1

            if max_muestras is not None and muestras_generadas >= max_muestras:
                return np.array(X), np.array(y)

    return np.array(X), np.array(y)



def generar_caricia_dedo(df):
    voltajes1 = df['Voltaje 1 (V)'].values
    voltajes2 = df['Voltaje 2 (V)'].values
    etiquetas = df['Etiqueta'].values
    X, y = [], []

    secuencia_total = ventanas_por_secuencia * ventana_tamano
    offsets = [-2, -1.75, -1.5, 1.25, -1, -0.75, -0.5, -0.25, 0]


    p1 = detectar_picos(voltajes1, umbral_caricia)
    p2 = detectar_picos(voltajes2, umbral_caricia)

    # Caso: pico solo en voltajes1
    for pico in p1:
        if any(abs(pico - p) < tiempo_entre_ventanas for p in p2):
            continue  # Saltar si también hay pico en voltajes2 (sería caricia_mano)

        for offset in offsets:
            inicio = int(pico + offset * ventana_tamano)
            fin = int(inicio + secuencia_total) 


            if inicio < 0 or fin > len(voltajes1):
                continue

            seg1 = voltajes1[inicio:fin]
            seg2 = voltajes2[inicio:fin]

            ventanas1 = np.array([seg1[i:i + ventana_tamano] for i in range(0, len(seg1), ventana_tamano)])
            ventanas2 = np.array([seg2[i:i + ventana_tamano] for i in range(0, len(seg2), ventana_tamano)])

            if ventanas1.shape == (ventanas_por_secuencia, ventana_tamano):
                X.append(np.stack([ventanas1, ventanas2], axis=-1))
                y.append("caricia_dedo")

    # Caso: pico solo en voltajes2
    for pico in p2:
        if any(abs(pico - p) < tiempo_entre_ventanas for p in p1):
            continue

        for offset in offsets:
            inicio = int(pico + offset * ventana_tamano)
            fin = int(inicio + secuencia_total)


            if inicio < 0 or fin > len(voltajes2):
                continue

            seg1 = voltajes1[inicio:fin]
            seg2 = voltajes2[inicio:fin]

            ventanas1 = np.array([seg1[i:i + ventana_tamano] for i in range(0, len(seg1), ventana_tamano)])
            ventanas2 = np.array([seg2[i:i + ventana_tamano] for i in range(0, len(seg2), ventana_tamano)])

            if ventanas1.shape == (ventanas_por_secuencia, ventana_tamano):
                X.append(np.stack([ventanas1, ventanas2], axis=-1))
                y.append("caricia_dedo")

    return np.array(X), np.array(y)


def generar_caricia_mano(df):
    voltajes1 = df['Voltaje 1 (V)'].values
    voltajes2 = df['Voltaje 2 (V)'].values
    X, y = [], []

    secuencia_total = ventanas_por_secuencia * ventana_tamano
    #offsets = np.arange(-ventanas_por_secuencia + ventanas_por_secuencia/12, 0.01, ventanas_por_secuencia/12)
    offsets = [-1.25]

    p1 = detectar_picos(voltajes1, umbral_caricia)
    p2 = detectar_picos(voltajes2, umbral_caricia)

    for pico1 in p1:
        # Buscar picos en p2 que estén cerca de pico1
        picos_cercanos = [pico2 for pico2 in p2 if abs(pico1 - pico2) < tiempo_entre_ventanas]

        if not picos_cercanos:
            continue  # No hay pico simultáneo en p2 → no es caricia_mano

        # Usamos el más cercano
        pico2 = min(picos_cercanos, key=lambda x: abs(x - pico1))
        pico = min(pico1, pico2)  # Tomamos el más adelantado como referencia

        for offset in offsets:
            inicio = int(pico + offset * ventana_tamano)
            fin = int(inicio + secuencia_total)

            if inicio < 0 or fin > len(voltajes1):
                continue

            seg1 = voltajes1[inicio:fin]
            seg2 = voltajes2[inicio:fin]

            ventanas1 = np.array([seg1[i:i + ventana_tamano] for i in range(0, len(seg1), ventana_tamano)])
            ventanas2 = np.array([seg2[i:i + ventana_tamano] for i in range(0, len(seg2), ventana_tamano)])

            if ventanas1.shape == (ventanas_por_secuencia, ventana_tamano):
                X.append(np.stack([ventanas1, ventanas2], axis=-1))
                y.append("caricia_mano")

    return np.array(X), np.array(y)

def generar_golpe_dedo(df):
    voltajes1 = df['Voltaje 1 (V)'].values
    voltajes2 = df['Voltaje 2 (V)'].values
    X, y = [], []

    secuencia_total = ventanas_por_secuencia * ventana_tamano
    #offsets = [-2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25,  0]
    offsets = np.arange(-ventanas_por_secuencia + ventanas_por_secuencia/24, 0.01, ventanas_por_secuencia/24)

    p1 = detectar_picos(voltajes1, umbral_golpe)
    p2 = detectar_picos(voltajes2, umbral_golpe)

    # Caso: pico solo en voltajes1
    for pico in p1:
        if any(abs(pico - p) < tiempo_entre_ventanas for p in p2):
            continue  # Si hay pico cercano en voltajes2 → sería golpe_mano

        for offset in offsets:
            inicio = int(pico + offset * ventana_tamano)
            fin = int(inicio + secuencia_total)


            if inicio < 0 or fin > len(voltajes1):
                continue

            seg1 = voltajes1[inicio:fin]
            seg2 = voltajes2[inicio:fin]

            ventanas1 = np.array([seg1[i:i + ventana_tamano] for i in range(0, len(seg1), ventana_tamano)])
            ventanas2 = np.array([seg2[i:i + ventana_tamano] for i in range(0, len(seg2), ventana_tamano)])

            if ventanas1.shape == (ventanas_por_secuencia, ventana_tamano):
                X.append(np.stack([ventanas1, ventanas2], axis=-1))
                y.append("golpe_dedo")

    # Caso: pico solo en voltajes2
    for pico in p2:
        if any(abs(pico - p) < tiempo_entre_ventanas for p in p1):
            continue  # Si hay pico cercano en voltajes1 → sería golpe_mano

        for offset in offsets:
            inicio = int(pico + offset * ventana_tamano)
            fin = int(inicio + secuencia_total)


            if inicio < 0 or fin > len(voltajes2):
                continue

            seg1 = voltajes1[inicio:fin]
            seg2 = voltajes2[inicio:fin]

            ventanas1 = np.array([seg1[i:i + ventana_tamano] for i in range(0, len(seg1), ventana_tamano)])
            ventanas2 = np.array([seg2[i:i + ventana_tamano] for i in range(0, len(seg2), ventana_tamano)])

            if ventanas1.shape == (ventanas_por_secuencia, ventana_tamano):
                X.append(np.stack([ventanas1, ventanas2], axis=-1))
                y.append("golpe_dedo")

    return np.array(X), np.array(y)


def generar_golpe_mano(df):
    voltajes1 = df['Voltaje 1 (V)'].values
    voltajes2 = df['Voltaje 2 (V)'].values
    X, y = [], []

    secuencia_total = ventanas_por_secuencia * ventana_tamano
    #offsets = [-2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0]
    offsets = [-2.25, -1.25]
    #offsets = np.arange(-ventanas_por_secuencia + ventanas_por_secuencia/24, 0.01, ventanas_por_secuencia/24)


    p1 = detectar_picos(voltajes1, umbral_golpe)
    p2 = detectar_picos(voltajes2, umbral_golpe)

    for pico1 in p1:
        # Buscar picos en p2 que estén cerca de pico1
        picos_cercanos = [pico2 for pico2 in p2 if abs(pico1 - pico2) < tiempo_entre_ventanas]

        if not picos_cercanos:
            continue  # No hay pico simultáneo en p2 → no es golpe_mano

        pico2 = min(picos_cercanos, key=lambda x: abs(x - pico1))
        pico = min(pico1, pico2)  # Referencia común para aplicar offsets

        for offset in offsets:
            inicio = int(pico + offset * ventana_tamano)
            fin = int(inicio + secuencia_total)


            if inicio < 0 or fin > len(voltajes1):
                continue

            seg1 = voltajes1[inicio:fin]
            seg2 = voltajes2[inicio:fin]

            ventanas1 = np.array([seg1[i:i + ventana_tamano] for i in range(0, len(seg1), ventana_tamano)])
            ventanas2 = np.array([seg2[i:i + ventana_tamano] for i in range(0, len(seg2), ventana_tamano)])

            if ventanas1.shape == (ventanas_por_secuencia, ventana_tamano):
                X.append(np.stack([ventanas1, ventanas2], axis=-1))
                y.append("golpe_mano")

    return np.array(X), np.array(y)

import json
import os

def guardar_configuracion_experimento(
    ruta_csv,
    muestras_por_clase,
    total_muestras,
    modo_comunicacion,
    ruta_guardado
):
    config = {
        "preprocesado": {
             "ventana_tamano": ventana_tamano,
            "solapamiento": solapamiento,
            "ventanas_por_secuencia": ventanas_por_secuencia,
            "paso_secuencia": paso_secuencia,
            "pico_en_ventana": pico_en_ventana,
            "umbral_caricia": umbral_caricia,
            "umbral_golpe": umbral_golpe,
            "tiempo_entre_picos": tiempo_entre_picos,
            "tiempo_entre_ventanas": tiempo_entre_ventanas,
            "offsets": OFFSETS,
            "escalas": ESCALAS,
            "normalizacion": "manual"
        },
        "datos": {
            "ruta_csv": os.path.basename(ruta_csv),
            "muestras_por_clase": muestras_por_clase,
            "total_muestras": total_muestras,
            "modo_comunicacion": modo_comunicacion,
        }
    }

    with open(ruta_guardado, "w") as f:
        json.dump(config, f, indent=4)

    print(f"✅ Configuración guardada en {ruta_guardado}")
