import numpy as np
import pandas as pd
import random
import joblib
from collections import Counter
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler

# --- Parámetros globales ---

ventana_tamaño = 1000         # Número de muestras en cada ventana de análisis
solapamiento = 300            # Número de muestras solapadas entre ventanas consecutivas
ventanas_por_secuencia = 4   # Cuántas ventanas forman una secuencia completa para el modelo (10 = 1s)
paso_secuencia = 2            # Paso con el que se recorre la lista de ventanas al generar secuencias

pico_en_ventana = 1          # Cantidad de ventanas hacia atrás desde un pico para crear una secuencia (para golpes)
umbral = 2.2                  # Umbral mínimo de voltaje para detectar un pico como golpe
tiempo_entre_picos = 100       # Distancia mínima (en muestras) entre dos picos consecutivos

OFFSET = 1.5                  # Offset usado para normalizar el voltaje
ESCALA = 1.65                 # Escala usada para normalizar el voltaje (normaliza entre 0 y 1)


def cargar_csv(ruta_csv):
    df = pd.read_csv(ruta_csv, sep=';')
    df = df.dropna()
    df['Voltaje (V)'] = df['Voltaje (V)'].astype(float)
    return df

def generar_caricia(df):
    df = df[df['Etiqueta'] == 'caricia_dedo']
    X, y = [], []

    voltajes = df['Voltaje (V)'].values
    total_muestras = len(voltajes)

    for i in range(0, total_muestras - ventana_tamaño * ventanas_por_secuencia, ventana_tamaño - solapamiento):
        segmento = voltajes[i:i + ventana_tamaño * ventanas_por_secuencia]
        ventanas = np.array([
            segmento[j:j + ventana_tamaño]
            for j in range(0, len(segmento), ventana_tamaño)
        ])

        if ventanas.shape == (ventanas_por_secuencia, ventana_tamaño):
            X.append(ventanas)
            y.append("caricia_dedo")

    '''
    # --- VERSIÓN ORIGINAL  ---
    etiquetas = df['Etiqueta'].values
    X_ventanas, y_ventanas = [], []
    for i in range(0, len(voltajes) - ventana_tamaño, ventana_tamaño - solapamiento):
        v = voltajes[i:i + ventana_tamaño]
        e = etiquetas[i:i + ventana_tamaño]
        contador = Counter(e)
        etiqueta, repes = contador.most_common(1)[0]
        if repes / len(e) >= 0.9:
            X_ventanas.append(v)
            y_ventanas.append(etiqueta)
    for i in range(0, len(X_ventanas) - ventanas_por_secuencia + 1, paso_secuencia):
        secuencia = X_ventanas[i:i + ventanas_por_secuencia]
        etiquetas_seq = y_ventanas[i:i + ventanas_por_secuencia]
        contador = Counter(etiquetas_seq)
        mas_comunes = contador.most_common()
        if len(mas_comunes) > 1 and mas_comunes[0][1] == mas_comunes[1][1]:
            etiqueta_final = etiquetas_seq[len(etiquetas_seq) // 2]
        else:
            etiqueta_final = mas_comunes[0][0]
        X.append(secuencia)
        y.append(etiqueta_final)
    '''

    return np.array(X)[..., np.newaxis], np.array(y)


def generar_golpes(df):
    # Ya no se filtra por etiqueta. Se trabaja con el DataFrame completo
    voltajes = df['Voltaje (V)'].values
    etiquetas = df['Etiqueta'].values
    X, y = [], []

    picos, _ = find_peaks(voltajes, height=umbral, distance=tiempo_entre_picos)

    for pico in picos:
        for offset in [-7, -5, -3, 0]:
            inicio = pico + offset * ventana_tamaño
            fin = inicio + ventanas_por_secuencia * ventana_tamaño

            if inicio >= 0 and fin <= len(voltajes):
                segmento = voltajes[inicio:fin]
                etiquetas_segmento = etiquetas[inicio:fin]
                contador = Counter(etiquetas_segmento)
                etiqueta_mayoritaria, repes = contador.most_common(1)[0]

                if etiqueta_mayoritaria != "golpe_dedo" or repes / len(etiquetas_segmento) < 0.7:
                    continue  # Saltar si no es golpe claro

                ventanas = np.array([
                    segmento[i:i + ventana_tamaño]
                    for i in range(0, len(segmento), ventana_tamaño)
                ])

                if ventanas.shape == (ventanas_por_secuencia, ventana_tamaño):
                    X.append(ventanas)
                    y.append("golpe_dedo")

    return np.array(X)[..., np.newaxis], np.array(y)

    '''
    # --- VERSIÓN ORIGINAL ---
    df = df[df['Etiqueta'] == 'golpe_dedo']
    voltajes = df['Voltaje (V)'].values
    etiquetas = df['Etiqueta'].values
    ...
    '''

def generar_ninguna(df, n_secuencias):
    df = df[df['Etiqueta'] == 'ninguna']
    voltajes = df['Voltaje (V)'].values
    X, y = [], []
    total_muestras = ventana_tamaño * ventanas_por_secuencia
    for _ in range(n_secuencias):
        idx = random.randint(0, len(voltajes) - total_muestras)
        segmento = voltajes[idx:idx + total_muestras]
        ventanas = np.array([
            segmento[i:i + ventana_tamaño] for i in range(0, len(segmento), ventana_tamaño)
        ])
        if ventanas.shape == (ventanas_por_secuencia, ventana_tamaño):
            X.append(ventanas)
            y.append("ninguna")
    return np.array(X)[..., np.newaxis], np.array(y)


def normalizar(X):
    return (X - OFFSET) / ESCALA