import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, TimeDistributed, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import Counter
from scipy.signal import find_peaks
import random

# --- Parámetros ---
ventana_tamaño = 1000
solapamiento = 200
ventanas_por_secuencia = 15
paso_secuencia = 2
pico_en_ventana = 5   # Ventanas antes de un pico
umbral = 2.2           # Mínimo voltaje para detectar un pico
tiempo_entre_picos = 50 # Distancia mínima entre picos
OFFSET = 1.5
ESCALA = 1.65

ruta_csv = 'C:\\Users\\dcast\\Desktop\\Iker C\\UNIVERSIDAD POLITÉCNICA DE MADRID\\5º AÑO\\2º CUATRIMESTRE\\TFG\\Código\\Python\\Detección de señal\\datos_caricia_ninguna.csv'

# --- Cargar datos ---
df = pd.read_csv(ruta_csv, sep=';')
df = df.dropna()
df['Voltaje (V)'] = df['Voltaje (V)'].astype(float)

# --- Separar por clase ---
df_caricia = df[df['Etiqueta'] == 'caricia_dedo']
df_golpe = df[df['Etiqueta'] == 'golpe_dedo']
df_nada = df[df['Etiqueta'] == 'ninguna']

# --- Funciones especializadas para cada clase ---
def generar_caricia(df):
    X, y = [], []
    voltajes = df['Voltaje (V)'].values
    etiquetas = df['Etiqueta'].values
    X_ventanas, y_ventanas = [], []
    for i in range(0, len(voltajes) - ventana_tamaño, ventana_tamaño - solapamiento):
        v = voltajes[i:i + ventana_tamaño]
        e = etiquetas[i:i + ventana_tamaño]
        contador = Counter(e)                              # Cuenta el número de etiquetas que hay de cada clase en la ventana
        etiqueta, repes = contador.most_common(1)[0]       # Devuelve la etiqueta más frecuente junto al número de veces que aparece
        if repes / len(e) >= 0.9:                          # Si la etiqueta más frecuente es igual o mayor al 90% de todas las etiquetas
            X_ventanas.append(v)
            y_ventanas.append(etiqueta)
    for i in range(0, len(X_ventanas) - ventanas_por_secuencia + 1, paso_secuencia):
        secuencia = X_ventanas[i:i + ventanas_por_secuencia]
        etiquetas_seq = y_ventanas[i:i + ventanas_por_secuencia]
        contador = Counter(etiquetas_seq)
        mas_comunes = contador.most_common()                                     # Devuelve una lista de tuplas (etiqueta, cantidad) ordenada de mayor a menor frecuencia.
        if len(mas_comunes) > 1 and mas_comunes[0][1] == mas_comunes[1][1]:      # Comprueba si hay empate entre las dos etiquetas más frecuentes.
            etiqueta_final = etiquetas_seq[len(etiquetas_seq) // 2]              # Elige como etiqueta final la del elemento central de la secuencia.
        else:
            etiqueta_final = mas_comunes[0][0]
        X.append(secuencia)
        y.append(etiqueta_final)
    return np.array(X)[..., np.newaxis], np.array(y)

def generar_golpes(df):
    voltajes = df['Voltaje (V)'].values
    X, y = [], []
    picos, _ = find_peaks(voltajes, height=umbral, distance=tiempo_entre_picos)  # Detecta picos que superen el umbral y que esten distanciados mínimo determinadas muestras de otro 
    for pico in picos:
        inicio = pico - pico_en_ventana * ventana_tamaño          # Creamos la secuencia de ventanas a estudiar tratando de centrar el pico en ella
        fin = inicio + ventanas_por_secuencia * ventana_tamaño
        if inicio >= 0 and fin <= len(voltajes):                  # Nos aseguramos que no nos pasamos del límite del array de voltajes
            segmento = voltajes[inicio:fin]
            ventanas = np.array([                                 # Se dividen las secuencias en ventanas para aplicar los filtros correspondienters luego
                segmento[i:i + ventana_tamaño] 
                for i in range(0, len(segmento), ventana_tamaño)
            ])
            if ventanas.shape == (ventanas_por_secuencia, ventana_tamaño):  # Asegurar que el número de ventanas y su tamaño son los esperados
                X.append(ventanas)
                y.append("golpe_dedo")
    return np.array(X)[..., np.newaxis], np.array(y)

def generar_ninguna(df, n_secuencias):
    voltajes = df['Voltaje (V)'].values
    X, y = [], []
    total_muestras = ventana_tamaño * ventanas_por_secuencia

    for _ in range(n_secuencias):
        idx = random.randint(0, len(voltajes) - total_muestras)          # Obtiene un punto de inicio aleatorio dentro de los voltajes con etiqueta ninguna
        segmento = voltajes[idx:idx + total_muestras]
        ventanas = np.array([
            segmento[i:i + ventana_tamaño] for i in range(0, len(segmento), ventana_tamaño)  # Extrae la secuencia y la divide en ventanas
        ])
        if ventanas.shape == (ventanas_por_secuencia, ventana_tamaño):
            X.append(ventanas)
            y.append("ninguna")

    return np.array(X)[..., np.newaxis], np.array(y)



# --- Generar secuencias por clase ---  Se verifica que las etiquetas manueles están limpias y consistentes
X_caricia, y_caricia = generar_caricia(df_caricia) 
X_golpe, y_golpe = generar_golpes(df_golpe)
X_nada, y_nada = generar_ninguna(df_nada, n_secuencias=max(len(X_caricia), len(X_golpe))) # Se obtiene el mismo numero de secuencias que el máximo de secuencias que tiene otro gesto para mantener el datasheet balanceado

# --- Unir y normalizar ---
X_secuencias = np.concatenate([X_caricia, X_golpe, X_nada])
y_secuencias = np.concatenate([y_caricia, y_golpe, y_nada])

'''
# Normalización centrada
media = np.mean(X_secuencias)
desviacion = np.std(X_secuencias)
X_secuencias = (X_secuencias - media) / desviacion

# Guardar media y desviación para usar durante la predicción
np.save("media_entrenamiento.npy", media)
np.save("std_entrenamiento.npy", desviacion)
'''

# --- Normalización ---
X_secuencias = (X_secuencias - OFFSET) / ESCALA

# --- Resto del flujo ---
print(f"Secuencias aceptadas: {X_secuencias.shape[0]}")
conteo_etiquetas = Counter(y_secuencias)
print("Distribución de clases:")
for etiqueta, cantidad in conteo_etiquetas.items():
    print(f"{etiqueta}: {cantidad}")

# Visualización --- Se grafican las primeras 3 secuencias de cada clase para verificar visualmente los patrones
clases = np.unique(y_secuencias)
N = 3
for clase in clases:
    idxs = np.where(y_secuencias == clase)[0][:N]
    plt.figure(figsize=(15, 3 * N))
    for i, idx in enumerate(idxs):
        plt.subplot(N, 1, i + 1)
        secuencia = X_secuencias[idx].squeeze()
        señal = secuencia.flatten()
        plt.plot(señal, lw=1)
        plt.title(f"Clase: {clase} – Ejemplo {i + 1}")
        plt.xlabel("Tiempo (muestras)")
        plt.ylabel("Voltaje")
    plt.tight_layout()
    plt.show()

le = LabelEncoder()
y_encoded = le.fit_transform(y_secuencias)
y_cat = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X_secuencias, y_cat, test_size=0.2, stratify=y_encoded, random_state=42, shuffle=True)

entrada = Input(shape=(ventanas_por_secuencia, ventana_tamaño, 1))
x = TimeDistributed(Conv1D(16, kernel_size=5, activation='relu', padding='same'))(entrada) #probar 'tanh' en vez de 'relu'
x = TimeDistributed(MaxPooling1D(2))(x)
x = TimeDistributed(Flatten())(x)
x = LSTM(32)(x)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)  #probar 'tanh' en vez de 'relu'
salida = Dense(3, activation='softmax')(x)

model = Model(inputs=entrada, outputs=salida)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath='mejor_modelo.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test), callbacks=[early_stop, checkpoint])

model.save("modelo_teng_secuencias.keras")
print("Modelo guardado en modelo_teng_secuencias.keras")
np.save("etiquetas_secuencias.npy", le.classes_)