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

# --- Parámetros ---
ventana_tamaño = 1000           # 100 ms con 10kHz
solapamiento = 200              # 20% de solapamiento
ventanas_por_secuencia = 25     # 1,5 s agrupando 15 ventanas consecutivas
paso_secuencia = 5              # avanza 5 ventanas, solapa 10
OFFSET = 1.5
ESCALA = 1.65

ruta_csv = 'C:\\Users\\dcast\\Desktop\\Iker C\\UNIVERSIDAD POLITÉCNICA DE MADRID\\5º AÑO\\2º CUATRIMESTRE\\TFG\\Código\\Python\\Detección de señal\\datos_caricia_ninguna.csv'

# --- Cargar datos ---
df = pd.read_csv(ruta_csv, sep=';')
voltajes = df['Voltaje (V)'].values
etiquetas = df['Etiqueta'].values

# --- Ventaneo ---
X_ventanas = []
y_ventanas = []
for i in range(0, len(voltajes) - ventana_tamaño, ventana_tamaño - solapamiento):
    ventana = voltajes[i:i + ventana_tamaño]
    etiquetas_ventana = etiquetas[i:i + ventana_tamaño]
    contador = Counter(etiquetas_ventana)
    etiqueta_mas_comun, repeticiones = contador.most_common(1)[0]
    if repeticiones / len(etiquetas_ventana) >= 0.9:
        X_ventanas.append(ventana)
        y_ventanas.append(etiqueta_mas_comun)

# --- Convertir listas a arrays ---
X_ventanas = np.array(X_ventanas)
y_ventanas = np.array(y_ventanas)

# --- Normalizar los datos ---
X_ventanas = X_ventanas[..., np.newaxis]  # Añadir canal
X_ventanas = X_ventanas - OFFSET   # Eliminar el offset
X_ventanas = X_ventanas / ESCALA # Normalizar de [-1.65, 1.65] a [-1, 1]

# --- Agrupar ventanas en secuencias ---
X_secuencias = []
y_secuencias = []

for i in range(0, len(X_ventanas) - ventanas_por_secuencia + 1, paso_secuencia):
    secuencia = X_ventanas[i:i + ventanas_por_secuencia]
    etiquetas_secuencia = y_ventanas[i:i + ventanas_por_secuencia]

    contador = Counter(etiquetas_secuencia)
    mas_comunes = contador.most_common()

    if len(mas_comunes) > 1 and mas_comunes[0][1] == mas_comunes[1][1]:
        etiqueta_mas_frecuente = etiquetas_secuencia[len(etiquetas_secuencia) // 2]
    else:
        etiqueta_mas_frecuente = mas_comunes[0][0]

    X_secuencias.append(secuencia)
    y_secuencias.append(etiqueta_mas_frecuente)

# --- Convertir a arrays finales ---
X_secuencias = np.array(X_secuencias)
y_secuencias = np.array(y_secuencias)


print(f"Secuencias aceptadas: {X_secuencias.shape[0]}")


# --- Muestra la distribución de clases de etiquetas ---
conteo_etiquetas = Counter(y_secuencias)
print("Distribución de clases:")
for etiqueta, cantidad in conteo_etiquetas.items():
    print(f"{etiqueta}: {cantidad}")

# --- Visualización de secuencias etiquetadas ---
clases = np.unique(y_secuencias)
N = 3  # número de ejemplos por clase

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

# --- Codificar etiquetas ---
le = LabelEncoder()
y_encoded = le.fit_transform(y_secuencias)
y_cat = to_categorical(y_encoded)

# --- División entrenamiento/test ---
X_train, X_test, y_train, y_test = train_test_split(X_secuencias, y_cat, test_size=0.2, stratify=y_encoded, random_state=42)

# --- Modelo CNN+LSTM por ventana + LSTM global ---
entrada = Input(shape=(ventanas_por_secuencia, ventana_tamaño, 1))
x = TimeDistributed(Conv1D(16, kernel_size=5, activation='relu', padding='same'))(entrada)
x = TimeDistributed(MaxPooling1D(2))(x)
x = TimeDistributed(Flatten())(x)
x = LSTM(32)(x)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)
salida = Dense(3, activation='softmax')(x)

model = Model(inputs=entrada, outputs=salida)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Callback: EarlyStopping ---
# Detiene el entrenamiento si la pérdida de validación ('val_loss') no mejora después de 4 épocas consecutivas.
# Además, restaura automáticamente los pesos del modelo al punto con mejor val_loss.
early_stop = EarlyStopping(
    monitor='val_loss',          # Métrica a monitorear 
    patience=4,                  # Número de épocas sin mejora antes de detener
    restore_best_weights=True    # Recupera los mejores pesos (los de menor val_loss)
)

# --- Callback: ModelCheckpoint ---
# Guarda automáticamente el mejor modelo encontrado durante el entrenamiento (según val_loss).
checkpoint = ModelCheckpoint(
    filepath='mejor_modelo.keras',  
    monitor='val_loss',          
    save_best_only=True,         
    mode='min',                  
    verbose=1                   
)

# --- Entrenamiento ---
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint]
)

# --- Guardar modelo ---
model.save("modelo_teng_secuencias.keras")
print("Modelo guardado en modelo_teng_secuencias.keras")

# --- Guardar etiquetas ---
np.save("etiquetas_secuencias.npy", le.classes_)
