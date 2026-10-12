import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from collections import Counter

# --- Parámetros ---
ventana_tamaño = 1000  # 100 ms con 10kHz
solapamiento = 500     # 50% de solapamiento
ruta_csv = 'C:\\Users\\dcast\\Desktop\\Iker C\\UNIVERSIDAD POLITÉCNICA DE MADRID\\5º AÑO\\2º CUATRIMESTRE\\TFG\\Código\\Python\\Detección de señal\\datos_teng_etiquetados.csv'

# --- Cargar datos ---
df = pd.read_csv(ruta_csv, sep=';')
voltajes = df['Voltaje (V)'].values
etiquetas = df['Etiqueta'].values

# --- Ventaneo ---
X = []
y = []
for i in range(0, len(voltajes) - ventana_tamaño, ventana_tamaño - solapamiento):
    ventana = voltajes[i:i + ventana_tamaño]
    etiquetas_ventana = etiquetas[i:i + ventana_tamaño]
    etiquetas_unicas = set(etiquetas_ventana)
    
    # Aceptamos solo si la ventana tiene una única etiqueta válida (ninguna también se incluye) y tiene que ser homogénea
    if len(set(etiquetas_ventana)) == 1:
        X.append(ventana)
        y.append(etiquetas_ventana[0])


X = np.array(X)
X = X[..., np.newaxis]  # Añadir dimensión de canal (channel como RGB)
print(f"Ventanas aceptadas: {X.shape[0]}")

# --- Muestra la distribución de clases de etiquetas ---
conteo_etiquetas = Counter(y)
print("Distribución de clases:")
for etiqueta, cantidad in conteo_etiquetas.items():
    print(f"{etiqueta}: {cantidad}")


# --- Codificar etiquetas ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# --- División entrenamiento/test ---
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# --- Modelo CNN + LSTM ---
model = Sequential([
    Conv1D(32, kernel_size=5, activation='relu', padding='same', input_shape=(ventana_tamaño, 1)),
    MaxPooling1D(2),
    Conv1D(64, kernel_size=5, activation='relu', padding='same'),
    MaxPooling1D(2),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Entrenamiento ---
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# --- Guardar modelo ---
model.save("modelo_teng_cnn_lstm.h5")
print("Modelo guardado en modelo_teng_cnn_lstm.h5")

# --- Guardar etiquetas ---
np.save("etiquetas.npy", le.classes_)
