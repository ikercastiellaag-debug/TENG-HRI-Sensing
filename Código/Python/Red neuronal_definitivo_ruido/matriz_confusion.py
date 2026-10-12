import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from collections import Counter


# --- Parámetros de ventaneo ---
ventana_tamaño = 1000
solapamiento = 100
ventanas_por_secuencia = 4
paso_secuencia = 2
OFFSET = 1.5
ESCALA = 1.65

# --- Rutas ---
ruta_csv = 'C:\\Users\\dcast\\Desktop\\Iker C\\UNIVERSIDAD POLITÉCNICA DE MADRID\\5º AÑO\\2º CUATRIMESTRE\\TFG\\Código\\Python\\Detección de señal\\muestras_matriz_confusion.csv'           
modelo_path = 'modelos\\mejor_modelo.keras'
etiquetas_path = 'datos\\etiquetas_secuencias.npy'

# --- Cargar modelo entrenado y etiquetas ---
modelo = load_model(modelo_path)
clases = np.load(etiquetas_path)

# --- Cargar nuevos datos ---
df = pd.read_csv(ruta_csv, sep=';')
voltajes = df['Voltaje (V)'].values
etiquetas = df['Etiqueta'].values

# --- Ventaneo ---
X_ventanas, y_ventanas = [], []
for i in range(0, len(voltajes) - ventana_tamaño, ventana_tamaño - solapamiento):
    ventana = voltajes[i:i + ventana_tamaño]
    etiquetas_ventana = etiquetas[i:i + ventana_tamaño]
    contador = Counter(etiquetas_ventana)
    etiqueta_mas_comun, repeticiones = contador.most_common(1)[0]
    if repeticiones / len(etiquetas_ventana) >= 0.9:
        X_ventanas.append(ventana)
        y_ventanas.append(etiqueta_mas_comun)

'''
# Normalización centrada
media = np.load("media_entrenamiento.npy")
desviacion = np.load("std_entrenamiento.npy")
'''

# --- Normalización ---
X_ventanas = np.array(X_ventanas)
X_ventanas = (X_ventanas - OFFSET) / ESCALA
X_ventanas = X_ventanas[..., np.newaxis]
y_ventanas = np.array(y_ventanas)

# --- Agrupar en secuencias ---
X_secuencias, y_secuencias = [], []
for i in range(0, len(X_ventanas) - ventanas_por_secuencia + 1, paso_secuencia):
    secuencia = X_ventanas[i:i + ventanas_por_secuencia]
    etiquetas_secuencia = y_ventanas[i:i + ventanas_por_secuencia]
    contador = Counter(etiquetas_secuencia)
    mas_comunes = contador.most_common()
    if len(mas_comunes) > 1 and mas_comunes[0][1] == mas_comunes[1][1]:
        etiqueta_final = etiquetas_secuencia[len(etiquetas_secuencia) // 2]
    else:
        etiqueta_final = mas_comunes[0][0]
    X_secuencias.append(secuencia)
    y_secuencias.append(etiqueta_final)

X = np.array(X_secuencias)
y = np.array(y_secuencias)

# --- Codificar etiquetas ---
le = LabelEncoder()
le.classes_ = clases
y_encoded = le.transform(y)
y_cat = to_categorical(y_encoded)

# --- Predicciones ---
pred_cat = modelo.predict(X)
pred_encoded = np.argmax(pred_cat, axis=1)

# --- Resultados ---
print("\n--- Classification Report ---")
print(classification_report(y_encoded, pred_encoded, target_names=clases))

# --- Matriz de confusión con anotaciones ---
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_encoded, pred_encoded)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', ax=ax, colorbar=True)
plt.title("Matriz de Confusión - Nuevos Datos")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
