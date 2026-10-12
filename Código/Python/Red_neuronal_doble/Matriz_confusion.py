import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from collections import Counter
from Preprocesado_modular import (
    cargar_csv,
    generar_ninguna,
    generar_caricia_dedo,
    generar_caricia_mano,
    generar_golpe_dedo,
    generar_golpe_mano,
    normalizar_ventanas,
    guardar_configuracion_experimento
)

# --- Parámetros de ventaneo ---
ventana_tamaño = 1000
solapamiento = 300
ventanas_por_secuencia = 3
paso_secuencia = 2

# --- Rutas ---
#IMPORTANTE CAMBIAR SIEMPRE
ruta_csv = "C:\\Users\\dcast\\Documents\\Muestras\\robotmov_tela_cable.csv"
modelo_path = "C:\\Users\\dcast\\Desktop\\Iker C\\UNIVERSIDAD POLITÉCNICA DE MADRID\\5º AÑO\\2º CUATRIMESTRE\\TFG\\Código\\Python\\MODELOS\\Robotmov_tela_cable\\mejor_modelo.keras"
etiquetas_path = "C:\\Users\\dcast\\Desktop\\Iker C\\UNIVERSIDAD POLITÉCNICA DE MADRID\\5º AÑO\\2º CUATRIMESTRE\\TFG\\Código\\Python\\MODELOS\\Robotmov_tela_cable\\etiquetas_secuencias.npy"
ruta_json="C:\\Users\\dcast\\Desktop\\Iker C\\UNIVERSIDAD POLITÉCNICA DE MADRID\\5º AÑO\\2º CUATRIMESTRE\\TFG\\Informacion de red neuronal\\resultado_robotmov_tela_cable.json"

# --- Cargar modelo entrenado y etiquetas ---
modelo = load_model(modelo_path)
clases = np.load(etiquetas_path)

# --- Cargar CSV completo ---
df = pd.read_csv(ruta_csv, sep=';')

# --- Filtrar por clase ---
df_nada         = df[df['Etiqueta'] == 'ninguna']
df_caricia_dedo = df[df['Etiqueta'] == 'caricia_dedo']
df_caricia_mano = df[df['Etiqueta'] == 'caricia_mano']
df_golpe_dedo   = df[df['Etiqueta'] == 'golpe_dedo']
df_golpe_mano   = df[df['Etiqueta'] == 'golpe_mano']

# --- Generar secuencias válidas ---
X0, y0 = generar_ninguna(df_nada)
X1, y1 = generar_caricia_dedo(df_caricia_dedo)
X2, y2 = generar_caricia_mano(df_caricia_mano)
X3, y3 = generar_golpe_dedo(df_golpe_dedo)
X4, y4 = generar_golpe_mano(df_golpe_mano)

# ---Guardar información de RESULTADOS---
muestras_por_clase = {}
for nombre, x, y_ in [
    ("ninguna", X0, y0),
    ("caricia_dedo", X1, y1),
    ("caricia_mano", X2, y2),
    ("golpe_dedo", X3, y3),
    ("golpe_mano", X4, y4)
]:
    print(f"Clase {nombre}: {len(y_)} muestras")
    muestras_por_clase[nombre] = len(y_)

# --- Concatenar todo ---
X = np.concatenate([X0, X1, X2, X3, X4])
y = np.concatenate([y0, y1, y2, y3, y4])

# --- Normalización ---
X = normalizar_ventanas(X)

# --- Codificar etiquetas ---
le = LabelEncoder()
le.classes_ = clases
y_encoded = le.transform(y)
y_cat = to_categorical(y_encoded)

# --- Evaluar modelo (para obtener pérdida) ---
loss, acc = modelo.evaluate(X, y_cat, verbose=0)

# --- Medir tiempo de inferencia ---
start = time.perf_counter()
pred_cat = modelo.predict(X, verbose=0)
end = time.perf_counter()
latencia_total = end - start
latencia_muestra = latencia_total / len(X)

pred_encoded = np.argmax(pred_cat, axis=1)

# --- Resultados ---
print("\n--- Classification Report ---")
print(classification_report(y_encoded, pred_encoded, target_names=clases))

# --- Matriz de confusión ---
cm = confusion_matrix(y_encoded, pred_encoded)

# --- Falsos positivos ---
### NUEVO: calcular FPs
false_positives = cm.sum(axis=0) - np.diag(cm)
total_fp = false_positives.sum()

# --- Métricas globales ---
acc_sklearn = accuracy_score(y_encoded, pred_encoded)
f1_macro = f1_score(y_encoded, pred_encoded, average='macro')
recall_macro = np.mean(np.diag(cm) / np.sum(cm, axis=1))

print("\n--- Métricas Globales ---")
print(f"Accuracy: {acc_sklearn:.4f}")
print(f"F1 macro: {f1_macro:.4f}")
print(f"Recall macro: {recall_macro:.4f}")
print(f"Loss: {loss:.4f}")
print(f"Número de muestras: {len(X)}")
print(f"Falsos positivos totales: {total_fp}")
print(f"Latencia media por muestra: {latencia_muestra*1000:.2f} ms")

# --- Matriz de confusión con gráfico ---
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', ax=ax, colorbar=True)
plt.title("Matriz de Confusión - Multicanal")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---Guardar innformación de la prueba realizada---
guardar_configuracion_experimento(
    ruta_csv=ruta_csv,
    muestras_por_clase=muestras_por_clase,
    total_muestras=len(y_),
    modo_comunicacion="bluetooth",          #cambiar según modo de funcionamiento
    ruta_guardado=ruta_json
)
with open(ruta_json, "r", encoding="utf-8") as f:
    config = json.load(f)

# Generar reporte como dict
reporte_clases = classification_report(
    y_encoded, pred_encoded, target_names=clases, output_dict=True
)

reporte_redondeado = {}
for clase, metrica in reporte_clases.items():
    if isinstance(metrica, dict):
        reporte_redondeado[clase] = {k: round(v, 4) for k, v in metrica.items()}
    else:
        reporte_redondeado[clase] = round(metrica, 4)


config["resultados"] = {
    "accuracy": round(acc_sklearn, 4),
    "f1_macro": round(f1_macro, 4),
    "recall_macro": round(recall_macro, 4),
    "loss": round(loss, 4),
    "falsos_positivos": int(total_fp),
    "latencia_ms_por_muestra": round(latencia_muestra * 1000, 2),
    "observaciones": "Precisión alta, latencia baja, confusiones puntuales.",
    "por_clase": reporte_redondeado
}


with open(ruta_json, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)

print(f"✅ Configuración y resultados guardados en {ruta_json}")
