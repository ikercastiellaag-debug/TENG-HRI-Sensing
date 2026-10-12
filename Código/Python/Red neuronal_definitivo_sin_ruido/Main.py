from Preprocesado_modular import cargar_csv, generar_caricia, generar_golpes, generar_ninguna, normalizar
from Entrenamiento_y_visualizacion import preparar_datos, entrenar_modelo, mostrar_info_dataset, visualizar_secuencias
from Modelo import crear_modelo
# from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# --- Ruta de los datos ---
ruta_csv = 'C:\\Users\\dcast\\Desktop\\Iker C\\UNIVERSIDAD POLITÉCNICA DE MADRID\\5º AÑO\\2º CUATRIMESTRE\\TFG\\Código\\Python\\Detección de señal\\prueba_sin_ruido.csv'

# --- Cargar y preparar datos ---
df = cargar_csv(ruta_csv)
df_caricia = df[df['Etiqueta'] == 'caricia_dedo']
df_golpe = df[df['Etiqueta'] == 'golpe_dedo']
df_nada = df[df['Etiqueta'] == 'ninguna']

X_caricia, y_caricia = generar_caricia(df)
X_golpe,   y_golpe   = generar_golpes(df)
X_nada,    y_nada    = generar_ninguna(df, n_secuencias=max(len(X_caricia), len(X_golpe)))

# --- Unir y normalizar ---
X = np.concatenate([X_caricia, X_golpe, X_nada])
y = np.concatenate([y_caricia, y_golpe, y_nada])
X = normalizar(X)

mostrar_info_dataset(X, y)
visualizar_secuencias(X, y)

# --- Preparar datos y etiquetas ---
X_train, X_test, y_train, y_test, le = preparar_datos(X, y)
np.save("datos/etiquetas_secuencias.npy", le.classes_)
'''
# --- Calcular class_weight ---
y_train_int = np.argmax(y_train, axis=1)
pesos = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_int), y=y_train_int)
class_weights = dict(enumerate(pesos))
print("Pesos por clase:", class_weights)
'''
# --- Crear modelo ---
modelo = crear_modelo(input_shape=X.shape[1:], num_clases=y_train.shape[1])
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelo.summary()

# --- Entrenar con class_weight ---
hist = entrenar_modelo(modelo, X_train, y_train, X_test, y_test) # class_weight=class_weights

# --- Guardar modelo ---
os.makedirs("modelos", exist_ok=True)
modelo.save("modelos/modelo_teng_secuencias.keras")
print("Modelo guardado en modelos/modelo_teng_secuencias.keras")
