import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import matplotlib.pyplot as plt

def preparar_datos(X, y, test_size=0.2, random_state=42):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_cat = to_categorical(y_encoded)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=test_size, stratify=y_encoded, random_state=random_state, shuffle=True
    )
    return X_train, X_test, y_train, y_test, le

def entrenar_modelo(model, X_train, y_train, X_test, y_test, path_modelo='modelos\mejor_modelo.keras'): # class_weight=None
    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=path_modelo, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=8,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, checkpoint],
        # class_weight=class_weight 
    )
    return history


def mostrar_info_dataset(X, y):
    print(f"Secuencias aceptadas: {X.shape[0]}")
    conteo_etiquetas = Counter(y)
    print("Distribución de clases:")
    for etiqueta, cantidad in conteo_etiquetas.items():
        print(f"{etiqueta}: {cantidad}")

def visualizar_secuencias(X, y, clases=None, batch_size=10):
    if clases is None:
        clases = np.unique(y)
    for clase in clases:
        idxs = np.where(y == clase)[0]
        N = min(len(idxs), 200)
        for inicio in range(0, N, batch_size):
            fin = min(inicio + batch_size, N)
            plt.figure(figsize=(15, 3 * (fin - inicio)))
            for i, idx in enumerate(idxs[inicio:fin]):
                plt.subplot(fin - inicio, 1, i + 1)
                secuencia = X[idx].squeeze()
                señal = secuencia.flatten()
                plt.plot(señal, lw=1)
                plt.title(f"Clase: {clase} – Ejemplo {inicio + i + 1}")
                plt.xlabel("Tiempo (muestras)")
                plt.ylabel("Voltaje")
            plt.tight_layout()
            plt.show()