import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from Preprocesado_modular import OFFSETS, ESCALAS, umbral_caricia, umbral_golpe


# Mostrar información del dataset

def mostrar_info_dataset(X, y):
    print("--- Información del Dataset ---")
    print("Forma de X:", X.shape)
    print("Clases disponibles:", np.unique(y))
    conteo = Counter(y)
    for clase, cantidad in conteo.items():
        print(f"Clase {clase}: {cantidad} muestras")

# Visualizar secuencias de ambos sensores

def visualizar_secuencias(X, y, num_muestras=4):
    clases = np.unique(y)

    for clase in clases:
        indices = np.where(y == clase)[0]
        seleccionados = np.random.choice(indices, min(num_muestras, len(indices)), replace=False)

        fig, axs = plt.subplots(num_muestras, 2, figsize=(12, 3 * num_muestras))
        fig.suptitle(f"Ejemplos para la clase {clase}", fontsize=16)

        for i, idx in enumerate(seleccionados):
            for sensor in range(2):
                offset = OFFSETS[sensor]
                escala = ESCALAS[sensor]
                señal_concatenada = np.concatenate([ventana[:, sensor] for ventana in X[idx]])
                axs[i, sensor].plot(señal_concatenada)

                # Umbrales adaptados al sensor correspondiente
                axs[i, sensor].axhline((umbral_caricia - offset) / escala, color='orange', linestyle='--', label='umbral caricia')
                axs[i, sensor].axhline((umbral_golpe - offset) / escala, color='red', linestyle='--', label='umbral golpe')

                axs[i, sensor].set_title(f"Muestra {idx} - Sensor {sensor+1}")
                axs[i, sensor].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


# Preparar los datos para entrenamiento

def preparar_datos(X, y, etiquetas):

    # División en entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Conversión a one-hot encoding para clasificación
    y_train_cat = to_categorical(y_train, num_classes=len(etiquetas))
    y_test_cat = to_categorical(y_test, num_classes=len(etiquetas))

    return X_train, X_test, y_train_cat, y_test_cat, etiquetas

# Entrenamiento del modelo

def entrenar_modelo(modelo, X_train, y_train, X_test, y_test, path_modelo):
    checkpoint = ModelCheckpoint(path_modelo, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    historia = modelo.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=16,
                          callbacks=[checkpoint, early_stop])
    return historia
