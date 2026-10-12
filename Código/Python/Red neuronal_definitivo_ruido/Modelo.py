from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, TimeDistributed, Flatten, Bidirectional, BatchNormalization
from tensorflow.keras.models import Model

def crear_modelo(input_shape=(15, 1000, 1), num_clases=3):
     # Entrada de forma: (n_frames=15, muestras_por_frame=1000, 1 canal)
    entrada = Input(shape=input_shape)

    # Primer bloque convolucional: detecta patrones locales cortos
    x = TimeDistributed(Conv1D(32, kernel_size=3, activation='relu', padding='same'))(entrada)
    x = TimeDistributed(BatchNormalization())(x)  # Normaliza activaciones para mayor estabilidad

    # Segundo bloque convolucional: detecta patrones más amplios
    x = TimeDistributed(Conv1D(32, kernel_size=5, activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)

    # Pooling: reduce resolución temporal, destaca los máximos, elimina ruido
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)

    # Aplanamiento por frame: convierte cada paso temporal en un vector
    x = TimeDistributed(Flatten())(x)

    # LSTM bidireccional: modela relaciones temporales en ambos sentidos
    x = Bidirectional(LSTM(64))(x)  # Salida final de 128 unidades (64+64)

    # Regularización: evita overfitting apagando neuronas durante entrenamiento
    x = Dropout(0.5)(x)

    # Capa totalmente conectada: traduce la representación en espacio más compacto
    x = Dense(64, activation='relu')(x)

    # Capa de salida: devuelve probabilidad para cada clase
    salida = Dense(num_clases, activation='softmax')(x)

    modelo = Model(inputs=entrada, outputs=salida)
    return modelo