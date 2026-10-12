from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Bidirectional, LSTM, LeakyReLU



def crear_modelo(input_shape=(3, 1000, 2), num_clases=5):
    # Entrada de forma: (n_frames=4, muestras_por_frame=1000, 2 canales representando los sensores)
    entrada = Input(shape=input_shape)

    # Primer bloque convolucional: detecta patrones locales en los sensores combinados
    x = TimeDistributed(Conv1D(32, kernel_size=3, padding='same'), name="conv1d_1")(entrada)
    x = TimeDistributed(LeakyReLU(alpha=0.05), name="leaky_relu_1")(x)  # Conserva valores negativos
    x = TimeDistributed(BatchNormalization(), name="batch_norm_1")(x)  # Normaliza activaciones para mayor estabilidad

    # Segundo bloque convolucional: detecta patrones más amplios combinando ambos sensores
    x = TimeDistributed(Conv1D(32, kernel_size=5, padding='same'), name="conv1d_2")(x)
    x = TimeDistributed(LeakyReLU(alpha=0.05), name="leaky_relu_2")(x)
    x = TimeDistributed(BatchNormalization(), name="batch_norm_2")(x)

    # Pooling: reduce resolución temporal, destaca los máximos, elimina ruido
    x = TimeDistributed(MaxPooling1D(pool_size=2), name="max_pooling")(x)

    # Aplanamiento por frame: convierte cada paso temporal en un vector
    x = TimeDistributed(Flatten(), name="flatten")(x)

    # LSTM bidireccional: modela relaciones temporales en ambos sentidos
    x = Bidirectional(LSTM(64), name="Bidirectional_lstm")(x)  # Salida final de 128 unidades (64+64)

    # Regularización: evita overfitting apagando neuronas durante entrenamiento
    x = Dropout(0.5)(x)

    # Capa totalmente conectada: traduce la representación en espacio más compacto
    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.05)(x)

    # Capa de salida: devuelve probabilidad para cada clase
    salida = Dense(num_clases, activation='softmax', name="output_softmax")(x)

    modelo = Model(inputs=entrada, outputs=salida)
    return modelo
