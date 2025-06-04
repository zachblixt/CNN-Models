import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU

def yolo_v1(input_shape=(448, 448, 3), S=7, B=2, C=20):
    input_layer = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=2, padding='same')(input_layer)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(192, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(128, (1, 1), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(256, (1, 1), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    for _ in range(4):
        x = Conv2D(256, (1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(512, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(512, (1, 1), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(1024, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    for _ in range(2):
        x = Conv2D(512, (1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(1024, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(1024, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(1024, (3, 3), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(1024, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(1024, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Flatten()(x)
    x = Dense(4096)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(S * S * (C + B * 5), activation='linear')(x)

    output = tf.keras.layers.Reshape((S, S, C + B * 5))(x)

    model = Model(inputs=input_layer, outputs=output)
    return model

model = yolo_v1()
model.summary()
