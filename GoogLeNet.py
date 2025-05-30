import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, Input, concatenate
from tensorflow.keras.models import Model

def inception_module(x, filters):
    f1, f3r, f3, f5r, f5, proj = filters

    path1 = Conv2D(f1, (1, 1), padding='same', activation='relu')(x)

    path2 = Conv2D(f3r, (1, 1), padding='same', activation='relu')(x)
    path2 = Conv2D(f3, (3, 3), padding='same', activation='relu')(path2)

    path3 = Conv2D(f5r, (1, 1), padding='same', activation='relu')(x)
    path3 = Conv2D(f5, (5, 5), padding='same', activation='relu')(path3)

    path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    path4 = Conv2D(proj, (1, 1), padding='same', activation='relu')(path4)

    return concatenate([path1, path2, path3, path4], axis=-1)

def googlenet(input_shape=(224, 224, 3), num_classes=1000):
    input_layer = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, [64, 96, 128, 16, 32, 32])   # Inception 3a
    x = inception_module(x, [128, 128, 192, 32, 96, 64]) # Inception 3b
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, [192, 96, 208, 16, 48, 64])  # Inception 4a
    x = inception_module(x, [160, 112, 224, 24, 64, 64]) # Inception 4b
    x = inception_module(x, [128, 128, 256, 24, 64, 64]) # Inception 4c
    x = inception_module(x, [112, 144, 288, 32, 64, 64]) # Inception 4d
    x = inception_module(x, [256, 160, 320, 32, 128, 128]) # Inception 4e
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, [256, 160, 320, 32, 128, 128]) # Inception 5a
    x = inception_module(x, [384, 192, 384, 48, 128, 128]) # Inception 5b

    x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid')(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model

# Instantiate the model
model = googlenet(input_shape=(224, 224, 3), num_classes=1000)
model.summary()
