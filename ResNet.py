import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def conv_block(x, filters, kernel_size=3, stride=1):
    shortcut = x

    # First convolution
    x = Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second convolution
    x = Conv2D(filters, kernel_size, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Adjust shortcut if needed
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Add shortcut
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

def resnet18(input_shape=(224, 224, 3), num_classes=1000):
    inputs = Input(shape=input_shape)

    # Initial conv and max pool
    x = Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Residual blocks
    x = conv_block(x, 64, stride=1)
    x = conv_block(x, 64, stride=1)

    x = conv_block(x, 128, stride=2)
    x = conv_block(x, 128, stride=1)

    x = conv_block(x, 256, stride=2)
    x = conv_block(x, 256, stride=1)

    x = conv_block(x, 512, stride=2)
    x = conv_block(x, 512, stride=1)

    # Global average pooling and output
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Create and summarize the model
model = resnet18(input_shape=(224, 224, 3), num_classes=1000)
model.summary()
