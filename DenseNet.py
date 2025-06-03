import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, AveragePooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def conv_block(x, growth_rate):
    x1 = BatchNormalization()(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(4 * growth_rate, (1, 1), use_bias=False)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(growth_rate, (3, 3), padding='same', use_bias=False)(x1)
    return x1

def dense_block(x, num_filters, growth_rate, layers):
    for _ in range(layers):
        cb = conv_block(x, growth_rate)
        x = Concatenate()([x, cb])
        num_filters += growth_rate
    return x, num_filters

def transition_layer(x, num_filters):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, (1, 1), use_bias=False)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x

def DenseNet(input_shape, num_classes, growth_rate=12, num_layers_per_block=[6, 12, 24, 16]):
    num_filters = 2 * growth_rate
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters, (7, 7), strides=(2, 2), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((3, 3), strides=(2, 2))(x)

    for i, layers in enumerate(num_layers_per_block):
        x, num_filters = dense_block(x, num_filters, growth_rate, layers)
        if i != len(num_layers_per_block) - 1:
            x = transition_layer(x, num_filters // 2)
            num_filters = num_filters // 2

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Example usage
input_shape = (224, 224, 3)
num_classes = 1000
model = DenseNet(input_shape, num_classes)
model.summary()
