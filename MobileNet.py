import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Input

def mobilenet(input_shape=(224, 224, 3), num_classes=1000):
    def conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
        x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)
        return x

    def depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1)):
        x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=depth_multiplier, strides=strides)(inputs)
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)
        x = Conv2D(pointwise_conv_filters, (1, 1), padding='same', strides=(1, 1))(x)
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)
        return x

    inputs = Input(shape=input_shape)
    x = conv_block(inputs, 32, strides=(2, 2))
    x = depthwise_conv_block(x, 64)
    x = depthwise_conv_block(x, 128, strides=(2, 2))
    x = depthwise_conv_block(x, 128)
    x = depthwise_conv_block(x, 256, strides=(2, 2))
    x = depthwise_conv_block(x, 256)
    x = depthwise_conv_block(x, 512, strides=(2, 2))
    for _ in range(5):
        x = depthwise_conv_block(x, 512)
    x = depthwise_conv_block(x, 1024, strides=(2, 2))
    x = depthwise_conv_block(x, 1024)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, x)
    return model

# Create and compile the model
model = mobilenet(input_shape=(224, 224, 3), num_classes=1000)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()
