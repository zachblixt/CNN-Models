
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, Lambda, Input, BatchNormalization, ReLU
from tensorflow.keras.models import Model

def channel_shuffle(x, groups):
    height, width, channels = x.shape[1:]
    channels_per_group = channels // groups

    x = tf.reshape(x, [-1, height, width, groups, channels_per_group])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [-1, height, width, channels])
    return x

def group_conv(x, filters, kernel_size, strides, groups):
    channel_axis = -1
    in_channels = x.shape[channel_axis]
    group_list = []

    for i in range(groups):
        start = i * in_channels // groups
        end = (i + 1) * in_channels // groups
        group_list.append(Conv2D(filters // groups, kernel_size, strides=strides, padding='same')(x[:, :, :, start:end]))

    x = Lambda(lambda z: tf.concat(z, axis=channel_axis))(group_list)
    return x

def shufflenet_unit(x, filters, strides, groups):
    shortcut = x

    if strides == 2:
        shortcut = DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
        shortcut = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Lambda(lambda z: channel_shuffle(z, groups))(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    if strides == 1:
        if shortcut.shape[-1] != x.shape[-1]:
            shortcut = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
        x = Add()([x, shortcut])
    else:
        x = Lambda(lambda z: tf.concat(z, axis=-1))([x, shortcut])

    x = ReLU()(x)
    return x

def build_shufflenet(input_shape, num_classes, scale_factor=1.0, groups=3):
    input = Input(shape=input_shape)
    x = Conv2D(int(24 * scale_factor), kernel_size=(3, 3), strides=(2, 2), padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(int(24 * scale_factor), kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    stage_repeats = [3, 7, 3]
    stage_out_channels = [int(c * scale_factor) for c in [240, 480, 960]]

    for stage, repeats in enumerate(stage_repeats):
        for i in range(repeats):
            strides = 2 if i == 0 else 1
            x = shufflenet_unit(x, stage_out_channels[stage], strides, groups)

    x = Conv2D(num_classes, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Lambda(lambda z: tf.reduce_mean(z, axis=[1, 2]))(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=output)
    return model

input_shape = (224, 224, 3)
num_classes = 1000
model = build_shufflenet(input_shape, num_classes)

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, decay=4e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
