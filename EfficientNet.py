import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Input, Add
from tensorflow.keras.models import Model

def swish(x):
    return x * tf.keras.backend.sigmoid(x)

def bottleneck_block(x, expansion_factor, output_channels, strides, block_name):
    input_channels = x.shape[-1]
    expanded = Conv2D(input_channels * expansion_factor, kernel_size=1, padding='same', use_bias=False, name=f'{block_name}_expand')(x)
    expanded = BatchNormalization(name=f'{block_name}_expand_bn')(expanded)
    expanded = tf.keras.layers.Activation(swish, name=f'{block_name}_expand_swish')(expanded)

    depthwise = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', use_bias=False, name=f'{block_name}_dw')(expanded)
    depthwise = BatchNormalization(name=f'{block_name}_dw_bn')(depthwise)
    depthwise = tf.keras.layers.Activation(swish, name=f'{block_name}_dw_swish')(depthwise)

    project = Conv2D(output_channels, kernel_size=1, padding='same', use_bias=False, name=f'{block_name}_project')(depthwise)
    project = BatchNormalization(name=f'{block_name}_project_bn')(project)

    if strides == 1 and input_channels == output_channels:
        project = Add(name=f'{block_name}_skip')([x, project])

    return project

def efficientnet_b0(input_shape=(224, 224, 3), num_classes=1000):
    inputs = Input(shape=input_shape)
    
    # Stem
    x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False, name='stem_conv')(inputs)
    x = BatchNormalization(name='stem_bn')(x)
    x = tf.keras.layers.Activation(swish, name='stem_swish')(x)

    # Example Bottleneck Blocks (greatly simplified version)
    x = bottleneck_block(x, expansion_factor=1, output_channels=16, strides=1, block_name='block1')
    x = bottleneck_block(x, expansion_factor=6, output_channels=24, strides=2, block_name='block2')
    x = bottleneck_block(x, expansion_factor=6, output_channels=40, strides=2, block_name='block3')
    
    # Head
    x = Conv2D(1280, kernel_size=1, use_bias=False, name='head_conv')(x)
    x = BatchNormalization(name='head_bn')(x)
    x = tf.keras.layers.Activation(swish, name='head_swish')(x)

    x = GlobalAveragePooling2D(name='global_pool')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=inputs, outputs=x, name='EfficientNetB0_scratch')
    return model

# Example usage
model = efficientnet_b0(input_shape=(224, 224, 3), num_classes=1000)
model.summary()
