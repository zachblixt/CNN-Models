
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, PReLU, MaxPooling2D, AveragePooling2D, Concatenate, Add, UpSampling2D, Lambda, Dropout, Cropping2D, Resizing
from tensorflow.keras.models import Model

def initial_block(x):
    conv = Conv2D(13, (3, 3), strides=2, padding='same')(x)
    pool = MaxPooling2D((2, 2), strides=2)(x)
    concat = Concatenate()([conv, pool])
    return concat

def bottleneck(x, filters, downsample=False, dilated=False, asymmetric=False):
    y = BatchNormalization()(x)
    y = PReLU()(y)
    if downsample:
        y = Conv2D(filters, (2, 2), strides=2, padding='same')(y)
    else:
        y = Conv2D(filters, (1, 1), padding='same')(y)
    
    y = BatchNormalization()(y)
    y = PReLU()(y)
    
    if dilated:
        y = Conv2D(filters, (3, 3), padding='same', dilation_rate=2)(y)
    elif asymmetric:
        y = Conv2D(filters, (5, 1), padding='same')(y)
        y = Conv2D(filters, (1, 5), padding='same')(y)
    else:
        y = Conv2D(filters, (3, 3), padding='same')(y)
    
    y = BatchNormalization()(y)
    y = PReLU()(y)
    y = Conv2D(filters, (1, 1), padding='same')(y)
    
    if downsample:
        x = MaxPooling2D((2, 2), strides=2)(x)
        # Align spatial dimensions before concatenation
        if y.shape[1] != x.shape[1] or y.shape[2] != x.shape[2]:
            x = Resizing(y.shape[1], y.shape[2])(x)
        y = Concatenate()([y, x])
    else:
        # Add a 1x1 Conv2D projection to match input and output shapes
        if x.shape[-1] != filters:
            x = Conv2D(filters, (1, 1), padding='same')(x)
        y = Add()([y, x])
    
    return y

def encoder(x):
    x = initial_block(x)
    x = bottleneck(x, 64, downsample=True)
    for _ in range(4):
        x = bottleneck(x, 64)
    
    x = bottleneck(x, 128, downsample=True)
    for _ in range(2):
        x = bottleneck(x, 128)
    x = bottleneck(x, 128, dilated=True)
    x = bottleneck(x, 128, asymmetric=True)
    x = bottleneck(x, 128, dilated=True)
    for _ in range(2):
        x = bottleneck(x, 128)
    
    x = bottleneck(x, 256, downsample=True)
    for _ in range(2):
        x = bottleneck(x, 256)
    x = bottleneck(x, 256, dilated=True)
    x = bottleneck(x, 256, asymmetric=True)
    x = bottleneck(x, 256, dilated=True)
    for _ in range(2):
        x = bottleneck(x, 256)
    
    return x

def decoder(x):
    x = bottleneck(x, 128, downsample=True)
    x = UpSampling2D((2, 2))(x)
    x = bottleneck(x, 64)
    x = UpSampling2D((2, 2))(x)
    x = bottleneck(x, 16)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (3, 3), padding='same')(x)
    return x

def ENet(input_shape):
    inputs = Input(shape=input_shape)
    x = encoder(inputs)
    x = decoder(x)
    model = Model(inputs, x)
    return model

# Example usage
input_shape = (360, 640, 3)
model = ENet(input_shape)
model.summary()
