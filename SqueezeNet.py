import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Input, Dropout, Activation
from tensorflow.keras.models import Model

def fire_module(x, squeeze_filters, expand_filters_1x1, expand_filters_3x3, name=None):
    squeeze = Conv2D(squeeze_filters, (1,1), activation='relu', padding='same', name=name+'_squeeze')(x)

    expand_1x1 = Conv2D(expand_filters_1x1, (1,1), activation='relu', padding='same', name=name+'_expand_1x1')(squeeze)
    expand_3x3 = Conv2D(expand_filters_3x3, (3,3), activation='relu', padding='same', name=name+'_expand_3x3')(squeeze)

    output = concatenate([expand_1x1, expand_3x3], axis=-1, name=name+'_concat')
    return output

def SqueezeNet(input_shape=(224, 224, 3), classes=1000):
    input_img = Input(shape=input_shape)

    # Initial conv layer
    x = Conv2D(96, (7,7), strides=(2,2), activation='relu', padding='same', name='conv1')(input_img)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='maxpool1')(x)

    # Fire modules
    x = fire_module(x, squeeze_filters=16, expand_filters_1x1=64, expand_filters_3x3=64, name='fire2')
    x = fire_module(x, squeeze_filters=16, expand_filters_1x1=64, expand_filters_3x3=64, name='fire3')
    x = fire_module(x, squeeze_filters=32, expand_filters_1x1=128, expand_filters_3x3=128, name='fire4')
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='maxpool4')(x)

    x = fire_module(x, squeeze_filters=32, expand_filters_1x1=128, expand_filters_3x3=128, name='fire5')
    x = fire_module(x, squeeze_filters=48, expand_filters_1x1=192, expand_filters_3x3=192, name='fire6')
    x = fire_module(x, squeeze_filters=48, expand_filters_1x1=192, expand_filters_3x3=192, name='fire7')
    x = fire_module(x, squeeze_filters=64, expand_filters_1x1=256, expand_filters_3x3=256, name='fire8')
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='maxpool8')(x)

    x = fire_module(x, squeeze_filters=64, expand_filters_1x1=256, expand_filters_3x3=256, name='fire9')
    x = Dropout(0.5, name='dropout9')(x)

    # Final conv layer
    x = Conv2D(classes, (1,1), padding='valid', name='conv10')(x)
    x = Activation('relu')(x)

    # Global average pooling to reduce to (classes,)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = Activation('softmax')(x)

    model = Model(inputs=input_img, outputs=output, name='SqueezeNet')
    return model

# Example usage:
model = SqueezeNet(input_shape=(224,224,3), classes=1000)
model.summary()
