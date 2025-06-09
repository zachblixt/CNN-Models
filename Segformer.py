import tensorflow as tf
from tensorflow.keras import layers, Model

def transformer_encoder(inputs, num_heads, key_dim, ff_dim, dropout=0.1):
    # Multi-Head Self Attention
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Network
    x = layers.Dense(ff_dim, activation='relu')(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def segformer(input_shape, num_classes, num_layers, num_heads, key_dim, ff_dim):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Hierarchical Transformer Encoder
    for _ in range(num_layers):
        x = transformer_encoder(x, num_heads, key_dim, ff_dim)

    # Lightweight All-MLP Decoder
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, x)

# Example usage
input_shape = (256, 256, 3)
num_classes = 21
num_layers = 4
num_heads = 8
key_dim = 64
ff_dim = 256

model = segformer(input_shape, num_classes, num_layers, num_heads, key_dim, ff_dim)
model.summary()
