from tensorflow.keras import layers, models

def ProtoNetEmbedding(input_shape=(84, 84, 3), embedding_dim=64):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, kernel_size=3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(64, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(64, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(embedding_dim, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)  # Output shape: (embedding_dim,)

    model = models.Model(inputs=inputs, outputs=x)
    return model

# Example usage:
if __name__ == "__main__":
    model = ProtoNetEmbedding()
    model.summary()
