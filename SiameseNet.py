import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.models import Model
import numpy as np

# Base network to generate embeddings
def create_base_network(input_shape):
    input = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (7, 7), activation='relu')(input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (5, 5), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128)(x)  # Final embedding

    return Model(input, x, name="EmbeddingNetwork")

# Contrastive Loss function
def contrastive_loss(margin=1.0):
    def loss(y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return loss

# Input shape (e.g., grayscale image 100x100)
input_shape = (100, 100, 1)

# Create shared embedding network
base_network = create_base_network(input_shape)

# Define the inputs
input_a = layers.Input(shape=input_shape)
input_b = layers.Input(shape=input_shape)

# Generate the embeddings for both inputs
embedding_a = base_network(input_a)
embedding_b = base_network(input_b)

# Compute the absolute difference (L1 distance)
l1_distance = layers.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([embedding_a, embedding_b])

# Optionally, add a small fully connected layer or just compute distance directly
output = layers.Dense(1, activation='sigmoid')(l1_distance)

# Define the model
siamese_model = Model(inputs=[input_a, input_b], outputs=output)

# Compile model with contrastive loss
siamese_model.compile(loss=contrastive_loss(margin=1.0), optimizer='adam', metrics=['accuracy'])

# Summary
siamese_model.summary()
