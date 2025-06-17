import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.models import Model
import numpy as np

# Define the embedding network (shared across anchor, positive, and negative)
def create_base_network(input_shape):
    input = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (7, 7), activation='relu')(input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (5, 5), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128)(x)  # Output embedding (128-D)

    return Model(input, x, name="EmbeddingNetwork")

# Triplet Loss Function
def triplet_loss(margin=0.5):
    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:, 0:128], y_pred[:, 128:256], y_pred[:, 256:384]
        pos_dist = K.sum(K.square(anchor - positive), axis=1)
        neg_dist = K.sum(K.square(anchor - negative), axis=1)
        basic_loss = pos_dist - neg_dist + margin
        loss = K.maximum(basic_loss, 0.0)
        return K.mean(loss)
    return loss

# Input shape (e.g. 100x100 grayscale image)
input_shape = (100, 100, 1)

# Instantiate the base network
embedding_network = create_base_network(input_shape)

# Create the inputs
anchor_input = layers.Input(name="anchor", shape=input_shape)
positive_input = layers.Input(name="positive", shape=input_shape)
negative_input = layers.Input(name="negative", shape=input_shape)

# Generate the embeddings
encoded_anchor = embedding_network(anchor_input)
encoded_positive = embedding_network(positive_input)
encoded_negative = embedding_network(negative_input)

# Concatenate embeddings into one tensor
merged_output = layers.concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=1)

# Define the full model for training
triplet_model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_output)

# Compile the model with the custom triplet loss
triplet_model.compile(loss=triplet_loss(margin=0.5), optimizer='adam')

# Summary
triplet_model.summary()
