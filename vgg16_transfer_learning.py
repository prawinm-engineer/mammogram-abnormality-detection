"""
Transfer Learning using VGG16 for Mammogram Image Classification
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG16

def build_vgg16_model(input_shape=(150, 150, 3)):
    """
    Builds a VGG16-based transfer learning model.
    """

    # Load pre-trained VGG16 model without top layer
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze convolutional layers
    base_model.trainable = False

    # Build classifier on top
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
