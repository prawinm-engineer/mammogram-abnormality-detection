"""
Data preprocessing utilities for mammogram images
"""

import cv2
import numpy as np

def load_and_preprocess(image_path, target_size=(50, 50)):
    """
    Loads a mammogram image, resizes it, and normalizes pixel values.

    Parameters:
        image_path (str): Path to the image file
        target_size (tuple): Desired image size

    Returns:
        numpy.ndarray: Preprocessed image
    """

    # Read image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    # Resize image
    image = cv2.resize(image, target_size)

    # Normalize pixel values
    image = image.astype("float32") / 255.0

    return image
