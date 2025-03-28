import cv2
import numpy as np

def preprocess_image(image):
    """Convert image to MNIST-compatible format"""
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize and normalize
    image = cv2.resize(image, (28, 28))
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=(0, -1))  # Add batch and channel dims