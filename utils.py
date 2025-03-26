import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))          # Resize to MNIST dimensions
    img = img.astype('float32') / 255.0      # Normalize
    img = np.expand_dims(img, axis=-1)       # Add channel dimension (28,28,1)
    return img