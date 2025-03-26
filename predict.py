import tensorflow as tf
import numpy as np
from utils import preprocess_image

# Load pre-trained model
model = tf.keras.models.load_model('models/pretrained_model.h5')  # or use tf.keras.applications

# Example: Predict on a single image
def predict_digit(image_path):
    img = preprocess_image(image_path)  # Resize, normalize, etc.
    prediction = model.predict(img[np.newaxis, ...])
    return np.argmax(prediction)

if __name__ == '__main__':
    image_path = 'sample.png'  # Replace with your image
    print(f"Predicted digit: {predict_digit(image_path)}")