from flask import Flask, request, jsonify
from utils import preprocess_image
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load pre-trained model
model = tf.keras.models.load_model('models/inception_resnet_mnist.h5')

def preprocess_image(image):
    """Convert uploaded image to MNIST-compatible format"""
    try:
        # Read and preprocess image
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (299, 299))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"Image processing error: {str(e)}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        # Process image and predict
        processed_img = preprocess_image(file)
        if processed_img is None:
            return jsonify({'error': 'Invalid image format'}), 400
            
        prediction = model.predict(processed_img)
        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        
        return jsonify({
            'digit': digit,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)