from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('models/mnist_cnn.h5')

@app.route('/')
def home():
    return "Digit Recognition API - POST to /predict"

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28)).reshape(1, 28, 28, 1).astype('float32') / 255
    pred = model.predict(img)
    return jsonify({
        'digit': int(np.argmax(pred)),
        'confidence': float(np.max(pred))
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)