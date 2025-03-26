# Standard way to import TensorFlow and Keras components
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, datasets, utils, applications
import numpy as np
import cv2

class CustomModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Example custom layers
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

def preprocess_data(images, labels):
    """Preprocess data for custom model"""
    # Convert grayscale to RGB and resize if needed
    images = np.array([cv2.cvtColor(cv2.resize(img, (32, 32)), cv2.COLOR_GRAY2RGB) for img in images])
    images = images.astype('float32') / 255.0
    labels = utils.to_categorical(labels, 10)
    return images, labels

def train_custom_model():
    # Load data
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    
    # Preprocess
    x_train, y_train = preprocess_data(train_images, train_labels)
    x_test, y_test = preprocess_data(test_images, test_labels)
    
    # Initialize model
    model = CustomModel()
    
    # Compile
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Train
    model.fit(x_train, y_train,
              batch_size=32,
              epochs=5,
              validation_split=0.2)
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save
    model.save('custom_model.h5')

if __name__ == '__main__':
    train_custom_model()