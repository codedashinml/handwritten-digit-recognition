import tensorflow as tf
from tensorflow.keras import layers, datasets, utils

def train_model():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    
    model = tf.keras.Sequential([
        layers.Conv2D(32, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=5)
    model.save('model.h5')

if __name__ == '__main__':
    train_model()