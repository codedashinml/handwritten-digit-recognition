import tensorflow as tf
from tensorflow.keras import layers

class CustomModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.flatten(x)
        return self.dense(x)

if __name__ == "__main__":
    model = CustomModel()
    model.build((None, 28, 28, 1))
    model.summary()