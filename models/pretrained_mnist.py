import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2

def preprocess_data(images, labels):
    # Convert grayscale to RGB and resize
    rgb_images = []
    for img in images:
        resized = cv2.resize(img, (299, 299))
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        rgb_images.append(rgb)
    return np.array(rgb_images)/255.0, to_categorical(labels, 10)

# Load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess
x_train, y_train = preprocess_data(train_images, train_labels)
x_test, y_test = preprocess_data(test_images, test_labels)

# Load base model
base_model = InceptionResNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(299, 299, 3)
)

# Freeze layers
base_model.trainable = False

# Add new head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(10, activation='softmax')(x)
model = Model(base_model.input, outputs)

# Compile and train
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=32,
          epochs=2,  # Reduced for quick testing
          validation_split=0.1)

# Evaluate
model.evaluate(x_test, y_test)

# Save
model.save('mnist_inception.h5')