import tensorflow as tf
import keras
from keras import layers
import numpy as np
import os

import matplotlib.pyplot as plt

# Constanst
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 1
TEST_DIR = "data/testing"

# Load Model
loaded_model = keras.models.load_model("models/trash-classifier-model-v0_2.keras")

ds_test = keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH), # Reshapes images if not in this size
    shuffle=False
)

class_names = ds_test.class_names

test_loss, test_acc = loaded_model.evaluate(ds_test, verbose=2)
print("Accuracy: ", test_acc)
print("Loss: ", test_loss)

# Predict and display results
for images, labels in ds_test.take(6):
    print(images.shape)
    predictions = loaded_model.predict(images)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class_idx]  
    true_class_idx = np.argmax(labels.numpy(), axis=1)[0]

    print(f"Predicted: {class_names[predicted_class_idx]}, (Confidence {confidence}) True: {class_names[true_class_idx]}")