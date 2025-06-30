import tensorflow as tf
import keras
from keras import layers
import numpy as np
import os

import matplotlib.pyplot as plt

# Constanst
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16
TEST_DIR = "data/testing"

class_names = sorted(os.listdir(TEST_DIR))  # Use folders in 'testing' as class names
# Load Model
loaded_model = keras.models.load_model("models/trash-classifier-model-v0_1.keras")

ds_test = keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH), # Reshapes images if not in this size
    shuffle=False
)

test_loss, test_acc = loaded_model.evaluate(ds_test, verbose=2)
print(test_acc)
print(test_loss)

# Optionally: predict and print class probabilities for a batch
for images, labels in ds_test.take(1):
    predictions = loaded_model.predict(images)
    print("Predictions shape:", predictions.shape)
    print("Predictions for first 5 images:")
    print(predictions[:5])
    predicted_classes = np.argmax(predictions, axis=1)
    print(predicted_classes)
    print("True labels for first 5 images:")
    print(labels[:5].numpy())
    