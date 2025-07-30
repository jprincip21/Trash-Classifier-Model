import tensorflow as tf
import keras
from keras import layers
import numpy as np

loaded_model = keras.models.load_model("models/trash-classifier-model-v0_1.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model=loaded_model)

model_tflite = converter.convert()

with open("models/trash-classifier-model-v0_1.tflite", "wb") as f:
    f.write(model_tflite)