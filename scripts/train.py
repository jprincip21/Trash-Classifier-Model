import tensorflow as tf
import keras
from keras import layers

import matplotlib.pyplot as plt

# Constanst
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32

# Pulling Images
ds_train = keras.preprocessing.image_dataset_from_directory(
    'data/',
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH), # Reshapes images if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training"
)

# Pulling Images
ds_validation = keras.preprocessing.image_dataset_from_directory(
    'data/',
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH), # Reshapes images if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training"
)


model = keras.Sequential(
    keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                               include_top=False,
                                               weights='imagenet')
)