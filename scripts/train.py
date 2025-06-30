import tensorflow as tf
import keras
from keras import layers
import numpy as np

import matplotlib.pyplot as plt

# Constanst
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32

# Pulling Images
ds_train = keras.preprocessing.image_dataset_from_directory(
    'data/training/',
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
    'data/training/',
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH), # Reshapes images if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation"
)

base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = keras.Sequential([
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='softmax')  # 5 waste classes
    
])

model.compile(
    optimizer="adam",
    loss=keras.losses.categorical_crossentropy,
    metrics=['accuracy'])

model.fit(
    ds_train,
    epochs=4,
    validation_data=ds_validation)

test_loss, test_acc = model.evaluate(ds_validation, verbose=2)
print(test_acc)

model.save("models/trash-classifier-model-v0_1.keras")
