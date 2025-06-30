import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Constanst
IMG_HEIGHT = 384
IMG_WIDTH = 512
BATCH_SIZE = 16

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    zoom_range=(0.95, 0.95),
    horizontal_flip=True,
    vertical_flip=True,
    data_format='channels_last',
    validation_split=0.1,
    dtype=tf.float32,

)

train_generator = datagen.flow_from_directory(
    "data/",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='sparse',
    shuffle=True,
    subset="training"
)

eval_generator = datagen.flow_from_directory(
    "data/",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='sparse',
    shuffle=True,
    subset="validation"
)

model = keras.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)), # Input Shape
    layers.Conv2D(32, (3, 3), activation="relu"), # Number of filters, sample size
    
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),

    # Dense Layers
    layers.Flatten(), # Flatten the layer above into a straight line (1 Dimensional)
    layers.Dense(128, activation='relu'), # 128 neuron Dense Layer
    layers.Dense(5, activation='softmax'), # 5 Neurons (We have 5 Classes)

])

def augment(x, y):
    image = tf.image.random_brightness(x, max_delta=0.05)
    return image, y

# ds_train = ds_train.map(augment)

# Training
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

history = model.fit(ds_train, epochs=4, 
                    validation_data=(ds_eval))

test_loss, test_acc = model.evaluate(ds_eval, verbose=2)
print(test_acc)
