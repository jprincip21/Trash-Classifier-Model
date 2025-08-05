import tensorflow as tf
import numpy as np
from PIL import Image
import os


TFLITE_MODEL_PATH = "models/trash-classifier-model-v0_2.tflite"

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# preprocess input image
def preprocess_image(image_path, input_shape):
    image = Image.open(image_path).convert('RGB').resize((input_shape[1], input_shape[2]))
    input_data = np.array(image, dtype=np.float32)

    # Normalize if your model expects normalized input [0,1]
    # input_data = input_data / 255.0

    # NO NEED TO DO THIS, Model does it in first layer

    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)

    return input_data

# Folder with test images
TEST_DIR = "data/testing"

# Run inference on each image in the test directory
for label_dir in os.listdir(TEST_DIR):
    label_path = os.path.join(TEST_DIR, label_dir)
    if not os.path.isdir(label_path):
        continue

    for image_file in os.listdir(label_path):
        image_path = os.path.join(label_path, image_file)

        input_data = preprocess_image(image_path, input_details[0]['shape'])

        # Set the tensor to point to the input data to be inferred
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_label_idx = np.argmax(output_data[0])
        confidence = output_data[0][predicted_label_idx]

        print(f"Image: {image_file}, Predicted: {predicted_label_idx}, Confidence: {confidence:.4f}")
