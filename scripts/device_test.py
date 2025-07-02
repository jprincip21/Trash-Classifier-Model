# Script to display which devices are detected by Tensorflow (CPU & GPU)
import tensorflow as tf

print("Devices Available: ", tf.config.list_physical_devices())