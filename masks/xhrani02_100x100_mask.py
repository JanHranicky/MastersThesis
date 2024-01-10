from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

image = Image.open('./img/xhrani02_100x100.png')
# Convert the image to grayscale
image = image.convert('L')
# Convert the image to a NumPy array
image_array = np.array(image)
# Threshold the image to get black and white pixels
threshold = 128  # Adjust this threshold as needed


logo_mask =  tf.convert_to_tensor(np.where(image_array < threshold, 1, 0),dtype=tf.float32)
bcknd_mask = tf.convert_to_tensor(np.where(image_array > threshold, 1, 0),dtype=tf.float32)