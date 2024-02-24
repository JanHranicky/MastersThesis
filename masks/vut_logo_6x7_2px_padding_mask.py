from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf 
 
logo_mask = [
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0],
    [0,0,0,0,1,1,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
]

logo_mask =  tf.convert_to_tensor(logo_mask,dtype=tf.float32)
background_mask = 1 - logo_mask