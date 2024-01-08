import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
import os
import random

def pil2tf(im):
  return tf.keras.utils.img_to_array(im,dtype="float32")

def pil2grayscale(img):
  return img.convert('L')

def batch_to_array(b):
  a = np.asanyarray(b)
  arr = []
  for x in a:
    arr.append(x)
  return arr

def visualize_batch(b,path=None,iter=None,visualize=True):
  a = np.asanyarray(tf.cast(b,dtype=tf.int32))
  imgs = []
  for x in a:
    #if x[2]:
      #x = tf.keras.utils.array_to_img(x)
    imgs.append(x)
  cnt = len(imgs)
  plot_images(imgs,2,cnt//2,path,iter,visualize)

def plot_images(imgs,n_col,n_row,path=None,iter=None,visualize=True):
  _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
  axs = axs.flatten()
  for img, ax in zip(imgs, axs):
      ax.imshow(img,cmap='gray', vmin=0, vmax=255)
  if path is not None:
    plt.savefig(f'{path}/batch_{iter}.png')
  if visualize:
    plt.show()
  plt.close()
  
def plot_single_image(img):
  arr = np.asarray(img)
  plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
  plt.show()

def convert_to_comparable_shape(b,gt_img_channels):
  if gt_img_channels == 1:
    if len(b.shape) == 3: #scatchy way to do it, but tensors will only be either in batch = 4 dims or single = 3 dims
      return b[:,:,0]
    return b[:,:,:,0]
  if len(b.shape) == 3: #scatchy way to do it, but tensors will only be either in batch = 4 dims or single = 3 dims
    return b[:,:,0:gt_img_channels]
  return b[:,:,:,0:gt_img_channels]

def tf2pil(t,grayscale=False):
  if not grayscale:
    t = tf.where(tf.math.is_nan(t), tf.ones_like(t) * 65535, t); #if w is nan use 1 * NUMBER else use element in w
    return tf.keras.utils.array_to_img(to_rgb(t))
  return Image.fromarray(np.uint8(tf2grayscale(t)),mode="L")

def tf2grayscale(b):
  if len(b.shape) == 3: #scatchy way to do it, but tensors will only be either in batch = 4 dims or single = 3 dims
    return b[:,:,0]
  return b[:,:,:,0]

def to_rgba(t):
  if len(t.shape) == 3: #scatchy way to do it, but tensors will only be either in batch = 4 dims or single = 3 dims
    return t[:,:,0:4]
  return t[:,:,:,0:4]

def to_single_channel(t):
  if len(t.shape) == 3: #scatchy way to do it, but tensors will only be either in batch = 4 dims or single = 3 dims
    return t[:,:,0]
  return t[:,:,:,0]

def to_rgb(t):
  if len(t.shape) == 3: #scatchy way to do it, but tensors will only be either in batch = 4 dims or single = 3 dims
    return t[:,:,0:3]
  return t[:,:,:,0:3]

def get_seed_tensor(w,h,channels,max_dim=None):
  blank_t = np.zeros([w,h,channels])
  if max_dim is None:
    blank_t[w//2, h//2, :] = 255
  else:
    blank_t[w//2, h//2, :] = (255 % max_dim)
  blank_t = tf.convert_to_tensor(blank_t,dtype="float32")
  return blank_t

def init_batch(n,w,h,c=16,max_dim=None):
  '''
  Returns a batch of empty tensors except for a single spot in the middle which has value of 1
  '''
  blank_t = get_seed_tensor(w,h,c,max_dim)
  blank_t = tf.expand_dims(blank_t, 0)
  return tf.repeat(blank_t,n,0)


def make_gif(name,frames):
    frame_one = frames[0]
    frame_one.save(name+".gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

def run_model_for_i(ca,w,h,c,steps,grayscale=False,checkpoint_path = None):
    x = init_batch(1,w,h,c)
    
    if checkpoint_path:
        ca.load_weights(checkpoint_path)
        
    frames = [tf2pil(x[0].numpy(),grayscale)]
    for i in range(steps):
        x = ca(x)
        frames.append(tf2pil(x[0].numpy(),grayscale))
    return frames
  
def open_image(path):
  return np.asarray(Image.open(path))

def generate_empty_img(width,height):
    return np.full((width,height,4),255,dtype=int)
  
def img_to_discrete_space_tf(img,state_num,target_channels,multiplier=1):
    img = pil2tf(img)

    if img.shape[-1] > target_channels:
      if target_channels == 1:
        img = img[:,:,0]
      else:
        img = img[:,:,0:target_channels]
    
    img = tf.math.floormod(img,tf.ones_like(img)*state_num) * multiplier
    return img
  
def display_tensor(c_map,t):
  colors_mapped = tf.gather(c_map, tf.cast(t,dtype=tf.int32))
  # Display the image using Matplotlib
  plt.imshow(colors_mapped.numpy())
      
  plt.axis('off')  # Turn off axis labels
  plt.title('Tensor Displayed as Image with Corresponding Colors')
  plt.ioff()
  plt.show()


def generate_random_colors(num_colors):
    # Generate 'num_colors' random colors in RGB format
    random_colors = []
    for _ in range(num_colors):
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        random_colors.append((red, green, blue))
    return random_colors
