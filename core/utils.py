import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
import os

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

def visualize_batch(b,convert_f,path=None,iter=None):
  a = np.asanyarray(convert_f(b))
  imgs = []
  for x in a:
    #if x[2]:
      #x = tf.keras.utils.array_to_img(x)
    imgs.append(x)
  cnt = len(imgs)
  plot_images(imgs,2,2,path,iter)

def plot_images(imgs,n_col,n_row,path=None,iter=None):
  _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
  axs = axs.flatten()
  for img, ax in zip(imgs, axs):
      ax.imshow(img,cmap='gray', vmin=0, vmax=255)
  if path is not None:
    plt.savefig(f'{path}/batch_{iter}.png')
  plt.show()
  
def plot_single_image(img):
  arr = np.asarray(img)
  plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
  plt.show()

def tf2pil(t,i,grayscale=False):
  if not grayscale:
    return tf.keras.utils.array_to_img(to_rgb(t))
  return Image.fromarray(tf2grayscale(t),mode="L")

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

def get_seed_tensor(w,h,channels):
  blank_t = np.zeros([w,h,channels])
  blank_t[w//2, h//2, :] = 255
  blank_t = tf.convert_to_tensor(blank_t,dtype="float32")
  return blank_t

def init_batch(n,w,h,c=16):
  '''
  Returns a batch of empty tensors except for a single spot in the middle which has value of 1
  '''
  blank_t = get_seed_tensor(w,h,c)
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

    frames = [tf2pil(batch_to_array(x)[0],grayscale)]
    for i in range(steps):
        x = ca(x)
        frames.append(tf2pil(batch_to_array(x)[0],i,grayscale))
    return frames