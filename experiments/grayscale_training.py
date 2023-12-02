from core import model,utils
import tensorflow as tf
from PIL import Image,ImageOps
from IPython.display import clear_output,display
import numpy as np 
import os
import IPython.display as display
from matplotlib import pyplot as plt


MODEL_NAME = '4_CHANNELS_GRAYSCALE'
GT_IMG_PATH = '../img/vut_logo_small.png'
TRAIN_STEP_INTERVAL = (75,100)
EPOCHS = 30000

VISUALIZE = True
VISUALIZE_ITERS = 100
SAVE_ITERS = 100
GENERATE_GIF_ITERS = 100
CHANNELS = 4

gt_img_name = GT_IMG_PATH.split('/')[-1].split('.')[0]
ca = model.CA(channel_n=CHANNELS)
loss_f = tf.keras.losses.MeanSquaredError()
learning_rate = 0.001

gt_img = Image.open(GT_IMG_PATH)
gt_img = ImageOps.grayscale(gt_img)
#utils.plot_single_image(gt_img)

height,width = gt_img.size
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [2000], [learning_rate, learning_rate*0.1])
trainer = tf.keras.optimizers.Adam(lr_sched)

@tf.function
def train_step(x):
  iter_n = tf.random.uniform([], TRAIN_STEP_INTERVAL[0], TRAIN_STEP_INTERVAL[1], tf.int32)
  with tf.GradientTape() as g:
    for i in tf.range(iter_n):
      x = ca(x)
    loss = tf.math.reduce_mean(loss_f(gt_img, utils.tf2grayscale(x)))
  grads = g.gradient(loss, ca.weights)
  grads = [g/(tf.norm(g)+1e-8) for g in grads]
  trainer.apply_gradients(zip(grads, ca.weights))
  return x, loss

loss_values = []
for i in range(101):
  x0 = utils.init_batch(4,width,height,CHANNELS)

  x, loss = train_step(x0)
  loss_val = np.log10(loss.numpy())
  loss_values.append(loss_val)
  
  #print('\r step: %d, log10(loss): %.3f'%(i, loss_val), end='')
  if VISUALIZE and i%VISUALIZE_ITERS == 0:
    #clear_output(wait=True)
    utils.visualize_batch(x,utils.tf2grayscale,f'./checkpoints/{MODEL_NAME}_{gt_img_name}',str(i))
  if np.isnan(loss_val):
    break
  if i%SAVE_ITERS == 0:
    ca.save_weights(f'./checkpoints/'+MODEL_NAME+'_'+gt_img_name+'/'+str(i))
    plt.plot(loss_values)
    plt.title(f'Loss function epoch num. {i}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.savefig(f'./checkpoints/{MODEL_NAME}_{gt_img_name}/loss_{i}.png')
#  if i%GENERATE_GIF_ITERS == 0:
#    imgs = utils.run_model_for_i(ca,width,height,CHANNELS,TRAIN_STEP_INTERVAL[1],True)
#    utils.make_gif('./checkpoints/'+MODEL_NAME+'_'+gt_img_name+'/'+str(i),imgs)
