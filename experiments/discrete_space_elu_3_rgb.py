#export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

import tensorflow as tf
from datetime import datetime
import os
from PIL import Image
from core import utils
from random import randrange
import random
import matplotlib.pyplot as plt
import numpy as np
import keras
from masks import xhrani02_mask

@keras.saving.register_keras_serializable()
class CA(tf.keras.Model):
  ''' 
  Represents the model of cellular automata
  '''
  def __init__(self,channel_n=1,model_name="CA"):
    super().__init__() #Need to be called to initialize the super tf.keras.Model class in order to use tensorflow utilities

    self.channel_n = channel_n
    self.model_name = model_name
    self.rule_model = self.set_rule_model(channel_n)
    self.perceive_conv = tf.keras.layers.DepthwiseConv2D(
      kernel_size=3,
      depth_multiplier=3,
      strides=[1, 1],
      padding='SAME'
    )
    self(tf.zeros([1,3,3,channel_n])) #dummy call to initialiaze model, the dummy shape does not have to be the same as final data. But it's dimensionality should be similiar

  def set_rule_model(self,channel_n):
    return tf.keras.Sequential([
      tf.keras.layers.Conv2D(filters=128,kernel_size=1,activation='elu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Conv2D(filters=64,kernel_size=1,activation='elu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Conv2D(filters=32,kernel_size=1,activation='elu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Conv2D(filters=channel_n,kernel_size=1,
      kernel_initializer=tf.zeros_initializer),
    ])

  @tf.function
  def call(self,x):
    y = self.perceive_conv(x)
    dx = self.rule_model(y)
    x += dx

    return x


def img_to_discrete_space_tf(img,state_num,multiplier=1):
    img = utils.pil2tf(img)[:,:,0]
    img = tf.math.floormod(img,tf.ones_like(img)*state_num) * multiplier
    return img


def generate_random_colors(num_colors):
    # Generate 'num_colors' random colors in RGB format
    random_colors = []
    for _ in range(num_colors):
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        random_colors.append((red, green, blue))
    return random_colors

def display_tensor(c_map,t):
    colors_mapped = tf.gather(c_map, tf.cast(t,dtype=tf.int32))
    # Display the image using Matplotlib
    plt.imshow(colors_mapped.numpy())
    plt.axis('off')  # Turn off axis labels
    plt.title('Tensor Displayed as Image with Corresponding Colors')
    plt.ioff()
    plt.show()

def save_progress(path, ca,i,loss_values):
    ca.save_weights(path+'/'+str(i))
    ca.save(path+'/'+str(i)+'.keras')
    
    plt.plot(loss_values)
    plt.title(f'Loss function epoch num. {i}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.savefig(f'{path}/loss_{i}.png')
    
def mask_loss(img,batch):
  img = tf.cast(img,dtype=tf.float32)

  diff = (batch - img)**2
  diff = tf.reduce_mean(diff,axis=0)
  
  bckdn = xhrani02_mask.background_mask * diff
  logo = xhrani02_mask.login_mask * diff
  
  less_mask = tf.less(logo, BCKND_ERROR * tf.ones_like(bckdn))
  less_indices = tf.where(less_mask)
  less_indices_cnt = less_indices.shape[0]
  if less_indices_cnt is not None:
    tf.tensor_scatter_nd_update(bckdn,less_indices,tf.zeros(shape=(less_indices.shape[0],)))
  
  return tf.reduce_mean(bckdn+logo)

GT_IMG_PATH = './img/xhrani02.png'
STATE_NUM = 16581375 #255*255*255 combinations
MULTIPLIER = 1

BCKND_ERROR = 50

BATCH_SIZE = 16
EPOCH_NUM = 300000
TRAIN_INTERVAL = (75,100)

#LOSS = mask_loss
LOSS = tf.keras.losses.MeanSquaredError()
LR = 0.001

date_time = datetime.now().strftime("%m_%d_%Y")
date_time = datetime.now().strftime("%m_%d_%Y")
gt_img = Image.open(GT_IMG_PATH)
gt_tf = img_to_discrete_space_tf(gt_img,STATE_NUM,MULTIPLIER)
width,height = gt_tf.shape[0],gt_tf.shape[1]

ca = CA(model_name=date_time+'_'+os.path.basename(__file__).split('.')[0])
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [2000], [LR, LR*0.1])
trainer = tf.keras.optimizers.Adam(lr_sched)

color_list = generate_random_colors(STATE_NUM)
gt_img_name = GT_IMG_PATH.split('/')[-1].split('.')[0]
checkpoint_path = f'./checkpoints/{ca.model_name}_{gt_img_name}'


def train_step(x):
    with tf.GradientTape() as tape:
        for i in range(randrange(TRAIN_INTERVAL[0],TRAIN_INTERVAL[1])):
            x = ca(x)
        loss = LOSS(utils.convert_to_comparable_shape(x,1),gt_tf)
        
        # Compute gradients
        gradients = tape.gradient(loss, ca.trainable_variables)
        gradients = [(tf.clip_by_value(grad, -1., 1.)) for grad in gradients]
        # Update weights
        trainer.apply_gradients(zip(gradients, ca.trainable_variables))
        
    #tf.print(x[0])
    #tf.print(gt_tf)
    #display_tensor(color_list,gt_tf)
    x = tf.cast(tf.math.floormod(tf.cast(x,dtype=tf.int32),tf.ones_like(x,dtype=tf.int32)*10),dtype=tf.float32)
    return x,loss

loss_values = []
for e in range(EPOCH_NUM):
    x0 = utils.init_batch(BATCH_SIZE,width,height,ca.channel_n,STATE_NUM*MULTIPLIER)

    x,loss = train_step(x0)
    print(f'[e,loss] = [{e},{loss.numpy()}]')
    #if e%1000 == 0:
    #    tf.print(x[0])
    #    display_tensor(color_list,utils.convert_to_comparable_shape(x[0],1))
    loss_values.append(np.log10(loss.numpy()))
    
    if e%5000 == 0:
        save_progress(checkpoint_path,ca,e,loss_values)
        
#ca.load_weights("./checkpoints/12_27_2023_discrete_spacenew_xhrani02/25000") 
#x = utils.init_batch(BATCH_SIZE,width,height,ca.channel_n,STATE_NUM)
#
#for i in range(100):
#    x = ca(x)
#    
#x = tf.math.floormod(tf.cast(x,dtype=tf.int32),tf.ones_like(x,dtype=tf.int32)*STATE_NUM)
#display_tensor(color_list,utils.convert_to_comparable_shape(x[0],1))
