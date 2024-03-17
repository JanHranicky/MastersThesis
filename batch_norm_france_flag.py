from core import model,utils,trainer,added_conv_model
import tensorflow as tf
from PIL import Image,ImageOps
from IPython.display import clear_output,display
import numpy as np 
import os
import IPython.display as display
from matplotlib import pyplot as plt

class discreteOutTrainer(trainer.Trainer):
    def __init__(self,model,loss_f,gt_img,gt_img_name,grayscale=False,data_pool_training=False,lr=0.001,epoch_num=300000,visualize=True,visualize_iters=10000,save_iters=5000,generate_gif_iters=5000,train_step_interval=(75,100)):
        super().__init__(model,loss_f,gt_img,gt_img_name,grayscale,data_pool_training,lr,epoch_num,visualize,visualize_iters,save_iters,generate_gif_iters,train_step_interval)
        
    @tf.function
    def train_step(self,x,trainer):
      iter_n = tf.random.uniform([], self.train_step_interval[0], self.train_step_interval[1], tf.int32)
      with tf.GradientTape() as g:
        for i in tf.range(iter_n):
          x = self.model(x)
          l_x = utils.convert_to_comparable_shape(x,len(self.gt_img.getbands()))
        loss = tf.math.reduce_mean(self.loss_f(self.gt_img, l_x))
      grads = g.gradient(loss, self.model.weights)
      trainer.apply_gradients(zip(grads, self.model.weights))
      return tf.cast(tf.cast(x,dtype=tf.int32),dtype=tf.float32), loss


GT_IMG_PATH = './img/flag_of_france.png'
gt_img = Image.open(GT_IMG_PATH)

layers = tf.keras.Sequential([
      tf.keras.layers.Conv2D(filters=128,kernel_size=1,activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Conv2D(filters=16,kernel_size=1,
      kernel_initializer=tf.zeros_initializer),
    ])
ca = added_conv_model.CA(channel_n=16,model_name="13_11_2023_batch_norm_5_10_steps",rule_model=layers)
loss_f = tf.keras.losses.MeanSquaredError()

t = discreteOutTrainer(ca,
                    loss_f,gt_img,
                    GT_IMG_PATH.split('/')[-1].split('.')[0],
                    generate_gif_iters=10000,
                    data_pool_training=True,
                    visualize=False,
                    visualize_iters=10000,
                    train_step_interval=(5,10)
                    )
t.train()