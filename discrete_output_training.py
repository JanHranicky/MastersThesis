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
          if self.grayscale:
            l_x = utils.tf2grayscale(x)
          else:
            l_x = x
        loss = tf.math.reduce_mean(self.loss_f(self.gt_img, l_x))
      grads = g.gradient(loss, self.model.weights)
      #grads = [g/(tf.norm(g)+1e-8) for g in grads]
      trainer.apply_gradients(zip(grads, self.model.weights))
      return x, loss  


GT_IMG_PATH = './img/vut_logo_small.png'
gt_img = Image.open(GT_IMG_PATH)
gt_img = ImageOps.grayscale(gt_img)

ca = added_conv_model.CA(channel_n=16,model_name="13_11_2023_discrete_output")
loss_f = tf.keras.losses.MeanSquaredError()

t = discreteOutTrainer(ca,
                    loss_f,gt_img,
                    GT_IMG_PATH.split('/')[-1].split('.')[0],
                    generate_gif_iters=10000,
                    grayscale=True,
                    data_pool_training=True,
                    visualize=False,
                    visualize_iters=10000
                    )
t.train()