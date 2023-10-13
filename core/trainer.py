from core import model,utils
import tensorflow as tf
from PIL import Image,ImageOps
from IPython.display import clear_output,display
import numpy as np 
import os
import IPython.display as display
from matplotlib import pyplot as plt
import pathlib


class Trainer():
  def __init__(self,model,loss_f,gt_img,gt_img_name,lr=0.001,epoch_num=30000,visualize=True,visualize_iters=100,save_iters=5000,generate_gif_iters=5000,train_step_interval=(75,100)) -> None:
    self.model = model
    self.loss_f = loss_f
    self.gt_img = gt_img
    self.gt_img_name = gt_img_name
    self.lr = lr
    self.epoch_num = epoch_num
    self.visualize = visualize
    self.visualize_iters = visualize_iters
    self.save_iters = save_iters
    self.generate_gif_iters = generate_gif_iters
    self.train_step_interval = train_step_interval
    self.checkpoint_path = f'./checkpoints/{self.model.model_name}_{self.gt_img_name}'

  def visualize_batch(self,x,i):
    if self.visualize and i%self.visualize_iters == 0:
      #clear_output(wait=True)
      utils.visualize_batch(x,utils.tf2grayscale,self.checkpoint_path,str(i))
      
  def save_progress(self,i,loss_values):
    if i%self.save_iters == 0:
      self.model.save_weights(self.checkpoint_path+'/'+str(i))
      
      plt.plot(loss_values)
      plt.title(f'Loss function epoch num. {i}')
      plt.xlabel('Epoch')
      plt.ylabel('Loss value')
      plt.savefig(f'{self.checkpoint_path}/loss_{i}.png')
    
  @tf.function
  def train_step(self,x,trainer):
    iter_n = tf.random.uniform([], self.train_step_interval[0], self.train_step_interval[1], tf.int32)
    with tf.GradientTape() as g:
      for i in tf.range(iter_n):
        x = self.model(x)
      loss = tf.math.reduce_mean(self.loss_f(self.gt_img, utils.tf2grayscale(x)))
    grads = g.gradient(loss, self.model.weights)
    grads = [g/(tf.norm(g)+1e-8) for g in grads]
    trainer.apply_gradients(zip(grads, self.model.weights))
    return x, loss  
  
  def train(self):
    pathlib.Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True) 

    height,width = self.gt_img.size
    lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [2000], [self.lr, self.lr*0.1])
    trainer = tf.keras.optimizers.Adam(lr_sched)
    
    loss_values = []
    for i in range(self.epoch_num):
      x0 = utils.init_batch(4,width,height,self.model.channel_n)

      x, loss = self.train_step(x0,trainer)
      loss_val = np.log10(loss.numpy())
      loss_values.append(loss_val)
      
      self.visualize_batch(x,i)
      if np.isnan(loss_val):
        break
      self.save_progress(i,loss_values)
