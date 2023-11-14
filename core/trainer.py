from core import model,utils,data_pool
import tensorflow as tf
from PIL import Image,ImageOps
from IPython.display import clear_output,display
import numpy as np 
import os
import IPython.display as display
from matplotlib import pyplot as plt
import pathlib


class Trainer():
  def __init__(self,model,loss_f,gt_img,gt_img_name,grayscale=False,data_pool_training=False,lr=0.001,epoch_num=30000,visualize=True,visualize_iters=100,save_iters=5000,generate_gif_iters=5000,train_step_interval=(75,100)) -> None:
    self.model = model
    self.batch_size = 8
    self.loss_f = loss_f
    self.gt_img = gt_img
    self.gt_img_name = gt_img_name
    self.grayscale = grayscale
    self.data_pool_training = data_pool_training
    self.height,self.width = self.gt_img.size
    if data_pool_training:
      self.dp = data_pool.DataPool(self.width,self.height,self.model.channel_n)
    self.lr = lr
    self.epoch_num = epoch_num
    self.visualize = visualize
    self.visualize_iters = visualize_iters
    self.save_iters = save_iters
    self.generate_gif_iters = generate_gif_iters
    self.train_step_interval = train_step_interval
    self.checkpoint_path = f'./checkpoints/{self.model.model_name}_{self.gt_img_name}'

  def visualize_batch(self,x,i):
    if i%self.visualize_iters == 0:
      #clear_output(wait=True)
      utils.visualize_batch(x,utils.tf2grayscale,self.checkpoint_path,str(i),self.visualize)
      
  def save_progress(self,i,loss_values):
    if i%self.save_iters == 0:
      self.model.save_weights(self.checkpoint_path+'/'+str(i))
      
      plt.plot(loss_values)
      plt.title(f'Loss function epoch num. {i}')
      plt.xlabel('Epoch')
      plt.ylabel('Loss value')
      plt.savefig(f'{self.checkpoint_path}/loss_{i}.png')
      
  def generate_gif(self,i,w,h):
    if not self.generate_gif or i%self.generate_gif_iters != 0: return
    imgs = utils.run_model_for_i(self.model,w,h,self.model.channel_n,self.train_step_interval[1],grayscale=self.grayscale)
    utils.make_gif(self.checkpoint_path+'/'+str(i),imgs)
    
  @tf.function
  def train_step(self,x,trainer):
    iter_n = tf.random.uniform([], self.train_step_interval[0], self.train_step_interval[1], tf.int32)
    with tf.GradientTape() as g:
      for i in tf.range(iter_n):
        x = self.model(x)
        l_x = utils.convert_to_comparable_shape(x,len(self.gt_img.getbands()))
      loss = tf.math.reduce_mean(self.loss_f(self.gt_img, l_x))
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
      if self.data_pool_training:
        x0 = self.dp.get_batch(self.batch_size)
        converted = utils.convert_to_comparable_shape(x0,len(self.gt_img.getbands()))
        highest_loss_i = self.dp.get_highest_loss_index(self.gt_img,converted,self.loss_f)
        x0 = self.dp.insert_seed_tensor(x0,highest_loss_i)
      else:
        x0 = utils.init_batch(self.batch_size,width,height,self.model.channel_n)
        
      x, loss = self.train_step(x0,trainer)
      loss_val = np.log10(loss.numpy())
      print(f'epoch: {i}, loss_val={loss_val}')
      loss_values.append(loss_val)
      
      if self.data_pool_training:
        self.dp.commit(x)
      
      self.visualize_batch(x,i)
      if np.isnan(loss_val):
        break
      self.save_progress(i,loss_values)
      self.generate_gif(i,width,height)
