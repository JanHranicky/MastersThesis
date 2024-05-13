from core import utils,data_pool
import tensorflow as tf
from PIL import Image
import numpy as np 
from matplotlib import pyplot as plt
import pathlib
import random

class DncaTrainer():
  def __init__(self,
               model,
               loss_f,
               gt_img,
               gt_img_name,
               state_num,
               batch_size=16,
               compare_channels=1,
               data_pool_training=False,
               lr=0.001, 
               epoch_num=100000,
               save_iters=5000,
               generate_gif_iters=5000,
               train_step_interval=(75,100),
               run=None,
               folder='./checkpoints/',
               full_range=False
               ):
    self.batch_size = batch_size
    self.state_num = state_num
    self.compare_channels = compare_channels
    self.model = model
    self.loss_f = loss_f
    
    self.full_range = full_range and self.model.channel_n >= 3
    
    self.epoch_num = epoch_num
    self.lr = lr
    self.train_step_interval = train_step_interval
    
    self.gt_img = gt_img
    self.gt_img_name = gt_img_name
    self.height, self.width = self.gt_img.size
    
    if not self.full_range:
      self.gt_tf = utils.img_to_discrete_tensor(gt_img,state_num)
      self.color_dict = utils.extract_color_dict(gt_img,self.gt_tf)
    else:
      self.gt_tf = tf.convert_to_tensor(gt_img, dtype=tf.float32)
    
    self.data_pool_training = data_pool_training
    if self.data_pool_training:
      self.dp = data_pool.DataPool(self.width,self.height,self.model.channel_n) 
      
    self.checkpoint_path = f'{folder}{self.model.model_name}+{self.gt_img_name}'
    if run:
        run_path = 'run_'+str(run)
        self.checkpoint_path = self.checkpoint_path+'/'+ run_path
    
    self.save_iters = save_iters
    self.generate_gif_iters = generate_gif_iters
  
  @tf.function
  def train_step(self,x,trainer):
    iter_n = tf.random.uniform([], self.train_step_interval[0], self.train_step_interval[1], tf.int32)
    with tf.GradientTape() as g:
      for i in tf.range(iter_n):
        x = self.model(x)
        
      loss = self.loss_f(self.gt_tf,x)

    grads = g.gradient(loss, self.model.weights)
    grads = [g/(tf.norm(g)+1e-8) for g in grads]
    trainer.apply_gradients(zip(grads, self.model.weights))
    
    return x, loss, iter_n
    
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
        #converted = utils.convert_to_comparable_shape(x0,1 if not self.full_range else 3)
        highest_loss_i = self.dp.get_highest_loss_index(self.gt_tf,x0,self.loss_f)
        x0 = self.dp.insert_seed_tensor(x0,highest_loss_i)
      else:
        x0 = utils.init_batch(self.batch_size,width,height,self.model.channel_n)
        
      x, loss, last_iter = self.train_step(x0,trainer)
      
      if loss.numpy() == 0:
        self.save_progress(i,loss_values,True)
        self.generate_gif(self.save_iters,width,height,last_iter,result=True)
        exit()
          
      loss_val = np.log10(loss.numpy())
      print(f'epoch: {i}, loss={loss.numpy()}, log_10(loss)={loss_val}')
      loss_values.append(loss_val)
        
      if self.data_pool_training:
        self.dp.commit(x)
        
      if np.isnan(loss_val):
        break
      self.save_progress(i,loss_values)
      self.generate_gif(i,width,height,last_iter)
    
    #save the last iteration no matter the save_iters and generate_gif_iters parameters
    if self.epoch_num%self.save_iters != 0:
      self.save_progress(self.epoch_num,loss_values,final_iteration=True)
    if self.generate_gif and self.epoch_num%self.generate_gif_iters != 0:
      self.generate_gif(self.epoch_num,width,height,last_iter,final_iteration=True)
    
  def save_progress(self,i,loss_values,result=False,final_iteration=False):
    if i%self.save_iters == 0 or result or final_iteration:
      safe_name = str(i) if not result else "result_"+str(i)+"_steps"
      #safe_name += ".weights.h5"
      self.model.save_weights(self.checkpoint_path+'/'+safe_name)
      self.model.save(self.checkpoint_path+'/'+safe_name+'.keras')

      np.save(self.checkpoint_path+'/convergence_arr.npy', loss_values)
      
      plt.plot(loss_values)
      plt.title(f'Loss function epoch num. {i}')
      plt.xlabel('Epoch')
      plt.ylabel('Loss value')
      plt.savefig(f'{self.checkpoint_path}/loss.png')
    
  def generate_gif(self,i,width,height,iter,result=False,final_iteration=False):
    if not self.generate_gif or i%self.generate_gif_iters != 0 and (not result and not final_iteration): return
    
    frames = []
    x = utils.init_batch(1,width,height,self.model.channel_n)
    for _ in range(iter):
      x = self.model(x)

      if not self.full_range:
        f = Image.fromarray(np.uint8(x[0][:,:,0].numpy()),mode="L")
        frames.append(utils.grayscale_to_rgb(f,self.color_dict))
      else:
        f = utils.tf2pil(x[0].numpy())
        frames.append(f)
    
    gif_name = str(i) if not result else "result_"+str(iter.numpy())+"_steps"
    
    utils.make_gif(str(self.checkpoint_path)+'/'+gif_name,frames)