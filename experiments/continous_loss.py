from core import utils,trainer,bigger_model
import tensorflow as tf
from PIL import Image,ImageOps
from IPython.display import clear_output,display
import numpy as np 
import os
import IPython.display as display
from matplotlib import pyplot as plt
import pathlib
from datetime import datetime

class CA(tf.keras.Model):
  '''
  Represents the model of cellular automata
  '''
  def __init__(self,channel_n=16,cell_reset_prob=0.5,model_name="CA"):
    super().__init__() #Need to be called to initialize the super tf.keras.Model class in order to use tensorflow utilities

    self.cell_reset_prob = cell_reset_prob
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
      tf.keras.layers.Conv2D(filters=128,kernel_size=1,activation='relu'),
      tf.keras.layers.Conv2D(filters=channel_n,kernel_size=1,
      kernel_initializer=tf.zeros_initializer),
    ])

  @tf.function
  def call(self,x):
    y = self.perceive_conv(x)
    dx = self.rule_model(y)
    x += dx

    return x


def convert_to_comparable_shape(a,b):
  return utils.convert_to_comparable_shape(a,1),utils.convert_to_comparable_shape(b,1)
  if b.shape[-1] != 3:
    return utils.convert_to_comparable_shape(a,1),utils.convert_to_comparable_shape(b,1)
  else:
    return utils.convert_to_comparable_shape(a,3),b

class discreteOutTrainer(trainer.Trainer):
    def __init__(self,model,loss_f,gt_img,gt_img_name,grayscale=False,data_pool_training=False,lr=0.001,epoch_num=300000,visualize=True,visualize_iters=10000,save_iters=5000,generate_gif_iters=5000,train_step_interval=(75,100)):
        super().__init__(model,loss_f,gt_img,gt_img_name,grayscale,data_pool_training,lr,epoch_num,visualize,visualize_iters,save_iters,generate_gif_iters,train_step_interval)
        self.prev_step_loss = 0
        
    def train_step(self,x,trainer):
      iter_n = tf.random.uniform([], self.train_step_interval[0], self.train_step_interval[1], tf.int32)
      with tf.GradientTape() as g:
        total_loss = 0
        for i in tf.range(iter_n):
          x = self.model(x)
          
          err_percentage = 0
          if i < self.train_step_interval[0]:
            err_percentage = 1-((i+1)/self.train_step_interval[0])
          
          total_loss += tf.reduce_mean(mse_with_margin(self.gt_img,x,err_percentage))
          
        #l_x, l_img = convert_to_comparable_shape(x,utils.pil2tf(self.gt_img))
        #loss = tf.math.reduce_mean(self.loss_f(l_img, l_x))
      grads = g.gradient(total_loss, self.model.weights)
      trainer.apply_gradients(zip(grads, self.model.weights))
      
      #batch_in_range = tf.cast(tf.math.floormod(tf.cast(x,dtype=tf.int32),tf.ones_like(x,dtype=tf.int32)*255),dtype=tf.float32)
      return x, total_loss, iter_n
    
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
          
        x, loss, steps = self.train_step(x0,trainer)
        self.prev_step_loss = loss
        
        
        loss_val = np.log10(loss.numpy())
        print(f'epoch: {i}, loss_val={loss_val}')
        loss_values.append(loss_val)
        
        if self.data_pool_training:
          self.dp.commit(x)
        
        self.visualize_batch(x,i)
        if np.isnan(loss_val):
          break
        self.save_progress(i,loss_values)
        self.generate_gif(i,width,height,steps)
        
    def visualize_batch(self,x,i):
      if i%self.visualize_iters == 0:
        #clear_output(wait=True)
        utils.visualize_batch(utils.convert_to_comparable_shape(x,1),self.checkpoint_path,str(i),self.visualize)

    def generate_gif(self,i,w,h,steps):
      def run_model_for_i(ca,w,h,c,steps,grayscale=False,checkpoint_path = None):
        x = utils.init_batch(1,w,h,c)
        
        if checkpoint_path:
            ca.load_weights(checkpoint_path)
            
        frames = [utils.tf2pil(x[0].numpy(),True)]
        for i in range(steps):
            x = ca(x)
            frames.append(utils.tf2pil(x[0].numpy(),True))
        return frames
      
      if not self.generate_gif or i%self.generate_gif_iters != 0: return
      imgs = run_model_for_i(self.model,w,h,self.model.channel_n,steps,grayscale=self.grayscale)
      utils.make_gif(self.checkpoint_path+'/'+str(i),imgs)


def mse_with_margin(img,batch,margin):
  img = tf.cast(img,dtype=tf.float32)[:,:,0:1]

  allowed_err = (img * float(margin))[:,:,0]
  diff = (batch - img)**2
  diff = tf.reduce_mean(diff,axis=-1)
    
  less_mask = tf.less(diff, allowed_err)
  less_indices = tf.where(less_mask)
  less_indices_cnt = less_indices.shape[0]
  
  if less_indices_cnt is not None:
    diff = tf.tensor_scatter_nd_update(diff,less_indices,tf.zeros(shape=(less_indices.shape[0],)))

  return diff

GT_IMG_PATH = './img/xhrani02.png'
date_time = datetime.now().strftime("%m_%d_%Y")
gt_img = Image.open(GT_IMG_PATH)

ca = CA(channel_n=1,model_name=date_time+'_'+os.path.basename(__file__).split('.')[0]+"_og_model_long_run")
loss_f = tf.keras.losses.MeanSquaredError()

t = discreteOutTrainer(ca,
                    loss_f,gt_img,
                    GT_IMG_PATH.split('/')[-1].split('.')[0],
                    generate_gif_iters=10000,
                    data_pool_training=False,
                    visualize=False,
                    visualize_iters=10000,
                    epoch_num=1000000
                    )
t.train()
