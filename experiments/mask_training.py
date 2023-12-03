from core import model,utils,trainer,added_conv_model
import tensorflow as tf
from PIL import Image,ImageOps
from IPython.display import clear_output,display
import numpy as np 
import os
import IPython.display as display
from matplotlib import pyplot as plt
import pathlib
from datetime import datetime
from masks import vut_logo_mask

class discreteOutTrainer(trainer.Trainer):
    def __init__(self,model,loss_f,gt_img,gt_img_name,grayscale=False,data_pool_training=False,lr=0.001,epoch_num=300000,visualize=True,visualize_iters=10000,save_iters=5000,generate_gif_iters=5000,train_step_interval=(75,100)):
        super().__init__(model,loss_f,gt_img,gt_img_name,grayscale,data_pool_training,lr,epoch_num,visualize,visualize_iters,save_iters,generate_gif_iters,train_step_interval)
        self.prev_step_loss = 0
        
    @tf.function
    def train_step(self,x,trainer):
      iter_n = tf.random.uniform([], self.train_step_interval[0], self.train_step_interval[1], tf.int32)
      with tf.GradientTape() as g:
        for i in tf.range(iter_n):
          x = self.model(x)
          l_x = utils.convert_to_comparable_shape(x,len(self.gt_img.getbands()))
        loss = tf.math.reduce_mean(self.loss_f(self.gt_img, l_x)) + self.prev_step_loss
      grads = g.gradient(loss, self.model.weights)
      trainer.apply_gradients(zip(grads, self.model.weights))
      return tf.cast(tf.cast(x,dtype=tf.int32),dtype=tf.float32), loss
    
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
        self.generate_gif(i,width,height)

GT_IMG_PATH = './img/vut_logo_small.png'
date_time = datetime.now().strftime("%m_%d_%Y")
gt_img = Image.open(GT_IMG_PATH)

ca = added_conv_model.CA(channel_n=16,model_name=date_time+'_'+os.path.basename(__file__).split('.')[0],rule_model="batch")

def mask_loss(img,batch):
  img = tf.cast(img,dtype=tf.float32)

  diff = (batch - img)**2
  diff = tf.reduce_mean(diff,axis=-1)
  
  bckdn = vut_logo_mask.background_mask * diff
  logo = vut_logo_mask.logo_mask * diff
  
  less_mask = tf.less(bckdn, 100.0 * tf.ones_like(bckdn))
  less_indices = tf.where(less_mask)
  less_indices_cnt = less_indices.shape[0]
  if less_indices_cnt is not None:
    tf.tensor_scatter_nd_update(bckdn,less_indices,tf.zeros(shape=(less_indices.shape[0],)))
  
  return tf.reduce_mean(bckdn+logo)

loss_f = mask_loss
#loss_f = tf.keras.losses.MeanSquaredError()

  
t = discreteOutTrainer(ca,
                    loss_f,gt_img,
                    GT_IMG_PATH.split('/')[-1].split('.')[0],
                    generate_gif_iters=10000,
                    data_pool_training=True,
                    visualize=False,
                    visualize_iters=100
                    )
t.train()