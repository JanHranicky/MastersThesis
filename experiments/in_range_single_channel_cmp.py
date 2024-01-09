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
import random

class discreteOutTrainer(trainer.Trainer):
    def __init__(self,model,loss_f,gt_img,gt_img_name,state_num,grayscale=False,data_pool_training=False,lr=0.001,epoch_num=300000,visualize=True,visualize_iters=10000,save_iters=5000,generate_gif_iters=5000,train_step_interval=(75,100)):
        super().__init__(model,loss_f,gt_img,gt_img_name,grayscale,data_pool_training,lr,epoch_num,visualize,visualize_iters,save_iters,generate_gif_iters,train_step_interval)
        self.STATE_NUM = state_num
        self.batch_size = 2
        self.compare_channels = 1
        
        self.gt_tf = utils.img_to_discrete_space_tf(self.gt_img,self.STATE_NUM,self.compare_channels)
                
    @tf.function
    def train_step(self,x,trainer):
      iter_n = tf.random.uniform([], self.train_step_interval[0], self.train_step_interval[1], tf.int32)
      with tf.GradientTape() as g:
        for i in tf.range(iter_n):
          x = self.model(x)  
        loss = tf.math.reduce_mean(self.loss_f(x,self.gt_tf,self.STATE_NUM))
        
      grads = g.gradient(loss, self.model.weights)
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
          converted = utils.convert_to_comparable_shape(x0,self.compare_channels)
          highest_loss_i = self.dp.get_highest_loss_index(self.gt_img,converted,self.loss_f,args=(self.gt_tf,self.STATE_NUM))
          x0 = self.dp.insert_seed_tensor(x0,highest_loss_i)
        else:
          x0 = utils.init_batch(self.batch_size,width,height,self.model.channel_n)
          
        x, loss = self.train_step(x0,trainer)
        
        loss_val = np.log10(loss.numpy())
        print(f'epoch: {i}, loss_val={loss_val}')
        loss_values.append(loss_val)
        
        if self.data_pool_training:
          self.dp.commit(x)
        
        self.visualize_batch(tf.math.floormod(x,tf.ones_like(x,dtype=tf.float32)*self.STATE_NUM),i)
        if np.isnan(loss_val):
          break
        self.save_progress(i,loss_values)
        self.generate_gif(i,width,height)

GT_IMG_PATH = './img/xhrani02_100x100.png'
date_time = datetime.now().strftime("%m_%d_%Y")
gt_img = Image.open(GT_IMG_PATH)

CHANNEL_NUM = 1
STATES = 8

def custom_mse(x, gt, states):
    #print(x)
    #print(gt)
    #print(states)
    l_x = utils.match_last_channel(x,gt)
    #l_x = tf.cast(tf.math.round(x),dtype=tf.int32)
    #l_x = tf.math.floormod(l_x,tf.ones_like(l_x,dtype=tf.int32)*states)
    #print(l_x.shape)
    #print(gt.shape)
    #print("here")
    #loss = tf.reduce_mean(tf.square(l_x - gt))
    #print(f"returning loss {loss}")
    return tf.reduce_mean(tf.square(l_x - gt))

ca = added_conv_model.CA(channel_n=CHANNEL_NUM,model_name=date_time+'_'+os.path.basename(__file__).split('.')[0]+"_batch_"+str(STATES)+"_states_single_C_TEST",rule_model="batch")
#ca.load_weights("./checkpoints/01_08_2024_output_in_range_single_channel_batch_8_states_single_C_xhrani02_100x100/93000")

#loss_f = tf.keras.losses.MeanSquaredError()
loss_f = custom_mse

t = discreteOutTrainer(ca,
                    loss_f,gt_img,
                    GT_IMG_PATH.split('/')[-1].split('.')[0],
                    state_num=STATES,
                    generate_gif_iters=10000,
                    data_pool_training=True,
                    visualize=False,
                    visualize_iters=500,
                    save_iters=500
                    )
t.train()


#color_list = utils.generate_random_colors(STATES)
#utils.display_tensor(color_list,t.gt_tf)
#height,width = t.gt_img.size
#x = utils.init_batch(1,width,height,CHANNEL_NUM)
#
#for i in range(100):
#  x = ca(x)
#  
#
##x = tf.math.floormod(tf.math.round(x),tf.ones_like(x,dtype=tf.float32)*STATES)
#utils.display_tensor(color_list,x[0][:,:,0])
#tf.print(x,summarize=-1)


