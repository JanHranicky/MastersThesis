from core import model,utils,trainer,output_layer_model
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
    
        self.clip_value = 1.0
        self.c_list = [(random.randint(0, 255),
                      random.randint(0, 255),
                      random.randint(0, 255))
                     for _ in range(self.STATE_NUM+1)]
    #@tf.function
    def train_step(self,x,trainer):
      iter_n = tf.random.uniform([], self.train_step_interval[0], self.train_step_interval[1], tf.int32)
      with tf.GradientTape() as g:
        for i in tf.range(iter_n):
          x = self.model(x)
        #print(x)
        loss = tf.math.reduce_mean(self.loss_f(x,self.gt_tf,self.STATE_NUM))
        #tf.print('loss:',loss)
        #tf.debugging.check_numerics(loss, "Loss contains NaN or Inf values")

      grads = g.gradient(loss, self.model.weights)
      grads = [tf.clip_by_norm(grad,self.clip_value) for grad in grads]

      #tf.print('before clip',grads)
      #grads = [g/(tf.norm(g)+1e-8) for g in grads]
      #tf.print('after clip',grads)
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
          #converted = utils.convert_to_comparable_shape(x0,self.compare_channels)
          highest_loss_i = self.dp.get_highest_loss_index(self.gt_img,x0,self.loss_f,args=(self.gt_tf,self.STATE_NUM))
          x0 = self.dp.insert_seed_tensor(x0,highest_loss_i)
        else:
          x0 = utils.init_batch(self.batch_size,width,height,self.model.channel_n)
          
        x, loss = self.train_step(x0,trainer)
        
        loss_val = np.log10(loss.numpy())
        print(f'epoch: {i}, loss_val={loss_val}')
        loss_values.append(loss_val)
        
        if self.data_pool_training:
          self.dp.commit(x)
        
        #self.visualize_batch(tf.math.floormod(x,tf.ones_like(x,dtype=tf.float32)*self.STATE_NUM),i)
        if np.isnan(loss_val):
          break
        self.save_progress(i,loss_values)
        self.generate_gif(i,width,height)
   
    def grayscale_to_rgb(self,grayscale_image):
      # Create a new image with the same size as the original but in RGB mode
      rgb_image = Image.new("RGB", grayscale_image.size)
      
      # Iterate through each pixel in the image
      for x in range(grayscale_image.width):
          for y in range(grayscale_image.height):
              # Get the grayscale pixel value at (x, y)
              grayscale_value = grayscale_image.getpixel((x, y))
              
              # Map the grayscale value to an RGB value
              # For simplicity, let's set R, G, and B to the grayscale value
              rgb_value = self.c_list[grayscale_value]
              
              # Set the RGB value at (x, y) in the new image
              rgb_image.putpixel((x, y), rgb_value)
      
      return rgb_image      
    
    def make_gif(self,name,frames):
      frame_one = frames[0]
      frame_one.save(name+".gif", format="GIF", append_images=frames,
                save_all=True, duration=100, loop=0)
    
    def generate_gif(self,i,width,height):
      if not self.generate_gif or i%self.generate_gif_iters != 0: return
      
      frames = []
      x = utils.init_batch(1,width,height,CHANNEL_NUM)
      for _ in range(100):
        x = self.model(x)
        
        f = tf.math.floormod(x,tf.ones_like(x,dtype=tf.float32)*self.STATE_NUM)
        f = tf.math.round(f)[0][:,:,0]
        #tf.print(f)
        f = Image.fromarray(np.uint8(f.numpy()),mode="L")
        frames.append(self.grayscale_to_rgb(f))
      
      #print(frames)
      #print(str(self.checkpoint_path)+'/'+str(i))
      self.make_gif(str(self.checkpoint_path)+'/'+str(i),frames)

GT_IMG_PATH = './img/xhrani02_100x100.png'
date_time = datetime.now().strftime("%m_%d_%Y")
gt_img = Image.open(GT_IMG_PATH)

CHANNEL_NUM = 1
STATES = 8

def custom_mse(x, gt, states):
    l_x = utils.match_last_channel(x,gt)
    return tf.reduce_mean(tf.square(l_x - gt))

ca = output_layer_model.CA(channel_n=CHANNEL_NUM,model_name=date_time+'_'+os.path.basename(__file__).split('.')[0]+'_'+str(STATES)+"_states_"+str(CHANNEL_NUM)+"_layers",states=STATES)
#ca.load_weights("./checkpoints/01_10_2024_in_range_single_channel_cmp8_states_single_c_compare_4_channels_xhrani02_100x100/64500")

#loss_f = tf.keras.losses.MeanSquaredError()
loss_f = custom_mse

t = discreteOutTrainer(ca,
                    loss_f,gt_img,
                    GT_IMG_PATH.split('/')[-1].split('.')[0],
                    state_num=STATES,
                    generate_gif_iters=100,
                    data_pool_training=True,
                    visualize=False,
                    visualize_iters=100,
                    save_iters=500
                    )
t.train()


#color_list = utils.generate_random_colors(STATES)
#utils.display_tensor(color_list,t.gt_tf[...,0])
#height,width = t.gt_img.size
#x = utils.init_batch(1,width,height,CHANNEL_NUM)
#
#for i in range(100):
#  x = ca(x)
#  
#
#x = tf.math.floormod(tf.math.round(x),tf.ones_like(x,dtype=tf.float32)*STATES)
#utils.display_tensor(color_list,x[0][:,:,0])
#tf.print(x,summarize=-1)


