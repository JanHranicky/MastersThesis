from core import utils,output_modulo_model,data_pool
import tensorflow as tf
from PIL import Image
import numpy as np 
import os
from matplotlib import pyplot as plt
import pathlib
from datetime import datetime
import random
import argparse
from tensorflow.python.client import device_lib

def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains neural network as discrete neural cellular automaton')

    # Add arguments
    parser.add_argument('-c', '--channels', type=int, help='Number of channels of the model', default=2)
    parser.add_argument('-i', '--iters', type=int, help='Maximum number of iterations', default=10)
    parser.add_argument('-s', '--states', type=int, help='Size of the state space.', default=8)
    parser.add_argument('-t', '--train_interval', type=utils.parse_int_tuple, help='Train interval of the network', default=(20,30))
    parser.add_argument('-m', '--image', type=str, help='Path to GT image', default='./img/vut_logo_17x17_2px_padding.png')
    parser.add_argument('-r', '--run', type=int, help='Number of the run. If provided results will be stored in a subfolder', default=None)
    parser.add_argument('-f', '--folder', type=str, help='Folder in which the reults will be stored', default='./checkpoints/DNCA/')
    parser.add_argument('-g', '--full_range', type=bool, help='If set to true will validate all RGB channels of the image', default=False)

    return parser.parse_args()

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
               visualize_iters=10000,
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
      self.gt_tf = self.img_to_discrete_tensor(gt_img,state_num)
    else:
      self.gt_tf = tf.convert_to_tensor(gt_img, dtype=tf.float32)
    
    self.color_dict = {i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(self.state_num+1)}
    
    self.data_pool_training = data_pool_training
    if self.data_pool_training:
      self.dp = data_pool.DataPool(self.width,self.height,self.model.channel_n) 
      
    self.checkpoint_path = f'{folder}{self.model.model_name}_{self.gt_img_name}'
    if run:
        run_path = 'run_'+str(run)
        self.checkpoint_path = self.checkpoint_path+'/'+ run_path
    
    self.visualize_iters = visualize_iters
    self.save_iters = save_iters
    self.generate_gif_iters = generate_gif_iters
  
  
  def img_to_discrete_tensor(self, img, states):
    """Converts input image into tensor by summing it's values into a single channel and then using modulo operation to get the values into the wanted number of states

    Args:
        img (Pil.Image): input as a PIL Image
        states (integer): integer number of wanted states

    Returns:
        tf.Tensor: converted PIL Image into tf.Tensor
    """
    tensor = tf.reduce_sum(tf.convert_to_tensor(img, dtype=tf.float32), axis=-1)
    return tf.math.floormod(tensor, tf.ones_like(tensor) * states)
  
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
        self.generate_gif(self.save_iters,width,height,last_iter,True)
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
    
  def save_progress(self,i,loss_values,result=False):
    if i%self.save_iters == 0 or result:
      safe_name = str(i) if not result else "result_"+str(i)+"_steps"
      self.model.save_weights(self.checkpoint_path+'/'+safe_name)

      np.save(self.checkpoint_path+'/convergence_arr.npy', loss_values)
      
      plt.plot(loss_values)
      plt.title(f'Loss function epoch num. {i}')
      plt.xlabel('Epoch')
      plt.ylabel('Loss value')
      plt.savefig(f'{self.checkpoint_path}/loss.png')
  
  def make_gif(self,name,frames):
    frame_one = frames[0]
    frame_one.save(name+".gif", format="GIF", append_images=frames,
              save_all=True, duration=100, loop=0)  
  
  def generate_gif(self,i,width,height,iter,result=False):
    if not self.generate_gif or i%self.generate_gif_iters != 0: return
    
    frames = []
    x = utils.init_batch(1,width,height,self.model.channel_n)
    for _ in range(iter):
      x = self.model(x)

      if not self.full_range:
        f = Image.fromarray(np.uint8(x[0][:,:,0].numpy()),mode="L")
        frames.append(self.grayscale_to_rgb(f))
      else:
        f = Image.fromarray(np.uint8(x[0][:,:,:3].numpy()))
        print(f)
        frames.append(f)
        
    
    gif_name = str(i) if not result else "result_"+str(iter.numpy())+"_steps"
    self.make_gif(str(self.checkpoint_path)+'/'+gif_name,frames)

  def grayscale_to_rgb(self,grayscale_image):
    rgb_image = Image.new("RGB", grayscale_image.size)
      
    for x in range(grayscale_image.width):
        for y in range(grayscale_image.height):
            grayscale_value = grayscale_image.getpixel((x, y))
            
            if grayscale_value in self.color_dict:
              rgb_value = self.color_dict[grayscale_value]
            else:
              random_rgb_value = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
              rgb_value = random_rgb_value
              
              self.color_dict[grayscale_value] = random_rgb_value
              
            rgb_image.putpixel((x, y), rgb_value)
      
    return rgb_image  
      
if __name__ == '__main__':
  arguments = parse_arguments()
  
  date_time = datetime.now().strftime("%m_%d_%Y")
  gt_img = Image.open(arguments.image)
  
  def custom_mse(gt,x):
    l_x = utils.match_last_channel(x,gt)
    return tf.reduce_mean(tf.square(l_x - gt))

  ca = output_modulo_model.CA(channel_n=arguments.channels,model_name=date_time+'_modulo_'+os.path.basename(__file__).split('.')[0]+'_'+str(arguments.states)+"_states_"+str(arguments.channels)+"_layers_"+str(arguments.train_interval[0])+"_"+str(arguments.train_interval[1])+"_steps_full_range_"+str(arguments.full_range),states=arguments.states)
  #ca.load_weights("./checkpoints/01_10_2024_in_range_single_channel_cmp8_states_single_c_compare_4_channels_xhrani02_100x100/64500")

  loss_f = custom_mse

  t = DncaTrainer(
                  ca,
                  loss_f,
                  gt_img.convert("RGB"),
                  gt_img.filename.split('/')[-1].split('.')[0],
                  state_num=arguments.states,
                  generate_gif_iters=50000,
                  data_pool_training=True,
                  visualize_iters=50000,
                  save_iters=50000,
                  train_step_interval=arguments.train_interval,
                  run=arguments.run,
                  folder=arguments.folder,
                  epoch_num=arguments.iters,
                  full_range=arguments.full_range
                  )
  """
    def __init__(self,
               model,
               loss_f,
               gt_img,
               state_num,
               batch_size=16,
               compare_channels=2,
               data_pool_training=False,
               lr=0.001, 
               epoch_num=100000,
               visualize=True,
               visualize_iters=10000,
               save_iters=5000,
               generate_gif_iters=5000,
               train_step_interval=(75,100)
               ):
  """

  t.train()


