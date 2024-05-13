from core import utils
import tensorflow as tf

class CA(tf.keras.Model):
  '''
  Represents the model of cellular automata
  '''
  def __init__(self,target_img_path,channel_n=16,model_name="CA",states=8,full_range=False):
    super().__init__() #Need to be called to initialize the super tf.keras.Model class in order to use tensorflow utilities

    self.states = states
    self.full_range = full_range
    self.channel_n = channel_n
    self.model_name = model_name
    self.target_img_path = target_img_path
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
      tf.keras.layers.Conv2D(filters=128,kernel_size=1,activation='elu'),
      tf.keras.layers.Conv2D(filters=64,kernel_size=1,activation='elu'),
      tf.keras.layers.Conv2D(filters=32,kernel_size=1,activation='elu'),
      tf.keras.layers.Conv2D(filters=channel_n,kernel_size=1,
      kernel_initializer=tf.zeros_initializer),
    ])

  @tf.function
  def call(self,x):
    y = self.perceive_conv(x)
    dx = self.rule_model(y)
    
    dx = self.round_int(dx) #round_int activation
    return x+dx
  
  @tf.custom_gradient
  def round_int(self,x):
    def round_int_grad(dy):
      return dy
    f_x = tf.floor(x)
    return f_x,round_int_grad


