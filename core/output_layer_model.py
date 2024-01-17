from core import utils
import tensorflow as tf

class CA(tf.keras.Model):
  '''
  Represents the model of cellular automata
  '''
  def __init__(self,channel_n=16,model_name="CA",states=8):
    super().__init__() #Need to be called to initialize the super tf.keras.Model class in order to use tensorflow utilities

    self.states = states
    self.channel_n = channel_n
    self.model_name = model_name
    self.rule_model = self.set_rule_model(channel_n)
    
    self.b_norm_layer = tf.keras.layers.BatchNormalization()
    self.rule_model.add(self.b_norm_layer)
    
    self.perceive_conv = tf.keras.layers.DepthwiseConv2D(
      kernel_size=3,
      depth_multiplier=3,
      strides=[1, 1],
      padding='SAME'
    )
    self(tf.zeros([1,3,3,channel_n])) #dummy call to initialiaze model, the dummy shape does not have to be the same as final data. But it's dimensionality should be similiar


  def modulo_activation(self, x):
    return x
    mod_x = tf.math.floormod(x,tf.ones_like(x,dtype=tf.float32)*self.states)
    return mod_x
    return my_floor(mod_x)

  
  def set_rule_model(self,channel_n):
    return tf.keras.Sequential([
      tf.keras.layers.Conv2D(filters=128,kernel_size=1,activation='elu'),
      tf.keras.layers.Conv2D(filters=channel_n,kernel_size=1,
      kernel_initializer=tf.zeros_initializer, activation=self.modulo_activation),
    ])

  @tf.function
  def call(self,x):
    y = self.perceive_conv(x)
    dx = self.rule_model(y)

    return dx
  
@tf.custom_gradient
def my_floor(x):
  def my_floor_grad(dy):
    return 1*dy
  f_x = tf.floor(x)
  return f_x,my_floor_grad

