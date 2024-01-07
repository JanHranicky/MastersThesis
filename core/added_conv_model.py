from core import utils
import tensorflow as tf

class CA(tf.keras.Model):
  '''
  Represents the model of cellular automata
  '''
  def __init__(self,channel_n=16,cell_reset_prob=0.5,model_name="CA",rule_model=None):
    super().__init__() #Need to be called to initialize the super tf.keras.Model class in order to use tensorflow utilities

    self.cell_reset_prob = cell_reset_prob
    self.channel_n = channel_n
    self.model_name = model_name
    self.rule_model = self.set_rule_model(rule_model,channel_n)
    self.perceive_conv = tf.keras.layers.DepthwiseConv2D(
      kernel_size=3,
      depth_multiplier=3,
      strides=[1, 1],
      padding='SAME'
    )
    self(tf.zeros([1,3,3,channel_n])) #dummy call to initialiaze model, the dummy shape does not have to be the same as final data. But it's dimensionality should be similiar

  def set_rule_model(self,type,channel_n):
    if type is None:
      return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=128,kernel_size=1,activation='relu'),
        tf.keras.layers.Conv2D(filters=channel_n,kernel_size=1,
        kernel_initializer=tf.zeros_initializer)
      ])
    elif type == "batch":
      return tf.keras.Sequential([
      tf.keras.layers.Conv2D(filters=128,kernel_size=1,activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Conv2D(filters=channel_n,kernel_size=1,
      kernel_initializer=tf.zeros_initializer),
    ])

  @tf.function
  def call(self,x):
    y = self.perceive_conv(x)
    dx = self.rule_model(y)
    x += dx

    return x