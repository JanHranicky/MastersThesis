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
    self(tf.zeros([1,3,3,channel_n])) #dummy call to initialiaze model, the dummy shape does not have to be the same as final data. But it's dimensionality should be similiar

  def set_rule_model(self,type,channel_n):
    if type is not None:
      return type
    return tf.keras.Sequential([
      tf.keras.layers.Conv2D(filters=128,kernel_size=1,activation='relu'),
      tf.keras.layers.Conv2D(filters=channel_n,kernel_size=1,
      kernel_initializer=tf.zeros_initializer)
    ])

  @tf.function
  def perceive(self,x):
    '''
    Perceive function applies filter of identity and normalized sobel filters sobel_x and sobel_y on a cellular automata.
    The input cellular automata x should be of shape (N,W,H,16), where N is batch size and WxH are it's dimensions.
    The convolution result for each cell is concatenated in it's depth channel, i.e. the result shape is (N,W,H,48)
    '''
    sobel_x = tf.constant([[-1,0,1],[-2,0,2],[-1,0,1]], dtype='float32')
    sobel_x = tf.divide(sobel_x,8) #dividing sobel by 8 will normalize it, so that max value is 1
    sobel_y = tf.transpose(sobel_x) #Trasnposing the sobel filter for Y dimension
    identity = tf.constant([[0,0,0],[0,1,0],[0,0,0]], dtype='float32') #initializing a simple identity filter
    kernel = tf.stack([sobel_x,sobel_y,identity]) #stacks filters into (3,3,3) kernel
    kernel = tf.expand_dims(kernel, -2) #expands into shape (3,3,1,3)
    kernel = tf.repeat(kernel,self.channel_n,2) #repeats each row and results in shape (3,3,self.channel_n,3)

    return tf.nn.depthwise_conv2d(
    x,
    kernel,
    strides=[1, 1, 1, 1],
    padding='SAME'
    )

  def get_alive_mask(self,x):
    '''
    Compute alive mask and returns it. Alive mask is a (N,W,H,1) mask tensor of the cellular automata x.
    It has boolean value, True means there was an "alive" (having value >0.1 of alpha channel) cell in the xs cell neighbourhood
    '''
    alpha_c = x[:, :, :, 3:4]
    result = tf.nn.max_pool(alpha_c, 3, [1, 1, 1, 1], 'SAME')
    return result > 0.1

  def get_stochastic_update_mask(self,x):
    rand_mask = tf.random.uniform(tf.shape(x[:,:,:,:1])) #copies the shape of cellular automata but with a single depth channel
    rand_mask = rand_mask < self.cell_reset_prob #sets the value to True/False
    return tf.cast(rand_mask, tf.float32)

  @tf.function
  def call(self,x):
    pre_update_mask = self.get_alive_mask(x) #pre update life mask is calculated

    y = self.perceive(x) #perceive vector is calculated
    dx = self.rule_model(y)
    x += dx  * self.get_stochastic_update_mask(x)

    post_update_mask = self.get_alive_mask(x)
    life_mask = tf.cast(tf.math.logical_and(pre_update_mask,post_update_mask), tf.float32) #computes logical and of pre and post update life masks and casts them to float32 so if can be multiplated
    return x * life_mask
