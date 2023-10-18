import random
from core import utils
import tensorflow as tf
class DataPool:
    """
    Class that implements pool of data
    When initialized DataPool of default len 100 is created and filled with seed tensors of defined sizes
    Class enables fetching and commit batch so that pool training is possible
    """
    def __init__(self, w,h,c,data_len = 100):
        """Constructor of DataPool class, creates datapool of len data_len

        Args:
            w (int): width of stored tensors
            h (int): height of stored tensors
            c (int): channel num
            data_len (int, optional): _description_. Defaults to 100.
        """
        #save shape for later
        self.w = w 
        self.h = h
        self.c = c
        
        self.data = {index: value for index, value in enumerate([utils.get_seed_tensor(w,h,c)] * data_len)}

    def get_batch(self, batch_size):
        """Returns random sample from self.data in a batch_size range

        Args:
            batch_size (int): number of returned tensors

        Returns:
            tf.Tensor: batch of {batch_size} len tensors from DataPool
        """
        self.keys = random.sample(list(self.data.keys()), min(batch_size, len(self.data)))
        batch = [self.data[key] for key in self.keys]
        return tf.stack(batch, axis=0)

    def commit(self,new_data):
        """Updates data in DataPool with new_data parameter.
        Should be called in rotation with get_batch() function,
        i.e commit() updates keys of data which were previously
        returned by get_batch() function. 

        Args:
            new_data ([tf.Tensor]): modified data from last get_batch() call
        """
        for i,v in enumerate(new_data):
            if i == len(self.keys): #new_data has more data then were fetched in get_batch() return.
                break
            self.data[self.keys[i]] = v
            
    def get_highest_loss_index(self,gt_img,data,loss_f):
        """Returns index of the tensor with highest value of loss_f

        Args:
            gt_img (PIL.Image): gt image
            data (tf.Tensor): batch of tensors should have dimensionality [n,w,h,c], where n is batch size, WxHxC are dimensions of the image
            loss_f (function): loss function  that takes in two arguments and returns single number

        Returns:
            int: index of Tensor in data that has the highest value of loss_f
        """
        loss = tf.map_fn(fn = lambda t: loss_f(gt_img,t),elems=data)
        return tf.math.argmax(loss).numpy()

    def insert_seed_tensor(self,batch,index):
        """Inserts seed tensor into tensor batch at given index

        Args:
            batch (tf.Tensor): tensor training batch of dimensionality [n,w,h,c]
            index (int): index of dimension n, that is to be replaced with seed tensor

        Returns:
            tf.Tensor: modified batch, with tensor at index replaced with seed tensor
        """
        b_list = tf.unstack(batch)
        seed = utils.get_seed_tensor(self.w,self.h,self.c)
        b_list[index] = seed
        return tf.stack(b_list, axis=0)
    
    def as_tensor(self):
        return tf.stack([i for i in self.data.values()], axis=0)