import random
from core import utils

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
        self.data = {index: value for index, value in enumerate([utils.get_seed_tensor(w,h,c)] * data_len)}

    def get_batch(self, batch_size):
        """Returns random sample from self.data in a batch_size range

        Args:
            batch_size (int): number of returned tensors

        Returns:
            [tf.Tensor]: array of {batch_size} len of tensors from DataPool
        """
        self.keys = random.sample(list(self.data.keys()), min(batch_size, len(self.data)))
        batch = [self.data[key] for key in self.keys]
        return batch

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