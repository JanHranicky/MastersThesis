from core import model,utils,added_conv_model
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

STEP_NUM = 1000
CHANNELS = 16
CHECKPOINT_PATH = './checkpoints/added_conv_layer_ca_vut_logo_small/100000'
#CHECKPOINT_PATH = './checkpoints/Meta_persistent/persistent_pattern_ca'+str(CHANNELS)+'_vut_logo_small/100000'
GT_IMG_PATH = './img/vut_logo_small.png'
GIF_NAME = 'trainable_conv_layer'

width,height = Image.open(GT_IMG_PATH).size
x = utils.init_batch(1,width,height)

ca = added_conv_model.CA(channel_n=CHANNELS)
ca.load_weights(CHECKPOINT_PATH)

imgs = utils.run_model_for_i(ca,width,height,CHANNELS,1000,True)
utils.make_gif(GIF_NAME,imgs)