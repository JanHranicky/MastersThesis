
from core import model,utils,added_conv_model
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import keras

STEP_NUM = 1000
CHANNELS = 16
CHECKPOINT_PATH = './checkpoints/11_26_2023_batch_norm_multiplic_duck/295000'
#CHECKPOINT_PATH = './checkpoints/Meta_persistent/persistent_pattern_ca'+str(CHANNELS)+'_vut_logo_small/100000'
GT_IMG_PATH = './img/duck.png'
GIF_NAME = 'test_load'

width,height = Image.open(GT_IMG_PATH).size
x = utils.init_batch(1,width,height)

ca = added_conv_model.CA(channel_n=16,model_name="_batch_norm_multiplic",rule_model="batch")
ca.load_weights(CHECKPOINT_PATH)
ca = keras.models.load_model('./duck_model.keras')

ca.save('duck_model.keras')
imgs = utils.run_model_for_i(ca,height,width,CHANNELS,100)
utils.make_gif(GIF_NAME,imgs)