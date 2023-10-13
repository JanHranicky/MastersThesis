from core import model,utils,trainer
import tensorflow as tf
from PIL import Image,ImageOps
from IPython.display import clear_output,display
import numpy as np 
import os
import IPython.display as display
from matplotlib import pyplot as plt


GT_IMG_PATH = './img/vut_logo_small.png'
gt_img = Image.open(GT_IMG_PATH)
gt_img = ImageOps.grayscale(gt_img)

ca = model.CA(channel_n=16,model_name="discrete_output_CA")
loss_f = tf.keras.losses.MeanSquaredError()

t = trainer.Trainer(ca,
                    loss_f,gt_img,
                    GT_IMG_PATH.split('/')[-1].split('.')[0],
                    generate_gif_iters=100,
                    grayscale=True
                    )
t.train()