from core import utils,output_modulo_model,dnca_trainer
import tensorflow as tf
from PIL import Image
import os
from datetime import datetime
import argparse
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains neural network as discrete neural cellular automaton using Gradient descent')

    # Add arguments
    parser.add_argument('-c', '--channels', type=int, help='Number of channels of the model', default=2)
    parser.add_argument('-i', '--iters', type=int, help='Maximum number of iterations', default=300000)
    parser.add_argument('-s', '--states', type=int, help='Size of the state space.', default=8)
    parser.add_argument('-t', '--train_interval', type=utils.parse_int_tuple, help='Train interval of the network', default=(20,30))
    parser.add_argument('-m', '--image', type=str, help='Path to GT image', default='./img/vut_logo_17x17_2px_padding.png')
    parser.add_argument('-r', '--run', type=int, help='Number of the run. If provided results will be stored in a subfolder', default=None)
    parser.add_argument('-f', '--folder', type=str, help='Folder in which the reults will be stored', default='./checkpoints/GD/')
    parser.add_argument('-g', '--full_range', type=bool, help='If set to true will validate all RGB channels of the image', default=False)
    parser.add_argument('-l', '--lr', type=float, help='Learning rate of the model', default=0.001)

    return parser.parse_args()
      
if __name__ == '__main__':
  arguments = parse_arguments()
  
  date_time = datetime.now().strftime("%m_%d_%Y")
  gt_img = Image.open(arguments.image)
  
  model_name = "{}+gd+states_{}+channels_{}+train_interval_{}+full_range_{}".format(
    date_time,
    arguments.states, #gt_img name
    arguments.channels,
    arguments.train_interval,
    arguments.full_range
  )
  ca = output_modulo_model.CA(arguments.image,channel_n=arguments.channels,model_name=model_name,states=arguments.states)
  loss_f = utils.custom_mse

  t = dnca_trainer.DncaTrainer(
                  ca,
                  loss_f,
                  gt_img.convert("RGB"),
                  gt_img.filename.split('/')[-1].split('.')[0],
                  state_num=arguments.states,
                  generate_gif_iters=1000,
                  data_pool_training=True,
                  save_iters=1000,
                  lr=arguments.lr,
                  train_step_interval=arguments.train_interval,
                  run=arguments.run,
                  folder=arguments.folder,
                  epoch_num=arguments.iters,
                  full_range=arguments.full_range
                  )
  t.train()


