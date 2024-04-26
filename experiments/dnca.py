from core import utils,output_modulo_model,dnca_trainer
import tensorflow as tf
from PIL import Image
import os
from datetime import datetime
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains neural network as discrete neural cellular automaton')

    # Add arguments
    parser.add_argument('-c', '--channels', type=int, help='Number of channels of the model', default=2)
    parser.add_argument('-i', '--iters', type=int, help='Maximum number of iterations', default=300000)
    parser.add_argument('-s', '--states', type=int, help='Size of the state space.', default=8)
    parser.add_argument('-t', '--train_interval', type=utils.parse_int_tuple, help='Train interval of the network', default=(20,30))
    parser.add_argument('-m', '--image', type=str, help='Path to GT image', default='./img/vut_logo_17x17_2px_padding.png')
    parser.add_argument('-r', '--run', type=int, help='Number of the run. If provided results will be stored in a subfolder', default=None)
    parser.add_argument('-f', '--folder', type=str, help='Folder in which the reults will be stored', default='./checkpoints/DNCA/')
    parser.add_argument('-g', '--full_range', type=bool, help='If set to true will validate all RGB channels of the image', default=False)

    return parser.parse_args()
      
if __name__ == '__main__':
  arguments = parse_arguments()
  
  date_time = datetime.now().strftime("%m_%d_%Y")
  gt_img = Image.open(arguments.image)
  
  ca = output_modulo_model.CA(channel_n=arguments.channels,model_name=date_time+'_modulo_'+os.path.basename(__file__).split('.')[0]+'_'+str(arguments.states)+"_states_"+str(arguments.channels)+"_layers_"+str(arguments.train_interval[0])+"_"+str(arguments.train_interval[1])+"_steps_full_range_"+str(arguments.full_range),states=arguments.states)
  loss_f = utils.custom_mse

  t = dnca_trainer.DncaTrainer(
                  ca,
                  loss_f,
                  gt_img.convert("RGB"),
                  gt_img.filename.split('/')[-1].split('.')[0],
                  state_num=arguments.states,
                  generate_gif_iters=50000,
                  data_pool_training=True,
                  visualize_iters=50000,
                  save_iters=50000,
                  train_step_interval=arguments.train_interval,
                  run=arguments.run,
                  folder=arguments.folder,
                  epoch_num=arguments.iters,
                  full_range=arguments.full_range
                  )
  t.train()


