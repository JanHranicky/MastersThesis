from core import model,utils,trainer,output_modulo_model
import tensorflow as tf
import tensorflow_probability as tfp
from PIL import Image,ImageOps
from IPython.display import clear_output,display
import numpy as np 
import os
import IPython.display as display
from matplotlib import pyplot as plt
import pathlib
from datetime import datetime
import random
import sys 
import math
import argparse
from masks import vut_logo_mask,vut_logo_6x7_2px_padding_mask

def parse_int_tuple(arg):
    try:
        # Assuming the input format is (x, y)
        x, y = map(int, arg.strip('()').split(','))
        return x, y
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid int tuple format. Use (x, y)")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains neural network using differential evolution')

    # Add arguments
    parser.add_argument('-c', '--channels', type=int, help='Number of channels of the model', default=1)
    parser.add_argument('-i', '--iters', type=int, help='Maximum number of iterations', default=1000)
    parser.add_argument('-s', '--states', type=int, help='Size of the state space.', default=8)
    parser.add_argument('-t', '--train_interval', type=parse_int_tuple, help='Train interval of the network', default=(20,30))
    parser.add_argument('-d', '--std_dev', type=float, help='Stddev used to generate initial population', default=0.02)
    parser.add_argument('-p', '--pop_size', type=int, help='size of the population', default=40)
    parser.add_argument('-w', '--diff_weight', type=float, help='The parameter controlling the strength of mutation in the algorithm', default=0.5)
    parser.add_argument('-x', '--cross_prob', type=float, help='The probability of recombination per site', default=0.9)
    parser.add_argument('-g', '--seed', type=int, help='Seed for generator', default=random.randint(0,sys.maxsize))
    parser.add_argument('-m', '--image', type=str, help='Path to GT image', default='./img/vut_logo_17x17_2px_padding.png')
    parser.add_argument('-r', '--run', type=int, help='Number of the run. If provided results will be stored in a subfolder', default=None)

    # Parse the command line arguments
    args = parser.parse_args()

    # Access the parsed arguments using dot notation
    parameters = {
        'channels': args.channels,
        'iters': args.iters,
        'states': args.states,
        'train_interval': args.train_interval,
        'std_dev': args.std_dev,
        'pop_size': args.pop_size,
        'diff_weight': args.diff_weight,
        'cross_prob': args.cross_prob,
        'seed': args.seed,
        'image': args.image,
        'run': args.run
    }

    return parameters

arguments = parse_arguments()
print(arguments)

GT_IMG_PATH = arguments['image']
date_time = datetime.now().strftime("%m_%d_%Y")
gt_img = Image.open(GT_IMG_PATH)

COMPARE_CHANNELS = 1

height,width = gt_img.size
gt_tf = utils.img_to_discrete_space_tf(gt_img,arguments['states'],COMPARE_CHANNELS)
model_name = "{}+{}+{}+channels_{}+iters_{}+states_{}+train_interval_{}+std_dev_{}+pop_size_{}+diff_weight_{}+cross_prob_{}".format(
    date_time,
    "mask_loss",
    GT_IMG_PATH.split('/')[-1].split('.')[0], #gt_img name
    arguments['channels'],
    arguments['iters'],
    arguments['states'],
    arguments['train_interval'],
    arguments['std_dev'],
    arguments['pop_size'],
    arguments['diff_weight'],
    arguments['cross_prob']
)
ca = output_modulo_model.CA(channel_n=arguments['channels'],model_name=model_name,states=arguments['states'])
CHECKPOINT_PATH = f'./checkpoints/DE/'+ca.model_name
RUN_NUM = arguments['run']

#ca.load_weights("./checkpoints/01_20_2024_modulo_output_modulo_8_states_3_layers_vut_logo_17x17_2px_padding/15700")

def extract_weights_as_tensors(model):
    """Extracts weights of tensorflow model. Parses them into tensors and returns them in a list. 
    Used to represent model weights as a chromozone

    Args:
        model (tf.keras.Model): model to extract weights

    Returns:
        List(tf.Tensor): An array of tensors representing the weights of the model.
    """
    return [tf.convert_to_tensor(var) for var in model.get_weights()]

def chromozone2weights(c):
    """Transforms list of tensors into list of numpy arrays.
    Used to parse chromozone representation of model weights back into the format used by tf.keras.Model.set_weights()

    Args:
        c (List(tf.Tensor)): chromozone

    Returns:
        List(np.Array): representation suitable for tf.keras.Model.set_weights() function
    """
    return [var.numpy() for var in c]

def grayscale_to_rgb(grayscale_image):
    c_list = [(random.randint(0, 255),
                      random.randint(0, 255),
                      random.randint(0, 255))
                     for _ in range(arguments['states']+1)]
  # Create a new image with the same size as the original but in RGB mode
    rgb_image = Image.new("RGB", grayscale_image.size)
  
  # Iterate through each pixel in the image
    for x in range(grayscale_image.width):
        for y in range(grayscale_image.height):
          # Get the grayscale pixel value at (x, y)
            grayscale_value = grayscale_image.getpixel((x, y))
          
          # Map the grayscale value to an RGB value
          # For simplicity, let's set R, G, and B to the grayscale value
            rgb_value = c_list[grayscale_value]
          
          # Set the RGB value at (x, y) in the new image
            rgb_image.putpixel((x, y), rgb_value)
    
    return rgb_image    

def make_gif(name,frames):
  frame_one = frames[0]
  frame_one.save(name+".gif", format="GIF", append_images=frames,
            save_all=True, duration=100, loop=0)

def custom_mse(x, gt):
    l_x = utils.match_last_channel(x,gt)
    return tf.reduce_mean(tf.square(l_x - gt))

TRESHOLD = 1
LOSS = arguments['states']

def mask_loss(img,batch):
  l_x = utils.match_last_channel(batch,img)
  
  img = tf.cast(img,dtype=tf.float32)

  diff = (l_x - img)**2
  diff = tf.reduce_mean(diff,axis=-1)
  
  bckdn = vut_logo_6x7_2px_padding_mask.background_mask * diff
  logo = vut_logo_6x7_2px_padding_mask.logo_mask * diff
  
  more_mask = tf.greater(logo, 0.0 * tf.ones_like(logo))
  less_indices = tf.where(more_mask)
  less_indices_cnt = less_indices.shape[0]
  if less_indices_cnt is not None:
    logo = tf.tensor_scatter_nd_update(logo,less_indices,(tf.ones(shape=(less_indices.shape[0],)) * 7) )
  return tf.reduce_mean(bckdn+logo)

tf_weights = extract_weights_as_tensors(ca)

iteration = 0
lowest_loss = sys.maxsize
loss_values = []
# extrakce parameteru do seznamu z tensorflow modelu DONE
# populace bude tvorena vahami modelu
# objektivni funkce spusti model s danymi vahami na obrazek a vrati MSE

def test_objective(*c):
    global iteration
    global lowest_loss
    global loss_values
    
    print("iteration {}/{}".format(iteration,arguments['iters']))
    #argument is a list of tensors with batch size being the size of population
    weights = [chromozone2weights([tensor[i] for tensor in c]) for i in range(arguments['pop_size'])]
    #construct NN with these parameters and return lists of population size len with the value of objective function
    nn_scores = []
    for w in weights:
        ca.set_weights(w)
        
        x = utils.init_batch(1,width,height,ca.channel_n)
        for _ in range(tf.random.uniform([], arguments['train_interval'][0], arguments['train_interval'][1], tf.int32)):
            x = ca(x)
        #print(x)
        #loss = custom_mse(x,gt_tf)
        loss = mask_loss(gt_tf,x)

        if loss.numpy() < lowest_loss:
            global CHECKPOINT_PATH
            
            lowest_loss = loss.numpy()
            print(f'new lowest loss found {lowest_loss}')
            if not RUN_NUM:
                CHECKPOINT_PATH += '+seed_'+str(arguments['seed'])
                save_path = CHECKPOINT_PATH+'/'+str(iteration)+'_'+"{:.2f}".format(loss.numpy())
            else:
                run_path = 'run_'+str(RUN_NUM)+'+seed_'+str(arguments['seed'])
                save_path = CHECKPOINT_PATH+'/'+ run_path +'/'+str(iteration)+'_'+"{:.2f}".format(loss.numpy())
            ca.save_weights(save_path)
            
            frames = []
            x = utils.init_batch(1,width,height,arguments['channels'])
            for _ in range(arguments['train_interval'][1]):
                x = ca(x)
                
                f = tf.math.floormod(x,tf.ones_like(x,dtype=tf.float32)*arguments['states'])
                f = tf.math.round(f)[0][:,:,0]
                
                f = Image.fromarray(np.uint8(f.numpy()),mode="L")
                frames.append(grayscale_to_rgb(f))
            
            make_gif(save_path,frames)
            
        nn_scores.append(loss.numpy())
    
    loss_values = loss_values + nn_scores
    nn_scores = tf.convert_to_tensor(nn_scores)
    
    iteration += 1 
    
    #plt.plot(loss_values)
    #plt.title(f'Loss function iteration {iteration}')
    #plt.xlabel('Iteration')
    #plt.ylabel('Loss value')
    #plt.savefig(f'{CHECKPOINT_PATH}/loss.png')

    return nn_scores    

print('Starting algorithm')

trained_nn = tfp.optimizer.differential_evolution_minimize(
    test_objective,
    initial_position=tf_weights,
    population_size=arguments['pop_size'],
    population_stddev=arguments['std_dev'],
    max_iterations=arguments['iters'],
    differential_weight=arguments['diff_weight'],
    crossover_prob=arguments['cross_prob'],
    seed=arguments['seed']
)

print(trained_nn.converged)
#print(trained_nn.num_objective_evaluations)
print(trained_nn.objective_value)
print('finished')
