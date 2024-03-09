from core import model,utils,trainer,output_modulo_model, de
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
from timeit import default_timer as timer
import ray

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
    parser.add_argument('-x', '--cross_prob', type=float, help='The probability of recombination per site', default=0.5)
    parser.add_argument('-g', '--seed', type=int, help='Seed for generator', default=random.randint(0,sys.maxsize))
    parser.add_argument('-m', '--image', type=str, help='Path to GT image', default='./img/vut_logo_17x17_2px_padding.png')
    parser.add_argument('-r', '--run', type=int, help='Number of the run. If provided results will be stored in a subfolder', default=None)
    parser.add_argument('-f', '--folder', type=str, help='Folder in which the reults will be stored', default='./checkpoints/DE/')

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
        'run': args.run,
        'folder': args.folder,
    }

    return parameters

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


def categorical_crossentropy_loss(y_true, y_pred):
    # Ensure the shapes are compatible
    assert y_true.shape == y_pred.shape, "Shape mismatch between y_true and y_pred"

    # Reshape y_true and y_pred if needed
    if len(y_true.shape) == 4:  # If the batch dimension is present
        y_true = tf.squeeze(y_true, axis=[3])  # Remove the last axis (channel)
        y_pred = tf.squeeze(y_pred, axis=[3])

    # Compute categorical crossentropy loss
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    return loss

def cnt_loss(img,batch):
  l_x = utils.match_last_channel(batch,img)
  
  img = tf.cast(img,dtype=tf.float32)

  diff = (l_x - img)
  diff_cnt = tf.math.count_nonzero(diff)

  return diff_cnt


@tf.function
def evaluate_individual(gt_tf, ca, width, height, channel_n, train_interval):
    x = utils.init_batch(1, width, height, channel_n)
    total_iterations = tf.random.uniform([], train_interval[0], train_interval[1], tf.int32)
    
    loss = tf.constant(0, dtype=tf.int64)
    for i in range(total_iterations):
        x = ca(x)
        if i == total_iterations - 2:
            loss = cnt_loss(gt_tf, x)
    
    loss += cnt_loss(gt_tf, x)
    return loss

def process_individual(gt_tf, ca, w, width, height, channel_n, train_interval):
    ca.set_weights(w)
    return evaluate_individual(gt_tf, ca, width, height, channel_n, train_interval)


def split_array(arr, num_processes):
    # Calculate the number of elements each process should handle
    elements_per_process = len(arr) // num_processes
    remainder = len(arr) % num_processes

    # Split the array into subarrays for each process
    subarrays = []
    start_idx = 0

    for i in range(num_processes):
        end_idx = start_idx + elements_per_process + (1 if i < remainder else 0)
        subarrays.append(arr[start_idx:end_idx])
        start_idx = end_idx

    return subarrays

def objective_func(c, evaluators):
    start = timer()
    weights = [de.unflatten_tensor(i,shapes) for i in c]
        
    splitted_weights = split_array(weights,len(evaluators))
    results = ray.get([e.evaluate_pop_part.remote(w) for (w,e) in zip(splitted_weights,evaluators)])
    
    combined = [item for sublist in results for item in sublist]
    #nn_scores = tf.convert_to_tensor(nn_scores)
    
    end = timer()
    print(f'objective_func() exection took {end-start}s')
    return tf.convert_to_tensor(combined)    

@ray.remote
class CA_Evaluator(object):
    def __init__(self, channels, model_name, states):
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        self.model = output_modulo_model.CA(channel_n=channels,model_name=model_name,states=states)

    def set_weights(self,weights):
        self.model.set_weights(weights)

    def evaluate_pop_part(self,weights):
        nn_scores = []
        for w in weights:
            self.set_weights(w)
            nn_scores.append(evaluate_individual(gt_tf, self.model, width, height, self.model.channel_n, arguments['train_interval']))
        
        return nn_scores


if __name__ == '__main__':
    arguments = parse_arguments()
    print(f'Started Script with arguments: {arguments}')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    init_ray_start = timer()
    ray.init(num_gpus=1)
    init_ray_end = timer()
    print(f'Initializing Ray took {init_ray_end - init_ray_start}')

    GT_IMG_PATH = arguments['image']
    date_time = datetime.now().strftime("%m_%d_%Y")
    gt_img = Image.open(GT_IMG_PATH)

    COMPARE_CHANNELS = 1

    height,width = gt_img.size
    gt_tf = utils.img_to_discrete_space_tf(gt_img,arguments['states'],COMPARE_CHANNELS)
    model_name = "{}+{}+{}+channels_{}+iters_{}+states_{}+train_interval_{}+std_dev_{}+pop_size_{}+diff_weight_{}+cross_prob_{}".format(
        date_time,
        "de_cnt_loss",
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
    
    NUM_PROCESSES = 4
    
    copying_model_start = timer()
    ca_evaluator_arr = [CA_Evaluator.remote(arguments['channels'], model_name, arguments['states']) for _ in  range(NUM_PROCESSES)]
    copying_model_end = timer()

    print(f'Creating models took {copying_model_end-copying_model_start}s')
    
    TRESHOLD = 1
    LOSS = arguments['states']
    CHECKPOINT_PATH = arguments['folder']+ca.model_name
    RUN_NUM = arguments['run']
    F = arguments['diff_weight']
    CR = arguments['cross_prob']
    
    
    lowest_loss = sys.maxsize
    tf_weights = de.extract_weights_as_tensors(ca)
    
    print('Starting algorithm')

    og_weights = ca.get_weights()
    flat,shapes = de.flatten_tensor(tf_weights)

    old_pop = de.generate_pop(flat,arguments['pop_size'], arguments['std_dev'])
    old_pop_rating = objective_func(old_pop,ca_evaluator_arr)

    min_losses = []

    for i in range(arguments['iters']):
        iter_start = timer()
        
        r = random.uniform(0, 1)
        
        indices = de.generate_unique_indices(arguments['pop_size'])
        mixed_pop = de.mix_population(old_pop,indices,F)
        
        crossed_pop = de.cross_over_pop(old_pop,mixed_pop,CR)
        crossed_pop_rating = objective_func(crossed_pop,ca_evaluator_arr)
        
        old_pop, old_pop_rating = de.make_new_pop(old_pop,old_pop_rating,crossed_pop,crossed_pop_rating)

        rating_list = [r.numpy() for r in old_pop_rating]
        min_value = min(rating_list)
        min_losses.append(min_value)

        if min_value < lowest_loss:
            lowest_loss = min_value
            print(f'new lowest loss found {lowest_loss}')
            if not RUN_NUM:
                path = CHECKPOINT_PATH + '+seed_'+str(arguments['seed'])
                save_path = path
            else:
                run_path = 'run_'+str(RUN_NUM)+'+seed_'+str(arguments['seed'])
                save_path = CHECKPOINT_PATH+'/'+ run_path
            weight_save_format = str(i)+'_'+"{:.2f}".format(min_value)
            
            ca.set_weights(de.unflatten_tensor(old_pop[rating_list.index(min_value)],shapes))
            ca.save_weights(save_path+'/'+weight_save_format)
            
            np_min_losses = np.array(min_losses)
            np.save(save_path+'/convergence_arr.npy', np_min_losses)

            frames = []
            x = utils.init_batch(1,width,height,arguments['channels'])
            for _ in range(arguments['train_interval'][1]):
                x = ca(x)

                f = tf.math.floormod(x,tf.ones_like(x,dtype=tf.float32)*arguments['states'])
                f = tf.math.round(f)[0][:,:,0]

                f = Image.fromarray(np.uint8(f.numpy()),mode="L")
                frames.append(grayscale_to_rgb(f))

            make_gif(save_path+'/'+weight_save_format,frames)
        
        iter_end = timer()
        print(f'iteration execution took {iter_end-iter_start}s') 
        print('Iteration {}/{}. Lowest loss: {}. Current pop lowest loss {}. F={}, CR={}'.format(i,arguments['iters'],lowest_loss,min_value,F,CR))
