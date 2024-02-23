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

def parse_parameters():
    # Check if at least one command-line argument is provided
    if len(sys.argv) < 3:
        print("Please provide a parameters as a command-line argument.")
        exit()

    # Get the command-line argument (assuming it's the first one after the script name)
    parameter_str_channels = sys.argv[1]
    parameter_str_iterations = sys.argv[2]

    try:
        # Convert the parameter to an integer
        parameter_int = int(parameter_str_channels)

        # Save the integer or use it as needed
        print("Integer parameter:", parameter_int)
        print("Integer parameter:", int(parameter_str_iterations))

        return parameter_int, int(parameter_str_iterations)

    except ValueError:
        print("Error: The provided parameter is not a valid integer.")

GT_IMG_PATH = './img/vut_logo_17x17_2px_padding.png'
date_time = datetime.now().strftime("%m_%d_%Y")
gt_img = Image.open(GT_IMG_PATH)

CHANNEL_NUM, MAX_ITERS = parse_parameters()
STATES = 8
STEPS = (20,30)

POP_SIZE = 40
SEED = random.randint(0,sys.maxsize)

COMPARE_CHANNELS = 1

height,width = gt_img.size
gt_tf = utils.img_to_discrete_space_tf(gt_img,STATES,COMPARE_CHANNELS)
ca = output_modulo_model.CA(channel_n=CHANNEL_NUM,model_name=date_time+'_modulo_'+os.path.basename(__file__).split('.')[0]+'_'+str(STATES)+"_states_"+str(CHANNEL_NUM)+"_layers_"+str(STEPS[0])+"_"+str(STEPS[1])+"_steps",states=STATES)
CHECKPOINT_PATH = f'./checkpoints/DE/'+ca.model_name
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
                     for _ in range(STATES+1)]
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
    
    print(f'iteration {iteration}/{MAX_ITERS}')
    #argument is a list of tensors with batch size being the size of population
    weights = [chromozone2weights([tensor[i] for tensor in c]) for i in range(POP_SIZE)]
    #construct NN with these parameters and return lists of population size len with the value of objective function
    nn_scores = []
    for w in weights:
        ca.set_weights(w)
        
        x = utils.init_batch(1,width,height,ca.channel_n)
        for _ in range(tf.random.uniform([], STEPS[0], STEPS[1], tf.int32)):
            x = ca(x)
        #print(x)
        loss = custom_mse(x,gt_tf)
        if loss.numpy() < lowest_loss:
            lowest_loss = loss.numpy()
            print(f'new lowest loss found {lowest_loss}')
            ca.save_weights(CHECKPOINT_PATH+'/'+str(iteration))
            
            frames = []
            x = utils.init_batch(1,width,height,CHANNEL_NUM)
            for _ in range(STEPS[1]):
                x = ca(x)
                
                f = tf.math.floormod(x,tf.ones_like(x,dtype=tf.float32)*STATES)
                f = tf.math.round(f)[0][:,:,0]
                
                f = Image.fromarray(np.uint8(f.numpy()),mode="L")
                frames.append(grayscale_to_rgb(f))
            
            make_gif(str(CHECKPOINT_PATH)+'/'+str(iteration),frames)
            
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
    population_size=POP_SIZE,
    population_stddev=0.02,
    max_iterations=MAX_ITERS,
    seed=SEED)

print(trained_nn.converged)
#print(trained_nn.num_objective_evaluations)
print(trained_nn.objective_value)
print('finished')
