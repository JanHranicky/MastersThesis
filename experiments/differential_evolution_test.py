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

GT_IMG_PATH = './img/vut_logo_17x17_2px_padding.png'
date_time = datetime.now().strftime("%m_%d_%Y")
gt_img = Image.open(GT_IMG_PATH)

CHANNEL_NUM = 3
STATES = 8
STEPS = (20,30)

POP_SIZE = 40
SEED = random.randint(0,sys.maxsize)

COMPARE_CHANNELS = 1
MAX_ITERS = 10000

height,width = gt_img.size
gt_tf = utils.img_to_discrete_space_tf(gt_img,STATES,COMPARE_CHANNELS)
ca = output_modulo_model.CA(channel_n=CHANNEL_NUM,model_name=date_time+'_modulo_'+os.path.basename(__file__).split('.')[0]+'_'+str(STATES)+"_states_"+str(CHANNEL_NUM)+"_layers_"+str(STEPS[0])+"_"+str(STEPS[1])+"_steps",states=STATES)
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

def custom_mse(x, gt):
    l_x = utils.match_last_channel(x,gt)
    return tf.reduce_mean(tf.square(l_x - gt))

tf_weights = extract_weights_as_tensors(ca)

iteration = 0
lowest_loss = sys.maxsize
# extrakce parameteru do seznamu z tensorflow modelu DONE
# populace bude tvorena vahami modelu
# objektivni funkce spusti model s danymi vahami na obrazek a vrati MSE

def test_objective(*c):
    global iteration
    global lowest_loss
    print(f'itearation {iteration}/{MAX_ITERS}')
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
        nn_scores.append(loss.numpy())
    
    nn_scores = tf.convert_to_tensor(nn_scores)
    #print(nn_scores)
    
    iteration += 1 
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
