from core import utils,output_modulo_model, de
import tensorflow as tf
import numpy as np
from PIL import Image
import numpy as np 
from datetime import datetime
import random
import sys 
import argparse
from timeit import default_timer as timer

def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains neural network using differential evolution')

    # Add arguments
    parser.add_argument('-c', '--channels', type=int, help='Number of channels of the model', default=1)
    parser.add_argument('-i', '--iters', type=int, help='Maximum number of iterations', default=1000)
    parser.add_argument('-s', '--states', type=int, help='Size of the state space.', default=8)
    parser.add_argument('-t', '--train_interval', type=utils.parse_int_tuple, help='Train interval of the network', default=(20,30))
    parser.add_argument('-d', '--std_dev', type=float, help='Stddev used to generate initial population', default=0.02)
    parser.add_argument('-p', '--pop_size', type=int, help='size of the population', default=40)
    parser.add_argument('-x', '--cross_operator', type=str, help='Chosen crossoveroperator', default="binomial")
    parser.add_argument('-a', '--image', type=str, help='Path to GT image', default='./img/vut_logo_17x17_2px_padding.png')
    parser.add_argument('-r', '--run', type=int, help='Number of the run. If provided results will be stored in a subfolder', default=None)
    parser.add_argument('-f', '--folder', type=str, help='Folder in which the reults will be stored', default='./checkpoints/DE/')
    parser.add_argument('-m', '--archive_len', type=int, help='Size of the archive, which stores the historical control parameters', default=10)
    parser.add_argument('-e', '--repetion_per_run', type=int, help='Number of times the algorithm will run in this program', default=1)

    return parser.parse_args()

@tf.function
def evaluate_individual(gt_tf, ca, width, height, channel_n, train_interval):
    x = utils.init_batch(1, width, height, channel_n)
    total_iterations = tf.random.uniform([], train_interval[0], train_interval[1], tf.int32)
    
    loss = tf.constant(0, dtype=tf.int64)
    for _ in range(total_iterations):
        x = ca(x)

    loss = utils.custom_l2(x,gt_tf)
    return loss

def objective_func(c):
    start = timer()
    weights = [de.unflatten_tensor(i,shapes) for i in c]
    
    nn_scores = []
    for w in weights:
        ca.set_weights(w)
        nn_scores.append(evaluate_individual(gt_tf, ca, width, height, ca.channel_n, arguments.train_interval))
    
    nn_scores = tf.convert_to_tensor(nn_scores)
    
    end = timer()
    
    print(f'objective_func() exection took {end-start}s')
    return nn_scores    


def save_best_solution():
    if not RUN_NUM:
        save_path = CHECKPOINT_PATH
    else:
        run = (j * 8000) + RUN_NUM
        run_path = 'run_'+str(run)
        save_path = CHECKPOINT_PATH+'/'+ run_path
    weight_save_format = str(lowest_loss_iter)+'_'+"{:.2f}".format(lowest_loss)

    ca.set_weights(de.unflatten_tensor(lowest_loss_individual,shapes))
    ca.save_weights(save_path+'/'+weight_save_format)

    np_min_losses = np.array(min_losses)
    np.save(save_path+'/convergence_arr.npy', np_min_losses)
    
    frames = []
    c_dict = utils.extract_color_dict(gt_img,gt_tf)
    x = utils.init_batch(1,width,height,arguments.channels)
    for _ in range(arguments.train_interval[1]):
        x = ca(x)
        f = tf.math.floormod(x,tf.ones_like(x,dtype=tf.float32)*arguments.states)
        f = tf.math.round(f)[0][:,:,0]
        f = Image.fromarray(np.uint8(f.numpy()),mode="L")
        frames.append(utils.grayscale_to_rgb(f,c_dict))
    utils.make_gif(save_path+'/'+weight_save_format,frames)

arguments = parse_arguments()
print("Printing arugments:")
print(arguments)

GT_IMG_PATH = arguments.image
date_time = datetime.now().strftime("%m_%d_%Y")
gt_img = Image.open(GT_IMG_PATH)

height,width = gt_img.size
gt_tf = utils.img_to_discrete_tensor(gt_img.convert("RGB"),arguments.states)
model_name = "{}+{}+{}+channels_{}+iters_{}+states_{}+train_interval_{}+pop_size_{}".format(
    date_time,
    "shade",
    GT_IMG_PATH.split('/')[-1].split('.')[0], #gt_img name
    arguments.channels,
    arguments.iters,
    arguments.states,
    arguments.train_interval,
    arguments.pop_size,
)

ca = output_modulo_model.CA(channel_n=arguments.channels,model_name=model_name,states=arguments.states)
CHECKPOINT_PATH = arguments.folder+ca.model_name
RUN_NUM = arguments.run

lowest_loss = sys.maxsize
lowest_loss_iter = None
lowest_loss_individual = None

print('Starting algorithm')

tf_weights = de.extract_weights_as_tensors(ca)
flat,shapes = de.flatten_tensor(tf_weights)

old_pop = de.generate_pop(flat,arguments.pop_size, arguments.std_dev)
old_pop_rating = objective_func(old_pop)

min_losses = []
archive = de.Archive(arguments.archive_len)

for j in range(arguments.repetion_per_run):
    for i in range(arguments.iters):
        iter_start = timer()
        
        s_cr = None 
        s_f = None
        
        c_parameters = de.generate_control_parameters(arguments.pop_size, archive)
        p_best_individuals = de.generate_top_n_indiduals(old_pop_rating)
        indices = de.generate_unique_indices(arguments.pop_size)
        
        mixed_pop = de.current_to_pbest_mutation(old_pop,indices,c_parameters,p_best_individuals,arguments.cross_operator)
        mixed_pop_rating = objective_func(mixed_pop)
        
        new_pop, new_pop_rating, better_mutants = de.shade_new_pop(old_pop,old_pop_rating,mixed_pop,mixed_pop_rating)
        
        if better_mutants:
            new_f = de.mean_wl_f(old_pop_rating,mixed_pop_rating,better_mutants,c_parameters)
            new_cr = de.mean_wa_cr(old_pop_rating,mixed_pop_rating,better_mutants,c_parameters)
            
            archive.add((new_f,new_cr))

        #set the new population
        old_pop = new_pop
        old_pop_rating = new_pop_rating
            
        rating_list = [r.numpy() for r in old_pop_rating]
        min_value = min(rating_list)
        min_losses.append(min_value)
        
        if min_value < lowest_loss:
            lowest_loss = min_value
            lowest_loss_iter = i
            lowest_loss_individual = old_pop[rating_list.index(min_value)]
            print(f'new lowest loss found {lowest_loss}')
        
        iter_end = timer()
        print(f'iteration execution took {iter_end-iter_start}s') 
        print('Iteration {}/{}. Lowest loss: {}. Current pop lowest loss {}. Archive sttus={}'.format(i,arguments.iters,lowest_loss,min_value,str(archive)))
    
    save_best_solution()
