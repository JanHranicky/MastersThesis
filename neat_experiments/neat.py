import neat
import argparse
import numpy as np
from PIL import Image
import multiprocessing

def parse_int_tuple(arg):
    try:
        # Assuming the input format is (x, y)
        x, y = map(int, arg.strip('()').split(','))
        return x, y
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid int tuple format. Use (x, y)")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains neural network using differential evolution')

    parser.add_argument('-c', '--config', type=str, help='Path to the config file', required=True)
    parser.add_argument('-i', '--image', type=str, help='Path to the input image', required=True)
    parser.add_argument('-s', '--states', type=int, help='Number of states', default=8)
    parser.add_argument('-t', '--train_interval', type=parse_int_tuple, help='Train interval of the network', default=(20,30))

    return parser.parse_args()

def flatten_neighborhood(image, x, y):
    """
    Flatten a 3x3 neighborhood around the pixel at (x, y) into a vector.
    """
    width, height = image.shape
    neighborhood = []

    for j in range(y - 1, y + 2):
        for i in range(x - 1, x + 2):
            # Check if pixel is within image boundaries
            if 0 <= i < width and 0 <= j < height:
                neighborhood.append(image[i, j])
            else:
                # If pixel is outside image boundaries, append 0
                neighborhood.append(1)

    return neighborhood

def get_seed_array(w, h, channels, max_dim=None):
    blank_arr = np.zeros((w, h, channels), dtype=np.float32)
    if max_dim is None:
        blank_arr[w//2, h//2, :] = 255
    else:
        blank_arr[w//2, h//2, :] = 255 % max_dim
    return blank_arr

def init_batch(n, w, h, c=16, max_dim=None):
    '''
    Returns a batch of empty arrays except for a single spot in the middle which has value of 1
    '''
    blank_arr = get_seed_array(w, h, c, max_dim)
    batch = np.expand_dims(blank_arr, 0)
    return np.repeat(batch, n, axis=0)

def img_to_discrete_np(img, states):
    array = np.sum(np.array(img, dtype=np.float32), axis=-1)
    return np.mod(array, np.ones_like(array) * states)

def process_step(nn,x):
    output = []
    width, height = x.shape
    for i in range(height):
        row = []
        for j in range(width):
            neighborhood_vector = flatten_neighborhood(x, j, i)
            #print(f"Pixel at ({j}, {i}): {neighborhood_vector}")
            row.append(nn.activate(neighborhood_vector)[0])
        output.append(row)
    
    return np.array(output)

def l2_loss(array1, array2, top_threshold=np.finfo(float).max):
    squared_diff = np.square(array1 - array2)
    l2_loss = np.sum(squared_diff)
    
    if np.isnan(l2_loss) or np.isinf(l2_loss):
        l2_loss = top_threshold
    
    return l2_loss
   
def loss_func(genome,cfg):
    #for genome_id, genome in genomes:
    nn = neat.nn.RecurrentNetwork.create(genome,cfg)
    width,height = gt_np.shape
    x = np.mod(np.sum(init_batch(1, width, height, 1)[0], axis=-1), arguments.states)
    iters = np.random.randint(arguments.train_interval[0], arguments.train_interval[1])
    for _ in range(iters):
        x = process_step(nn,x)

    #print(x) 
       
    #diff = (x - gt_np)
    #diff_cnt = np.count_nonzero(diff)        
    #genome.fitness = diff_cnt
    #return diff_cnt
    
    l2_loss_np = l2_loss(x,gt_np)
    negative = -1 * l2_loss_np
    print(f'evaluated genome_id {genome.key}, fitness: {negative}')
    return negative
        
def run_neat():
    p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1000))
    
    #winner = p.run(loss_func, 300)
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), loss_func)
    winner = p.run(pe.evaluate, 50)
    
    print(f'Printing winner:')
    print(winner)
    
def load_image(img_path):
    try:
        image = Image.open(img_path)
    except Exception as e:
        print("Error while loading the input image", e)
    
    return image

def load_config(cfg_path):
    try:
        cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         cfg_path)
    except Exception as e:
        print("Error while loading neat config", e)
    
    return cfg

#store arguments, image and config as global variables in order to acess them in loss function
#found no better way to do this :/
arguments = parse_arguments()
image = load_image(arguments.image)
gt_np = img_to_discrete_np(image,arguments.states)

config = load_config(arguments.config)

if __name__ == '__main__':
    run_neat()
    
    