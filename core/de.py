import tensorflow as tf
from timeit import default_timer as timer
import random 
import numpy as np

def flatten_tensor(tl):
    """Flattens tensor list into a single tensor. Returns list of original shapes for the purpose of transforming the tensors back

    Args:
        tl ([tf.Tensor]): list of tensors to flatten and join

    Returns:
        tf.Tensor: flattened and joined tensor
        [tf.Shape]: list of original tensor shapes
    """
    flat_t = tf.concat([tf.reshape(tensor, shape=(-1,)) for tensor in tl], axis=0)
    shapes = [t.shape for t in tl]
    return flat_t,shapes

def unflatten_tensor(t,shapes):
    """Transforms single tensor into smaller tensors.
    Intented to transform tensors transformed by flatten_tensor() back to their original shape

    Args:
        t (tf.Tensor): tensor to transform
        shapes ([tf.Shape]): list of shapes of the output tensors

    Returns:
        [tf.Tensor]: list of tensors shaped using the shapes argument
    """
    unflattened_tensors = []
    current_index = 0

    for s in shapes:
        size = tf.reduce_prod(s)
        unflattened_tensor = tf.reshape(t[current_index:current_index+size], shape=s).numpy()
        unflattened_tensors.append(unflattened_tensor)
        current_index += size

    return unflattened_tensors

def extract_weights_as_tensors(model):
    """Extracts weights of tensorflow model. Parses them into tensors and returns them in a list. 
    Used to represent model weights as a chromozone

    Args:
        model (tf.keras.Model): model to extract weights

    Returns:
        List(tf.Tensor): An array of tensors representing the weights of the model.
    """
    return [tf.convert_to_tensor(var) for var in model.get_weights()]



def generate_pop(original_tensor_list, N, stddev):
    """
    Add noise to each element of the given list of TensorFlow tensors N times.

    Parameters:
    - original_tensor_list: List of TensorFlow tensors
    - N: Integer, number of noisy tensors to generate for each original tensor

    Returns:
    - List of lists of N TensorFlow tensors with added noise
    """

    def add_noise_to_element(element):
        noise = tf.random.normal(shape=tf.shape(element), mean=0.0, stddev=stddev, dtype=tf.float32)
        return element + noise
    start = timer()
    pop = []
    for _ in range(N):
            pop.append(add_noise_to_element(original_tensor_list))    
    end = timer()
    print(f'generate_pop() execution took {end - start}s')
    return pop


def generate_unique_indices(num_tensors):
    """Given the len of population and the index of current tensor returns three different indexees that will be used in the mixing algorithm

    Args:
        num_tensors (Int): size of the population
        provided_index (Int): selected index

    Raises:
        ValueError: population size must be at least 4

    Returns:
        [Int]: list of selected indeces
    """
    # Check if num_tensors is greater than or equal to 4
    if num_tensors < 4:
        raise ValueError("Number of tensors should be at least 4 to generate three unique indices.")

    indices = []
    for i in range(num_tensors):
        # Generate a list of all possible indices
        all_indices = list(range(num_tensors))

        # Remove the provided index from the list
        all_indices.remove(i)

        # Randomly shuffle the remaining indices
        shuffled_indices = tf.random.shuffle(all_indices)

        # Take the first three indices from the shuffled list
        indices.append(shuffled_indices[:3].numpy())

    return indices

def mix_population(population, indices,F):
    """Generates mixed population of the DE algorithm 
        for each individual generates a mutant using formula
        
        vi,j:=xr1,j+F(xr2,j-xr3,j) 

        where xr1,xr2 and xr3 are randomly selected individuals in the population
    Args:
        population ([tf.Tensor]): population of the DE algorithhm
        indices ([[Int]]): list of trouples that index the indiviudals chosen for the mutation
        F (float): control parameter of the algoritm. Control how strong is the mutation

    Returns:
        [tf.Tensor]: list of newly mutated tensors
    """
    mixed_pop = []
    
    for i in indices:
        x1, x2, x3 = [population[index] for index in i]
        v = x1 + F * (x2-x3)
        
        mixed_pop.append(v)
        
    return mixed_pop    


def cross_over_pop(pop,mutated_pop, CR):
    if len(pop) != len(mutated_pop):
        raise ValueError("Both input lists should have the same length.")

    mixed_tensors = []

    for tensor1, tensor2 in zip(pop, mutated_pop):
        # Generate random values in the range [0, 1)
        random_values = tf.random.uniform(shape=tensor1.shape, minval=0.0, maxval=1.0)
        # Use a threshold of 0.5 to choose elements from each tensor
        mixed_tensor = tf.where(random_values < CR, tensor1, tensor2)
        mixed_tensors.append(mixed_tensor)

    return mixed_tensors


def make_new_pop(pop, pop_ratings, crossed_pop, crossed_pop_rating):
    new_pop = []
    new_pop_rating = []
    
    for r1, r2, i1, i2 in zip(pop_ratings, crossed_pop_rating, pop, crossed_pop):
        r1_better = r1 < r2
        mixed_tensor = tf.where(r1_better, i1, i2)
        
        new_pop.append(mixed_tensor)
        new_pop_rating.append(r1 if r1_better else r2)
    
    return new_pop, new_pop_rating


def make_new_pop_with_sucess_rate(pop, pop_ratings, crossed_pop, crossed_pop_rating):
    new_pop = []
    new_pop_rating = []
    
    pop_len = len(crossed_pop)
    mutation_sucess_rate = 0
    
    for r1, r2, i1, i2 in zip(pop_ratings, crossed_pop_rating, pop, crossed_pop):
        r1_better = r1 < r2
        if not r1_better:
            mutation_sucess_rate += 1
        mixed_tensor = tf.where(r1_better, i1, i2)
        
        new_pop.append(mixed_tensor)
        new_pop_rating.append(r1 if r1_better else r2)
    
    return new_pop, new_pop_rating, mutation_sucess_rate/pop_len


class Archive:
    def __init__(self, length):
        self.length = length
        self.data = [(0.5,0.5)] * length
        self.current_index = 0

    def replace_none(self,old_value,new_value):
        if new_value is None:
            return old_value
        return new_value

    def add(self, value):
        old_f,old_cr = self.data[self.current_index]
        new_f,new_cr = value
            
        new_f = self.replace_none(old_f,new_f)
        new_cr = self.replace_none(old_cr,new_cr)
        
        self.data[self.current_index] = (new_f, new_cr)
        self.current_index = (self.current_index + 1) % self.length

    def get(self,index):
        if index > self.length - 1:
            raise "Index out of range"
        return self.data[index]
    
    def get_random(self):
        i = random.randint(0,self.length-1)
        return self.data[i]

    def __str__(self):
        str_data = ""
        for i in range(self.length):
            data = self.get(i)
            str_data += str(self.get(i)) + ", "
        
        return str_data

def generate_control_parameters(pop_size,archive):
    c_parameter_list = []
    
    for i in range(pop_size):
        m_f, m_cr = archive.get_random()
        g_cr = 0.1 * np.random.randn() + m_cr
        g_cr = max(0,min(1,g_cr))
        
        
        g_f = 0.1 * np.random.standard_cauchy() + m_f
        while g_f <= 0: #in case negative value was generated 
            g_f = 0.1 * np.random.standard_cauchy() + m_f
        g_f = min(1,g_f) #truncate to be maximum of 1 
        
        c_parameter_list.append((g_f,g_cr))
        
    return c_parameter_list


def top_n_indices(fitness_values, n):
    """
    Returns the indices of the top N best individuals based on fitness values.
    
    Parameters:
        fitness_values (numpy.ndarray): Array containing fitness values of individuals.
        n (int): Number of top individuals to select.
        
    Returns:
        numpy.ndarray: Array of indices corresponding to the top N best individuals.
    """
    sorted_indices = np.argsort(fitness_values)
    top_n = sorted_indices[:n]  # Select top N indices
    return top_n

def generate_top_n_indiduals(fitness_values):
    """Chooses best individuals for the current-to-pbest mutation operator

    Args:
        fitness_values ([Number]): list of fitness values

    Returns:
        [Int]: index of chosed individuals for mutation operator
    """
    P_MAX = 0.2
    P_MIN = 2 / len(fitness_values)
    
    pop_size = len(fitness_values)
    num_min = int(pop_size * P_MIN)
    num_max = int(pop_size * P_MAX)
    
    fitness_sorted_pop = np.argsort(fitness_values)
    
    best_individuals = []
    for _ in range(pop_size):
        p_best = int(random.uniform(num_min,num_max))
        best_individuals.append(random.choice(fitness_sorted_pop[:p_best]))
        
    return best_individuals

def binomial_crossover(trial,donor, cr_rate):
    random_values = tf.random.uniform(shape=donor.shape, minval=0.0, maxval=1.0)
    D_tensor = tf.tensor_scatter_nd_update(tf.cast(tf.zeros_like(donor),dtype=tf.bool), [[random.randint(0,donor.shape[0]-1)]], [True]) #tensor with all False values except single random position being True
    cr_mask = tf.cast(tf.logical_or(random_values <= cr_rate, D_tensor), tf.float32) #mask combining single random position and CR mask

    crossed_trial_vector = (trial * cr_mask) + (donor * (1 - cr_mask)) #take elements from trial vector v_i where mask is 1.0 and from donor vector x1 where mask is 0.0
    
    return crossed_trial_vector

def interm_crossover(trial,donor):
    a = tf.random.uniform(shape=donor.shape, minval=0.0, maxval=1.0)

    return a*trial + (1-a)*donor

def current_to_pbest_mutation(population, indices, parameters, p_best, cross_operator):
    trial_vectors = []
    for i,p,pb in zip(indices,parameters,p_best):
        x1, x2, x3 = [population[index] for index in i]
        f_i = float(p[0])
        x_p = population[pb]
        
        v_i = x1 + f_i * (x_p - x1) + f_i * (x2 - x3)
        
        if cross_operator == "interm":
            crossed_trial_vector = interm_crossover(v_i,x1)
        else:
            crossed_trial_vector = binomial_crossover(v_i,x1,p[1])
        
        trial_vectors.append(crossed_trial_vector)
        
    return trial_vectors


def shade_new_pop(pop, pop_ratings, crossed_pop, crossed_pop_rating):
    new_pop = []
    new_pop_rating = []
    better_mutants = []
    for i, r1, r2, i1, i2 in zip(range(len(pop)),pop_ratings, crossed_pop_rating, pop, crossed_pop):
        r1_better = r1 < r2
        
        new_pop.append(i1 if r1_better else i2)
        new_pop_rating.append(r1 if r1_better else r2)
        if not r1_better:
            better_mutants.append(i)
    
    return new_pop, new_pop_rating, better_mutants
    
def mean_wl_f(old_fitness,new_fitness,better_mutant_indices,c_parameters):
    fitness_diff_arr = [tf.math.abs(old_fitness[bm] - new_fitness[bm]) for bm in better_mutant_indices]
    weight_sum = tf.reduce_sum(fitness_diff_arr)
    if weight_sum == 0: #0 in case only one mutant was better and had exactly the same fitness as its parent
        return None
    
    weights = [f/weight_sum for f in fitness_diff_arr]
    top = [w*c_parameters[i][0]**2 for (w,i) in zip(weights,better_mutant_indices)]
    bot = [w*c_parameters[i][0] for (w,i) in zip(weights,better_mutant_indices)]
    
    return tf.reduce_sum(top)/tf.reduce_sum(bot)
    
def mean_wa_cr(old_fitness,new_fitness,better_mutant_indices,c_parameters):
    fitness_diff_arr = [tf.math.abs(old_fitness[bm] - new_fitness[bm]) for bm in better_mutant_indices]
    weight_sum = tf.reduce_sum(fitness_diff_arr)
    if weight_sum == 0: #0 in case only one mutant was better and had exactly the same fitness as its parent
        return None
    
    weights = [f/weight_sum for f in fitness_diff_arr]
    
    products = [w*c_parameters[i][0]**2 for (w,i) in zip(weights,better_mutant_indices)]
    
    return tf.reduce_sum(products)
    