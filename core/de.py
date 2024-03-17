import tensorflow as tf
from timeit import default_timer as timer

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