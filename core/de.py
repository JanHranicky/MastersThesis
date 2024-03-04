import tensorflow as tf

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



def generate_pop(original_tensor_list, N):
    """
    Add noise to each element of the given list of TensorFlow tensors N times.

    Parameters:
    - original_tensor_list: List of TensorFlow tensors
    - N: Integer, number of noisy tensors to generate for each original tensor

    Returns:
    - List of lists of N TensorFlow tensors with added noise
    """

    def add_noise_to_element(element):
        noise = tf.random.normal(shape=tf.shape(element), mean=0.0, stddev=0.1, dtype=tf.float32)
        return element + noise
    pop = []
    for _ in range(N):
            pop.append(add_noise_to_element(original_tensor_list))    
    return pop