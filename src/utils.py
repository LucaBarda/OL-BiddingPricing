import numpy as np

def set_seed(seed):
    np.random.seed(seed)
    return

def generate_adv_sequence(len, min, max):
    return_array = np.zeros(len)
    for i in range(len):
        return_array[i] = np.random.uniform(min, max)
    return return_array

def normalize_zero_one(x, min_x, max_x):
    return (x - min_x) / (max_x - min_x)

def denormalize_zero_one(x, min_x, max_x):
    return min_x + (max_x - min_x) * x