import numpy as np

def set_seed(seed):
    np.random.seed(seed)
    return

def generate_adv_conv_prob_sequence(len_seq):
    return_array = np.zeros(len_seq)
    alpha = 1
    for i in range(len_seq):
        beta = np.random.uniform(2, 8) #adversarial
        conversion_prob = np.random.beta(alpha, beta)
        # reward = conversion_prob * num_buyers * (price - cost_per_good)    
        return_array[i] = conversion_prob
    
    return return_array