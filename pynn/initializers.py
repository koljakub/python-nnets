import numpy as np

def initialize_he(layers):
    """
    He initialization.
    https://arxiv.org/pdf/1502.01852.pdf
    :return:
    """
    weights = []
    biases = []
    for i in range(1, len(layers)):
        weights.append(np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2.0 / layers[i - 1]))
        biases.append(np.zeros((layers[i], 1)))
    return (weights, biases)