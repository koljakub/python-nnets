import numpy as np

def relu(z):
    return np.abs(z) * (z > 0)

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis = 0)

def identity(x):
    return x