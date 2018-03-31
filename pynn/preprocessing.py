import numpy as np


def get_batches(data_X, data_Y, batch_size):
    m = data_X.shape[1]
    batches = []
    for i in range(0, m, batch_size):
        batches.append((data_X[:, i:(i + batch_size)], data_Y[:, i:i + batch_size]))
    return batches

def one_hot(data_y, n_classes):
    return np.eye(n_classes)[data_y]