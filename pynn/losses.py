import numpy as np

def negative_log_loss(y_hat, y):
    return -np.sum(y * np.log(y_hat + np.finfo(float).eps), axis=0)