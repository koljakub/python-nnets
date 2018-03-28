import numpy as np


class Initializer:

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)


class HeInitializer(Initializer):
    """
    He initialization.
    https://arxiv.org/pdf/1502.01852.pdf
    :return:
    """

    def __init__(self, seed):
        Initializer.__init__(self, seed)

    def initialize(self, shape):
        W = np.random.randn(shape[0], shape[1]) * np.sqrt(2.0 / shape[1])
        b = np.zeros((shape[0], 1))
        return (W, b)