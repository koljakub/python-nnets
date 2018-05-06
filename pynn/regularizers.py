import abc
import numpy as np
from pynn.layers import *


class Regularizer(object):

    def __init__(self, rate, name):
        self.rate = rate
        self.name = name

    @abc.abstractmethod
    def regularize(self, layer):
        return

    def __str__(self):
        return self.name


class L1Regularizer(Regularizer):

    def __init__(self, rate):
        Regularizer.__init__(self, rate, "L1 Regularizer")

    def regularize(self, layer):
        layer.d_W = layer.d_W + self.rate * np.sign(layer.W)


class L2Regularizer(Regularizer):

    def __init__(self, rate):
        Regularizer.__init__(self, rate, "L2 Regularizer")

    def regularize(self, layer):
        layer.d_W = layer.d_W + self.rate * layer.W
