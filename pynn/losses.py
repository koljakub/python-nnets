import numpy as np


class LossFunction:

    def __init__(self, name):
        if (name in globals()):
            self.name = name
            self.callable_f = get_function(name)
        else:
            raise ValueError("Loss function {} not defined!".format(name))

    def __call__(self, y_hat, y):
        return self.callable_f(y_hat, y)

    def get_initial_delta(self, activation_f):
        """
        The function computes initial delta value for the Backpropagation algorithm.

        Let L be some loss function. Let Z be a preactivation of some activation function g, such
        that A = g(Z). Formally, the initial delta value can be expressed as a partial derivative
        using the chain rule as follows: dL / dg * dg / dZ.

        The initial delta is abstracted into a lambda function which can be applied to the concrete
        inputs of the Backpropagation algorithm - the vector (matrix) of predictions and the vector
        (matrix) of true values.
        :param loss:
        :param activation_f:
        :return:
        """
        if (self.name == "negative_log_loss" and (activation_f.name == "softmax" or activation_f.name == "sigmoid")):
            return lambda y_hat, y: y_hat - y


def get_function(func_name):
    return globals()[func_name]


def negative_log_loss(y_hat, y):
    return -np.sum(y * np.log(y_hat + np.finfo(float).eps), axis=0)