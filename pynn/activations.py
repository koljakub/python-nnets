import numpy as np


class ActivationFunction:

    def __init__(self, name):
        if (name in globals()):
            self.name = name
            self.callable_f = get_function(name)
        else:
            raise ValueError("Activation function {} not defined!".format(name))

    def __call__(self, x):
        return self.callable_f(x)


class ActivationFunctionDerivative:

    def __init__(self, function_name):
        if (function_name in globals()):
            self.name = "d_" + function_name
            self.callable_d_f = get_derivative(function_name)
        else:
            raise ValueError("Activation function {} not defined!".format(function_name))

    def __call__(self, x):
        return self.callable_d_f(x)


def get_function(func_name):
    return globals()[func_name]


def get_derivative(func_name):
    return globals()["d_" + func_name]


def relu(z):
    return np.abs(z) * (z > 0)


def d_relu(z):
    return np.ones_like(z)


def softmax(z):
    z = np.exp(z - np.max(z, axis=0))
    return z / np.sum(z, axis=0)


def d_softmax(z):
    raise NotImplementedError()


def identity(z):
    return z


def d_identity(z):
    return np.ones_like(z)
