from pynn.initializers import *
from pynn.activations import *


class Layer:

    def __init__(self, n_units, activation_f):
        self.n_units = n_units
        self.activation_f = activation_f
        self.prev_layer = None

    def connect_layer(self, layer):
        self.prev_layer = layer

    def __str__(self):
        return self.__class__.__name__ + " layer " + str(self.n_units) + "units"


class Input(Layer):

    def __init__(self, n_units):
        Layer.__init__(self, n_units, identity)

    def foo(self):
        print(self.n_units)


class Dense(Layer):

    def __init__(self, n_units, initializer=HeInitializer(42), activation_f=relu):
        Layer.__init__(self, n_units, activation_f)
        self.initializer = initializer
        self.A = None

    def initialize_parameters(self, prev_n_units):
        self.W, self.b = self.initializer.initialize((self.n_units, prev_n_units))

    def feedforward(self, A_prev):
        self.A = self.activation_f(self.W.dot(A_prev) + self.b)
        return self.A

    def backpropagate(self):
        pass


if __name__ == '__main__':
    input_layer = Input(n_units=64)
    dense1 = Dense(n_units=15, activation_f = relu)
    dense1.initialize_parameters(5)
    x = np.random.randn(5, 1)
    print(dense1.feedforward(x))