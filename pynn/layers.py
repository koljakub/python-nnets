import abc
from pynn.initializers import *
from pynn.activations import *


class Layer:

    def __init__(self, n_units, activation_f):
        self.n_units = n_units
        self.activation_f = ActivationFunction(activation_f)
        self.d_activation_f = ActivationFunctionDerivative(activation_f)
        self.prev_layer = None
        self.next_layer = None
        self.A = None
        self.W = None
        self.b = None
        self.d_A = None
        self.d_W = None
        self.d_b = None
        self.delta = None

    def connect_previous_layer(self, prev_layer):
        self.prev_layer = prev_layer

    def connect_next_layer(self, next_layer):
        self.next_layer = next_layer

    @abc.abstractmethod
    def forward(self, A_prev):
        return

    @abc.abstractmethod
    def backward(self):
        return

    def __str__(self):
        return self.__class__.__name__ + " layer ({}) | {} units".format(self.activation_f.name, self.n_units)


class Input(Layer):

    def __init__(self, n_units):
        Layer.__init__(self, n_units, "identity")

    def forward(self, X):
        self.A = self.activation_f(X)

    def backward(self):
        raise NotImplementedError("Input layer does not have any parameters which can be updated via backward step.")


class Dense(Layer):

    def __init__(self, n_units, initializer=HeInitializer(42), activation_f="relu"):
        Layer.__init__(self, n_units, activation_f)
        self.initializer = initializer

    def initialize_parameters(self, prev_n_units):
        self.W, self.b = self.initializer.initialize((self.n_units, prev_n_units))

    def forward(self, A_prev):
        Z = self.W.dot(A_prev) + self.b
        self.A = self.activation_f(Z)
        self.d_A = self.d_activation_f(Z)

    def backward(self):
        self.delta = self.next_layer.W.T.dot(self.next_layer.delta) * self.d_A
        self.d_W = self.delta.dot(self.prev_layer.A.T)
        self.d_b = np.sum(self.delta, axis=1, keepdims=True)


class Output(Dense):

    def __init__(self, n_units, initializer=HeInitializer(42), activation_f="softmax"):
        Dense.__init__(self, n_units, initializer, activation_f)
        self.initial_delta = None

    def forward(self, A_prev):
        Z = self.W.dot(A_prev) + self.b
        self.A = self.activation_f(Z)

    def backward(self):
        self.d_W = self.initial_delta.dot(self.prev_layer.A.T)
        self.d_b = np.sum(self.initial_delta, axis=1, keepdims=True)
        self.delta = self.initial_delta
