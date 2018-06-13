import abc
from pynn.initializers import *
from pynn.activations import *
from pynn.regularizers import *


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
        self.model = None
        self.trainable = True

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
        return self.__class__.__name__ + " layer ({}) | {} units".format(self.activation_f, self.n_units)


class Input(Layer):

    def __init__(self, n_units):
        Layer.__init__(self, n_units, "identity")
        self.trainable = False

    def forward(self, X):
        self.A = self.activation_f(X)

    def backward(self):
        raise NotImplementedError("Input layer does not have any parameters which can be updated via backward step.")


class Dense(Layer):

    def __init__(self, n_units, activation_f="relu", initializer=HeInitializer(42), regularizer=None):
        Layer.__init__(self, n_units, activation_f)
        self.initializer = initializer
        self.regularizer = regularizer

    def initialize_parameters(self, prev_n_units):
        self.W, self.b = self.initializer.initialize((self.n_units, prev_n_units))

    def forward(self, A_prev):
        Z = self.W.dot(A_prev) + self.b
        self.A = self.activation_f(Z)
        self.d_A = self.d_activation_f(Z)

    def backward(self):
        if(isinstance(self.next_layer, Dropout)):
            self.d_A = self.d_A * self.next_layer.mask
        self.delta = self.next_layer.W.T.dot(self.next_layer.delta) * self.d_A
        self.d_W = self.delta.dot(self.prev_layer.A.T)
        self.d_b = np.sum(self.delta, axis=1, keepdims=True)
        if(not (self.regularizer is None)):
            self.regularizer.regularize(self)


    def __str__(self):
        return Layer.__str__(self) + (" | Regularizer: {}".format(self.regularizer))


class Output(Dense):

    def __init__(self, n_units, initializer=HeInitializer(42), activation_f="softmax"):
        Dense.__init__(self, n_units=n_units, initializer=initializer, activation_f=activation_f)
        self.initial_delta = None

    def forward(self, A_prev):
        Z = self.W.dot(A_prev) + self.b
        self.A = self.activation_f(Z)

    def backward(self):
        self.d_W = self.initial_delta.dot(self.prev_layer.A.T)
        self.d_b = np.sum(self.initial_delta, axis=1, keepdims=True)
        self.delta = self.initial_delta


class Dropout(Dense):

    def __init__(self, keep_prob):
        Dense.__init__(self, n_units=None, initializer=None, activation_f="identity")
        self.keep_prob = float(keep_prob)
        self.mask = None
        self.trainable = False

    def initialize_parameters(self, prev_n_units):
        self.n_units = self.prev_layer.n_units
        self.W = None

    def forward(self, A_prev):
        if(self.model.is_training):
            self.mask = np.random.binomial(n=1, p=self.keep_prob, size=A_prev.shape) / self.keep_prob
            self.A = self.activation_f(A_prev * self.mask)
        else:
            self.A = self.prev_layer.A

    def backward(self):
        self.delta = self.next_layer.delta
        self.W = self.next_layer.W
