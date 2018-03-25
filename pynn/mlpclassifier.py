import numpy as np
from pynn.activations import *


class MLPClassifier:

    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
        self.parameters = {}
        self._initialize_parameters()

    def _initialize_parameters(self):
        """
        He initialization.
        https://arxiv.org/pdf/1502.01852.pdf
        :return:
        """
        for i in range(1, self.num_layers):
            self.parameters["W" + str(i)] = np.random.randn(self.layers[i], self.layers[i-1]) * np.sqrt(2.0 / self.layers[i-1])
            self.parameters["b" + str(i)] = np.zeros((self.layers[i], 1))

    def _feedforward(self, X):
        """
        Feedforward pass.
        ReLU hidden activations + Softmax output activations.
        :param X:
        :return:
        """
        activations = []
        A = X
        for i in range(1, self.num_layers - 1):
            Z = self.parameters["W" + str(i)].dot(A) + self.parameters["b" + str(i)]
            A = relu(Z)
            activations.append(A)
        Z = self.parameters["W" + str(i+1)].dot(A) + self.parameters["b" + str(i+1)]
        activations.append(softmax(Z))
        return activations