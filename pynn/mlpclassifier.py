import numpy as np
from pynn.activations import *
from collections import deque


class MLPClassifier:

    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
        self.weights = []
        self.biases = []
        self._initialize_parameters()

    def _initialize_parameters(self):
        """
        He initialization.
        https://arxiv.org/pdf/1502.01852.pdf
        :return:
        """
        for i in range(1, self.num_layers):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i - 1]) * np.sqrt(2.0 / self.layers[i - 1]))
            self.biases.append(np.zeros((self.layers[i], 1)))

    def _feedforward(self, X):
        """
        Feedforward pass.
        ReLU hidden activations + Softmax output activations.
        :param X:
        :return:
        """
        activations = []
        A = X
        activations.append(A)

        for i in range(0, len(self.weights) - 1):
            Z = self.weights[i].dot(A) + self.biases[i]
            A = relu(Z)
            activations.append(A)
        Z = self.weights[-1].dot(A) + self.biases[-1]
        activations.append(softmax(Z))
        return activations

    def _backpropagate(self, X, Y):
        """
        Backpropagation alg.
        :return:
        """
        batch_size = X.shape[1]
        grad_w = deque()
        grad_b = deque()

        activations = self._feedforward(X)
        delta = activations[-1] - Y
        grad_w.append((1.0 / batch_size) * np.outer(delta, activations[-2].T))
        grad_b.append((1.0 / batch_size) * np.sum(delta, axis = 1, keepdims = True))
        for i in range(self.num_layers - 2, 0, -1):
            delta = self.weights[i].T.dot(delta)
            grad_w.appendleft((1.0 / batch_size) * np.outer(delta, activations[i - 1].T))
            grad_b.appendleft((1.0 / batch_size) * np.sum(delta, axis=1, keepdims=True))
        return (grad_w, grad_b)

    def predict(self, X):
        return np.argmax(self._feedforward(X.T)[-1], axis = 0)

# if __name__ == '__main__':
#     net = MLPClassifier([50, 5, 4, 3])
#     X = np.random.randn(1, 50)
#     Y = np.array([1, 0, 0]).reshape(3, 1)
#     gradient = net._backpropagate(X.T, Y)
#     print(gradient)