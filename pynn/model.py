from pynn.optimizers import *


class Model:

    def __init__(self, loss_function, optimizer):
        self.loss_function = LossFunction(loss_function)
        self.optimizer = optimizer
        self.layers = []
        self.is_training = False

    def feedforward(self, X):
        A = X
        for layer in self.layers:
            layer.forward(A)
            A = layer.A
        return A

    def add_layer(self, layer):
        self.layers.append(layer)
        layer.model = self

    def initialize(self):
        for i in range(1, len(self.layers)):
            self.layers[i].connect_previous_layer(self.layers[i - 1])
            self.layers[i - 1].connect_next_layer(self.layers[i])
            self.layers[i].initialize_parameters(self.layers[i - 1].n_units)

    def print_model_details(self):
        print("\n- NN architecture:\n")
        for layer in self.layers:
            print("  {}".format(layer))
        print("\n- Loss function: {}\n".format(self.loss_function.name))
        print("- Optimizer: {}\n".format(self.optimizer.name))

    def train(self, train_X, train_Y, batch_size, epochs, learning_rate, verbose=True):
        self.is_training = True
        losses = self.optimizer.optimize(self, train_X.T, train_Y.T, batch_size, epochs, learning_rate, verbose)
        self.is_training = False
        return losses

    def predict(self, X):
        return self.feedforward(X.T).T
