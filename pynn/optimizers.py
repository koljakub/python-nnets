import abc
from pynn.losses import *
from pynn.preprocessing import *


class Optimizer:

    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def optimize(self, model, train_X, train_Y, batch_size, epochs, alpha, verbose):
        return


class GradientDescentOptimizer(Optimizer):

    def __init__(self, beta=0.9):
        Optimizer.__init__(self, "Gradient Descent (with momentum) optimizer")
        self.beta = beta

    def optimize(self, model, train_X, train_Y, batch_size, epochs, alpha, verbose):
        batches = get_batches(train_X, train_Y, batch_size)
        epoch_losses = []
        v_weights = []
        v_biases = []
        for layer in model.layers:
            v_weights.append(np.zeros_like(layer.W))
            v_biases.append(np.zeros_like(layer.b))

        for i in range(epochs):
            epoch_loss = 0
            for batch in batches:
                predictions = model.feedforward(batch[0])
                model.layers[-1].initial_delta = model.loss_function.get_initial_delta(model.layers[-1].activation_f)(
                    predictions, batch[1])
                for j in range(len(model.layers) - 1, 0, -1):
                    layer = model.layers[j]
                    layer.backward()
                    if (not layer.trainable): continue
                    v_weights[j] = self.beta * v_weights[j] + (1 - self.beta) * layer.d_W
                    v_biases[j] = self.beta * v_biases[j] + (1 - self.beta) * layer.d_b
                    layer.W = layer.W - alpha * (1.0 / batch_size) * v_weights[j]
                    layer.b = layer.b - alpha * (1.0 / batch_size) * v_biases[j]
                batch_loss = (1.0 / batch_size) * np.sum(negative_log_loss(model.feedforward(batch[0]), batch[1]))
                epoch_loss += batch_loss
            epoch_loss = (1.0 / len(batches)) * epoch_loss
            epoch_losses.append(epoch_loss)
            if (verbose):
                print("Epoch {} | Loss: {}").format(i, epoch_loss)
        return epoch_losses


class AdamOptimizer(Optimizer):

    def __init__(self, beta1=0.9, beta2=0.999):
        Optimizer.__init__(self, "Adaptive Moment Estimation optimizer (ADAM)")
        self.beta1 = beta1
        self.beta2 = beta2

    def optimize(self, model, train_X, train_Y, batch_size, epochs, alpha, verbose):
        batches = get_batches(train_X, train_Y, batch_size)
        epoch_losses = []
        v_weights = []
        s_weights = []
        v_biases = []
        s_biases = []
        for layer in model.layers:
            v_weights.append(np.zeros_like(layer.W))
            s_weights.append(np.zeros_like(layer.W))
            v_biases.append(np.zeros_like(layer.b))
            s_biases.append(np.zeros_like(layer.b))
        eps = np.finfo(np.float32).eps

        for i in range(1, epochs + 1):
            epoch_loss = 0
            for batch in batches:
                predictions = model.feedforward(batch[0])
                model.layers[-1].initial_delta = model.loss_function.get_initial_delta(model.layers[-1].activation_f)(
                    predictions, batch[1])
                for j in range(len(model.layers) - 1, 0, -1):
                    layer = model.layers[j]
                    layer.backward()
                    if (not layer.trainable): continue
                    v_weights[j] = self.beta1 * v_weights[j] + (1 - self.beta1) * layer.d_W
                    v_biases[j] = self.beta1 * v_biases[j] + (1 - self.beta1) * layer.d_b
                    s_weights[j] = self.beta2 * s_weights[j] + (1 - self.beta2) * np.square(layer.d_W)
                    s_biases[j] = self.beta2 * s_biases[j] + (1 - self.beta2) * np.square(layer.d_b)
                    v_weights_corrected = (1.0 / (1 - self.beta1 ** i)) * v_weights[j]
                    v_biases_corrected = (1.0 / (1 - self.beta1 ** i)) * v_biases[j]
                    s_weights_corrected = (1.0 / (1 - self.beta2 ** i)) * s_weights[j]
                    s_biases_corrected = (1.0 / (1 - self.beta2 ** i)) * s_biases[j]
                    layer.W = layer.W - alpha * (1.0 / batch_size) * (
                                v_weights_corrected / (np.sqrt(s_weights_corrected) + eps))
                    layer.b = layer.b - alpha * (1.0 / batch_size) * (
                                v_biases_corrected / (np.sqrt(s_biases_corrected) + eps))
                batch_loss = (1.0 / batch_size) * np.sum(negative_log_loss(model.feedforward(batch[0]), batch[1]))
                epoch_loss += batch_loss
            epoch_loss = (1.0 / len(batches)) * epoch_loss
            epoch_losses.append(epoch_loss)
            if (verbose):
                print("Epoch {} | Loss: {}").format(i, epoch_loss)
        return epoch_losses
