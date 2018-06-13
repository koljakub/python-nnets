from sklearn.datasets import load_digits
from pynn.model import Model
from pynn.layers import *
from pynn.optimizers import *


def mnist_feedforward_ann():
    model = Model("negative_log_loss", AdamOptimizer())
    model.add_layer(Input(64))
    model.add_layer(Dense(256))
    model.add_layer(Dropout(0.8))
    model.add_layer(Dense(256))
    model.add_layer(Dropout(0.8))
    model.add_layer(Output(10))
    model.initialize()
    model.print_model_details()

    digits = load_digits()
    data_X = digits.data / 255.0
    data_y = digits.target
    p = np.random.permutation(len(data_y))
    data_X = data_X[p]
    data_y = data_y[p]
    train_X = data_X[0:1000]
    train_Y = np.eye(10)[data_y[0:1000]]
    test_X = data_X[1000:]
    test_y = data_y[1000:]
    model.train(train_X, train_Y, batch_size=32, epochs=250, learning_rate=0.03)
    predictions = model.predict(test_X)
    print("\nAccuracy (test set): {}".format(np.sum(np.argmax(predictions, axis=1) == test_y) / float(len(test_X))))


if __name__ == '__main__':
    mnist_feedforward_ann()
