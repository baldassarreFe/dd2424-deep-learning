from abc import ABC, abstractmethod

import numpy as np

import layers
from network import Network


class Optimizer(ABC):
    def __init__(self, network):
        self.network = network
        self.acc_train = []
        self.acc_val = []
        self.cost_train = []
        self.cost_val = []
        pass

    @abstractmethod
    def train(self, training, validation, epochs, batch_size):
        pass

    def update_metrics(self, training, validation):
        cost, accuracy = self.cost_accuracy(training)
        self.cost_train.append(cost)
        self.acc_train.append(accuracy)

        cost, accuracy = self.cost_accuracy(validation)
        self.cost_val.append(cost)
        self.acc_val.append(accuracy)

    def cost_accuracy(self, dataset):
        Y = self.network.evaluate(dataset.images)
        return (self.network.cost(dataset.one_hot_labels, None, Y),
                self.network.accuracy(dataset.one_hot_labels, None, Y))


class VanillaSGD(Optimizer):
    def __init__(self, network, initial_learning_rate,
                 decay_factor=1, shuffle=False):
        super().__init__(network)
        self.learning_rate = initial_learning_rate
        self.decay_factor = decay_factor
        self.shuffle = shuffle
        self.learning_rates = []

    def train(self, training, validation, epochs=1, batch_size=None):
        batch_size = batch_size or training.size

        for epoch_num in range(epochs):
            self.train_epoch(training, batch_size)
            self.update_metrics(training, validation)
            self.learning_rate *= self.decay_factor

    def train_epoch(self, training, batch_size):
        if self.shuffle:
            indexes = np.random.permutation(training.size)
        else:
            indexes = np.arange(training.size)

        for start in range(0, training.size, batch_size):
            self.run_batch(training, indexes[start: start + batch_size])
            self.execute_update()

    def run_batch(self, training, indexes):
        inputs = training.images[:, indexes]
        one_hot_targets = training.one_hot_labels[:, indexes]

        self.network.evaluate(inputs)
        self.network.backward(one_hot_targets)

    def execute_update(self):
        for layer in self.network.layers:
            if type(layer) is layers.Linear:
                layer.W -= self.learning_rate * layer.grad_W
                layer.b -= self.learning_rate * layer.grad_b

    def update_metrics(self, training, validation):
        super().update_metrics(training, validation)
        self.learning_rates.append(self.learning_rate)


class MomentumSGD(VanillaSGD):
    def __init__(self, network, initial_learning_rate,
                 decay_factor=1.0, shuffle=False, momentum=0.0):
        super().__init__(network, initial_learning_rate, decay_factor, shuffle)
        self.momentum = momentum

        # Create matrices to store the gradient history
        # for every weight/bias in every linear layer of
        # the network
        self.history = {}
        for layer in self.network.layers:
            if type(layer) is layers.Linear:
                self.history[layer] = {
                    'W': np.zeros_like(layer.W),
                    'b': np.zeros_like(layer.b)
                }

    def execute_update(self):
        for layer in self.network.layers:
            if type(layer) is layers.Linear:
                # Update history
                self.history[layer]['W'] = \
                    self.momentum * self.history[layer]['W'] + \
                    self.learning_rate * layer.grad_W
                self.history[layer]['b'] = \
                    self.momentum * self.history[layer]['b'] + \
                    self.learning_rate * layer.grad_b

                # Update weighs
                layer.W -= self.history[layer]['W']
                layer.b -= self.history[layer]['b']


if __name__ == '__main__':
    import initializers
    from utils import costs_accuracies_plot, show_plot
    from cifar import CIFAR10

    cifar = CIFAR10()
    training = cifar.get_batches('data_batch_1')
    validation = cifar.get_batches('data_batch_2')

    # Test VanillaSGD
    net = Network()
    net.add_layer(layers.Linear(cifar.input_size, cifar.output_size, 0, initializers.Xavier()))
    net.add_layer(layers.Softmax(cifar.output_size))

    opt = VanillaSGD(net, initial_learning_rate=0.001, decay_factor=0.95, shuffle=True)

    opt.train(training, validation, 10, 100)

    costs_accuracies_plot(opt.acc_train, opt.acc_val, opt.cost_train, opt.cost_val, 'images/vanilla.png')
    show_plot('images/vanilla.png')

    # Test MomentumSGD
    net = Network()
    net.add_layer(layers.Linear(cifar.input_size, cifar.output_size, 0, initializers.Xavier()))
    net.add_layer(layers.Softmax(cifar.output_size))

    opt = MomentumSGD(net, initial_learning_rate=0.001, decay_factor=0.95, shuffle=True, momentum=0.9)

    opt.train(training, validation, 10, 100)

    costs_accuracies_plot(opt.acc_train, opt.acc_val, opt.cost_train, opt.cost_val, 'images/momentum.png')
    show_plot('images/momentum.png')
