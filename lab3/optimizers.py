from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm

import layers
from network import Network


class Optimizer(ABC):
    def __init__(self, network):
        self.network = network
        self.epoch_nums = []
        self.acc_train = []
        self.acc_val = []
        self.cost_train = []
        self.cost_val = []

    @abstractmethod
    def train(self, training, validation, epochs, batch_size):
        pass

    def update_metrics(self, training, validation, epoch_num):
        self.epoch_nums.append(epoch_num)

        cost, accuracy = self.network.cost_accuracy(training)
        self.cost_train.append(cost)
        self.acc_train.append(accuracy)

        cost, accuracy = self.network.cost_accuracy(validation)
        self.cost_val.append(cost)
        self.acc_val.append(accuracy)


class VanillaSGD(Optimizer):
    def __init__(self, network, initial_learning_rate,
                 decay_factor=1.0, shuffle=False):
        super().__init__(network)
        self.learning_rate = initial_learning_rate
        self.decay_factor = decay_factor
        self.shuffle = shuffle
        self.learning_rates = []

    def train(self, training, validation, epochs=1, batch_size=None):
        batch_size = batch_size or training.size

        for epoch_num in tqdm(range(epochs), desc='Epochs'):
            self.train_epoch(training, batch_size)
            self.update_metrics(training, validation, epoch_num)
            self.learning_rate *= self.decay_factor

    def train_epoch(self, training, batch_size):
        if self.shuffle and (batch_size != training.size):
            indexes = np.random.permutation(training.size)
        else:
            indexes = np.arange(training.size)

        for start in range(0, training.size, batch_size):
            self.train_batch(training, indexes[start: start + batch_size])
            self.execute_update()

    def train_batch(self, training, indexes):
        inputs = training.images[:, indexes]
        one_hot_targets = training.one_hot_labels[:, indexes]

        self.network.evaluate(inputs, train=True)
        self.network.backward(one_hot_targets)

    def execute_update(self):
        for layer in self.network.layers:
            if type(layer) is layers.Linear:
                layer.W -= self.learning_rate * layer.grad_W
                layer.b -= self.learning_rate * layer.grad_b

    def update_metrics(self, training, validation, epoch_num):
        super().update_metrics(training, validation, epoch_num)
        self.learning_rates.append(self.learning_rate)


class MomentumSGD(VanillaSGD):
    def __init__(self, network, initial_learning_rate,
                 decay_factor=1.0, shuffle=False, momentum=0.0):
        super().__init__(network, initial_learning_rate, decay_factor, shuffle)
        self.momentum = momentum

        # Create matrices to store the gradient history
        # for every weight/bias in every linear layer of
        # the network
        self.updatables = [{
                               'layer': layer,
                               'mom_W': np.zeros_like(layer.W),
                               'mom_b': np.zeros_like(layer.b)
                           }
                           for layer in self.network.layers
                           if type(layer) is layers.Linear]

    def execute_update(self):
        for his in self.updatables:
            # Update history
            his['mom_W'] = \
                self.momentum * his['mom_W'] + \
                self.learning_rate * his['layer'].grad_W
            his['mom_b'] = \
                self.momentum * his['mom_b'] + \
                self.learning_rate * his['layer'].grad_b

            # Update weighs
            his['layer'].W -= his['mom_W']
            his['layer'].b -= his['mom_b']


if __name__ == '__main__':
    import initializers
    import datasets
    from utils import costs_accuracies_plot, show_plot


    def test_vanilla(cifar):
        training = cifar.get_named_batches('data_batch_1')
        validation = cifar.get_named_batches('data_batch_2')

        net = Network()
        net.add_layer(layers.Linear(cifar.input_size, cifar.output_size, 0, initializers.Xavier()))
        net.add_layer(layers.Softmax(cifar.output_size))

        opt = VanillaSGD(net, initial_learning_rate=0.01, decay_factor=0.99, shuffle=True)

        opt.train(training, validation, 100, 500)

        costs_accuracies_plot(opt.epoch_nums, opt.acc_train, opt.acc_val, opt.cost_train, opt.cost_val,
                              'images/vanilla.png')
        show_plot('images/vanilla.png')


    def test_momentum(cifar):
        training = cifar.get_named_batches('data_batch_1')
        validation = cifar.get_named_batches('data_batch_2')

        net = Network()
        net.add_layer(layers.Linear(cifar.input_size, cifar.output_size, 0, initializers.Xavier()))
        net.add_layer(layers.Softmax(cifar.output_size))

        opt = MomentumSGD(net, initial_learning_rate=0.01, decay_factor=0.99, shuffle=True, momentum=0.8)

        opt.train(training, validation, 100, 500)

        costs_accuracies_plot(opt.epoch_nums, opt.acc_train, opt.acc_val, opt.cost_train, opt.cost_val,
                              'images/momentum.png')
        show_plot('images/momentum.png')


    def test_overfitting(cifar, momentum):
        training = cifar.get_named_batches('data_batch_1').subset(100)

        net = Network()
        net.add_layer(layers.Linear(cifar.input_size, 50, 0, initializers.Xavier()))
        net.add_layer(layers.ReLU(50))
        net.add_layer(layers.Linear(50, cifar.output_size, 0, initializers.Xavier()))
        net.add_layer(layers.Softmax(cifar.output_size))

        opt = MomentumSGD(net, initial_learning_rate=0.005, momentum=momentum)

        opt.train(training, training, 400)

        costs_accuracies_plot(opt.epoch_nums, opt.acc_train, opt.acc_val, opt.cost_train, opt.cost_val,
                              'images/overfit_mom{}.png'.format(momentum))
        show_plot('images/overfit_mom{}.png'.format(momentum))

    def test_batch_norm(cifar, learning_rate):
        training = cifar.get_named_batches('data_batch_1').subset(1000)

        net = Network()
        net.add_layer(layers.Linear(cifar.input_size, 50, 0, initializers.Xavier()))
        net.add_layer(layers.BatchNormalization(50))
        net.add_layer(layers.ReLU(50))
        net.add_layer(layers.Linear(50, cifar.output_size, 0, initializers.Xavier()))
        net.add_layer(layers.Softmax(cifar.output_size))

        opt = MomentumSGD(net, initial_learning_rate=learning_rate, momentum=0.0)

        opt.train(training, training, 400)

        costs_accuracies_plot(opt.epoch_nums, opt.acc_train, opt.acc_val, opt.cost_train, opt.cost_val,
                              'images/overfit_bn{}.png'.format(learning_rate))
        #show_plot('images/overfit_bn{}.png'.format(learning_rate))

    def test_three_layers_no_batch(cifar):
        training = cifar.get_named_batches('data_batch_1')
        validation = cifar.get_named_batches('data_batch_5').subset(1000)

        net = Network()
        net.add_layer(layers.Linear(cifar.input_size, 50, 0, initializers.Xavier()))
        net.add_layer(layers.ReLU(50))
        net.add_layer(layers.Linear(50, 30, 0, initializers.Xavier()))
        net.add_layer(layers.ReLU(30))
        net.add_layer(layers.Linear(30, cifar.output_size, 0, initializers.Xavier()))
        net.add_layer(layers.Softmax(cifar.output_size))

        opt = MomentumSGD(net, initial_learning_rate=0.05, decay_factor=0.998, momentum=0.8)

        opt.train(training, validation, epochs=15, batch_size=250)

        costs_accuracies_plot(opt.epoch_nums, opt.acc_train, opt.acc_val, opt.cost_train, opt.cost_val,
                              'images/three_layer_no_batch.png')
        # show_plot('images/three_layer_no_batch.png')

    def test_three_layers_batch(cifar, initial_learning_rate):
        training = cifar.get_named_batches('data_batch_1').subset(5000)
        validation = cifar.get_named_batches('data_batch_5').subset(1000)

        net = Network()
        net.add_layer(layers.Linear(cifar.input_size, 50, 0, initializers.Xavier()))
        net.add_layer(layers.BatchNormalization(50))
        net.add_layer(layers.ReLU(50))
        net.add_layer(layers.Linear(50, 30, 0, initializers.Xavier()))
        #net.add_layer(layers.BatchNormalization(30))
        net.add_layer(layers.ReLU(30))
        net.add_layer(layers.Linear(30, cifar.output_size, 0, initializers.Xavier()))
        net.add_layer(layers.Softmax(cifar.output_size))

        opt = MomentumSGD(net, initial_learning_rate=initial_learning_rate,
                          decay_factor=0.998, momentum=0.8)
        #opt = VanillaSGD(net, initial_learning_rate, decay_factor=0.998)

        opt.train(training, validation, epochs=50, batch_size=25)

        costs_accuracies_plot(opt.epoch_nums, opt.acc_train, opt.acc_val, opt.cost_train, opt.cost_val,
                              'images/three_layer_batch{}.png'.format(initial_learning_rate))
        # show_plot('images/three_layer_batch.png')


    cifar = datasets.CIFAR10()

    """
    test_vanilla(cifar)
    test_momentum(cifar)

    for momentum in [.3, .6, .8, .95]:
        test_overfitting(cifar, momentum)

    for learning_rate in [0.5, 0.05, 0.005, 0.0005]:
        test_batch_norm(cifar, learning_rate)
    """
    # test_three_layers_no_batch(cifar)
    for learning_rate in [0.001, 0.005, 0.01]:
        test_three_layers_batch(cifar, learning_rate)

