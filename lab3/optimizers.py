from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm

from layers import Linear
from network import Network


class Optimizer(ABC):
    def __init__(self, network: Network):
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

        self.updatables = [l for l in self.network.layers if type(l) is Linear]

    def train(self, training, validation, epochs=1, batch_size=None):
        batch_size = batch_size or training.size

        for epoch_num in tqdm(range(epochs), desc='Epochs'):
            self.train_epoch(training, batch_size)
            self.update_metrics(training, validation, epoch_num)
            self.learning_rate *= self.decay_factor

    def train_epoch(self, training, batch_size):
        if self.shuffle:
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
        for layer in self.updatables:
            self._update_linear(layer)

    def _update_linear(self, layer):
        update = self.learning_rate * layer.grad_W
        if __debug__:
            self.learning_rate_warning(layer.W, update)
        layer.W -= update

        update = self.learning_rate * layer.grad_b
        if __debug__:
            self.learning_rate_warning(layer.b, update)
        layer.b -= update

    def update_metrics(self, training, validation, epoch_num):
        super().update_metrics(training, validation, epoch_num)
        self.learning_rates.append(self.learning_rate)

    @staticmethod
    def learning_rate_warning(param, update):
        param_scale = np.linalg.norm(param.ravel())
        if param_scale != 0:
            updates_scale = np.linalg.norm(update.ravel())
            ratio = updates_scale / param_scale
            if ratio > 1e-2:
                print('Update ratio:', ratio, 'learning rate might be too high')
            elif ratio < 1e-4:
                print('Update ratio:', ratio, 'learning rate might be too low')


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
                           for layer in self.updatables]

    def execute_update(self):
        for his in self.updatables:
            # Update history of updates
            his['mom_W'] = \
                self.momentum * his['mom_W'] + \
                self.learning_rate * his['layer'].grad_W
            his['mom_b'] = \
                self.momentum * his['mom_b'] + \
                self.learning_rate * his['layer'].grad_b

            if __debug__:
                self.learning_rate_warning(his['layer'].W, his['mom_W'])
                self.learning_rate_warning(his['layer'].b, his['mom_b'])

            # Update weighs
            his['layer'].W -= his['mom_W']
            his['layer'].b -= his['mom_b']
