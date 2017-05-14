import numpy as np
from tqdm import tqdm

from .optimizer import Optimizer


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

        for epoch_num in tqdm(range(1, epochs + 1), desc='Epochs'):
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
        """
        Compares the parameter scale with the update scale and checks that
        they are in an acceptable ratio (1e-4 < ratio < 1e-2)
        :param param:
        :param update:
        :return:
        """
        param_scale = np.linalg.norm(param.ravel())
        if param_scale != 0:
            updates_scale = np.linalg.norm(update.ravel())
            ratio = updates_scale / param_scale
            if ratio > 1e-2:
                print('Update ratio:', ratio, 'learning rate might be too high')
            elif ratio < 1e-4:
                print('Update ratio:', ratio, 'learning rate might be too low')
