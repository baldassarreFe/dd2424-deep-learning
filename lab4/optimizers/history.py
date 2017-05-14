from abc import ABC

import numpy as np

from optimizers import VanillaSGD


class HistorySGD(VanillaSGD, ABC):
    def __init__(self, network, initial_learning_rate,
                 decay_factor=1.0, shuffle=False):
        super().__init__(network, initial_learning_rate, decay_factor, shuffle)

        # Create matrices to store the gradient history
        # for every weight/bias in every linear layer of
        # the network
        self.updatables = [{
                               'layer': layer,
                               'hist_W': np.zeros_like(layer.W),
                               'hist_b': np.zeros_like(layer.b)
                           }
                           for layer in self.updatables]
