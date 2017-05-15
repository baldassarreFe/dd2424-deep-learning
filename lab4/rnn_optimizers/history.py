from abc import ABC

import numpy as np

from rnn_optimizers import RnnVanilla


class RnnHistory(RnnVanilla, ABC):
    def __init__(self, rnn, initial_learning_rate,
                 decay_factor=1.0, stateful=False, clip=None):
        super().__init__(rnn, initial_learning_rate,
                         decay_factor, stateful, clip)

        # Create matrices to store the gradient history
        # for every weight/bias in every linear layer of
        # the network
        self.hist = {name: np.zeros_like(weights)
                     for weights, _, name in self.rnn.weights_gradients_pairs()}
