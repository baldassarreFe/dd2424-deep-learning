import numpy as np

from .history import RnnHistory


class RnnRmsProp(RnnHistory):
    def __init__(self, rnn, initial_learning_rate,
                 decay_factor=1.0, stateful=False, gamma=0.0, clip=None):
        super().__init__(rnn, initial_learning_rate,
                         decay_factor, stateful, clip)
        self.gamma = gamma

    def execute_update(self):
        for weights, grad, name in self.rnn.weights_gradients_pairs():
            if self.clip is not None:
                grad = np.clip(grad, -self.clip, +self.clip)
            self.hist[name] = self.gamma * self.hist[name] + \
                              (1 - self.gamma) * grad ** 2
            update = self.learning_rate * grad / \
                     np.sqrt(self.hist[name] + np.finfo(float).eps)
            # if __debug__:
            #    self.learning_rate_warning(weights, update)
            weights -= update
