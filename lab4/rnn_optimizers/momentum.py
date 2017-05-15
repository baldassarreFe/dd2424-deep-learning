import numpy as np

from .history import RnnHistory


class RnnMomentum(RnnHistory):
    def __init__(self, rnn, initial_learning_rate,
                 decay_factor=1.0, stateful=False, momentum=0.0, clip=None):
        super().__init__(rnn, initial_learning_rate,
                         decay_factor, stateful, clip)
        self.momentum = momentum

    def execute_update(self):
        for weights, grad, name in self.rnn.weights_gradients_pairs():
            if self.clip is not None:
                grad = np.clip(grad, -self.clip, +self.clip)
            self.hist[name] = self.momentum * self.hist[name] + \
                              self.learning_rate * grad
            # if __debug__:
            #    self.learning_rate_warning(weights, self.hist[name])
            weights -= self.hist[name]
