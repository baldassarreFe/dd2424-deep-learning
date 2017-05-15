import numpy as np

from .history import RnnHistory


class RnnAdaGrad(RnnHistory):
    def execute_update(self):
        for weights, grad, name in self.rnn.weights_gradients_pairs():
            if self.clip is not None:
                grad = np.clip(grad, -self.clip, +self.clip)
            self.hist[name] += grad ** 2
            update = self.learning_rate * grad / \
                     np.sqrt(self.hist[name] + np.finfo(float).eps)
            # if __debug__:
            #    self.learning_rate_warning(weights, update)
            weights -= update
