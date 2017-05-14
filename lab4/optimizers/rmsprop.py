import numpy as np

from .history import HistorySGD


class RmsPropSGD(HistorySGD):
    def __init__(self, network, initial_learning_rate,
                 decay_factor=1.0, shuffle=False, gamma=0.0):
        super().__init__(network, initial_learning_rate, decay_factor, shuffle)
        self.gamma = gamma

    def execute_update(self):
        for his in self.updatables:
            # Update history
            his['hist_W'] = self.gamma * his['hist_W'] + \
                            (1 - self.gamma) * his['layer'].grad_W ** 2
            his['hist_b'] = self.gamma * his['hist_b'] + \
                            (1 - self.gamma) * his['layer'].grad_b ** 2

            update_W = self.learning_rate * his['layer'].grad_W / \
                       np.sqrt(his['hist_W'] + np.finfo(float).eps)
            update_b = self.learning_rate * his['layer'].grad_b / \
                       np.sqrt(his['hist_b'] + np.finfo(float).eps)

            if __debug__:
                self.learning_rate_warning(his['layer'].W, update_W)
                self.learning_rate_warning(his['layer'].b, update_b)

            # Update weighs
            his['layer'].W -= update_W
            his['layer'].b -= update_b
