import numpy as np

from .history import HistorySGD


class AdaGradSGD(HistorySGD):
    def execute_update(self):
        for his in self.updatables:
            # Update history
            his['hist_W'] += his['layer'].grad_W ** 2
            his['hist_b'] += his['layer'].grad_b ** 2

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
