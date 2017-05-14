from .history import HistorySGD


class MomentumSGD(HistorySGD):
    def __init__(self, network, initial_learning_rate,
                 decay_factor=1.0, shuffle=False, momentum=0.0):
        super().__init__(network, initial_learning_rate, decay_factor, shuffle)
        self.momentum = momentum

    def execute_update(self):
        for his in self.updatables:
            # Update history of updates
            his['hist_W'] = \
                self.momentum * his['hist_W'] + \
                self.learning_rate * his['layer'].grad_W
            his['hist_b'] = \
                self.momentum * his['hist_b'] + \
                self.learning_rate * his['layer'].grad_b

            if __debug__:
                self.learning_rate_warning(his['layer'].W, his['hist_W'])
                self.learning_rate_warning(his['layer'].b, his['hist_b'])

            # Update weighs
            his['layer'].W -= his['hist_W']
            his['layer'].b -= his['hist_b']
