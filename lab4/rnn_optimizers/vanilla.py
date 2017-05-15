import numpy as np

from .optimizer import RnnOptimizer


class RnnVanilla(RnnOptimizer):
    def __init__(self, rnn, initial_learning_rate,
                 decay_factor=1.0, stateful=False, clip=None):
        super().__init__(rnn, stateful)
        self.learning_rate = initial_learning_rate
        self.decay_factor = decay_factor
        self.clip = clip
        self.learning_rates = []

    def train(self, sequence_pairs, epochs=1,
              callback=None, callback_every=0,
              epoch_callback=None):
        for _ in range(epochs):
            prev_state = np.zeros(self.rnn.state_size)
            for i, sp in enumerate(sequence_pairs):
                probs, states = self.rnn.forward(sp.input, prev_state)
                self.rnn.backward(sp.output)
                self.execute_update()
                cost = self.rnn.cost(sp.output)
                self.update_metrics(cost)
                if self.stateful:
                    prev_state = states[-1]
                else:
                    prev_state = np.zeros(self.rnn.state_size)
                if callback is not None and i % callback_every == 0:
                    callback(self, probs[-1], states[-1])
                self.learning_rate *= self.decay_factor
            if epoch_callback:
                epoch_callback(self)

    def execute_update(self):
        for weights, grad, name in self.rnn.weights_gradients_pairs():
            if self.clip is not None:
                grad = np.clip(grad, -self.clip, +self.clip)
            update = self.learning_rate * grad
            # if __debug__:
            #    self.learning_rate_warning(weights, update)
            weights -= update

    def update_metrics(self, cost):
        super().update_metrics(cost)
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
