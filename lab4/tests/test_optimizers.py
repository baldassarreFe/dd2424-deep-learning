import unittest

import numpy as np

from datasets import InputTargetSequence
from initializers import Xavier
from initializers import Zeros
from network import CharRNN
from rnn_optimizers import RnnMomentum
from rnn_optimizers.adagrad import RnnAdaGrad
from rnn_optimizers.rmsprop import RnnRmsProp
from rnn_optimizers.vanilla import RnnVanilla


class TestOptimizers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_output_size = 4
        cls.state_size = 5
        cls.sequence_pairs = [
            cls.get_sequence(10, cls.input_output_size) for t in range(100)]

    @staticmethod
    def get_sequence(timesteps, size):
        np.random.seed(123)
        sequence = np.zeros((timesteps + 1, size))
        for t in range(timesteps):
            sequence[t, np.random.randint(0, size)] = 1
        return InputTargetSequence(sequence[:-1], sequence[1:])

    def setUp(self):
        self.rnn = CharRNN(
            input_output_size=self.input_output_size,
            state_size=self.state_size,
            initializer_W=Xavier(),
            initializer_U=Xavier(),
            initializer_V=Xavier(),
            initializer_b=Zeros(),
            initializer_c=Zeros()
        )

    def test_vanilla(self):
        opt = RnnVanilla(self.rnn, .001, .999, stateful=False)
        opt.train(self.sequence_pairs, epochs=40)
        self.assertCostDecreasing(opt.smooth_costs)

    def test_momentum(self):
        opt = RnnMomentum(self.rnn, .01, .999, stateful=False, momentum=.8)
        opt.train(self.sequence_pairs, epochs=40)
        self.assertCostDecreasing(opt.smooth_costs)

    def test_adagrad(self):
        opt = RnnAdaGrad(self.rnn, .01, .999, stateful=False)
        opt.train(self.sequence_pairs, epochs=40)
        self.assertCostDecreasing(opt.smooth_costs)

    def test_rmsprop(self):
        opt = RnnRmsProp(self.rnn, .01, .999, stateful=False, gamma=.99)
        opt.train(self.sequence_pairs, epochs=40)
        self.assertCostDecreasing(opt.smooth_costs)

    def assertCostDecreasing(self, costs):
        half = len(costs) // 2
        self.assertGreater(sum(costs[:half])/half, sum(costs[half:])/half)
