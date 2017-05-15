import unittest

import numpy as np

from initializers import Xavier, Zeros
from network import RecurrentNeuralNetwork


class TestRNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_size = 2
        cls.output_size = 3
        cls.state_size = 5
        cls.rnn = RecurrentNeuralNetwork(
            input_size=cls.input_size,
            output_size=cls.output_size,
            state_size=cls.state_size,
            initializer_W=Xavier(),
            initializer_U=Xavier(),
            initializer_V=Xavier(),
            initializer_b=Zeros(),
            initializer_c=Zeros()
        )

    def test_predict_prob(self):
        p, h = self.rnn.predict_prob(
            x=np.array([0, 1]),
            prev_state=np.random.random(size=self.state_size)
        )
        self.valid_output(p)
        self.valid_state(h)

    def test_predict_class(self):
        c, h = self.rnn.predict_class(
            x=np.array([1, 0]),
            prev_state=np.random.random(size=self.state_size)
        )
        self.valid_class(c)
        self.valid_state(h)

    def test_forward(self):
        sequence = np.array([
            [0, 1],
            [1, 0],
            [0, 1],
            [0, 1],
            [1, 0],
            [0, 1],
        ])
        probs, states = self.rnn.forward(
            sequence=sequence,
            prev_state=np.random.random(size=self.state_size)
        )
        for t in range(5):
            self.valid_output(probs[t])
            self.valid_state(states[t])

        targets = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
        ])
        cost = self.rnn.cost(targets)

    def test_backprop(self):
        sequence = np.array([
            [0, 1],
            [1, 0],
            [0, 1],
            [0, 1],
            [1, 0],
            [0, 1],
        ])
        probs, states = self.rnn.forward(
            sequence=sequence,
            prev_state=np.random.random(size=self.state_size)
        )

        targets = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
        ])

        self.rnn.backward(targets)

    def valid_class(self, c):
        self.valid_output(c)
        self.assertTrue((c == 1).sum(), 1)

    def valid_state(self, h):
        self.assertEqual(h.size, self.state_size)

    def valid_output(self, p):
        self.assertEqual(p.size, self.output_size)
        self.assertAlmostEqual(p.sum(), 1)


if __name__ == '__main__':
    unittest.main()
