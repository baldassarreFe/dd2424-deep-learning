import unittest

import numpy as np

from initializers import Xavier, Zeros
from network import CharRNN
from tests.test_rnn import TestRNN


class TestCharRNN(TestRNN):
    @classmethod
    def setUpClass(cls):
        cls.rnn = None
        cls.input_size = 2
        cls.output_size = 2
        cls.state_size = 5
        cls.rnn = CharRNN(
            input_output_size=cls.input_size,
            state_size=cls.state_size,
            initializer_W=Xavier(),
            initializer_U=Xavier(),
            initializer_V=Xavier(),
            initializer_b=Zeros(),
            initializer_c=Zeros()
        )

    def test_generate(self):
        seq, last_state = self.rnn.generate(
            x=np.array([1, 0]),
            prev_state=np.random.random(size=self.state_size),
            timesteps=10
        )
        for t in range(10):
            self.valid_class(seq[t])
        self.valid_state(last_state)

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
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [1, 0],
            [0, 1],
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
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [1, 0],
            [0, 1],
        ])
        self.rnn.backward(targets)


if __name__ == '__main__':
    unittest.main()
