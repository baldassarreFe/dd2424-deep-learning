import unittest

import numpy as np

from datasets import Goblet
from utils import build_network, generate_text


class TestTextGeneration(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.rnn = build_network()
        cls.goblet = Goblet('goblet_book.txt')

    def test_text_generation(self):
        seq, last_state = generate_text(self.rnn, self.goblet)
        print(seq)
        self.assertEqual(last_state.size, self.rnn.state_size)

        seq, last_state = generate_text(
            self.rnn, self.goblet, 'A',
            np.random.random(self.rnn.state_size), 100)
        print(seq)
        self.assertEqual(last_state.size, self.rnn.state_size)


if __name__ == '__main__':
    unittest.main()
