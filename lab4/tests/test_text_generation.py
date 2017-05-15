import unittest

import numpy as np

from datasets import TextSource
from initializers import Xavier
from initializers import Zeros
from network import CharRNN
from utils import generate_text


class TestTextGeneration(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.goblet = TextSource('sources/goblet_book.txt')
        cls.rnn = CharRNN(
            input_output_size=cls.goblet.num_classes,
            state_size=100,
            initializer_W=Xavier(),
            initializer_U=Xavier(),
            initializer_V=Xavier(),
            initializer_b=Zeros(),
            initializer_c=Zeros()
        )

    def test_text_generation(self):
        seq, last_state = generate_text(self.rnn, self.goblet)
        print(seq)
        self.assertEqual(last_state.size, self.rnn.state_size)

        seq, last_state = generate_text(
            self.rnn, self.goblet, self.goblet.encode('A'),
            np.random.random(self.rnn.state_size), 100)
        print(seq)
        self.assertEqual(last_state.size, self.rnn.state_size)


if __name__ == '__main__':
    unittest.main()
