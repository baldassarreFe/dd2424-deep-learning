import unittest
from itertools import islice
from datasets import Goblet


class TestGoblet(unittest.TestCase):
    def test_goblet(self):
        goblet = Goblet('goblet_book.txt')
        print(goblet.encoded_text.shape)

        res = goblet.encode('Harry')
        print(res.shape, goblet.decode_to_strings(res))
        res = goblet.encode('Harry', 'Potter')
        print([r.shape for r in res], goblet.decode_to_strings(*res))

        res = goblet.encode(['H', 'a', 'r', 'r', 'y'])
        print(res.shape, goblet.decode_to_strings(res))
        res = goblet.encode(['S', 'o', 'm', 'e'],
                            ['s', 't', 'r', 'i', 'n', 'g'])
        print([r.shape for r in res], goblet.decode_to_strings(*res))

        for seq in goblet.get_sequences():
            print(seq.input.shape, seq.output.shape)

        for seq in islice(goblet.get_sequences(), 0, 5):
            x, y = goblet.decode_to_strings(seq.input, seq.output)
            print()
            print('=' * 25)
            print(x)
            print(y)
            print('=' * 25)
