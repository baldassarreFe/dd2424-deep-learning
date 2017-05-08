from collections import namedtuple
from itertools import islice

from sklearn import preprocessing

InputTargetSequence = namedtuple('InputTargetSequence', 'input output')


class Goblet:
    def __init__(self, filename):
        self.char_sequence = list(self.load_text(filename))
        self.label_encoder = preprocessing.LabelBinarizer()
        self.encoded_text = self.label_encoder.fit_transform(
            self.char_sequence).T
        self.num_classes, self.sequence_length = self.encoded_text.shape

    def encode(self, *values):
        """
        Encode one/more string/char sequences:
        - encode('Harry')
        - encode(['P', 'o', 't', 't', 'e', 'r'])
        - encode('Harry', 'Potter')
        - encode(['H', 'a', 'r', 'r', 'y'], ['P', 'o', 't', 't', 'e', 'r'])
        :param values:
        :return:
        """
        if len(values) == 1:
            return self.label_encoder.transform(list(values[0])).T
        else:
            return [self.encode(s) for s in values]

    def decode_to_strings(self, *sequences):
        if len(sequences) == 1:
            return ''.join(self.label_encoder.inverse_transform(sequences[0].T))
        else:
            return [self.decode_to_strings(s) for s in sequences]

    def get_sequences(self, length=25):
        for i in range(0, self.sequence_length - length, length):
            yield InputTargetSequence(
                input=self.encoded_text[:, i:i + length],
                output=self.encoded_text[:, i + 1:i + length + 1]
            )

    @staticmethod
    def load_text(filename):
        with open(filename, 'r') as f:
            return f.read()


if __name__ == '__main__':
    goblet = Goblet('goblet_book.txt')
    print(goblet.encoded_text.shape)

    res = goblet.encode('Harry')
    print(res.shape, goblet.decode_to_strings(res))
    res = goblet.encode('Harry', 'Potter')
    print([r.shape for r in res], goblet.decode_to_strings(*res))

    res = goblet.encode(['H', 'a', 'r', 'r', 'y'])
    print(res.shape, goblet.decode_to_strings(res))
    res = goblet.encode(['S', 'o', 'm', 'e'], ['s', 't', 'r', 'i', 'n', 'g'])
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
