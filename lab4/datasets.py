from collections import namedtuple

from sklearn import preprocessing
import numpy as np

InputTargetSequence = namedtuple('InputTargetSequence', 'input output')


class Goblet:
    num_classes = 80

    def __init__(self, filename):
        self.char_sequence = list(self.load_text(filename))
        self.label_encoder = preprocessing.LabelBinarizer()
        self.encoded_text = self.label_encoder.fit_transform(
            self.char_sequence)
        self.sequence_length = self.encoded_text.shape[1]

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
            return np.squeeze(self.label_encoder.transform(list(values[0])))
        else:
            return [self.encode(s) for s in values]

    def decode_to_strings(self, *sequences):
        if len(sequences) == 1:
            return ''.join(self.label_encoder.inverse_transform(sequences[0]))
        else:
            return [self.decode_to_strings(s) for s in sequences]

    def get_sequences(self, length=25):
        for i in range(0, self.sequence_length - length, length):
            yield InputTargetSequence(
                input=self.encoded_text[i:i + length],
                output=self.encoded_text[i + 1:i + length + 1]
            )

    @staticmethod
    def load_text(filename):
        with open(filename, 'r') as f:
            return f.read()

