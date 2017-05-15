import argparse

import numpy as np
from sklearn import preprocessing

from utils import restore_char_rnn


def main(rnn_checkpoint, chars_checkpoint, number):
    all_chars = np.load(chars_checkpoint)['labels']

    label_encoder = preprocessing.LabelBinarizer()
    label_encoder.fit(all_chars)

    rnn = restore_char_rnn(rnn_checkpoint)
    first_char = np.squeeze(label_encoder.transform(['\0']))
    initial_state = np.zeros(rnn.state_size)
    for i in range(number):
        seq, _ = rnn.generate(first_char, initial_state, 140)
        gen = ''.join(label_encoder.inverse_transform(seq)).strip()
        print(gen, end='\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('labels',
                        type=str,
                        help='restore labels from this file')
    parser.add_argument('checkpoint',
                        type=str,
                        help='restore weights from this checkpoint')
    parser.add_argument('-n', '--number',
                        type=int,
                        default=10,
                        metavar='N',
                        help='generate N tweets')
    args = parser.parse_args()
    main(args.checkpoint, args.labels, args.number)
