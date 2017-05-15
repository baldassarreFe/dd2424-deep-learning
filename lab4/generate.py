import argparse

import numpy as np
from tqdm import tqdm

from datasets import TextSource
from utils import generate_text, restore_char_rnn


def print_text(rnn, last_probs, last_state, data, length):
    first_char = np.zeros_like(last_probs)
    first_char[np.random.choice(rnn.input_size, p=last_probs)] = 1
    text, _ = generate_text(rnn, data, first_char=first_char,
                            initial_state=last_state, length=length)
    print(text)


def main(args):
    data = TextSource(args.source)
    rnn = restore_char_rnn(args.checkpoint)

    sequence_pairs = list(data.get_sequences(25))

    prev_state = np.zeros(rnn.state_size)
    warm_up = np.random.randint(1, len(sequence_pairs))
    sequence_pairs = sequence_pairs[:warm_up]
    for sp in tqdm(sequence_pairs,
                   desc='Warm up on {} sequences'.format(warm_up)):
        probs, states = rnn.forward(sp.input, prev_state)
        prev_state = states[-1]

    print_text(rnn, probs[-1], prev_state, data, args.length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source',
                        type=str,
                        help='the source for the text')
    parser.add_argument('checkpoint',
                        type=str,
                        help='restore weights from this checkpoint')
    parser.add_argument('-l', '--length',
                        type=int,
                        default=200,
                        metavar='LENGTH',
                        help='generate a paragraph of this LENGTH')
    args = parser.parse_args()
    main(args)
