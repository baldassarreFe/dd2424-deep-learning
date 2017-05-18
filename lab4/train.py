import argparse
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from os.path import basename

from datasets import TextSource
from initializers import Xavier
from initializers import Zeros
from network import CharRNN
from rnn_optimizers.adagrad import RnnAdaGrad
from utils import generate_text, save_char_rnn, restore_char_rnn


def callback(opt, last_probs, last_state, data, start):
    first_char = np.zeros_like(last_probs)
    first_char[np.random.choice(opt.rnn.input_size, p=last_probs)] = 1
    text, _ = generate_text(opt.rnn, data,
                            first_char=first_char,
                            initial_state=last_state)
    print('Sequences {} cost {:.3f} learning rate {:.2e} elapsed {:.0f}s:\n{}\n'
          .format(opt.steps, opt.smooth_costs[-1], opt.learning_rates[-1],
                  time.time() - start, text))
    plt.plot(opt.smooth_costs, 'b-')
    plt.pause(.05)


def epoch_callback(opt):
    save_char_rnn(opt.rnn, 'weights_{}'.format(opt.steps))
    plt.axvline(x=opt.steps, color='r')
    plt.pause(.05)


def setup_plot():
    plt.plot([])
    plt.grid(True, which='major', color='k', linestyle='-', alpha=0.3)
    plt.grid(True, which='minor', color='k', linestyle='-', alpha=0.1)
    plt.minorticks_on()
    plt.title('Evolution of cost over the number of sequences seen')
    plt.xlabel('Sequences seen')
    plt.ylabel('Smoothed cost')
    plt.ion()


def main(args):
    data = TextSource(args.source)
    if args.checkpoint:
        rnn = restore_char_rnn(args.checkpoint)
    else:
        rnn = CharRNN(
            input_output_size=data.num_classes,
            state_size=args.state,
            initializer_W=Xavier(),
            initializer_U=Xavier(),
            initializer_V=Xavier(),
            initializer_b=Zeros(),
            initializer_c=Zeros()
        )
    opt = RnnAdaGrad(rnn, 0.1, stateful=True, clip=5)

    setup_plot()

    sequence_pairs = list(data.get_sequences(25))
    print('Training on {}:\n'
          '- {} total characters\n'
          '- {} unique characters\n'
          '- {} sequences of length 25'
          .format(args.source, data.total_chars,
                  data.num_classes, len(sequence_pairs)))
    opt.train(sequence_pairs, epochs=40,
              callback=partial(callback, data=data, start=time.time()),
              callback_every=4321,
              epoch_callback=epoch_callback)

    plt.savefig('plots/{}.png'.format(basename(args.source)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source',
                        type=str,
                        help='the source for the text')
    parser.add_argument('-c', '--checkpoint',
                        type=str,
                        metavar='CHECKPOINT',
                        help='restore weights from CHECKPOINT')
    parser.add_argument('-s', '--state',
                        type=int,
                        default=100,
                        metavar='HIDDEN',
                        help='set HIDDEN as size of the RNN state')
    args = parser.parse_args()
    main(args)
