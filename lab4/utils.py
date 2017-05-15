import itertools

import matplotlib.pyplot as plt
import numpy as np

from initializers import Zeros
from network import CharRNN


def generate_text(rnn, data, first_char=None, initial_state=None, length=200):
    if first_char is None:
        first_char = np.zeros(rnn.input_size)
        first_char[np.random.randint(0, rnn.input_size)] = 1
    if initial_state is None:
        initial_state = np.zeros(rnn.state_size)
    seq, last_state = rnn.generate(first_char, initial_state, length)
    return data.decode_to_strings(seq), last_state


def cost_plot(opt, destfile):
    plt.plot(opt.epoch_nums, opt.cost_train, 'r-', label='Train')
    plt.plot(opt.epoch_nums, opt.cost_val, 'b-', label='Validation')
    plt.title('Smoothed cost function')
    plt.xlabel('Sequences')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid('on')

    plt.savefig('images/' + destfile)
    plt.clf()
    plt.close()
    return 'images/' + destfile


def save_char_rnn(rnn, filename):
    np.savez_compressed(
        filename,
        **{name: weight for weight, _, name in rnn.weights_gradients_pairs()},
        input_output_size=rnn.input_size,
        state_size=rnn.state_size
    )


def restore_char_rnn(filename):
    loaded = np.load(filename)
    rnn = CharRNN(
        input_output_size=loaded['input_output_size'],
        state_size=loaded['state_size'],
        initializer_W=Zeros(),
        initializer_U=Zeros(),
        initializer_V=Zeros(),
        initializer_b=Zeros(),
        initializer_c=Zeros(),
    )
    for weight, _, name in rnn.weights_gradients_pairs():
        weight += loaded[name]
    return rnn


def id_gen(prefix=''):
    for i in itertools.count():
        yield '{}_{}'.format(prefix, i)
