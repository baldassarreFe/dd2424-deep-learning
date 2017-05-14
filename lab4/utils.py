import itertools
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from datasets import Goblet
from initializers import Xavier, Zeros
from network import CharRNN
from optimizers import MomentumSGD


def build_network():
    return CharRNN(
        input_output_size=Goblet.num_classes,
        state_size=100,
        initializer_W=Xavier(),
        initializer_U=Xavier(),
        initializer_V=Xavier(),
        initializer_b=Zeros(),
        initializer_c=Zeros()
    )


def generate_text(rnn, goblet, first_char=None, initial_state=None, length=200):
    if first_char is None:
        first_char = np.random.choice(list('I am lord Voldemort'))
    x = goblet.encode(first_char)
    if initial_state is None:
        initial_state = np.zeros(rnn.state_size)
    seq, last_state = rnn.generate(x, initial_state, length)
    return goblet.decode_to_strings(seq), last_state


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


def id_gen(prefix=''):
    for i in itertools.count():
        yield '{}_{}'.format(prefix, i)