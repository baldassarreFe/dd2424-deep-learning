# Disable interactive backend to train on a remote machine
import matplotlib

matplotlib.use('Agg')

import os
import pandas as pd
import numpy as np

from utils import create_and_train, id_gen


def tests():
    deep_id = id_gen('deep')

    base = {
        'hidden_layer_sizes': [50, 30],
        'decay_factor': 0.99,
        'momentum': 0.8
    }

    for bn in [False, True]:
        yield create_and_train(batch_normalized=bn,
                               regularization=0,
                               initial_learning_rate=0.01,
                               **base,
                               train_id=next(deep_id),
                               target='fine')

    # Coarse search with batch norm (random values)
    for reg in [0.000192, 0.000672, 0.003772, 0.032430, 0.266132]:
        for ilr in [0.40573, 0.029202, 0.012721, 0.003215]:
            yield create_and_train(batch_normalized=True,
                                   regularization=reg,
                                   initial_learning_rate=ilr,
                                   **base,
                                   train_id=next(deep_id),
                                   target='coarse')

    # Fine search with batch norm
    for reg in [0.012361, 0.032430, 0.053913]:
        for ilr in [0.043773, 0.029202, 0.016231, 0.001051]:
            yield create_and_train(batch_normalized=True,
                                   regularization=reg,
                                   initial_learning_rate=ilr,
                                   **base,
                                   train_id=next(deep_id),
                                   target='fine')

    # Last one
    yield create_and_train(batch_normalized=True,
                           regularization=0.0001,
                           initial_learning_rate=0.029202,
                           **base,
                           train_id=next(deep_id),
                           target='final')


np.random.seed(123)

for result in tests():
    df = pd.DataFrame([result])
    if not os.path.isfile('deep.csv'):
        df.to_csv('deep.csv', header=True, mode='w')
    else:
        df.to_csv('deep.csv', header=False, mode='a')

print_these = ['batch_norm', 'regularization', 'initial_learning_rate',
               'plot', 'final_acc_train', 'final_acc_val']
df = pd.read_csv('deep.csv')
print(df[print_these].sort_values(by='final_acc_val', ascending=False))
