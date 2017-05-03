# Disable interactive backend to train on a remote machine
import matplotlib

matplotlib.use('Agg')

import os
import pandas as pd
import numpy as np

from utils import create_and_train, id_gen


def tests():
    fast_id = id_gen('fast')

    base = {
        'hidden_layer_sizes': [50],
        'decay_factor': 0.99,
        'regularization': 0.0001,
        'momentum': 0.8,
        'target': 'fine'
    }

    for bn in [False, True]:
        for ilr in [0.40573, 0.029202, 0.003215]:
            yield create_and_train(batch_normalized=bn,
                                   initial_learning_rate=ilr,
                                   **base,
                                   train_id=next(fast_id))


np.random.seed(123)

for result in tests():
    df = pd.DataFrame([result])
    if not os.path.isfile('fast.csv'):
        df.to_csv('fast.csv', header=True, mode='w')
    else:
        df.to_csv('fast.csv', header=False, mode='a')

print_these = ['batch_norm', 'initial_learning_rate',
               'plot', 'final_acc_train', 'final_acc_val']
df = pd.read_csv('fast.csv')
print(df[print_these].sort_values(by='final_acc_val', ascending=False))
