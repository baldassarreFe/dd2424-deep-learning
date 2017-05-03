# Disable interactive backend to train on a remote machine
import matplotlib

matplotlib.use('Agg')

import os
import pandas as pd

from utils import create_and_train, id_gen


def tests():
    over_id = id_gen('overfit')

    base = {
        'initial_learning_rate': 0.01,
        'decay_factor': 0.99,
        'regularization': 0,
        'target': 'overfit'
    }

    # One layer
    yield create_and_train([], False, momentum=.8, train_id=next(over_id), **base)

    # Two layers
    # With or without batch norm
    for bn in [False, True]:
        # Different momentum values
        for m in [.5, .7, .9]:
            yield create_and_train([50], batch_normalized=bn, momentum=m,
                                   train_id=next(over_id), **base)
    # Three layers
    # With or without batch norm
    for bn in [False, True]:
        yield create_and_train([50, 30], batch_normalized=bn, momentum=.8,
                               train_id=next(over_id), **base)

    # Four layers
    # With or without batch norm
    for bn in [False, True]:
        yield create_and_train([50, 30, 15], batch_normalized=bn, momentum=.8,
                               train_id=next(over_id), **base)


for result in tests():
    df = pd.DataFrame([result])
    if not os.path.isfile('overfit.csv'):
        df.to_csv('overfit.csv', header=True, mode='w')
    else:
        df.to_csv('overfit.csv', header=False, mode='a')

print_these = ['batch_norm', 'initial_learning_rate',
               'plot', 'final_acc_train', 'final_acc_val']
df = pd.read_csv('overfit.csv')
print(df[print_these].sort_values(by='final_acc_val', ascending=False))
