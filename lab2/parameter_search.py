import itertools

# Disable interactive backend to train on a remote machine
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from tqdm import tqdm

import datasets
from utils import create_and_train

# Dataset definition
cifar = datasets.CIFAR10()
training = cifar.get_named_batches('data_batch_1')
validation = cifar.get_named_batches('data_batch_5')


def simple_create_and_train(regularization, learning_rate, iterations, plot_name):
    return create_and_train(training, validation, iterations, 50,
                            regularization, learning_rate, 0.99, 0.8, plot_name)


def coarse_search():
    # Parameter search space
    regularization = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5]
    initial_learning_rate = [0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.001]

    parameters = list(itertools.product(regularization, initial_learning_rate))

    # Training
    for i, p in enumerate(tqdm(parameters, desc='Coarse search')):
        result = create_and_train(training,
                                  validation,
                                  epochs=5,
                                  hidden_size=50,
                                  regularization=p[0],
                                  initial_learning_rate=p[1],
                                  decay_factor=0.998,
                                  momentum=.8,
                                  train_id='coarse_{}'.format(i))
        pd.DataFrame([result]).to_csv('coarse.csv', mode='a', header=False)


def fine_search():
    # Parameter search space
    regularization = np.linspace(0.005, 0.0005, num=7)
    initial_learning_rate = np.linspace(0.35, 0.25, num=4)

    parameters = list(itertools.product(regularization, initial_learning_rate))

    # Training
    for i, p in enumerate(tqdm(parameters, desc='Fine search')):
        result = create_and_train(training,
                                  validation,
                                  epochs=10,
                                  hidden_size=50,
                                  regularization=p[0],
                                  initial_learning_rate=p[1],
                                  decay_factor=0.998,
                                  momentum=.8,
                                  train_id='fine_{}'.format(i))
        pd.DataFrame([result]).to_csv('fine.csv', mode='a', header=False)


def best_net():
    training = cifar.get_named_batches('data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5')
    validation = training
    test = cifar.get_named_batches('test_batch')

    result = create_and_train(training,
                              validation,
                              epochs=50,
                              hidden_size=50,
                              regularization=0.00275,
                              initial_learning_rate=.35,
                              decay_factor=0.998,
                              momentum=.8,
                              train_id='final',
                              test=test)
    pd.DataFrame([result]).to_csv('final.csv', mode='a', header=False)


def show_results(csv):
    df = pd.read_csv(csv)
    return df[['regularization',
               'initial_learning_rate',
               'final_acc_train',
               'final_acc_val']] \
        .sort_values(by='final_acc_val', ascending=False)


def show_best_results():
    df = pd.read_csv('final.csv')
    return df[['regularization',
               'initial_learning_rate',
               'final_acc_train',
               'final_acc_val',
               'final_acc_test']]


if __name__ == '__main__':
    coarse_search()
    print(show_results('coarse.csv'))

    fine_search()
    print(show_results('fine.csv'))

    best_net()
    print(show_best_results())
