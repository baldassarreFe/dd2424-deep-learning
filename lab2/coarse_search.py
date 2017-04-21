import itertools

import pandas as pd
from tqdm import tqdm

import datasets
from utils import create_and_train

# Dataset definition
cifar = datasets.CIFAR10()
training = training = cifar.get_named_batches('data_batch_1')
validation = cifar.get_named_batches('data_batch_5')


def coarse_create_and_train(regularization, learning_rate, plot_name):
    return create_and_train(training, validation, 5, 50,
                            regularization, learning_rate, 0.99, 0.8, plot_name)

# Parameter search space
regularization = [0.0, 0.001, 0.003, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.3, 0.5]
initial_learning_rate = [0.2, 0.1, 0.06, 0.03, 0.01, 0.008, 0.006, 0.004, 0.001, 0.0005]

parameters = list(itertools.product(regularization, initial_learning_rate))

# Training
tasks = (coarse_create_and_train(*p, 'coarse_{}'.format(i))
         for i, p in enumerate(tqdm(parameters, desc='Coarse search')))
results = pd.DataFrame(tasks)

print(results)

results.to_csv('coarse.csv')
