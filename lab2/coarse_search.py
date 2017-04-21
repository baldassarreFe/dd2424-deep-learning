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
    return create_and_train(training, validation, 1, 50,
                            regularization, learning_rate, 0.99, 0.8, plot_name)

# Parameter search space
regularization = [0.0, 0.005, 0.01, 0.05, 0.1, 0.3]
initial_learning_rate = [0.1, 0.05, 0.01, 0.005, 0.001]

parameters = list(itertools.product(regularization, initial_learning_rate))

# Training
tasks = (coarse_create_and_train(*p, 'coarse_{}'.format(i))
         for i, p in enumerate(tqdm(parameters, desc='Coarse search')))
results = pd.DataFrame(tasks)

print(results)

results.to_csv('coarse.csv')
