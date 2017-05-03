import itertools
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import confusion_matrix

from datasets import Batch, CIFAR10
from initializers import Xavier
from layers import Linear, Softmax, BatchNormalization, ReLU
from network import Network
from optimizers import MomentumSGD

# Misc. setup
plt.rcParams['figure.figsize'] = (14.0, 8.0)

if not os.path.isdir('images'):
    os.mkdir('images')
    with open('.gitignore', 'a+') as out:
        out.write('images\n')

np.random.seed(123)
cifar = CIFAR10()
training = cifar.get_named_batches('data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4')
validation = cifar.get_named_batches('data_batch_5')
test = cifar.get_named_batches('test_batch')


def build_network(hidden_layer_sizes: List[int], batch_normalized: bool, regularization: float) -> Network:
    net = Network()
    layer_sizes = [CIFAR10.input_size] + hidden_layer_sizes + [CIFAR10.output_size]
    for i, (size_in, size_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        net.add_layer(Linear(size_in, size_out, regularization, Xavier(), name='Li' + str(i + 1)))
        if i < len(layer_sizes) - 2:
            if batch_normalized:
                net.add_layer(BatchNormalization(size_out, name='Bn' + str(i + 1)))
            net.add_layer(ReLU(size_out, name='Re' + str(i + 1)))
        else:
            net.add_layer(Softmax(size_out, name='S'))
    return net


def confusion_matrix_plot(net: Network, dataset: Batch, labels, destfile):
    Y = net.evaluate(dataset.images)
    cm = confusion_matrix(y_true=dataset.numeric_labels, y_pred=np.argmax(Y, axis=0))
    # print(cm)

    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation='vertical')
    plt.yticks(tick_marks, labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('images/' + destfile)
    plt.clf()
    plt.close()
    return 'images/' + destfile


def costs_accuracies_plot(opt, destfile):
    plt.subplot(1, 2, 1)
    plt.plot(opt.epoch_nums, opt.cost_train, 'r-', label='Train')
    plt.plot(opt.epoch_nums, opt.cost_val, 'b-', label='Validation')
    plt.title('Cost function')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid('on')

    plt.subplot(1, 2, 2)
    plt.plot(opt.epoch_nums, opt.acc_train, label='Train')
    plt.plot(opt.epoch_nums, opt.acc_val, label='Validation')
    plt.title('Accuracy')
    plt.ylim(0, 1 if max(opt.acc_train + opt.acc_train) > .5 else 0.5)
    plt.gca().yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid('on')

    plt.savefig('images/' + destfile)
    plt.clf()
    plt.close()
    return 'images/' + destfile


def id_gen(prefix=''):
    for i in itertools.count():
        yield '{}_{}'.format(prefix, i)


def create_and_train(hidden_layer_sizes: List[int],
                     batch_normalized: bool,
                     regularization: float,
                     initial_learning_rate: float,
                     decay_factor: float,
                     momentum: float,
                     train_id: str,
                     target: str):
    # Build network
    net = build_network(hidden_layer_sizes, batch_normalized, regularization)
    network_desc = {
        'hidden_layers': hidden_layer_sizes,
        'batch_norm': batch_normalized,
        'regularization': regularization
    }

    # Train
    opt = MomentumSGD(net, initial_learning_rate, decay_factor, True, momentum)
    if target == 'coarse':
        tr = training.subset(5000)
        val = validation.subset(1000)
        epochs = 10
    elif target == 'fine':
        tr = training.subset(10000)
        val = validation.subset(1000)
        epochs = 15
    elif target == 'overfit':
        tr = val = training.subset(100)
        epochs = 400
    else:
        tr = training
        val = validation
        epochs = 20

    opt.train(tr, val, epochs, 500)
    training_desc = {
        'epochs': epochs,
        'initial_learning_rate': initial_learning_rate,
        'decay_factor': decay_factor,
        'momentum': momentum,
        'final_cost_train': opt.cost_train[-1],
        'final_acc_train': opt.acc_train[-1],
        'final_cost_val': opt.cost_val[-1],
        'final_acc_val': opt.acc_val[-1],
        'plot': costs_accuracies_plot(opt, '{}.png'.format(train_id))
    }

    # Use the test dataset
    test_res = {
        'final_cost_test': 0,
        'final_acc_test': 0,
        'confusion_matrix': ''
    }
    if target == 'final':
        test_res['final_cost_test'], test_res['final_acc_test'] = net.cost_accuracy(test)
        test_res['confusion_matrix'] = confusion_matrix_plot(net, test,
                                                             CIFAR10().labels,
                                                             '{}_conf.png'.format(train_id))

    return {**network_desc, **training_desc, **test_res}
