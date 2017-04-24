import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import confusion_matrix

import initializers
import layers
import optimizers
from datasets import Batch, CIFAR10
from network import Network

plt.rcParams['figure.figsize'] = (14.0, 8.0)

if not os.path.isdir('images'):
    os.mkdir('images')
    with open('.gitignore', 'a+') as out:
        out.write('images\n')


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
    plt.savefig(destfile)
    plt.clf()
    plt.close()
    return destfile


def costs_accuracies_plot(epoch_nums, acc_train, acc_val, cost_train, cost_val, destfile):
    plt.subplot(1, 2, 1)
    plt.plot(epoch_nums, cost_train, 'r-', label='Train')
    plt.plot(epoch_nums, cost_val, 'b-', label='Validation')
    plt.title('Cost function')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid('on')

    plt.subplot(1, 2, 2)
    plt.plot(epoch_nums, acc_train, label='Train')
    plt.plot(epoch_nums, acc_val, label='Validation')
    plt.title('Accuracy')
    plt.ylim(0, 1 if max(acc_train + acc_train) > .5 else 0.5)
    plt.gca().yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid('on')

    plt.savefig(destfile)
    plt.clf()
    plt.close()
    return destfile


def show_plot(plot_file):
    plot_img = plt.imread(plot_file)
    plt.imshow(plot_img)
    plt.axis('off')
    plt.show()


def cost_accuracy_string(net: Network, training: Batch, validation: Batch):
    Y = net.evaluate(training.images)
    accuracy = net.accuracy(training.one_hot_labels, outputs=Y)
    cost = net.cost(training.one_hot_labels, outputs=Y)
    res = 'Training:\n' + \
          '- accuracy: {:.3%}\n'.format(accuracy) + \
          '- cost:     {:.3f}\n'.format(cost)

    Y = net.evaluate(validation.images)
    accuracy = net.accuracy(validation.one_hot_labels, outputs=Y)
    cost = net.cost(validation.one_hot_labels, outputs=Y)
    res += 'Validation:\n' + \
           '- accuracy: {:.3%}\n'.format(accuracy) + \
           '- cost:     {:.3f}\n'.format(cost)
    return res


def create_and_train(training: Batch,
                     validation: Batch,
                     epochs: int,
                     hidden_size: int,
                     regularization: float,
                     initial_learning_rate: float,
                     decay_factor: float,
                     momentum: float,
                     train_id: str,
                     test: Batch = None):
    """
    Create and train a 2 layer network:
    - subtract mean of the training set
    - linear layer
    - relu
    - linear layer
    - softmax

    The only parameters that are fixed are the layer initializers
    and the batch size.

    :param train_id:
    :param training:
    :param validation:
    :param epochs:
    :param hidden_size:
    :param regularization:
    :param initial_learning_rate:
    :param decay_factor:
    :param momentum:
    :return:
    """
    # Mean of the training set
    mu = training.mean()

    # Definition of the network
    net = Network()
    net.add_layer(layers.BatchNormalization(CIFAR10.input_size, mu))
    net.add_layer(layers.Linear(CIFAR10.input_size, hidden_size, regularization, initializers.Xavier()))
    net.add_layer(layers.ReLU(hidden_size))
    net.add_layer(layers.Linear(hidden_size, CIFAR10.output_size, regularization, initializers.Xavier()))
    net.add_layer(layers.Softmax(CIFAR10.output_size))

    # Training
    opt = optimizers.MomentumSGD(net, initial_learning_rate, decay_factor, True, momentum)
    opt.train(training, validation, epochs, 10000)

    # Plotting
    plot = costs_accuracies_plot(opt.epoch_nums, opt.acc_train, opt.acc_val,
                                 opt.cost_train, opt.cost_val,
                                 'images/{}.png'.format(train_id))

    result = {
        'epochs': epochs,
        'hidden_size': hidden_size,
        'regularization': regularization,
        'initial_learning_rate': initial_learning_rate,
        'decay_factor': decay_factor,
        'momentum': momentum,
        # 'net': net,
        # 'opt': opt,
        'epoch_nums': opt.epoch_nums,
        'cost_train': opt.cost_train,
        'acc_train': opt.acc_train,
        'cost_val': opt.cost_val,
        'acc_val': opt.acc_val,
        'final_cost_train': opt.cost_train[-1],
        'final_acc_train': opt.acc_train[-1],
        'final_cost_val': opt.cost_val[-1],
        'final_acc_val': opt.acc_val[-1],
        'plot': plot
    }

    # Test set
    if test is not None:
        result['final_cost_test'], result['final_acc_test'] = net.cost_accuracy(test)
        result['confusion_matrix'] = confusion_matrix_plot(net, test,
                                                           CIFAR10().labels,
                                                           'images/{}_conf.png'.format(train_id))

    return result
