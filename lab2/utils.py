import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from datasets import Batch
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


def costs_accuracies_plot(acc_train, acc_val, cost_train, cost_val, destfile):
    plt.subplot(1, 2, 1)
    plt.plot(cost_train, 'r-', label='Train')
    plt.plot(cost_val, 'b-', label='Validation')
    plt.title('Cost function')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid('on')

    plt.subplot(1, 2, 2)
    plt.plot(acc_train, label='Train')
    plt.plot(acc_val, label='Validation')
    plt.title('Accuracy')
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
