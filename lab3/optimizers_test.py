from datasets import CIFAR10
from initializers import *
from layers import *
from optimizers import *
from utils import costs_accuracies_plot_with_opt

np.random.seed(123)

cifar = CIFAR10()
training = cifar.get_named_batches('data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4')
validation = cifar.get_named_batches('data_batch_5').subset(1000)


def one_layer(regularization):
    net = Network()
    net.add_layer(Linear(CIFAR10.input_size, CIFAR10.output_size, regularization, Xavier()))
    net.add_layer(Softmax(CIFAR10.output_size))
    return net


def two_layers(regularization):
    net = Network()
    net.add_layer(Linear(CIFAR10.input_size, 50, regularization, Xavier()))
    net.add_layer(ReLU(50))
    net.add_layer(Linear(50, CIFAR10.output_size, regularization, Xavier()))
    net.add_layer(Softmax(CIFAR10.output_size))
    return net


def two_layers_batch(regularization):
    net = Network()
    net.add_layer(Linear(CIFAR10.input_size, 50, regularization, Xavier()))
    net.add_layer(BatchNormalization(50))
    net.add_layer(ReLU(50))
    net.add_layer(Linear(50, CIFAR10.output_size, regularization, Xavier()))
    net.add_layer(Softmax(CIFAR10.output_size))
    return net


def three_layers(regularization):
    net = Network()
    net.add_layer(Linear(CIFAR10.input_size, 50, regularization, Xavier()))
    net.add_layer(ReLU(50))
    net.add_layer(Linear(50, 30, regularization, Xavier()))
    net.add_layer(ReLU(30))
    net.add_layer(Linear(30, CIFAR10.output_size, regularization, Xavier()))
    net.add_layer(Softmax(CIFAR10.output_size))
    return net


def three_layers_batch(regularization):
    net = Network()
    net.add_layer(Linear(CIFAR10.input_size, 50, regularization, Xavier()))
    net.add_layer(BatchNormalization(50))
    net.add_layer(ReLU(50))
    net.add_layer(Linear(50, 30, regularization, Xavier()))
    net.add_layer(BatchNormalization(30))
    net.add_layer(ReLU(30))
    net.add_layer(Linear(30, CIFAR10.output_size, regularization, Xavier()))
    net.add_layer(Softmax(CIFAR10.output_size))
    return net


def four_layers_batch(regularization):
    net = Network()
    net.add_layer(Linear(CIFAR10.input_size, 50, regularization, Xavier()))
    net.add_layer(BatchNormalization(50))
    net.add_layer(ReLU(50))
    net.add_layer(Linear(50, 30, regularization, Xavier()))
    net.add_layer(BatchNormalization(30))
    net.add_layer(ReLU(30))
    net.add_layer(Linear(30, 15, regularization, Xavier()))
    net.add_layer(BatchNormalization(15))
    net.add_layer(ReLU(15))
    net.add_layer(Linear(15, CIFAR10.output_size, regularization, Xavier()))
    net.add_layer(Softmax(CIFAR10.output_size))
    return net


def train_vanilla(net, initial_learning_rate, decay_factor, name):
    opt = VanillaSGD(net, initial_learning_rate, decay_factor, True)
    opt.train(training, validation, epochs=15, batch_size=1000)
    name = 'images/{}_vanilla_{}.png'.format(name, initial_learning_rate)
    costs_accuracies_plot_with_opt(opt, name)


def train_momentum(net, initial_learning_rate, decay_factor, momentum, name):
    opt = MomentumSGD(net, initial_learning_rate, decay_factor, True, momentum)
    opt.train(training, validation, epochs=15, batch_size=1000)
    name = 'images/{}_momentum_{}.png'.format(name, momentum)
    costs_accuracies_plot_with_opt(opt, name)


def overfit_vanilla(net, initial_learning_rate, decay_factor, name):
    opt = VanillaSGD(net, initial_learning_rate, decay_factor, True)
    opt.train(training.subset(100), training.subset(100), epochs=400, batch_size=100)
    name = 'images/{}_overfit_vanilla_{}.png'.format(name, initial_learning_rate)
    costs_accuracies_plot_with_opt(opt, name)


def overfit_momentum(net, initial_learning_rate, decay_factor, momentum, name):
    opt = MomentumSGD(net, initial_learning_rate, decay_factor, True, momentum)
    opt.train(training.subset(100), training.subset(100), epochs=400, batch_size=100)
    name = 'images/{}_overfit_momentum_{}.png'.format(name, momentum)
    costs_accuracies_plot_with_opt(opt, name)


"""
    One layer tests
"""
train_vanilla(one_layer(regularization=0),
              initial_learning_rate=0.01,
              decay_factor=0.99, name='1L')

train_vanilla(one_layer(regularization=0.8),
              initial_learning_rate=0.01,
              decay_factor=0.99, name='1L_reg')

train_momentum(one_layer(regularization=0),
               initial_learning_rate=0.01,
               decay_factor=0.99,
               momentum=0.8, name='1L')

train_momentum(one_layer(regularization=0.8),
               initial_learning_rate=0.01,
               decay_factor=0.99,
               momentum=0.8, name='1L_reg')

overfit_momentum(one_layer(regularization=0),
                 initial_learning_rate=0.005,
                 decay_factor=1,
                 momentum=0.8, name='1L')

"""
    Two layers tests
"""
train_vanilla(two_layers(regularization=0),
              initial_learning_rate=0.01,
              decay_factor=0.99, name='2L')

train_vanilla(two_layers(regularization=0.05),
              initial_learning_rate=0.01,
              decay_factor=0.99, name='2L_reg')

train_momentum(two_layers(regularization=0),
               initial_learning_rate=0.01,
               decay_factor=0.99,
               momentum=0.8, name='2L')

train_momentum(two_layers(regularization=0.05),
               initial_learning_rate=0.01,
               decay_factor=0.99,
               momentum=0.8, name='2L_reg')

for m in [.5, .7, .9]:
    overfit_momentum(two_layers(regularization=0),
                     initial_learning_rate=0.005,
                     decay_factor=1,
                     momentum=m, name='2L')
"""
    Three layers tests
"""
train_momentum(three_layers(regularization=0),
               initial_learning_rate=0.01,
               decay_factor=0.99,
               momentum=0.8, name='3L')

"""
    Batch norm tests
"""
train_vanilla(two_layers_batch(regularization=0),
              initial_learning_rate=0.007,
              decay_factor=0.98, name='bn_2L')

train_momentum(two_layers_batch(regularization=0),
               initial_learning_rate=0.01,
               decay_factor=0.99,
               momentum=0.8, name='bn_2L')

overfit_vanilla(two_layers_batch(regularization=0),
                initial_learning_rate=0.005,
                decay_factor=1,
                name='bn_2L')

overfit_momentum(two_layers_batch(regularization=0),
                 initial_learning_rate=0.01,
                 decay_factor=1,
                 momentum=0.8, name='bn_2L')

overfit_momentum(three_layers_batch(regularization=0),
                 initial_learning_rate=0.01,
                 decay_factor=1,
                 momentum=0.8, name='bn_3L')

overfit_momentum(four_layers_batch(regularization=0),
                 initial_learning_rate=0.01,
                 decay_factor=1,
                 momentum=0.8, name='bn_4L')
