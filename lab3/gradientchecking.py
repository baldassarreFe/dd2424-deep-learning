import numpy as np
from tqdm import tqdm

import datasets
import initializers
import layers
from network import Network


def compute_grads_for_matrix(one_hot_targets, inputs,
                             matrix, network: Network):
    # Initialize an empty matrix to contain the gradients
    grad = np.zeros_like(matrix)
    h = 1e-6

    # Iterate over the matrix changing one entry at the time
    desc = 'Gradient computations for a {} matrix, {} samples' \
        .format(matrix.shape, inputs.shape[1])
    with tqdm(desc=desc, total=matrix.size) as progress_bar:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i, j] += h
                plus_cost = network.cost(one_hot_targets, inputs)
                matrix[i, j] -= 2 * h
                minus_cost = network.cost(one_hot_targets, inputs)
                grad[i, j] = (plus_cost - minus_cost) / (2 * h)
                matrix[i, j] += h
                progress_bar.update()
    return grad


def print_grad_diff(grad, grad_num, title='Gradient difference'):
    err = np.abs(grad - grad_num)
    rel_err = err / np.maximum(np.finfo('float').eps, np.abs(grad) + np.abs(grad_num))
    coord_worst = np.unravel_index(np.argmax(rel_err), rel_err.shape)
    print('\n{}: {:.2e}\n'.format(title, np.max(rel_err)))
    if np.max(rel_err) > 1e-4:
        print('grad {:.3e}'.format(grad[coord_worst]))
        print('num {:.3e}'.format(grad_num[coord_worst]))
        print('diff {:.3e}'.format(err[coord_worst]))


def one_layer_no_reg():
    """
    One layer network without regularization
    """
    net = Network()
    linear = layers.Linear(cifar.input_size, cifar.output_size, 0, initializers.Xavier())
    net.add_layer(linear)
    net.add_layer(layers.Softmax(cifar.output_size))
    outputs = net.evaluate(training.images)
    net.backward(training.one_hot_labels)
    cost = net.cost(training.one_hot_labels, outputs=outputs)
    # Weights matrix
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear.W, net)
    print_grad_diff(linear.grad_W, grad_num, '1L W1')
    # Biases matrix
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear.b, net)
    print_grad_diff(linear.grad_b, grad_num, '1L b1')


def one_layer_with_reg():
    """
    One layer network with regularization
    """
    net = Network()
    linear = layers.Linear(cifar.input_size, cifar.output_size, 0.2, initializers.Xavier())
    net.add_layer(linear)
    net.add_layer(layers.Softmax(cifar.output_size))
    outputs = net.evaluate(training.images)
    net.backward(training.one_hot_labels)
    cost = net.cost(training.one_hot_labels, outputs=outputs)
    # Weights matrix
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear.W, net)
    print_grad_diff(linear.grad_W, grad_num, '1L reg W1')
    # Biases matrix
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear.b, net)
    print_grad_diff(linear.grad_b, grad_num, '1L reg b1')


def two_layer_with_reg():
    # Two layer network with regularization
    net = Network()
    linear1 = layers.Linear(cifar.input_size, 15, 0.1, initializers.Xavier(), name='Linear 1')
    net.add_layer(linear1)
    net.add_layer(layers.ReLU(15))
    linear2 = layers.Linear(15, cifar.output_size, 0.3, initializers.Xavier(), name='Linear 2')
    net.add_layer(linear2)
    net.add_layer(layers.Softmax(cifar.output_size))
    outputs = net.evaluate(training.images)
    net.backward(training.one_hot_labels)
    cost = net.cost(training.one_hot_labels, outputs=outputs)
    # Weights matrix, layer 1
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear1.W, net)
    print_grad_diff(linear1.grad_W, grad_num, '2L reg W1')
    # Biases matrix, layer 1
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear1.b, net)
    print_grad_diff(linear1.grad_b, grad_num, '2L reg b1')
    # Weights matrix, layer 2
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear2.W, net)
    print_grad_diff(linear2.grad_W, grad_num, '2L reg W2')
    # Biases matrix, layer 2
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear2.b, net)
    print_grad_diff(linear2.grad_b, grad_num, '2L reg b2')


def two_layer_with_bn():
    # Two layer network without regularization and with batch normalization
    net = Network()
    linear1 = layers.Linear(cifar.input_size, 15, 0.1, initializers.Xavier(), name='Linear 1')
    net.add_layer(linear1)
    net.add_layer(layers.BatchNormalization(15))
    net.add_layer(layers.ReLU(15))
    linear2 = layers.Linear(15, cifar.output_size, 0.3, initializers.Xavier(), name='Linear 2')
    net.add_layer(linear2)
    net.add_layer(layers.Softmax(cifar.output_size))

    outputs = net.evaluate(training.images)
    net.backward(training.one_hot_labels)
    cost = net.cost(training.one_hot_labels, outputs=outputs)

    # Weights matrix, layer 1
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear1.W, net)
    print_grad_diff(linear1.grad_W, grad_num, '2L bn W1')

    # Biases matrix, layer 1
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear1.b, net)
    print_grad_diff(linear1.grad_b, grad_num, '2L bn b1')

    # Weights matrix, layer 2
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear2.W, net)
    print_grad_diff(linear2.grad_W, grad_num, '2L bn W2')

    # Biases matrix, layer 2
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear2.b, net)
    print_grad_diff(linear2.grad_b, grad_num, '2L bn b2')


def three_layer_with_bn():
    # Three layer network without regularization and with batch normalization
    net = Network()
    linear1 = layers.Linear(cifar.input_size, 25, 0.1, initializers.Xavier(), name='Linear 1')
    net.add_layer(linear1)
    net.add_layer(layers.BatchNormalization(25))
    net.add_layer(layers.ReLU(25))
    linear2 = layers.Linear(25, 15, 0.1, initializers.Xavier(), name='Linear 2')
    net.add_layer(linear2)
    net.add_layer(layers.BatchNormalization(15))
    net.add_layer(layers.ReLU(15))
    linear3 = layers.Linear(15, cifar.output_size, 0.3, initializers.Xavier(), name='Linear 3')
    net.add_layer(linear3)
    net.add_layer(layers.Softmax(cifar.output_size))

    outputs = net.evaluate(training.images)
    net.backward(training.one_hot_labels)
    cost = net.cost(training.one_hot_labels, outputs=outputs)

    # Weights matrix, layer 1
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear1.W, net)
    print_grad_diff(linear1.grad_W, grad_num, '3L bn W1')

    # Biases matrix, layer 1
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear1.b, net)
    print_grad_diff(linear1.grad_b, grad_num, '3L bn b1')

    # Weights matrix, layer 2
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear2.W, net)
    print_grad_diff(linear2.grad_W, grad_num, '3L bn W2')

    # Biases matrix, layer 2
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear2.b, net)
    print_grad_diff(linear2.grad_b, grad_num, '3L bn b2')

    # Weights matrix, layer 3
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear3.W, net)
    print_grad_diff(linear3.grad_W, grad_num, '3L bn W3')

    # Biases matrix, layer 3
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear3.b, net)
    print_grad_diff(linear3.grad_b, grad_num, '3L bn b3')


def four_layer_with_bn():
    # Four layer network without regularization and with batch normalization
    net = Network()
    linear1 = layers.Linear(cifar.input_size, 30, 0.1, initializers.Xavier(), name='Linear 1')
    net.add_layer(linear1)
    net.add_layer(layers.BatchNormalization(30))
    net.add_layer(layers.ReLU(30))
    linear2 = layers.Linear(30, 20, 0.1, initializers.Xavier(), name='Linear 2')
    net.add_layer(linear2)
    net.add_layer(layers.BatchNormalization(20))
    net.add_layer(layers.ReLU(20))
    linear3 = layers.Linear(20, 15, 0.1, initializers.Xavier(), name='Linear 3')
    net.add_layer(linear3)
    net.add_layer(layers.BatchNormalization(15))
    net.add_layer(layers.ReLU(15))
    linear4 = layers.Linear(15, cifar.output_size, 0.3, initializers.Xavier(), name='Linear 4')
    net.add_layer(linear4)
    net.add_layer(layers.Softmax(cifar.output_size))

    outputs = net.evaluate(training.images)
    net.backward(training.one_hot_labels)
    cost = net.cost(training.one_hot_labels, outputs=outputs)

    # Weights matrix, layer 1
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear1.W, net)
    print_grad_diff(linear1.grad_W, grad_num, '4L bn W1')

    # Biases matrix, layer 1
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear1.b, net)
    print_grad_diff(linear1.grad_b, grad_num, '4L bn b1')

    # Weights matrix, layer 2
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear2.W, net)
    print_grad_diff(linear2.grad_W, grad_num, '4L bn W2')

    # Biases matrix, layer 2
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear2.b, net)
    print_grad_diff(linear2.grad_b, grad_num, '4L bn b2')

    # Weights matrix, layer 3
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear3.W, net)
    print_grad_diff(linear3.grad_W, grad_num, '4L bn W3')

    # Biases matrix, layer 3
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear3.b, net)
    print_grad_diff(linear3.grad_b, grad_num, '4L bn b3')

    # Weights matrix, layer 4
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear4.W, net)
    print_grad_diff(linear4.grad_W, grad_num, '4L bn W4')

    # Biases matrix, layer 4
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear4.b, net)
    print_grad_diff(linear4.grad_b, grad_num, '4L bn b4')


if __name__ == '__main__':
    cifar = datasets.CIFAR10()
    training = cifar.get_named_batches('data_batch_1').subset(20)

    one_layer_no_reg()
    one_layer_with_reg()

    two_layer_with_reg()
    two_layer_with_bn()

    three_layer_with_bn()

    four_layer_with_bn()
