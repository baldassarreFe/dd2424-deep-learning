import numpy as np
from tqdm import tqdm

from network import Network


def compute_grads_for_matrix(one_hot_targets, inputs,
                             matrix, network: Network,
                             initial_cost):
    # Initialize an empty matrix to contain the gradients
    grad = np.zeros_like(matrix)
    h = 1e-6  # np.finfo(float).eps

    # Iterate over the matrix changing one entry at the time
    desc = 'Gradient computations for a {} matrix, {} samples' \
        .format(matrix.shape, inputs.shape[1])
    with tqdm(desc=desc, total=matrix.size) as progress_bar:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i, j] += h
                final_cost = network.cost(one_hot_targets, inputs)
                grad[i, j] = (final_cost - initial_cost) / h
                matrix[i, j] -= h
                progress_bar.update()
    return grad


def print_grad_diff(grad, grad_num, title='Gradient difference'):
    print(title)
    print('- sum of abs differences: {:.3e}'.format(np.abs(grad - grad_num).sum()))
    print('- mean of abs values   grad: {:.3e}   grad_num: {:.3e}'
          .format(np.abs(grad).mean(), np.abs(grad_num).mean()))
    print('- min  of abs values   grad: {:.3e}   grad_num: {:.3e}'
          .format(np.abs(grad).min(), np.abs(grad_num).min()))
    print('- max  of abs values   grad: {:.3e}   grad_num: {:.3e}'
          .format(np.abs(grad).max(), np.abs(grad_num).max()))


if __name__ == '__main__':
    import layers
    import initializers
    import datasets

    cifar = datasets.CIFAR10()
    training = cifar.get_named_batches('data_batch_1').subset(50)
    
    # One layer network with regularization
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
                                        linear.W, net, cost)
    print_grad_diff(linear.grad_W, grad_num, 'Grad W')

    # Biases matrix
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear.b, net, cost)
    print_grad_diff(linear.grad_b, grad_num, 'Grad b')

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
                                        linear1.W, net, cost)
    print_grad_diff(linear1.grad_W, grad_num, 'Grad W1')

    # Biases matrix, layer 1
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear1.b, net, cost)
    print_grad_diff(linear1.grad_b, grad_num, 'Grad b1')

    # Weights matrix, layer 2
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear2.W, net, cost)
    print_grad_diff(linear2.grad_W, grad_num, 'Grad W2')

    # Biases matrix, layer 2
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear2.b, net, cost)
    print_grad_diff(linear2.grad_b, grad_num, 'Grad b2')
