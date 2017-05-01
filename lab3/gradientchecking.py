from tqdm import tqdm

from datasets import CIFAR10
from initializers import *
from layers import *
from network import Network


def compute_grads_for_matrix(one_hot_targets, inputs,
                             matrix, network: Network):
    # Initialize an empty matrix to contain the gradients
    grad = np.empty_like(matrix)
    h = 1e-5

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
    print('\n{}: {:.2e}\n'.format(title, np.max(rel_err)))

    coord_worst = np.unravel_index(np.argmax(rel_err), rel_err.shape)
    if rel_err[coord_worst] > 1e-4:
        print('grad {:.3e}'.format(grad[coord_worst]))
        print('num {:.3e}'.format(grad_num[coord_worst]))
        print('diff {:.3e}'.format(err[coord_worst]))


def one_layer_no_reg():
    """
    One layer network without regularization
    """
    net = Network()
    linear = Linear(CIFAR10.input_size, CIFAR10.output_size, 0, Xavier())
    net.add_layer(linear)
    net.add_layer(Softmax(CIFAR10.output_size))

    # Run one forward and backward pass
    net.evaluate(training.images, train=True)
    net.backward(training.one_hot_labels)

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
    linear = Linear(CIFAR10.input_size, CIFAR10.output_size, 0.2, Xavier())
    net.add_layer(linear)
    net.add_layer(Softmax(CIFAR10.output_size))

    # Run one forward and backward pass
    net.evaluate(training.images, train=True)
    net.backward(training.one_hot_labels)

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


def one_layer_with_bn():
    """
    One layer network with batch normalization
    """
    net = Network()
    linear = Linear(CIFAR10.input_size, CIFAR10.output_size, 0, Xavier())
    net.add_layer(linear)
    net.add_layer(BatchNormalization(CIFAR10.output_size))
    net.add_layer(Softmax(CIFAR10.output_size))

    # Run one forward and backward pass
    net.evaluate(training.images, train=True)
    net.backward(training.one_hot_labels)

    # Weights matrix
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear.W, net)
    print_grad_diff(linear.grad_W, grad_num, '1L bn W1')

    # Biases matrix
    grad_num = compute_grads_for_matrix(training.one_hot_labels,
                                        training.images,
                                        linear.b, net)
    print_grad_diff(linear.grad_b, grad_num, '1L bn b1')


def two_layer_with_reg():
    # Two layer network with regularization
    net = Network()
    linear1 = Linear(CIFAR10.input_size, 15, 0.1, Xavier(), name='Linear 1')
    net.add_layer(linear1)
    net.add_layer(ReLU(15))
    linear2 = Linear(15, CIFAR10.output_size, 0.3, Xavier(), name='Linear 2')
    net.add_layer(linear2)
    net.add_layer(Softmax(CIFAR10.output_size))

    # Run one forward and backward pass
    net.evaluate(training.images, train=True)
    net.backward(training.one_hot_labels)

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
    linear1 = Linear(CIFAR10.input_size, 15, 0, CenteredNormal(0.001), name='Linear 1')
    net.add_layer(linear1)
    net.add_layer(BatchNormalization(15))
    net.add_layer(ReLU(15))
    linear2 = Linear(15, CIFAR10.output_size, 0, CenteredNormal(0.1), name='Linear 2')
    net.add_layer(linear2)
    net.add_layer(Softmax(CIFAR10.output_size))

    # Run one forward and backward pass
    net.evaluate(training.images, train=True)
    net.backward(training.one_hot_labels)

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
    linear1 = Linear(CIFAR10.input_size, 25, 0, Xavier(), name='Linear 1')
    net.add_layer(linear1)
    net.add_layer(BatchNormalization(25))
    net.add_layer(ReLU(25))
    linear2 = Linear(25, 15, 0, Xavier(), name='Linear 2')
    net.add_layer(linear2)
    net.add_layer(BatchNormalization(15))
    net.add_layer(ReLU(15))
    linear3 = Linear(15, CIFAR10.output_size, 0, Xavier(), name='Linear 3')
    net.add_layer(linear3)
    net.add_layer(Softmax(CIFAR10.output_size))

    # Run one forward and backward pass
    net.evaluate(training.images, train=True)
    net.backward(training.one_hot_labels)

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
    linear1 = Linear(CIFAR10.input_size, 30, 0, Xavier(), name='Linear 1')
    net.add_layer(linear1)
    net.add_layer(BatchNormalization(30))
    net.add_layer(ReLU(30))
    linear2 = Linear(30, 20, 0.1, Xavier(), name='Linear 2')
    net.add_layer(linear2)
    net.add_layer(BatchNormalization(20))
    net.add_layer(ReLU(20))
    linear3 = Linear(20, 15, 0, Xavier(), name='Linear 3')
    net.add_layer(linear3)
    net.add_layer(BatchNormalization(15))
    net.add_layer(ReLU(15))
    linear4 = Linear(15, CIFAR10.output_size, 0, Xavier(), name='Linear 4')
    net.add_layer(linear4)
    net.add_layer(Softmax(CIFAR10.output_size))

    # Run one forward and backward pass
    net.evaluate(training.images, train=True)
    net.backward(training.one_hot_labels)

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


def batch_norm_grad_slow(bnl: BatchNormalization, gradients):
    mu = bnl.batch_mean.ravel()
    v = bnl.batch_var.ravel()
    v[v == 0] = np.finfo(float).eps
    V_m1_2 = np.diag(v ** -0.5)
    V_m3_2 = np.diag(v ** -1.5)

    D, N = bnl.X.shape

    dJdv = np.zeros(D)
    for i in range(N):
        x = bnl.X[:, i]
        g = gradients[i, :]
        d = np.diag(x - mu)
        t1 = np.dot(g, V_m3_2)
        t2 = np.dot(t1, d)
        dJdv += t2
    dJdv /= -2

    dJdmu = np.zeros(D)
    for i in range(N):
        g = gradients[i, :]
        t = np.dot(g, V_m1_2)
        dJdmu += t
    dJdmu *= -1

    grr = np.empty((N, D))
    for i in range(N):
        x = bnl.X[:, i]
        g = gradients[i, :]
        d = np.diag(x - mu)

        z1 = np.dot(g, V_m1_2)
        z2 = (2 / N) * np.dot(dJdv, d)
        z3 = (1 / N) * dJdmu

        grr[i, :] = z1 + z2 + z3

    return grr


def test_bn():
    # 2 dimensional input, 4 samples
    X = np.array([
        [1, 7, 3, 5],
        [0, 12, 4, 8],
    ])
    bn = BatchNormalization(input_size=X.shape[0])
    res = bn.forward(X, train=True)

    assert np.allclose(res.mean(axis=1, keepdims=True), 0)
    assert np.allclose(res.var(axis=1, keepdims=True), 1)

    # Each row is relative to the corresponding
    # column in X
    gradients = np.array([
        [0, 1],
        [3, 4],
        [6, 7],
        [9, 2],
    ])

    grr = batch_norm_grad_slow(bn, gradients)
    backprop = bn.backward(gradients)

    assert np.allclose(backprop, grr), \
        'Different\n' + str(grr) + '\n' + str(backprop)
    print('Passed')

if __name__ == '__main__':
    cifar = CIFAR10()
    training = cifar.get_named_batches('data_batch_1').subset(20)

    np.random.seed(123)

    # one_layer_no_reg()
    # one_layer_with_bn()
    # one_layer_with_reg()

    # two_layer_with_reg()
    two_layer_with_bn()

    # three_layer_with_bn()

    # four_layer_with_bn()

    test_bn()
