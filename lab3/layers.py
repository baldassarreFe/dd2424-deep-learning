from abc import ABC, abstractmethod

import numpy as np

import initializers
from initializers import Initializer

np.seterr(over='raise')


class Layer(ABC):
    def __init__(self, input_size, output_size, name):
        # Layer parameters
        self.input_size = input_size
        self.output_size = output_size
        self.name = name

    @abstractmethod
    def forward(self, X):
        """
        Propagates forward the activations in the network.
        :param X: a (input size x N ) matrix containing the N inputs
                  for this layer (each column corresponds to a sample)
        :return: a (output size x N ) matrix containing the N outputs
                 of this layer (each column corresponds to a sample)
        """
        pass

    @abstractmethod
    def backward(self, gradients):
        """
        Propagates back the gradients received from the following layer
        in the network.
        :param gradients: a (N x output size) matrix containing the N
                          gradients from the following layer in the network,
                          one gradient per row
        :return: a (N x input size) matrix containing the N gradients propagated
                 back from this layer, one gradient per row
        """
        pass

    def cost(self):
        return 0

    def __str__(self):
        return '{} (in: {}, out: {})' \
            .format(self.name, self.input_size, self.output_size)


class Linear(Layer):
    def __init__(self, input_size, output_size, weight_regularization,
                 weight_initializer: Initializer = None,
                 bias_initializer: Initializer = None,
                 name='Linear'):
        # Hyper parameters
        super().__init__(input_size, output_size, name)
        self.weight_regularization = weight_regularization

        # Trainable weights
        weight_initializer = weight_initializer or initializers.Zeros()
        bias_initializer = bias_initializer or initializers.Zeros()
        self.W = weight_initializer.new_matrix((self.output_size, self.input_size))
        self.b = bias_initializer.new_matrix((self.output_size, 1))

        # Gradients
        # - d loss / d W
        # - d loss / d b
        self.grad_W = np.empty_like(self.W, dtype=float)
        self.grad_b = np.empty_like(self.b, dtype=float)

        # Dummy input for back propagation
        self.X = np.empty(shape=(self.input_size, 1))

    def forward(self, X):
        # Store input for back propagation
        assert X.shape[0] == self.input_size
        self.X = X

        # Compute output: every column of the output is the
        # linear transformation Wx+b of the corresponding column
        # in the input
        return np.dot(self.W, X) + self.b

    def backward(self, gradients):
        # Size of the mini batch
        assert self.X.shape[1] == gradients.shape[0]
        N = self.X.shape[1]

        # Reset gradients
        self.grad_W = np.zeros_like(self.W, dtype=float)
        self.grad_b = np.zeros_like(self.b, dtype=float)

        # Compute gradients for every sample in the batch
        for i in range(N):
            x = self.X[:, i]
            g = gradients[i, :]

            self.grad_W += np.outer(g, x)
            self.grad_b += np.reshape(g, self.grad_b.shape)

        self.grad_W = self.grad_W / N + 2 * self.weight_regularization * self.W
        self.grad_b /= N

        # Propagate back the gradients
        return np.dot(gradients, self.W)

    def cost(self):
        return self.weight_regularization * np.power(self.W, 2).sum()


class ReLU(Layer):
    def __init__(self, input_size, name='ReLU'):
        # Hyper parameters
        super().__init__(input_size, input_size, name)

        # Dummy input for back propagation
        self.X = np.empty(shape=(self.input_size, 1))

    def forward(self, X):
        # Store input for back propagation
        assert X.shape[0] == self.input_size
        self.X = X

        # Compute output: every column in the output is the element wise
        # operation max(0, x), where x is the corresponding column
        # in the input
        return X * (X > 0)

    def backward(self, gradients):
        # Size of the mini batch
        assert self.X.shape[1] == gradients.shape[0]

        return np.multiply(gradients, self.X.T > 0)


class Softmax(Layer):
    def __init__(self, input_size, name='Softmax'):
        # Hyper parameters
        super().__init__(input_size, input_size, name)

        # Dummy output probabilities for back propagation
        self.P = np.empty(shape=(self.input_size, 1))

    def forward(self, X):
        assert X.shape[0] == self.input_size
        # Compute output: every column of the output is the
        # softmax of the corresponding column in the input
        try:
            e = np.exp(X)
            P = e / np.sum(e, axis=0)
        except FloatingPointError:
            # What happens here is that we get some of the values in x
            # that are too big so the exp overflows.
            # In this case we do the softmax operation column by column
            # to see where the problem is.
            P = self.softmax_column_by_column(X)

        # Store output probabilities for back propagation
        self.P = P

        return P

    def backward(self, one_hot_targets):
        # Size of the mini batch
        assert self.P.shape[1] == one_hot_targets.shape[1]
        N = one_hot_targets.shape[1]

        output_target_pairs = ((one_hot_targets[:, i], self.P[:, i]) for i in range(N))
        gradients = (self.softmax_gradient(y, p) for (y, p) in output_target_pairs)

        # Propagate back the gradients
        return np.vstack(gradients)

    def cost(self, one_hot_targets, P=None):
        # Size of the mini batch
        if P is None:
            P = self.P
        assert P.shape[1] == one_hot_targets.shape[1]
        N = one_hot_targets.shape[1]

        # This is element wise multiplication
        log_arg = np.multiply(one_hot_targets, P).sum(axis=0)
        log_arg[log_arg == 0] = np.finfo(float).eps

        return - np.log(log_arg).sum() / N

    def softmax_column_by_column(self, X):
        # Size of the mini batch
        N = X.shape[1]

        # Initialize result as a matrix of eps
        res = np.full_like(X, fill_value=np.finfo(float).eps)

        # For every sample try computing a softmax,
        # if this does not work, that column will be filled
        # with eps, except one value that make it sum to 1
        for i in range(N):
            x = X[:, i]
            try:
                e = np.exp(x)
                res[:, i] = e / e.sum()
            except FloatingPointError:
                res[np.argmax(x), :] = 1 - (self.input_size - 1) * np.finfo(float).eps

        # In this result every column sums to 1
        return res

    @staticmethod
    def softmax_gradient(y, p):
        # The actual formula for computing the gradient
        # g = - np.dot(y, (np.diag(p) - np.outer(p, p))) / np.dot(y, p)

        # Computations broken down for debugging
        t1 = np.outer(p, p)
        t2 = np.dot(y, (np.diag(p) - t1))
        t3 = np.dot(y, p)
        if t3 == 0:
            t3 = np.finfo(float).eps

        return - t2 / t3


class BatchNormalization(Layer):
    def __init__(self, input_size, mu=None, Sigma=None, alpha=0.99, name='BatchNormalization'):
        """
        A batch normalization layer
        :param input_size:
        :param mu: the initial mu to use (dimension input_size x 1),
                   if None it will be the mean of the first batch
        :param Sigma: the initial Sigma to use (dimension input_size x 1),
                      if None it will be the variance of the first batch
        :param alpha: a factor for the exp moving average
        :param name: a name for this layer
        """
        # Hyper parameters
        super().__init__(input_size, input_size, name)
        self.alpha = alpha

        # Initial mu and Sigma
        assert mu is None or mu.shape == (input_size, 1)
        assert Sigma is None or Sigma.shape == (input_size, 1)
        self.mu = mu
        self.Sigma = Sigma

        # Dummy input for back propagation
        self.X = np.empty(shape=(self.input_size, 1))

    def forward(self, X, train=False):
        """
        Computes this operation for each x in the batch
        (i.e. for every column of the input matrix X)

            out = diag(var)^(-1/2) (x - mu)

        This is actually equivalent to this operation,
        where the multiplication and the power are element-wise:

            out = var^(-1/2) .* (x - mu)

        :param X: the data batch, one sample per column
        :param train:
            - False to use the internal mean and variance
            - True to use the mean and variance of the input and update
                the internal mean and variance using:
                mu_avg <- alpha * mu_avg + (1-alpha) * mu_batch
                Sigma_avg <- alpha * Sigma_avg + (1-alpha) * Sigma_batch
        :return:
        """
        # Store input for back propagation
        assert X.shape[0] == self.input_size
        self.X = X

        if train or self.mu is None or self.Sigma is None:
            self.__update_mu_and_var(X)
        self.Sigma[self.Sigma == 0] = np.finfo(float).eps
        return self.Sigma ** -0.5 * (X - self.mu)

    def __update_mu_and_var(self, X):
        # Update mean
        batch_mu = X.mean(axis=1, keepdims=True)
        if self.mu is None:
            self.mu = batch_mu
        else:
            self.mu = self.alpha * self.mu + (1 - self.alpha) * batch_mu

        # Update variance (Numpy's variance uses N at the denominator
        # and not N-1, so for us it's already ok)
        batch_Sigma = X.var(axis=1, keepdims=True)
        if self.Sigma is None:
            self.Sigma = batch_Sigma
        else:
            self.Sigma = self.alpha * self.Sigma + (1 - self.alpha) * batch_Sigma

    def backward(self, gradients):
        # Size of the mini batch
        assert self.X.shape[1] == gradients.shape[0]
        N = self.X.shape[1]

        # Prepare intermediary results
        gradients = gradients.T
        Sigma_inv = self.Sigma ** -0.5
        centered_X = (self.X - self.mu)

        dJ_dv = - 0.5 * (gradients * (self.Sigma ** -1.5 * centered_X)) \
            .sum(axis=1, keepdims=True)
        dJ_dmu = - (gradients * Sigma_inv).sum(axis=1, keepdims=True)

        # Gradient computations
        gradients = gradients * Sigma_inv + \
                    2 / N * dJ_dv * centered_X + \
                    1 / N * dJ_dmu

        return gradients.T


if __name__ == '__main__':
    import datasets

    cifar = datasets.CIFAR10()
    training = cifar.get_named_batches('data_batch_1').subset(15)

    bn = BatchNormalization(input_size=cifar.input_size)
    out = bn.forward(training.images, train=True)

    print(out.mean(axis=1))
    print(out.var(axis=1))

    print(bn.backward(training.images.T).shape)
