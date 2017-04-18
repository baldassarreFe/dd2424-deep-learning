import numpy as np

import initializers

np.seterr(over='raise')


class Layer:
    def forward(self, X):
        return X

    def backward(self, gradients):
        return gradients

    def cost(self):
        return 0


class Linear(Layer):
    def __init__(self, input_size, output_size, weight_regularization,
                 weight_initializer=None, bias_initializer=None):
        # Hyper parameters
        self.input_size = input_size
        self.output_size = output_size
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
        self.X = X

        # Compute output: every column of the output is the
        # linear transformation Wx+b of the corresponding column
        # in the input
        return np.dot(self.W, X) + self.b

    def backward(self, gradients):
        # Size of the mini batch
        N = self.X.shape[1]
        N = gradients.shape[0]

        # Reset gradients
        self.grad_W = np.zeros_like(self.W, dtype=float)
        self.grad_b = np.zeros_like(self.b, dtype=float)

        # Compute gradients for every sample in the batch
        for i in range(N):
            x = self.X[:, i]
            g = gradients[i, :]

            self.grad_W += np.outer(g, x)
            self.grad_b += np.reshape(g, self.grad_b.shape)

        self.grad_W /= N
        self.grad_b /= N

        # Propagate back the gradients
        return np.dot(gradients, self.W)

    def cost(self):
        return self.weight_regularization * np.power(self.W, 2).sum()


class ReLU(Layer):
    def __init__(self, input_size):
        # Hyper parameters
        self.input_size = input_size
        self.output_size = input_size

        # Dummy input for back propagation
        self.X = np.empty(shape=(self.input_size, 1))

    def forward(self, X):
        # Store input for back propagation
        self.X = X

        # Compute output: every column in the output is the element wise
        # operation max(0, x), where x is the corresponding column
        # in the input
        return X * (X > 0)

    def backward(self, gradients):
        # Propagate back the gradients
        return np.dot(gradients, self.X > 0)


class Softmax(Layer):
    def __init__(self, input_size):
        # Hyper parameters
        self.input_size = input_size

        # Dummy output probabilities for back propagation
        self.P = np.empty(shape=(self.input_size, 1))

    def forward(self, X):
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
        N = self.P.shape[1]
        N = one_hot_targets.shape[1]

        output_target_pairs = ((one_hot_targets[:, i], self.P[:, i]) for i in range(N))
        gradients = (self.softmax_gradient(y, p) for (y, p) in output_target_pairs)

        # Propagate back the gradients
        return np.vstack(gradients)

    def cost(self, one_hot_targets, P=None):
        # Size of the mini batch
        if P is None:
            P = self.P
        N = P.shape[1]
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
    def __init__(self, input_size, mu=None, Sigma=None):
        # Hyper parameters
        self.input_size = input_size
        self.output_size = input_size

        # Initial mu and Sigma
        self.mu = mu or np.zeros(shape=(input_size, 1), dtype=float)
        self.Sigma = Sigma or np.eye(input_size, dtype=float)

    def forward(self, X, train=False):
        if train:
            mu = X.mean(axis=1)
            Sigma = X.var(axis=1)
        else:
            mu = self.mu
            Sigma = self.Sigma
        return np.dot(Sigma, X - mu)

    def backward(self, gradients):
        # Not implemented yet
        return gradients
