import unittest

import numpy as np

from initializers import Xavier
from initializers import Zeros
from network import RecurrentNeuralNetwork


class GradientChecking(unittest.TestCase):
    @staticmethod
    def get_sequence(timesteps, input_size, output_size):
        np.random.seed(123)
        input_sequence = np.zeros((timesteps, input_size))
        output_sequence = np.zeros((timesteps, output_size))
        for t in range(timesteps):
            input_sequence[t, np.random.randint(0, input_size)] = 1
            output_sequence[t, np.random.randint(0, output_size)] = 1
        return input_sequence, output_sequence

    @staticmethod
    def compute_grads_for_matrix(inputs, targets, initial_state, matrix,
                                 network, name):
        # Initialize an empty matrix to contain the gradients
        matrix = np.atleast_2d(matrix)
        grad = np.empty_like(matrix)
        h = 1e-4

        # Iterate over the matrix changing one entry at the time
        print('Gradient computations for {} {}, sequence length {}'
              .format(name, matrix.shape, inputs.shape[0]))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i, j] += h
                network.forward(inputs, initial_state)
                plus_cost = network.cost(targets)
                matrix[i, j] -= 2 * h
                network.forward(inputs, initial_state)
                minus_cost = network.cost(targets)
                grad[i, j] = (plus_cost - minus_cost) / (2 * h)
                matrix[i, j] += h
        return np.squeeze(grad)

    def print_grad_diff(self, grad, grad_num, name=''):
        err = np.abs(grad - grad_num)
        rel_err = err / np.maximum(np.finfo('float').eps,
                                   np.abs(grad) + np.abs(grad_num))
        print('Gradient difference {}: {:.2e}'.format(name, np.max(rel_err)))

        coord_worst = np.unravel_index(np.argmax(rel_err), rel_err.shape)
        self.assertLess(rel_err[coord_worst], 1e-4,
                        msg='grad {:.3e}\nnum {:.3e}\ndiff {:.3e}\n'
                        .format(grad[coord_worst], grad_num[coord_worst],
                                err[coord_worst]))

    def test_gradients(self):
        input_size = 10
        state_size = 40
        output_size = 30
        timesteps = 200

        input_sequence, output_sequence = self.get_sequence(
            timesteps, input_size, output_size)

        rnn = RecurrentNeuralNetwork(
            input_size=input_size,
            output_size=output_size,
            state_size=state_size,
            initializer_W=Xavier(),
            initializer_U=Xavier(),
            initializer_V=Xavier(),
            initializer_b=Zeros(),
            initializer_c=Zeros()
        )

        rnn.forward(input_sequence, np.zeros(state_size))
        rnn.backward(output_sequence)

        for (param, grad, name) in rnn.weights_gradients_pairs():
            grad_num = self.compute_grads_for_matrix(
                input_sequence, output_sequence, np.zeros(state_size),
                param, rnn, name)
            self.print_grad_diff(grad, grad_num, name)


if __name__ == '__main__':
    unittest.main()
