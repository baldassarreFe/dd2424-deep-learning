import numpy as np

class RecurrentNeuralNetwork:
    def __init__(self, input_size, output_size, state_size,
                 initializer_W, initializer_U, initializer_V,
                 initializer_b, initializer_c):
        # For input checking
        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size

        # Trainable parameters
        self.W = initializer_W.new_matrix((state_size, state_size))
        self.U = initializer_U.new_matrix((state_size, input_size))
        self.b = initializer_b.new_matrix(state_size)
        self.V = initializer_V.new_matrix((output_size, state_size))
        self.c = initializer_c.new_matrix(output_size)

        # Gradients
        self.grad_W = np.empty_like(self.W)
        self.grad_U = np.empty_like(self.U)
        self.grad_b = np.empty_like(self.b)
        self.grad_V = np.empty_like(self.V)
        self.grad_c = np.empty_like(self.c)

        # Bookkeeping for backpropagation (dummy sequence length of 1)
        # NOTE: prev_states[t=3] is the state at t=2
        self.timesteps = 1
        self.sequence = np.empty((self.timesteps, input_size))
        self.prev_states = np.empty((self.timesteps + 1, state_size))
        self.probs = np.empty((self.timesteps, output_size))

    def weights_gradients_pairs(self):
        yield (self.W, self.grad_W, 'W')
        yield (self.U, self.grad_U, 'U')
        yield (self.b, self.grad_b, 'b')
        yield (self.V, self.grad_V, 'V')
        yield (self.c, self.grad_c, 'c')

    def forward(self, sequence, prev_state):
        # Check input size
        assert sequence.shape[1] == self.input_size
        assert prev_state.size == self.state_size
        self.timesteps = sequence.shape[0]

        # Bookkeeping for backpropagation
        self.sequence = sequence
        self.prev_states = np.empty((self.timesteps + 1, self.state_size))
        self.probs = np.empty((self.timesteps, self.output_size))

        for t in range(self.timesteps):
            self.prev_states[t] = prev_state
            self.probs[t], prev_state = self.predict_prob(sequence[t],
                                                          prev_state)
        self.prev_states[-1] = prev_state

        return self.probs, self.prev_states

    def predict_prob(self, x, prev_state):
        assert x.size == self.input_size
        assert prev_state.size == self.state_size
        a = self.W @ prev_state + self.U @ x + self.b
        h = np.tanh(a)
        o = self.V @ h + self.c
        p = self._softmax(o)
        return p, h

    def predict_class(self, x, prev_state):
        probs, prev_state = self.predict_prob(x, prev_state)
        one_hot = np.zeros(self.output_size)
        one_hot[np.random.choice(self.output_size, p=probs)] = 1
        return one_hot, prev_state

    def _softmax(self, o):
        try:
            e = np.exp(o)
            res = e / e.sum()
        except FloatingPointError:
            res = np.full_like(o, fill_value=np.finfo(float).eps)
            res[np.argmax(o)] = 1 - \
                                (self.output_size - 1) * np.finfo(float).eps
        return res

    def evaluate(self, inputs, train=False):
        outputs = inputs
        for layer in self.layers:
            if type(layer) is BatchNormalization:
                outputs = layer.forward(outputs, train)
            else:
                outputs = layer.forward(outputs)
        return outputs

    def backward(self, targets):
        """
        Note: the network will use the intermediary results of
        the previous run to propagate the gradients back
        :param targets: the target sequence to compare the output against,
                        one timestep per row
        """
        assert self.probs.shape == targets.shape
        # dL/do
        dL_do = self.probs - targets

        # dL/dc
        self.grad_c = dL_do.sum(axis=0)

        # dL/dV
        self.grad_V = np.zeros_like(self.V)
        for t in range(self.timesteps):
            self.grad_V += np.outer(dL_do[t], self.prev_states[t+1])

        # dL/dW, dL/dU, dL/db computed iteratively going back in time
        self.grad_W = np.zeros_like(self.W)
        self.grad_U = np.zeros_like(self.U)
        self.grad_b = np.zeros_like(self.b)
        dL_da = np.zeros(self.state_size)
        for t in range(self.timesteps - 1, 0 - 1, -1):
            dL_dh = dL_do[t] @ self.V + dL_da @ self.W
            dL_da = dL_dh * (1 - self.prev_states[t + 1] ** 2)

            self.grad_W += np.outer(dL_da, self.prev_states[t])
            self.grad_U += np.outer(dL_da, self.sequence[t])
            self.grad_b += dL_da

        """
        dL_dh = dL_do[self.timesteps-1] @ self.V
        for t in range(self.timesteps-1, 0-1, -1):
            dL_da = dL_dh * (1 - self.prev_states[t+1]**2)
            dL_dh = dL_do[t] @ self.V + dL_da @ self.W
        """

    def cost(self, targets):
        assert self.probs.shape == targets.shape
        log_arg = (self.probs * targets).sum(axis=1)
        log_arg[log_arg == 0] = np.finfo(float).eps
        return - np.log(log_arg).sum()

    def __str__(self):
        return 'RNN {} -> {} -> {}'.format(
            self.input_size, self.state_size, self.output_size)


class CharRNN(RecurrentNeuralNetwork):
    def __init__(self, input_output_size, state_size,
                 initializer_W, initializer_U, initializer_V,
                 initializer_b, initializer_c):
        super().__init__(input_output_size, input_output_size, state_size,
                         initializer_W, initializer_U, initializer_V,
                         initializer_b, initializer_c)

    def generate(self, x, prev_state, timesteps):
        res = np.empty((timesteps, self.output_size), dtype=np.int)
        for t in range(timesteps):
            res[t], prev_state = self.predict_class(x, prev_state)
        return res, prev_state
