import numpy as np

import initializers
from layers import Softmax


class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def evaluate(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

    def backward(self, one_hot_targets, inputs=None):
        if inputs is not None:
            self.evaluate(inputs)

        for layer in self.layers[::-1]:
            if type(layer) is Softmax:
                gradients = layer.backward(one_hot_targets)
            else:
                gradients = layer.backward(gradients)

    def cost(self, one_hot_targets, inputs=None, outputs=None):
        cost = 0
        if outputs is None:
            outputs = self.evaluate(inputs)
        for layer in self.layers:
            if type(layer) is Softmax:
                cost += layer.cost(one_hot_targets, outputs)
            else:
                cost += layer.cost()

        return cost

    def accuracy(self, one_hot_targets, inputs=None, outputs=None):
        N = one_hot_targets.shape[1]
        if outputs is None:
            outputs = self.evaluate(inputs)

        predicted_labels = np.argmax(outputs, axis=0)
        true_labels = np.argmax(one_hot_targets, axis=0)
        return np.sum(predicted_labels == true_labels) / N

if __name__ == '__main__':
    import layers
    from cifar import CIFAR10

    cifar = CIFAR10()
    training = cifar.get_batches('data_batch_1')

    net = Network()
    net.add_layer(layers.Linear(cifar.input_size, 50, 0, initializers.PositiveNormal(0.01)))
    net.add_layer(layers.ReLU(50))
    net.add_layer(layers.Linear(50, cifar.output_size, 0, initializers.Xavier()))
    net.add_layer(layers.Softmax(cifar.output_size))

    Y = net.evaluate(training.images[:, :5])
    print(net.accuracy(training.one_hot_labels[:, :5], None, Y))
    print(net.cost(training.one_hot_labels[:, :5], None, Y))
