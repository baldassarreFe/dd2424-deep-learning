import numpy as np

from layers import Softmax, Layer, BatchNormalization


class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def evaluate(self, inputs, train=False):
        outputs = inputs
        for layer in self.layers:
            if type(layer) is BatchNormalization:
                outputs = layer.forward(outputs, train)
            else:
                outputs = layer.forward(outputs)
        return outputs

    def backward(self, one_hot_targets):
        """
        Note: the network will use the intermediary results of
        the previous run to propagate the gradients back
        :param one_hot_targets: the target to compare the
                                output against
        """

        for layer in self.layers[::-1]:
            if type(layer) is Softmax:
                gradients = layer.backward(one_hot_targets)
            else:
                gradients = layer.backward(gradients)

            if __debug__ and np.abs(gradients).max() > 1000:
                print('Huge gradient at', str(layer), np.abs(gradients).max())

    def cost(self, one_hot_targets, inputs=None, outputs=None):
        if outputs is None:
            outputs = self.evaluate(inputs, train=False)
        assert one_hot_targets.shape == outputs.shape

        cost = 0
        for layer in self.layers:
            if type(layer) is Softmax:
                cost += layer.cost(one_hot_targets, outputs)
            else:
                cost += layer.cost()
        return cost

    def accuracy(self, one_hot_targets, inputs=None, outputs=None):
        N = one_hot_targets.shape[1]
        if outputs is None:
            outputs = self.evaluate(inputs, train=False)

        predicted_labels = np.argmax(outputs, axis=0)
        true_labels = np.argmax(one_hot_targets, axis=0)
        return np.sum(predicted_labels == true_labels) / N

    def cost_accuracy(self, dataset):
        Y = self.evaluate(dataset.images, train=False)
        return (self.cost(dataset.one_hot_labels, None, Y),
                self.accuracy(dataset.one_hot_labels, None, Y))
