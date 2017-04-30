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

    def backward(self, one_hot_targets, inputs=None):
        """
        :param one_hot_targets: the target to compare the
                                output against
        :param inputs: if None the network will use the
                       output of the previous run
        """
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

    def cost_accuracy(self, dataset):
        Y = self.evaluate(dataset.images)
        return (self.cost(dataset.one_hot_labels, None, Y),
                self.accuracy(dataset.one_hot_labels, None, Y))


if __name__ == '__main__':
    import layers
    import datasets
    import initializers
    import matplotlib.pyplot as plt

    cifar = datasets.CIFAR10()
    training = cifar.get_named_batches('data_batch_1', limit=4)

    net = Network()
    net.add_layer(layers.Linear(cifar.input_size, 50, 0, initializers.Xavier()))
    net.add_layer(layers.ReLU(50))
    net.add_layer(layers.Linear(50, cifar.output_size, 0, initializers.Xavier()))
    net.add_layer(layers.Softmax(cifar.output_size))

    Y = net.evaluate(training.images)
    print('Cost:', net.cost(training.one_hot_labels, None, Y))
    print('Accuracy: {:.2%}'
          .format(net.accuracy(training.one_hot_labels, None, Y)))

    plt.subplot(1, 3, 1)
    plt.imshow(Y)
    plt.yticks(range(10), cifar.labels)
    plt.xlabel('Image number')
    plt.title('Probabilities')

    plt.subplot(1, 3, 2)
    plt.imshow(cifar.label_encoder.transform(np.argmax(Y, axis=0)).T)
    plt.yticks([])
    plt.xlabel('Image number')
    plt.title('Predicted classes')

    plt.subplot(1, 3, 3)
    plt.imshow(training.one_hot_labels)
    plt.yticks([])
    plt.xlabel('Image number')
    plt.title('Ground truth')

    plt.show()
