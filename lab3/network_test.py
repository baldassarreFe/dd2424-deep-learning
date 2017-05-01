import matplotlib.pyplot as plt

from datasets import CIFAR10
from initializers import Xavier
from layers import *
from network import Network

cifar = CIFAR10()
training = cifar.get_named_batches('data_batch_1').subset(10)

net = Network()
net.add_layer(Linear(CIFAR10.input_size, 50, 0, Xavier()))
net.add_layer(ReLU(50))
net.add_layer(Linear(50, CIFAR10.output_size, 0, Xavier()))
net.add_layer(Softmax(CIFAR10.output_size))

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
