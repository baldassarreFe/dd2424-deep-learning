import os
import pickle
import tarfile
import urllib.request
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

Batch = namedtuple('Batch', ['size', 'images', 'one_hot_labels', 'numeric_labels'])


class CIFAR10:
    def __init__(self):
        self.output_size = 10
        self.input_size = 32 * 32 * 3
        self.download_dataset()
        self.labels = self.load_labels()
        self.loaded_batches = {}
        self.label_encoder = preprocessing.LabelBinarizer()
        self.label_encoder.fit([x for x in range(self.output_size)])

    @staticmethod
    def load_labels():
        with open('cifar-10-batches-py/batches.meta', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            return [x.decode('ascii') for x in data[b'label_names']]

    @staticmethod
    def download_dataset():
        if not os.path.isdir('cifar-10-batches-py'):
            file, _ = urllib.request.urlretrieve(
                "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                "temp.tar.gz")
            with tarfile.open(file, "r:gz") as tar:
                tar.extractall()
            os.remove(file)
            with open('.gitignore', 'a+') as out:
                out.write('cifar-10-batches-py\n')

    def get_batch(self, batch_name) -> Batch:
        if batch_name not in self.loaded_batches:
            with open('cifar-10-batches-py/' + batch_name, 'rb') as f:
                data = pickle.load(f, encoding='bytes')

            self.loaded_batches[batch_name] = Batch(
                size=len(data[b'labels']),
                images=np.divide(data[b'data'], 255).T,
                one_hot_labels=self.label_encoder.transform(data[b'labels']).T,
                numeric_labels=data[b'labels']
            )
        return self.loaded_batches[batch_name]

    def get_batches(self, *args, limit=None) -> Batch:
        batches = [self.get_batch(name) for name in args]
        big_batch = Batch(
            size=sum((b.size for b in batches)),
            images=np.hstack([b.images for b in batches]),
            one_hot_labels=np.hstack([b.one_hot_labels for b in batches]),
            numeric_labels=np.hstack([b.numeric_labels for b in batches])
        )
        if limit is not None:
            return self.subset(big_batch, limit)
        else:
            return big_batch

    @staticmethod
    def subset(batch: Batch, size):
        return Batch(
            size=size,
            images=batch.images[:, :size],
            one_hot_labels=batch.one_hot_labels[:, :size],
            numeric_labels=batch.numeric_labels[:size]
        )

    def describe_dataset(self, dataset: Batch, name: str = 'Dataset') -> str:
        res = '{} (total images {})\n'.format(name, dataset.size)
        for l in np.unique(dataset.numeric_labels):
            res += '- {:.2%}  {}\n'.format(
                (dataset.numeric_labels == l).sum() / dataset.size,
                self.labels[l])
        return res

    def show_image(self, img, label: int = None, interpolation='gaussian'):
        squared_image = np.rot90(np.reshape(img, (32, 32, 3), order='F'), k=3)
        plt.imshow(squared_image, interpolation=interpolation)
        plt.axis('off')
        plt.title(self.labels[label] if label is not None else '')


if __name__ == '__main__':
    cifar = CIFAR10()
    training = cifar.get_batches('data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4')
    validation = cifar.get_batches('data_batch_5')
    test = cifar.get_batches('test_batch')

    print(cifar.describe_dataset(training, 'Training'))
    print(cifar.describe_dataset(validation, 'Validation'))
    print(cifar.describe_dataset(test, 'Test'))

    for plot_i, img_i in enumerate(np.random.choice(training.size, 15, replace=False)):
        plt.subplot(3, 5, plot_i + 1)
        cifar.show_image(training.images[:, img_i], training.numeric_labels[img_i])

    plt.show()
