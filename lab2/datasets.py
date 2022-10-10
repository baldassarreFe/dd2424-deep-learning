import os
import pickle
import tarfile
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing


class Batch:
    def __init__(self, images, one_hot_labels, numeric_labels, class_names):
        self.size = images.shape[1]
        self.images = images
        self.one_hot_labels = one_hot_labels
        self.numeric_labels = numeric_labels
        self.class_names = class_names

    def description(self, name: str = 'Dataset') -> str:
        res = '{} (total images {})\n'.format(name, self.size)
        for l in range(len(self.class_names)):
            res += '- {:.2%}  {}\n'.format(
                (self.numeric_labels == l).sum() / self.size,
                self.class_names[l])
        return res

    def subset(self, new_size):
        return Batch(
            images=self.images[:, :new_size],
            one_hot_labels=self.one_hot_labels[:, :new_size],
            numeric_labels=self.numeric_labels[:new_size],
            class_names=self.class_names
        )

    def mean(self):
        return self.images.mean(axis=1)


class CIFAR10:
    output_size = 10
    input_size = 32 * 32 * 3

    def __init__(self):
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
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar)
            os.remove(file)
            with open('.gitignore', 'a+') as out:
                out.write('cifar-10-batches-py\n')

    def get_named_batch(self, batch_name) -> Batch:
        if batch_name not in self.loaded_batches:
            with open('cifar-10-batches-py/' + batch_name, 'rb') as f:
                data = pickle.load(f, encoding='bytes')

            self.loaded_batches[batch_name] = Batch(
                images=np.divide(data[b'data'], 255).T,
                one_hot_labels=self.label_encoder.transform(data[b'labels']).T,
                numeric_labels=data[b'labels'],
                class_names=self.labels
            )
        return self.loaded_batches[batch_name]

    def get_named_batches(self, *args) -> Batch:
        batches = [self.get_named_batch(name) for name in args]
        return Batch(
            images=np.hstack([b.images for b in batches]),
            one_hot_labels=np.hstack([b.one_hot_labels for b in batches]),
            numeric_labels=np.hstack([b.numeric_labels for b in batches]),
            class_names=self.labels
        )

    def show_image(self, img, label: int = None, interpolation='gaussian'):
        squared_image = np.rot90(np.reshape(img, (32, 32, 3), order='F'), k=3)
        plt.imshow(squared_image, interpolation=interpolation)
        plt.axis('off')
        plt.title(self.labels[label] if label is not None else '')


if __name__ == '__main__':
    cifar = CIFAR10()
    training = cifar.get_named_batches('data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4')
    validation = cifar.get_named_batches('data_batch_5')
    test = cifar.get_named_batches('test_batch')

    print(training.description('Training'))
    print(validation.description('Validation'))
    print(test.description('Test'))

    for plot_i, img_i in enumerate(np.random.choice(training.size, 15, replace=False)):
        plt.subplot(3, 5, plot_i + 1)
        cifar.show_image(training.images[:, img_i], training.numeric_labels[img_i])

    plt.show()
