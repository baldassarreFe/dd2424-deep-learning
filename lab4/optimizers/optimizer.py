from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, network):
        self.network = network
        self.epoch_nums = []
        self.acc_train = []
        self.acc_val = []
        self.cost_train = []
        self.cost_val = []

    @abstractmethod
    def train(self, training, validation, epochs, batch_size):
        pass

    def update_metrics(self, training, validation, epoch_num):
        self.epoch_nums.append(epoch_num)

        cost, accuracy = self.network.cost_accuracy(training)
        self.cost_train.append(cost)
        self.acc_train.append(accuracy)

        cost, accuracy = self.network.cost_accuracy(validation)
        self.cost_val.append(cost)
        self.acc_val.append(accuracy)
