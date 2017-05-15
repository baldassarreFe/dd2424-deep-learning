from abc import ABC, abstractmethod


class RnnOptimizer(ABC):
    def __init__(self, rnn, stateful=False):
        self.rnn = rnn
        self.smooth_costs = []
        self.stateful = stateful
        self.steps = 0

    @abstractmethod
    def train(self, sequence_pairs, epochs=1):
        pass

    def update_metrics(self, cost):
        self.steps += 1
        if len(self.smooth_costs) > 0:
            cost = .999 * self.smooth_costs[-1] + .001 * cost
        self.smooth_costs.append(cost)
