from collections import defaultdict

import matplotlib.pyplot as plt


class DataTracker():
    def __init__(self, keys=None, label=None):
        self.keys = keys
        self.data = defaultdict(list)
        self.x = []
        self.label = label

    def append(self, step, data):
        self.x += [step]
        if self.keys is None:
            self.keys = data.keys()
        for k in self.keys:
            self.data[k] += [data[k]]

    def plot(self, key, **kwargs):
        label = kwargs.get('label', self.label)
        plt.plot(self.x, self.data[key], label=label, **kwargs)
