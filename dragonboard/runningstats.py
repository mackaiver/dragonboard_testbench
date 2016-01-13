from __future__ import division, print_function, absolut_import
import numpy as np


class RunningStats():
    def __init__(self, shape=1):
        self.shape = shape
        self.n = np.zeros_like(shape)
        self._mean = 0
        self._delta = 0
        self._M2 = 0

    def add(self, data):
        self.n += 1
        self._delta = data - self._mean
        self._mean = self._mean + self._delta / self.n
        self._M2 = self._M2 + self._delta * (data - self._mean)

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        if self.n < 2:
            raise ValueError('variance not defined for n<2')
        return self._M2 / (self.n - 1)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def sem(self):
        return 1 / np.sqrt(self.n) * self.std
