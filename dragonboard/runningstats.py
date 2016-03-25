from __future__ import division, print_function, absolute_import
import numpy as np


class RunningStats():
    def __init__(self, shape=1):
        self.shape = shape
        self._n = np.zeros(shape, dtype=int)
        self._mean = np.full(shape, np.nan)
        self._M2 = np.full(shape, np.nan)

    def add(self, data):
        data = np.array(data, copy=False)

        idx = np.logical_not(np.isnan(data))
        uninitialised = np.logical_and(idx, self._n == 0)

        self._mean[uninitialised] = 0
        self._M2[uninitialised] = 0
        self._n[idx] += 1

        delta = data[idx] - self._mean[idx]
        self._mean[idx] = self._mean[idx] + delta / self._n[idx]
        self._M2[idx] = self._M2[idx] + delta * (data[idx] - self._mean[idx])

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        var = self._M2 / (self.n - 1)
        var[self.n <= 2] = np.nan
        return var

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def sem(self):
        return 1 / np.sqrt(self.n) * self.std
