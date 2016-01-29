from __future__ import division, print_function, absolute_import
import numpy as np


class RunningStats():
    def __init__(self, shape=1):
        self.shape = shape
        self.n = np.zeros(shape)
        self._mean = np.full(shape, np.nan)
        self._delta = np.full(shape, np.nan)
        self._M2 = np.full(shape, np.nan)

    def add(self, data):
        idx = ~np.isnan(data)
        mask = (idx) & (self.n == 0)
        self._mean[mask] = 0
        self._delta[mask] = 0
        self._M2[mask] = 0

        self.n[idx] += 1
        self._delta[idx] = data[idx] - self._mean[idx]
        self._mean[idx] = self._mean[idx] + self._delta[idx] / self.n[idx]
        self._M2[idx] = self._M2[idx] + self._delta[idx] * (data[idx] - self._mean[idx])


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
