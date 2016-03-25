import numpy as np
from dragonboard import RunningStats


def test_mean_result():

    mean = 1
    std = 2

    rs = RunningStats(5)
    data = np.random.normal(mean, std, size=(1000, rs.shape))

    for row in data:
        rs.add(row)

    assert np.all(np.isclose(rs.mean, np.mean(data, axis=0)))


def test_std_result():

    mean = 1
    std = 2

    rs = RunningStats(5)
    data = np.random.normal(mean, std, size=(1000, rs.shape))

    for row in data:
        rs.add(row)

    assert np.all(np.isclose(rs.std, np.std(data, axis=0, ddof=1)))
