"""
Example: 
dragon browser analysis fun:

"""
import numpy as np

def print_mean_std(channel, gain, data, stop_cell, axis):
    print("{name}: {mean:0.1f} {std:0.1f}".format(
        name=channel,
        mean=data.mean(),
        std=data.std(),
        )
    )

def digital_leading_edge_discriminator(data, time, threshold=0, window_length=0):
    z = np.where(np.diff(np.signbit(data-threshold)))[0][0]
    s = slice(z-window_length, z+2+window_length)
    m, b = np.polyfit(time[s], data[s], deg=1)
    return (threshold-b)/m

def find_leading_edge_1500(channel, gain, data, stop_cell, axis):
    arrival_time = digital_leading_edge_discriminator(
        data=data,
        time=np.arange(len(data)),
        threshold=1500,
    )
    axis.axvline(arrival_time, color='k')

from scipy import interpolate

def integrate_around_arrival_time_2(channel, gain, data, stop_cell, axis):
    arrival_time = digital_leading_edge_discriminator(
        data=data,
        time=np.arange(len(data)),
        threshold=1500,
    )
    interpolant = interpolate.interp1d(np.arange(len(data)), data, kind='cubic')
    start = arrival_time - 2
    end = arrival_time + 2

    fill_x = np.linspace(start, end, 20)
    axis.fill_between(fill_x, interpolant(fill_x))
