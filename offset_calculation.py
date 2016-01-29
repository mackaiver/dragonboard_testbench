'''
programm calculates statistics (mean offsets)
on the fly and creates a list with all data.
'''

import numpy as np
import matplotlib.pyplot as plt
from dragonboard import File
from dragonboard.runningstats import RunningStats  # Max's method to do statistics on the fly
from tqdm import tqdm  # enable to show progress


def offset_calc(filename, pixelindex, gaintype):
    '''calculate mean offset & RMS for every capacitor and plot the data.'''
    # initialize stats array on which calculations are carried out
    stats = RunningStats(shape=4096)

    # give out pixelindex (= channel) and gaintype during
    # calculation to maintain overview of progress
    print(pixelindex, gaintype)

    # calculate mean using Max's method
    with File(filename) as f:
        for event in tqdm(f[1:]):
            data = np.full(4096, np.nan)
            stop_cell = event.header.stop_cells[pixelindex]
            # print(stop_cell)
            roi = event.roi
            data[:roi] = event.data[gaintype][pixelindex]

            # that [1] is insane. how was it before?
            stats.add(np.roll(data, stop_cell[1]))

            # give out text file with data
            np.savetxt(
                'offsets_{}_channel{}_{}-gain.csv'.format(
                    filename, pixelindex, gaintype
                ),
                np.column_stack([stats.mean, stats.std]),
                delimiter=',',
            )

        # plot means with RMS
        plt.title("channel {} {}".format(pixelindex, gaintype))
        plt.errorbar(
            np.arange(4096),
            stats.mean,
            yerr=stats.std,  # yerr = y-error bars
            fmt="+",  # fmt means format
            markersize=3,
            capsize=1,  # frame bars of error bars
        )
        plt.figure()
        # plt.xlim(0,4096)


if __name__ == '__main__':
    for pixelindex in range(2):
        # calculate mean offset & RMS for every capacitor and plot the data.
        offset_calc('Ped444706_1.dat', pixelindex, "low")
        offset_calc('Ped444706_1.dat', pixelindex, "high")
    plt.show()
