"""
##################################################
*          Dragon board readout software         *
* calculate offset constants from pedestal files *
##################################################

annotations:

- inputdirectory is the path to the directory of your pedestal files
- outputpath is the file path where the calibration constants are written to


Usage:
  offset_calculation.py <inputdirectory> <outputpath>
  offset_calculation.py (-h | --help)
  offset_calculation.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.

"""

import numpy as np
import matplotlib.pyplot as plt
from dragonboard import EventGenerator
from dragonboard.runningstats import RunningStats
import dragonboard
from tqdm import tqdm
import glob
import os
from docopt import docopt
import sys
import pickle

def offset_calc(inputdirectory):
    """ calculate mean offset for every capacitor. the data is saved as .csv files """

    calibration_constants = {}
    for pixelindex in range(dragonboard.io.num_channels):
        for gaintype in dragonboard.io.gaintypes:
            calibration_constants[(pixelindex, gaintype)] = RunningStats(shape=dragonboard.io.max_roi)

    glob_expression = os.path.join(inputdirectory, '*.dat')
    for filename in sorted(glob.glob(glob_expression)):
        try:
            f = open(filename, "rb")
            for event in tqdm(
                    iterable=EventGenerator(f),
                    desc=os.path.basename(filename),
                    leave=True,
                    unit="events"):
                for pixelindex in range(dragonboard.io.num_channels):
                    for gaintype in dragonboard.io.gaintypes:
                        stats = calibration_constants[(pixelindex, gaintype)]

                        data = np.full(dragonboard.io.max_roi, np.nan)
                        stop_cell = event.header.stop_cells[gaintype][pixelindex]
                        data[:event.roi] = event.data[gaintype][pixelindex]

                        stats.add(np.roll(data, stop_cell))
        except Exception as e:
            print(e)

    return calibration_constants


if __name__ == '__main__':
    arguments = docopt(__doc__, version='Dragon Board Offset Calculation v.1.1')

    if not glob.glob(os.path.join(arguments["<inputdirectory>"], '*.dat')):
        sys.exit("Error: no files found to perform offset calculation")

    calibration_constants = offset_calc(arguments["<inputdirectory>"])
    pickle.dump(calibration_constants, open(arguments["<outputpath>"], 'wb'))