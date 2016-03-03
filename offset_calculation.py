"""
##################################################
*          Dragon board readout software         *
* calculate offset constants from pedestal files *
##################################################

annotations:

- inputdirectory is the path to the directory of your pedestal files
- outputdirectory is the directory where the calibration constants are written to


Usage:
  offset_calculation.py <inputdirectory> <outputdirectory>
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


def offset_calc(inputdirectory):
    """ calculate mean offset for every capacitor. the data is saved as .csv files """

    calibration_constants = []

    for pixelindex in range(dragonboard.io.num_channels):

        for gaintype in dragonboard.io.gaintypes:

            stats = RunningStats(shape=dragonboard.io.max_roi)

            for filename in glob.glob(os.path.join(inputdirectory, '*.dat')):

                with open(filename, "rb") as f:         
                    
                    for event in tqdm(EventGenerator(f)):

                        data = np.full(dragonboard.io.max_roi, np.nan)
                        stop_cell = event.header.stop_cells[pixelindex]
                        data[:event.roi] = event.data[gaintype][pixelindex]

                        # # assert correct stop cell. Assumed for the if-structure: stop cells are arranged in array 
                        # # [(channel0_sc_low, channel0_sc_high), ..., (channel7_sc_low), channel7_sc_high)]. IS THAT APPROACH CORRECT?!
                        if gaintype == dragonboard.io.gaintypes[0]:
                            stop_cell_array_pos = 0

                        if gaintype == dragonboard.io.gaintypes[1]:
                            stop_cell_array_pos = 1

                        stats.add(np.roll(data, stop_cell[stop_cell_array_pos]))

            calibration_constants.append(stats.mean)

    return calibration_constants



def scan_pedestalfile_amount(inputdirectory):
    """ assert Pedestal files '*.dat' to exist """

    amount_of_files = 0

    for filename in glob.glob(os.path.join(inputdirectory, '*.dat')):

        amount_of_files += 1

    if amount_of_files == 0:

        sys.exit("Error: no files found to perform offset calculation")



def store_data(outputdirectory, calibration_constants):
    """ store acquired data to outputdirectory. data column structure: 0low, 0high, 1low, ..., 7high """

    np.savetxt(
        outputdirectory + "offsets.csv", 
        np.column_stack([
            calibration_constants[0],
            calibration_constants[1],
            calibration_constants[2],
            calibration_constants[3],
            calibration_constants[4],
            calibration_constants[5],
            calibration_constants[6],
            calibration_constants[7],
            calibration_constants[8],
            calibration_constants[9],
            calibration_constants[10],
            calibration_constants[11],
            calibration_constants[12],
            calibration_constants[13],
            calibration_constants[14],
            calibration_constants[15]
            ]),
        delimiter=','
    )



if __name__ == '__main__':

    arguments = docopt(__doc__, version='Dragon Board Offset Calculation v.1.1')
    inputdirectory = arguments["<inputdirectory>"]
    outputdirectory = arguments["<outputdirectory>"]

    scan_pedestalfile_amount(inputdirectory)

    store_data(outputdirectory, offset_calc(inputdirectory))