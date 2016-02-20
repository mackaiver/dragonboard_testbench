"""
##########################################################################
*                     Dragon board readout software                      *
* calculate offset constants and standard deviations from pedestal files *
##########################################################################

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

    superstats = [np.nan] * (dragonboard.io.num_channels * dragonboard.io.num_gains)
    iteration_counter = 0

    for pixelindex in range(dragonboard.io.num_channels):

        for gaintype in dragonboard.io.gaintypes:

            stats = RunningStats(shape=dragonboard.io.max_roi)

            for filename in glob.glob(os.path.join(inputdirectory, '*.dat')):

                with open(filename, "rb") as f:         
                    
                    # print("reading file: %s, channel %s, %s gain" % (filename, pixelindex, gaintype))

                    for event in tqdm(EventGenerator(f)):

                        data = np.full(dragonboard.io.max_roi, np.nan)
                        stop_cell = event.header.stop_cells[pixelindex]
                        data[:event.roi] = event.data[gaintype][pixelindex]

                        # assert correct stop cell. Assumed for the if-structure: stop cells are arranged in array 
                        # [(channel0_sc_low, channel0_sc_high), ..., (channel7_sc_low), channel7_sc_high)]. IS THAT APPROACH CORRECT?!
                        if gaintype == dragonboard.io.gaintypes[0]:

                            stop_cell_array_pos = 0

                        if gaintype == dragonboard.io.gaintypes[1]:

                            stop_cell_array_pos = 1

                        stats.add(np.roll(data, stop_cell[stop_cell_array_pos]))

            superstats[iteration_counter] = stats.mean
            iteration_counter += 1

    return superstats



def scan_pedestalfile_amount(inputdirectory):
    """ assert Pedestal files '*.dat' to exist """

    amount_of_files = 0

    for filename in glob.glob(os.path.join(inputdirectory, '*.dat')):

        amount_of_files += 1

    if amount_of_files == 0:

        sys.exit("Error: no files found to perform offset calculation")



def store_data(outputdirectory, superstats):
    """store acquired data to outputdirectory. data column structure: 0low, 0high, 1low, ..., 7high"""
 
    # stackstring = ""
    # for i in range(dragonboard.io.num_channels*dragonboard.io.num_gains):
            
    #     print(superstats[i])

    #     if i == dragonboard.io.num_channels*dragonboard.io.num_gains-1:

    #         stackstring = stackstring + "superstats[" + str(i) + "]"

    #     else:

    #         stackstring = stackstring + "superstats[" + str(i) + "], "

    # print(stackstring)

    np.savetxt(
       'offsets.csv', 
        np.column_stack([
            superstats[0],
            superstats[1],
            superstats[2],
            superstats[3],
            superstats[4],
            superstats[5],
            superstats[6],
            superstats[7],
            superstats[8],
            superstats[9],
            superstats[10],
            superstats[11],
            superstats[12],
            superstats[13],
            superstats[14],
            superstats[15]
            ]),
        delimiter=','
    )



if __name__ == '__main__':

    arguments = docopt(__doc__, version='Dragon Board Offset Calculation v.1.1')
    inputdirectory = arguments["<inputdirectory>"]
    outputdirectory = arguments["<outputdirectory>"]

    scan_pedestalfile_amount(inputdirectory)

    store_data(outputdirectory, offset_calc(inputdirectory))