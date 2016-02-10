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


def offset_calc(inputdirectory, outputdirectory):
    """ calculate mean offset & standard deviation for every capacitor and plot the data. the data is saved as .csv files.
        The .csv files are named offsets_channel<pixelindex>_<gaintype>-gain. 
        this FORMAT MUST NOT BE CHANGED
    """

    for pixelindex in range(dragonboard.io.num_channels):

        print(dragonboard.Event.data.fget)

        for gaintype in ["low", "high"]:

            stats = RunningStats(shape=dragonboard.io.max_roi)

            for filename in glob.glob(os.path.join(inputdirectory, 'Ped*.dat')):

                with open(filename, "rb") as f:

                    generator = EventGenerator(f) 
                    
                    #print("reading file: %s, channel %s, %s gain" % (filename, pixelindex, gaintype))

                    for event in generator:

                        data = np.full(dragonboard.io.max_roi, np.nan)
                        stop_cell = event.header.stop_cells[pixelindex]
                        roi = event.roi
                        data[:roi] = event.data[gaintype][pixelindex]

                        # assert correct stop cell. Assumed for the if-structure: stop cells are arranged in array 
                        # [(channel0_sc_low, channel0_sc_high), ..., (channel7_sc_low), channel7_sc_high)]. IS THAT APPROACH CORRECT?!
                        if gaintype == "low":

                            stop_cell_array_pos = 0

                        else:

                            stop_cell_array_pos = 1

                        stats.add(np.roll(data, stop_cell[stop_cell_array_pos]))



            np.savetxt(
                '{}offsets_channel{}_{}-gain.csv'.format(outputdirectory, pixelindex, gaintype), 
                np.column_stack([stats.mean, stats.std]),
                delimiter=','
            )




def scan_pedestalfile_amount(inputdirectory):
    """ assert Ped*.files to exist """

    amount_of_files = 0

    for filename in glob.glob(os.path.join(inputdirectory, 'Ped*.dat')):

        amount_of_files += 1

    if amount_of_files == 0:

        sys.exit("Error: no files found to perform offset calculation")




if __name__ == '__main__':

    arguments = docopt(__doc__, version='Dragon Board Offset Calculation v.1.0')
    inputdirectory = arguments["<inputdirectory>"]
    outputdirectory = arguments["<outputdirectory>"]

    scan_pedestalfile_amount(inputdirectory)

    offset_calc(inputdirectory, outputdirectory)