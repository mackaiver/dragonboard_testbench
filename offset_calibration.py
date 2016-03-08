"""
###################################################
* apply offset calibration for (a) given file.dat *
###################################################

this program uses calibration constants stored in .csv files

annotations:

- raw_datafile_directory is the path to your raw files
- calibration_constants_directory is the path to your offset.csv file
- output_directory is the path where you want to safe your calibrated files


Usage:
  offset_calibration.py <raw_datafile_directory> <calibration_constants_directory> <output_directory>
  offset_calibration.py (-h | --help)
  offset_calibration.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.

"""

import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
from dragonboard import EventGenerator
from dragonboard.runningstats import RunningStats
import dragonboard
from tqdm import tqdm
import glob
import os
from docopt import docopt
import sys


def apply_offset_calibration(raw_datafile_directory, calibration_constants_directory):
    """ applies offset calibration for a given raw_datafile_directory. returns calibrated_data. """

    calibration_constants = read_calibration_constants(calibration_constants_directory)
    calibrated_data = []
    event_header = []

    for filename in glob.glob(os.path.join(raw_datafile_directory, '*.dat')):

        for pixelindex in range(dragonboard.io.num_channels):

            for gaintype in dragonboard.io.gaintypes:

                with open(filename, "rb") as f:

                    for event in tqdm(EventGenerator(f)):

                        if gaintype == dragonboard.io.gaintypes[0]:
                            calib_const_array_pos = pixelindex * dragonboard.io.num_gains
                            stop_cell_array_pos = 0

                        if gaintype == dragonboard.io.gaintypes[1]:
                            calib_const_array_pos = pixelindex * dragonboard.io.num_gains + 1
                            stop_cell_array_pos = 1
                        
                        stop_cell = event.header.stop_cells[pixelindex]
                        const_roi = calibration_constants[calib_const_array_pos][stop_cell[stop_cell_array_pos]:stop_cell[stop_cell_array_pos]+event.roi]

                        if isinstance(event,list) == True:

                            for i in range(len(event)):

                                print(event.data[gaintype][pixelindex][i])

                                try:

                                    calibrated_data.append(np.subtract(event.data[gaintype][pixelindex][i], const_roi))
                                    event_header.append([filename[len(raw_datafile_directory):], gaintype, pixelindex, stop_cell[stop_cell_array_pos]])

                                except Exception:
                                    pass

                        else:

                            try:

                                calibrated_data.append(np.subtract(event.data[gaintype][pixelindex], const_roi))
                                event_header.append([filename[len(raw_datafile_directory):], gaintype, pixelindex, stop_cell[stop_cell_array_pos]])

                            except Exception:
                                pass                        
                        
    if len(calibrated_data) == 0:
        sys.exit("Error: could not calibrate data")

    calib_data_with_head = list(zip(calibrated_data, event_header))

    # for i in range(4):
    #     plt.step(calibrated_data[i], ":")
    #     plt.figure()
    #     # print(calibrated_data)    

    return calib_data_with_head



def scan_datafile_amount(raw_datafile_directory):
    """ assert raw data files '*.dat' to exist """

    amount_of_files = 0

    for filename in glob.glob(os.path.join(raw_datafile_directory, '*.dat')):

        amount_of_files += 1

    if amount_of_files == 0:

        sys.exit("Error: no raw data file(s) found to perform offset calibration")



def is_calibration_constants_existent(calibration_constants_directory):
    """ assert existence of offsets.csv """

    is_calib_file_existent = False

    for filename in glob.glob(os.path.join(calibration_constants_directory, '*.csv')):

        if filename == os.path.join(calibration_constants_directory, 'offsets.csv'):
    
            is_calib_file_existent += True

    if is_calib_file_existent == False:

        sys.exit("Error: no calibration constants found to perform offset calibration")



def read_calibration_constants(calibration_constants_directory):
    """ read calibration constants saved in offsets.csv. Data column structure: 0low, 0high, 1low, ..., 7high """

    calibration_constants = []

    for i in range(dragonboard.io.num_channels * dragonboard.io.num_gains):

        calibration_constants.append(
            np.loadtxt(
                os.path.join(
                    calibration_constants_directory, "offsets.csv"),
                    delimiter=",", usecols=([i]), unpack=True
            )
        )

    return calibration_constants



def store_calibrated_data(output_directory, calib_data_with_head):
    """ slice and store calibrated data as .csv files. format: [event],[sourcefile, gaintype, pixelindex, stopcell] """

    slicearray = []
    slicearray.append(0)
    itcounter = -1

    for i in calib_data_with_head[:len(calib_data_with_head)-1]:

        itcounter += 1

        if calib_data_with_head[itcounter][1][0] != calib_data_with_head[itcounter+1][1][0]:

            slicearray.append(itcounter+1)

    slicearray.append(len(calib_data_with_head))

    for i in range(len(slicearray)-1):

        with open(output_directory + calib_data_with_head[slicearray[i]+1][1][0].rpartition('.')[0] + '_calib.csv', 'w', newline='') as f:

            for element in tqdm(calib_data_with_head[slicearray[i]:slicearray[i+1]]):

                entry=" , ".join(str(x) for x in element)
                f.write(entry+"\n")



if __name__ == '__main__':
    arguments = docopt(__doc__, version = '1.0')
    raw_datafile_directory = arguments["<raw_datafile_directory>"]
    calibration_constants_directory = arguments["<calibration_constants_directory>"]
    output_directory = arguments["<output_directory>"]

    scan_datafile_amount(raw_datafile_directory)
    is_calibration_constants_existent(calibration_constants_directory)
    store_calibrated_data(output_directory, apply_offset_calibration(raw_datafile_directory, calibration_constants_directory))

    # apply_offset_calibration(raw_datafile_directory, calibration_constants_directory)
    # plt.show()