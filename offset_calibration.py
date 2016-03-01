"""
#################################################
* apply offset calibration for a given file.dat *
#################################################

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
import csv


def apply_offset_calibration(raw_datafile_directory, calibration_constants_directory):
    """ applies offset calibration for a given raw_datafile_directory. returns calibrated_data. """

    calibration_constants = read_calibration_constants(calibration_constants_directory)
    calibrated_data = []
    event_header = []

    for filename in glob.glob(os.path.join(raw_datafile_directory, '*.dat')):

        for pixelindex in range(dragonboard.io.num_channels):
        #for pixelindex in range(1):

            for gaintype in dragonboard.io.gaintypes:

                with open(filename, "rb") as f:

                    for event in tqdm(EventGenerator(f)):

                        if gaintype == dragonboard.io.gaintypes[0]:
                            calib_const_array_pos = pixelindex * dragonboard.io.num_gains
                            stop_cell_array_pos = 0

                        if gaintype == dragonboard.io.gaintypes[1]:
                            calib_const_array_pos = pixelindex * dragonboard.io.num_gains +1
                            stop_cell_array_pos = 1
                        
                        stop_cell = event.header.stop_cells[pixelindex]
                        const_roi = calibration_constants[calib_const_array_pos][stop_cell[stop_cell_array_pos]:stop_cell[stop_cell_array_pos]+event.roi]

                        calibrated_data.append(np.subtract(event.data[gaintype][pixelindex], const_roi))
                        event_header.append([filename[len(raw_datafile_directory):], gaintype, pixelindex, stop_cell[stop_cell_array_pos]])

    calib_data_with_head = list(zip(calibrated_data, event_header))
    
    # for i in range(10):
    #     #print(calib_data_with_head[i*200]) 
    #     plt.step(calibrated_data[i*200], ":")
    #     plt.figure()

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
    """ store the calibrated data as .csv file. format: [event],[sourcefile, gaintype, pixelindex, stopcell] """

    with open('calibrated_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerows(calib_data_with_head)



if __name__ == '__main__':
    arguments = docopt(__doc__, version='1.0')
    raw_datafile_directory = arguments["<raw_datafile_directory>"]
    calibration_constants_directory = arguments["<calibration_constants_directory>"]
    output_directory = arguments["<output_directory>"]

    scan_datafile_amount(raw_datafile_directory)
    is_calibration_constants_existent(calibration_constants_directory)
    store_calibrated_data(output_directory, apply_offset_calibration(raw_datafile_directory, calibration_constants_directory)) 

    plt.show()


# Junk

    #data = np.full(dragonboard.io.max_roi, np.nan)
    #data[:event.roi] = event.data[gaintype][pixelindex]
    #data = np.roll(data, stop_cell[stop_cell_array_pos])
    #calibrated_data.append(np.subtract(data, calibration_constants[calib_const_array_pos]))
                        
   # print("calibrating file: %s, channel %s, %s gain" % (raw_datafile_directory, pixelindex, gaintype))    

    # # plot of raw data
    # plt.xlabel('time slice / DRS4 cell')
    # plt.ylabel('ADC counts')
    # plt.axis([60,180,-100,350]) # ([xmin, xmax, ymin, ymax]), before any figure() or show() command!
    # plt.title("%s, channel %s, %s gain" %(raw_datafile_directory, pixelindex, gaintype))
    # plt.step(np.roll(data, stop_cell[stop_cell_array_pos]), ":",label="raw data")
    # plt.legend()
    # plt.figure()

    # plot of raw and calibrated data
    # plt.xlabel('time slice / DRS4 cell')
    # plt.ylabel('ADC counts')
    # plt.axis([60,180,-100,350]) # ([xmin, xmax, ymin, ymax]), before any figure() or show() command!
    # plt.title("%s, channel %s, %s gain" %(raw_datafile_directory, pixelindex, gaintype))
    # plt.step(np.roll(data, stop_cell[stop_cell_array_pos]), ":",label="raw data")
    # plt.step(calibrated_data, "r:",label="calibrated data")
    # plt.legend()    
    #plt.figure()

    

    # # plot calibrated data
    # plt.title("raw data: %s, channel %s, %s gain" %(raw_datafile_directory, pixelindex, gaintype))
    # plt.errorbar(
    #     np.arange(4096),
    #     np.roll(data, stop_cell[stop_cell_array_pos]),
    #     yerr=std, # yerr = y-error bars
    #     fmt="o:", # fmt means format
    #     markersize=3,
    #     capsize=1, # frame bars of error bars
    # )
    # #plt.axis([60,180,-100,350]) # ([xmin, xmax, ymin, ymax]), before any figure() or show() command!
    # plt.figure()
    

    # # plot calibrated data
    # plt.title("calibrated data: %s, channel %s, %s gain" %(raw_datafile_directory, pixelindex, gaintype))
    # plt.errorbar(
    #     np.arange(4096),
    #     calibrated_data,
    #     yerr=std, # yerr = y-error bars
    #     fmt="ro:", # fmt means format
    #     markersize=3,
    #     capsize=1, # frame bars of error bars
    # )
    # #plt.axis([60,180,-100,350])