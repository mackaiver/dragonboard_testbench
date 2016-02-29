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

   # print("calibrating file: %s, channel %s, %s gain" % (raw_datafile_directory, pixelindex, gaintype))    

    calibration_constants = read_calibration_constants(calibration_constants_directory)
    calibrated_data = [ ]

    # plt.plot(calibration_constants[0])
    # plt.figure()

    for filename in glob.glob(os.path.join(raw_datafile_directory, '*.dat')):

        # calibrated_data = np.full(1024*16, np.nan)

        #for pixelindex in range(dragonboard.io.num_channels):
        for pixelindex in range(1):

            print(pixelindex)

            for gaintype in dragonboard.io.gaintypes:

                print(gaintype)

                with open(filename, "rb") as f:

                    for event in tqdm(EventGenerator(f)):

                        if gaintype == dragonboard.io.gaintypes[0]:
                            calib_const_array_pos = pixelindex * dragonboard.io.num_gains
                            stop_cell_array_pos = 0

                        if gaintype == dragonboard.io.gaintypes[1]:
                            calib_const_array_pos = pixelindex * dragonboard.io.num_gains +1
                            stop_cell_array_pos = 1
                        
                        #data[stop_cell[stop_cell_array_pos]:stop_cell[stop_cell_array_pos]+event.roi] = event.data[gaintype][pixelindex]
                        data = np.full(dragonboard.io.max_roi, np.nan)
                        stop_cell = event.header.stop_cells[pixelindex]
                        data[:event.roi] = event.data[gaintype][pixelindex]
                        data = np.roll(data, stop_cell[stop_cell_array_pos])

                        #const_roi = calibration_constants[calib_const_array_pos]

                        #data = event.data[gaintype][pixelindex]
                        #np.roll(data, stop_cell[stop_cell_array_pos])
                        #calibrated_data.append(np.subtract(event.data[gaintype][pixelindex], const_roi))
                        calibrated_data.append(np.subtract(data, calibration_constants[calib_const_array_pos]))
                        #calibrated_data.append(np.subtract(data, calibration_constants[calib_const_array_pos]))
                        # calibrated_data.append(
                        #     np.subtract(np.roll(data, stop_cell[stop_cell_array_pos]), np.roll(calibration_constants[calib_const_array_pos],  stop_cell[stop_cell_array_pos]))
                        #     )
        for i in range(10):                        
            plt.step(calibrated_data[i], ":")
            plt.figure()
        #                 data = np.full(dragonboard.io.max_roi, np.nan)
        #                 stop_cell = event.header.stop_cells[pixelindex]
        #                 data[:event.roi] = event.data[gaintype][pixelindex]

        #             if gaintype == dragonboard.io.gaintypes[0]:

        #                 stop_cell_array_pos = 0

        #             if gaintype == dragonboard.io.gaintypes[1]:

        #                 stop_cell_array_pos = 1

        #             calibrated_data = np.subtract(np.roll(data, stop_cell[stop_cell_array_pos]), calibration_constants)

        # return calibrated_data
        #print(raw_data["high"][0])

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



def scan_datafile_amount(raw_datafile_directory):
    """ assert raw data files '*.dat' to exist """

    amount_of_files = 0

    for filename in glob.glob(os.path.join(raw_datafile_directory, '*.dat')):

        amount_of_files += 1

    if amount_of_files == 0:

        sys.exit("Error: no raw data file(s) found to perform offset calibration")



def is_calibration_constants_existent(calibration_constants_directory):

    is_calib_file_existent = False

    for filename in glob.glob(os.path.join(calibration_constants_directory, '*.csv')):

        if filename == os.path.join(calibration_constants_directory, 'offsets.csv'):
    
            is_calib_file_existent += True

    if is_calib_file_existent == False:

        sys.exit("Error: no calibration constants found to perform offset calibration")



def read_calibration_constants(calibration_constants_directory):
    """read calibration constants saved in offsets.csv. Data column structure: 0low, 0high, 1low, ..., 7high"""

    calibration_constants = []

    for i in range(dragonboard.io.num_channels * dragonboard.io.num_gains):

        calibration_constants.append(
            np.loadtxt(
                os.path.join(
                    calibration_constants_directory, "offsets.csv"),
                    delimiter=",", usecols=([i]), unpack=True
            )
        )

    #print(calibration_constants)
    # for i in range(16):
    #     print(len(calibration_constants[i]))
    # print(len(calibration_constants))
    # # for i in range(16):
    #     plt.plot(calibration_constants[i], "+")
    #     plt.figure()


    return calibration_constants


if __name__ == '__main__':
    arguments = docopt(__doc__, version='0.1')
    raw_datafile_directory = arguments["<raw_datafile_directory>"]
    calibration_constants_directory = arguments["<calibration_constants_directory>"]
    output_directory = arguments["<output_directory>"]

    scan_datafile_amount(raw_datafile_directory)
    is_calibration_constants_existent(calibration_constants_directory)
    apply_offset_calibration(raw_datafile_directory, calibration_constants_directory)

    plt.show()