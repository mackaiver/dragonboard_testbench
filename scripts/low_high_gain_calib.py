""" low_high_gain_calib

Reads in all dat files in a folder and caluclates for each dat file the ratio between the signal 
in the high gain to the low gain channel.
Therefore the following steps are performed:
- Rough baseline substraction (substracting the mean of the first 20 slices)
- finding the maximum amplitude in low and high gain channel in each pixel
- calculating the ratio between the max amplitudes for each pixel
- averaging over all events of the dat file
The calculated ratio (one per dat file) is than plotted against the average maximum amplitude of the
high gain channel for each pixel.
You would expect for low high gain amplitudes noise effects and therefore a small ratio,
for high high gain values saturation effect and therefore a small ratio,
and only for middle high gain values the correct ratio (which should be roughly 20)

The dat files should have injected pulses (or even photon signals) in all pixels. The height of the
pulses should differ from file to file (but not from event to event in one file)


Usage:
  crosstalk_study.py <folder>
  crosstalk_study.py (-h | --help)
  crosstalk_study.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.

"""

import numpy as np
import os
from dragonboard import EventGenerator,read
from dragonboard.io import get_roi,guess_event_size,num_channels,num_gains,max_roi
from docopt import docopt
import matplotlib.pyplot as plt
import pprint

def calculatedIntegral(data,posMax,intWindow=3):
    slices = np.arange(posMax-intWindow,posMax+intWindow+1)
    print posMax
    print slices
    integral = 0
    for sl in slices:
        integral += data[sl]
    return integral

def find_max_amplitude(data,min,max):
    pos = np.argmax(data[min:max])+min
    maxAmplitude = data[pos-min]
    return pos,maxAmplitude

def estimateBaseline(data,min=0,max=20):
    baseline = np.mean(data[min:max])
    return baseline

def main():
    arguments = docopt(__doc__, version='Dragon Data Browser 0.1alpha')
    folder = arguments["<folder>"]

    xdata = dict()
    ydata = dict()

    for pix in range(num_channels):
        xdata[pix] = list()
        ydata[pix] = list()

    for (dirpath, dirnames, filenames) in os.walk(folder):
        for filename in filenames:
            path = os.path.join(dirpath,filename)
            if not path.endswith(".dat"):
                continue

            print path

            generator = read(path)
            nEvents = len(generator)
            ratios = np.zeros((num_channels,nEvents))
            pulsePositionsLow = np.zeros((num_channels,nEvents))
            pulsePositionsHigh = np.zeros((num_channels,nEvents))
            pulseMaxLow = np.zeros((num_channels,nEvents))
            pulseMaxHigh = np.zeros((num_channels,nEvents))
            for i,event in enumerate(generator):
                for pix in range(num_channels):
                    dataLow = event.data["low"][pix].astype(float)
                    dataLow -= estimateBaseline(dataLow,0,20)
                    dataHigh = event.data["high"][pix].astype(float)
                    dataHigh -= estimateBaseline(dataHigh,0,20)
                    pulsePositionsLow[pix][i], pulseMaxLow[pix][i] = find_max_amplitude(dataLow,0,90)
                    pulsePositionsHigh[pix][i], pulseMaxHigh[pix][i] = find_max_amplitude(dataHigh,0,90)
                    ratios[pix][i] = float(pulseMaxHigh[pix][i])/pulseMaxLow[pix][i]

            averagedRatios = np.zeros(num_channels)
            averagedMaxHigh = np.zeros(num_channels)
            averagedMaxLow = np.zeros(num_channels)

            for pix in range(num_channels):
                averagedRatios[pix] = np.mean(ratios[pix])
                averagedMaxHigh[pix] = np.mean(pulseMaxHigh[pix])
                averagedMaxLow[pix] = np.mean(pulseMaxLow[pix])
                xdata[pix].append(averagedMaxHigh[pix])
                ydata[pix].append(averagedRatios[pix])


    for pix in range(num_channels):
        plt.plot(xdata[pix],ydata[pix],'*')
        plt.title("Pixel: {}".format(pix))
        plt.show()

        # for pix in range(num_channels):
        #     plt.hist(pulsePositionsLow[pix],bins=50,color="r", label="low")
        #     plt.hist(pulsePositionsHigh[pix],bins=50,color = "b", label="high")
        #     plt.title("Pixel: {}".format(pix))
        #     plt.legend()
        #     plt.show()
        #     plt.hist(ratios[pix],bins=50)
        #     plt.title("Pixel: {}".format(pix))
        #     plt.show()



if __name__ == '__main__':
    main()
