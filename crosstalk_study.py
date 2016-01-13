""" Plot Dragon Board Raw

Show drag data interactively. Just click on the plotting canvas to
see the next event.


Usage:
  crosstalk_study.py <filename> <gainChannel> <pixelWithInjectedPulse>
  crosstalk_study.py (-h | --help)
  crosstalk_study.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.

"""

import numpy as np
from dragonboard import EventGenerator
from dragonboard.io import get_roi,guess_event_size,num_channels,num_gains,max_roi
from docopt import docopt
import matplotlib.pyplot as plt

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
    maxAmplitude = data[pos]
    return pos,maxAmplitude

def main():
    arguments = docopt(__doc__, version='Dragon Data Browser 0.1alpha')
    gainChannel = arguments["<gainChannel>"]
    pulsePixel = int(arguments["<pixelWithInjectedPulse>"])
    leftWindow=20
    rightWindow=30
    averagedShiftedData = np.zeros((num_channels,rightWindow+leftWindow))
    with open(arguments["<filename>"], "rb") as f:
        generator = EventGenerator(f)
        num_events = 0

        for event in generator:
            num_events += 1
            pulsePos, pulseMax = find_max_amplitude(event.data[gainChannel][pulsePixel],20,80)
            xvalues = np.arange(pulsePos - leftWindow,pulsePos+rightWindow)

            for pix in range(num_channels):
                data = event.data[gainChannel][pix]
                averagedShiftedData[pix][:] += data[xvalues]

        xvalues = np.arange(-leftWindow,rightWindow)
        for pix in range(num_channels):
            averagedShiftedData[pix] /= num_events
            plt.plot(xvalues,averagedShiftedData[pix])
        plt.show()


        # for event in generator:
        #     for gainChannel in ["low","high"]:
        #         for pix in range(8):
        #             data = event.data[gainChannel][pix]
        #             pos,maxAmplitude = find_max_amplitude(data,20,80)
        #             integral = calculatedIntegral(data,pos)
        #             print "Pixel: {}, {} gain, Maximum: (position: {}, value: {}), Integral: {}".format(pix,gainChannel,pos,maxAmplitude,integral)


if __name__ == '__main__':
    main()
