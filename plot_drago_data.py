""" Plot Dragon Board Raw

Show drag data interactively. Just click on the plotting canvas to
see the next event.


Usage:
  plot_drago_data.py <filename>
  plot_drago_data.py (-h | --help)
  plot_drago_data.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.

"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from dragonboard import event_generator
from docopt import docopt


class DragonBrowser(object):

    def __init__(self, generator):
        self.generator = generator

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.fig.suptitle("Click somewhere to see the next event")
        self.fig.canvas.mpl_connect('key_release_event', self.onkeypress)

        self.drago_event = next(self.generator)
        self.update()

    def onkeypress(self, event):
        if event.key == 'right':
            self.drago_event = next(self.generator)
            self.update()

    def update(self):

        self.ax2.cla()
        self.ax1.cla()
        self.ax1.set_title("Low Gain Channel")
        self.ax2.set_title("High Gain Channel")
        event = self.drago_event
        for channel, (stop_cell, data) in enumerate(
            zip(event.header.stop_cells, event.data['low'])
        ):

            x = np.arange(stop_cell, stop_cell + event.roi) % 4096
            self.ax1.plot(x, data, '-', label=str(channel) + ' low')

        for channel, (stop_cell, data) in enumerate(
            zip(event.header.stop_cells, event.data['high'])
        ):

            x = np.arange(stop_cell, stop_cell + event.roi) % 4096
            self.ax2.plot(x, data, '-', label=str(channel) + ' high')

        self.ax1.legend()
        self.ax2.legend()
        self.fig.canvas.draw()

    def show(self):
        plt.show()


def main():
    arguments = docopt(__doc__, version='Dragon Data Browser 0.1alpha')
    with open(arguments["<filename>"], "rb") as f:
        generator = event_generator(f)

        browser = DragonBrowser(generator)
        browser.fig.suptitle(
            "DragoCam raw data. {}".format(arguments["<filename>"])
        )
        browser.show()


if __name__ == '__main__':
    main()
