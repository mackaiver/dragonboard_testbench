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
        self.fig.canvas.mpl_connect('button_release_event', self.onmouserelease)

        self.drago_event = next(self.generator)
        self.update()

    def onmouserelease(self, event):
        self.drago_event = next(self.generator)
        self.update()

    def update(self):

        self.ax2.cla()
        self.ax1.cla()
        header, read_depth, d = self.drago_event
        stop_cells = header[3]
        self.ax1.set_title("Raw Data")
        for i in range(14):
            sc = stop_cells[i % 8]
            x = np.arange(sc, sc+read_depth)
            self.ax1.plot(x, d[i], '.', label=str(i))

        self.ax2.set_title("DRS Tag Data")
        for i in range(14, 16):
            sc = stop_cells[i % 8]
            x = np.arange(sc, sc+read_depth)
            self.ax2.plot(x, d[i], '.', label=str(i))

        self.fig.canvas.draw()

    def show(self):
        plt.show()


def main():
    arguments = docopt(__doc__, version='Dragon Data Browser 0.1alpha')
    with open(arguments["<filename>"], "rb") as f:
        generator = event_generator(f)

        browser = DragonBrowser(generator)
        browser.ax1.set_title(
            "DragoCam raw data. {}".format(arguments["<filename>"])
        )
        browser.show()


if __name__ == '__main__':
    main()
