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
import read_lst_stuff
from  docopt import docopt


class DragoBrowser(object):

    def __init__(self, generator):
        self.generator = generator
        self.drago_event = next(generator)
        self.update()

    def onmouserelease(self, event):
        self.drago_event = next(generator)
        self.update()

    def update(self):

        ax2.cla()
        ax1.cla()
        header, read_depth, d = self.drago_event
        stop_cells = header[3]
        ax1.set_title("Low Gain Data")
        for i in range(14):
            sc = stop_cells[i%8]
            x = np.arange(sc, sc+read_depth)
            ax1.plot(x, d[i], '.', label=str(i))

        ax2.set_title("High Gain Data")
        for i in range(14, 16):
            sc = stop_cells[i%8]
            x = np.arange(sc, sc+read_depth)
            ax2.plot(x, d[i], '.', label=str(i))

        fig.canvas.draw()

if __name__ == '__main__':
    arguments = docopt(__doc__, version='Dragon Data Browser 0.1alpha')
    f = open(arguments["<filename>"], "rb")
    generator = read_lst_stuff.event_generator(f)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title("DragoCam raw data. {}".format(arguments["<filename>"]))
    browser = DragoBrowser(generator)
    fig.suptitle("Click somewhere to see the next event")
    fig.canvas.mpl_connect('button_release_event', browser.onmouserelease)

    plt.show()
