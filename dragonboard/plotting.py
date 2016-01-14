from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt


class DragonBrowser(object):

    def __init__(self, generator):
        self.generator = generator

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.fig.suptitle('Click somewhere to see the next event')
        self.fig.canvas.mpl_connect('key_release_event', self.onkeypress)

        self.dragon_event = next(self.generator)
        self.update()

    def onkeypress(self, event):
        print(event.key)
        if event.key == 'right':
            self.dragon_event = next(self.generator)
            self.update()

        if event.key == 'left':
            try:
                self.dragon_event = self.generator.previous()
            except ValueError:
                print('already at first event')
            else:
                self.update()

    def update(self):

        self.ax2.cla()
        self.ax1.cla()
        self.ax1.set_title('Low Gain Channel')
        self.ax2.set_title('High Gain Channel')
        event = self.dragon_event
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
