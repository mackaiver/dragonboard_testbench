import numpy as np
from PyQt4 import QtCore, QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt4 import NavigationToolbar2QT
from matplotlib.figure import Figure
from collections import defaultdict
from functools import partial
from matplotlib.colors import ColorConverter
import os

from .io import EventGenerator

color_converter = ColorConverter()


def mpl2rgb(color):
    return tuple(int(c * 255) for c in color_converter.to_rgb(color))


class FigureCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvasQTAgg.__init__(self, self.fig)

        self.setParent(parent)
        FigureCanvas.setSizePolicy(
            self,
            QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding,
        )
        FigureCanvas.updateGeometry(self)


class DragonBrowser(QtGui.QMainWindow):
    def __init__(self, filename=None, **kwargs):
        QtGui.QMainWindow.__init__(self, **kwargs)

        self.setWindowTitle('DragonBrowser')

        self.file = None
        self.open_new_file(filename)
        self.generator = EventGenerator(self.file)
        self.dragon_event = next(self.generator)
        self.gains = self.dragon_event.data.dtype.names
        self.n_channels = self.dragon_event.data.shape[0]
        self.init_gui()

    def open_new_file(self, filename):
        if self.file:
            self.file.close()
        if not filename:
            filename = QtGui.QFileDialog.getOpenFileName(
                self, 'Open file', os.environ['HOME']
            )
        self.file = open(filename, 'rb')

    def init_gui(self):
        self.canvas = FigureCanvas(self, 12.8, 7.2)
        self.setCentralWidget(self.canvas)
        self.fig = self.canvas.fig

        self.axs = {'high': self.fig.add_subplot(2, 1, 2)}
        self.axs['low'] = self.fig.add_subplot(2, 1, 1, sharex=self.axs['high'])
        self.axs['low'].set_title('Low Gain Channel')
        self.axs['high'].set_title('High Gain Channel')

        self.navbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar = self.addToolBar('Test')
        self.toolbar.addWidget(self.navbar)

        bottom_frame = QtGui.QFrame()
        layout = QtGui.QHBoxLayout(bottom_frame)
        self.text = QtGui.QLineEdit(bottom_frame)
        self.text.setReadOnly(True)
        self.text.setFocusPolicy(QtCore.Qt.NoFocus)
        layout.addWidget(self.text)

        self.plots = defaultdict(dict)
        for channel in range(self.n_channels):
            for gain in self.gains:
                plot, = self.axs[gain].plot([], [])
                self.plots[gain][channel] = plot

            button = QtGui.QPushButton(bottom_frame)
            button.setStyleSheet('background-color: rgb({}, {}, {});'.format(
                *mpl2rgb(plot.get_color())
            ))
            button.clicked.connect(partial(
                self.changeColor, channel=channel, button=button)
            )
            button.setFocusPolicy(QtCore.Qt.NoFocus)
            layout.addWidget(button)

            cb = QtGui.QCheckBox(str(channel), bottom_frame)
            cb.toggle()
            cb.setFocusPolicy(QtCore.Qt.NoFocus)
            cb.stateChanged.connect(
                partial(self.toggle_channel, channel=channel)
            )
            layout.addWidget(cb)
        self.statusBar().insertWidget(0, bottom_frame)

        button = QtGui.QPushButton(bottom_frame)
        button.clicked.connect(self.previous_event)
        button.setFocusPolicy(QtCore.Qt.NoFocus)
        button.setText('Previous')
        layout.addWidget(button)

        button = QtGui.QPushButton(bottom_frame)
        button.clicked.connect(self.next_event)
        button.setFocusPolicy(QtCore.Qt.NoFocus)
        button.setText('Next')
        layout.addWidget(button)


        for ax in self.axs.values():
            ax.set_xlabel('Physical Cell')
            ax.set_ylabel('ADC counts')

        self.fig.tight_layout()
        self.update()

    def changeColor(self, channel, button):
        diag = QtGui.QColorDialog(self)
        color = diag.getColor().name()
        button.setStyleSheet('background-color: {}'.format(color))

        for gain in self.gains:
            self.plots[gain][channel].set_color(color)
        self.canvas.draw()

    def toggle_channel(self, channel):
        for gain in self.gains:
            plot = self.plots[gain][channel]
            plot.set_visible(not plot.get_visible())
        self.canvas.draw()

    def update(self):
        event = self.dragon_event

        for gain in self.gains:
            for channel, stop_cell in enumerate(event.header.stop_cells):
                x = np.arange(stop_cell, stop_cell + event.roi) % 4096
                self.plots[gain][channel].set_data(x, event.data[gain][channel])

        for ax in self.axs.values():
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw()
        self.text.setText('Event: {}'.format(
            self.dragon_event.header.event_counter
        ))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Right:
            self.next_event()

        if event.key() == QtCore.Qt.Key_Left:
            self.previous_event()

    def next_event(self):
        self.dragon_event = next(self.generator)
        self.update()

    def previous_event(self):
        try:
            self.dragon_event = self.generator.previous()
        except ValueError:
            pass
        else:
            self.update()

    def closeEvent(self, event):
        self.file.close()
        event.accept()
        QtCore.QCoreApplication.instance().quit()
