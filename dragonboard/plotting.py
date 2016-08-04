from __future__ import division, print_function, absolute_import
import numpy as np
from PyQt4 import QtCore, QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt4 import NavigationToolbar2QT
from matplotlib.figure import Figure
from collections import defaultdict
from functools import partial
from matplotlib.colors import ColorConverter
import os
import sys
import inspect
from io import StringIO

from .io import EventGenerator
from .calibration import TimelapseCalibration, TimelapseCalibrationExtraOffsets

color_converter = ColorConverter()

sys.path = ["."] + sys.path
try:
    import dragon_analysis as analysis
except ImportError:
    from . import example_analysis as analysis
print(analysis.__file__)

mandatory_parameters = {'axis', 'stop_cell', 'data', 'gain', 'channel'}
analysis_functions = {
    name:func for name, func in inspect.getmembers(analysis, inspect.isfunction) if set(inspect.signature(func).parameters) == mandatory_parameters
    }

colors = [
    "#E24A33",
    "#348ABD",
    "#988ED5",
    "#777777",
    "#FBC15E",
    "#8EBA42",
    "#FFB5B8",
    "#E24A33",
    ]

class NavigationToolbar(NavigationToolbar2QT):
    toolitems = [t for t in NavigationToolbar2QT.toolitems if
                 t[0] in ('Home', 'Pan', 'Zoom', 'Subplots', 'Save', None)]


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
    def __init__(self, filename=None, calibfile=None, extra_offset_file=None, start=None, **kwargs):
        super().__init__(**kwargs)

        self.setWindowTitle('DragonBrowser')

        self.filename = filename or QtGui.QFileDialog.getOpenFileName(
                self, 'Open file', os.environ['HOME']
            )
        if not self.filename:
            sys.exit()

        if extra_offset_file is not None:
            self.calib = TimelapseCalibrationExtraOffsets(calibfile, extra_offset_file)
        elif calibfile is not None:
            self.calib = TimelapseCalibration(calibfile)
        else:
            self.calib = lambda x: x

        self.generator = EventGenerator(self.filename, start=start)
        self.dragon_event = self.calib(next(self.generator))
        self.gains = self.dragon_event.data.dtype.names
        self.n_channels = self.dragon_event.data.shape[0]
        self.is_channel_active = []
        self.channel_color_choice = []
        self.init_gui()

    def init_gui(self):
        right_part = QtGui.QWidget()
        main_window = QtGui.QWidget()
        main_hbox = QtGui.QHBoxLayout()
        right_vbox = QtGui.QVBoxLayout()
        main_window.setLayout(main_hbox)
        right_part.setLayout(right_vbox)
        self.setCentralWidget(main_window)
        

        self.canvas = FigureCanvas(parent=self, width=12.8, height=7.2)
        self.fig = self.canvas.fig

        self.axs = {'high': self.fig.add_subplot(2, 1, 2)}
        self.axs['low'] = self.fig.add_subplot(2, 1, 1, sharex=self.axs['high'])
        self.axs['low'].set_title('Low Gain Channel')
        self.axs['high'].set_title('High Gain Channel')
        self.axs['high'].set_xlim(-0.5, self.dragon_event.roi)

        self.navbar = NavigationToolbar(self.canvas, self)
        self.toolbar = self.addToolBar('Test')
        self.toolbar.addWidget(self.navbar)

        bottom_frame = QtGui.QFrame()
        layout = QtGui.QHBoxLayout(bottom_frame)
        self.event_text = QtGui.QLineEdit(bottom_frame)
        self.event_text.setReadOnly(True)
        self.event_text.setFocusPolicy(QtCore.Qt.NoFocus)
        layout.addWidget(self.event_text)

        
        for channel in range(self.n_channels):
            cb = QtGui.QCheckBox(str(channel), bottom_frame)
            self.is_channel_active.append(cb)

            cb.setFocusPolicy(QtCore.Qt.NoFocus)
            if channel != 7:
                cb.toggle()

            layout.addWidget(cb)

            color_choice = QtGui.QPushButton(bottom_frame)
            self.channel_color_choice.append(color_choice)
            color_choice.setStyleSheet('background-color: {}'.format(colors[channel]))
            color_choice.clicked.connect(partial(
                self.changeColor, channel=channel, button=color_choice)
            )
            color_choice.setFocusPolicy(QtCore.Qt.NoFocus)
            layout.addWidget(color_choice)

        for cb in self.is_channel_active:
            cb.stateChanged.connect(
                partial(self.toggle_channel, channel=channel)
            )


        self.rescale_box = QtGui.QCheckBox('Rescale', bottom_frame)
        self.rescale_box.setFocusPolicy(QtCore.Qt.NoFocus)
        self.rescale_box.toggle()
        layout.addWidget(self.rescale_box)
        
        self.cb_physical = QtGui.QCheckBox('Show Cell ID', bottom_frame)
        self.cb_physical.setFocusPolicy(QtCore.Qt.NoFocus)
        self.cb_physical.stateChanged.connect(self.update)
        layout.addWidget(self.cb_physical)

        btn_next_event = QtGui.QPushButton(bottom_frame)
        btn_next_event.clicked.connect(self.next_event)
        btn_next_event.setFocusPolicy(QtCore.Qt.NoFocus)
        btn_next_event.setText('Next Event')
        layout.addWidget(btn_next_event)

        self.analysis_choice = QtGui.QListWidget(right_part)
        self.analysis_choice.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        for name in analysis_functions:
            self.analysis_choice.addItem(name)
        self.analysis_choice.itemSelectionChanged.connect(self.update)
        
        self.analysis_output = QtGui.QPlainTextEdit(right_part)
                

        self.statusBar().insertWidget(0, bottom_frame)
        for ax in self.axs.values():
            ax.set_ylabel('ADC Counts')
        self.axs['high'].set_xlabel('Time Slice')

        self.fig.tight_layout()



        
        right_vbox.addWidget(self.analysis_choice)
        right_vbox.addWidget(self.analysis_output)

        main_hbox.addWidget(self.canvas)
        main_hbox.addWidget(right_part)




        self.update()

    def changeColor(self, channel, button):
        color = QtGui.QColorDialog(self).getColor().name()
        button.setStyleSheet('background-color: {}'.format(color))
        self.update()

    def toggle_channel(self, channel):
        self.update()

    def update(self):
        event = self.dragon_event

        for gain in self.gains:
            self.axs[gain].clear()
            for channel in range(event.data.shape[0]):
                if self.is_channel_active[channel].isChecked():
                    stop_cell = event.header.stop_cells[channel][gain]
                    x = np.arange(event.roi)
                    if self.cb_physical.isChecked():
                        x = (x + stop_cell) % 4096
                    self.axs[gain].plot(
                        x, 
                        event.data[gain][channel], 
                        label='Ch{}'.format(channel), 
                        drawstyle="steps",
                        color=self.channel_color_choice[channel].styleSheet()[18:]
                    )

        for ax in self.axs.values():
            ax.relim()
            if self.rescale_box.isChecked():
                ax.autoscale(enable=True)
            ax.autoscale_view()

        if self.cb_physical.isChecked():
            self.axs['high'].set_xlabel('Cell ID')
        else:
            self.axs['high'].set_xlabel('Sample ID')
            self.axs['high'].set_xlim(-0.5, event.roi)

        self.analysis()
        self.fig.canvas.draw()
        self.event_text.setText('Event: {}'.format(
            self.dragon_event.header.event_counter
        ))

    def analysis(self):
        try:
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()

            analysis_choice = [ analysis_functions[item.text()] for item in self.analysis_choice.selectedItems() ]

            gain = "high"
            for func in analysis_choice:
                for channel in range(self.dragon_event.data.shape[0]):
                    if self.is_channel_active[channel].isChecked():
                        stop_cell = self.dragon_event.header.stop_cells[channel][gain]
                        data = self.dragon_event.data[gain][channel]
                        func(channel=channel, gain=gain, data=data, stop_cell=stop_cell, axis=self.axs[gain])

            self.analysis_output.setPlainText(mystdout.getvalue())
        finally:
            sys.stdout = old_stdout

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Right:
            self.next_event()

    def next_event(self):
        try:
            self.dragon_event = self.calib(next(self.generator))
        except StopIteration:
            pass
        else:
            self.update()

    def closeEvent(self, event):
        event.accept()
        QtCore.QCoreApplication.instance().quit()
