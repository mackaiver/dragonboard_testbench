'''
Usage:
    dragonviewer [<inputfile>]
'''
import matplotlib
import matplotlib.style
matplotlib.use('Qt4Agg')
import sys
import signal
from docopt import docopt
from .plotting import DragonBrowser

from PyQt4 import QtGui
from PyQt4 import QtCore


def sigint_handler(*args):
    sys.stderr.write('\rReceived SIGINT, terminating\n')
    QtGui.QApplication.instance().quit()


def main():
    try:
        matplotlib.style.use('ggplot')
    except NameError:
        pass
    args = docopt(__doc__)
    qApp = QtGui.QApplication(sys.argv)
    signal.signal(signal.SIGINT, sigint_handler)

    widget = DragonBrowser(args['<inputfile>'])
    widget.show()

    # let the QApplication process signals from the python thread
    # see http://stackoverflow.com/a/4939113/3838691
    timer = QtCore.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    sys.exit(qApp.exec_())


if __name__ == '__main__':
    main()
