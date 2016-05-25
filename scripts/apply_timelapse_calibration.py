"""
Usage:
    apply_timelapse_calibration.py <inputfile> <fit_constants> <outpath> [options]

Options:
    -n <cores>       Cores to use [default: 1]
    -v <verbosity>   Verbosity [default: 10]
    -m <max_events>  Maximum number of Events

apply TimelapseCalibration for given .dat file(s):
inputfile: .dat file
fit constants: fit_delta_t.py output file
outpath: path where to put generated data.pickle
"""
from dragonboard import EventGenerator
from dragonboard.calibration import TimelapseCalibration
from joblib import Parallel, delayed
from docopt import docopt
import pickle
import os

def calc_data(event):
    """Calibrate data"""
    return calib(event)


if __name__ == '__main__':
    args = docopt(
        __doc__, version='Dragon Board Timelapse Calibration v.1.0'
    )

    calib = TimelapseCalibration(args['<fit_constants>'])

    events = EventGenerator(
        args['<inputfile>'],
        max_events=int(args['-m']) if args['-m'] else None,
    )

    with Parallel(int(args['-n']), verbose=int(args['-v'])) as pool:

        data = list(
            pool(
                delayed(calc_data)(event) for event in events
            )
        )

    file = (args['<outpath>'] + '{}_calib.pickle'.format(os.path.basename(args['<inputfile>'])))
    max_data_size = 10000

    for i in range(int(len(data) / max_data_size)):
        try:
            f = open(file, "rb")
            calib_data = []
            while 1:
                try:
                    calib_data.append(pickle.load(f))
                except EOFError:
                    break
        except:
            pass
        f = open(file, "wb")
        try:
            for data_item in calib_data:
                pickle.dump(data_item, f)
        except:
            pass
        pickle.dump(data[i * max_data_size:(i + 1) * max_data_size - 1], f)
        f.close()
