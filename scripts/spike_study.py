"""
Usage:
    spike_study.py <inputfile> [options]

Options:
    -n <cores>       Cores to use [default: 1]
    -v <verbosity>   Verbosity [default: 10]
    -m <min_adc>     Minimum ADC count value to be considered [default: 0]

analyze spike data for given inputfile.pickle:
inputfile: .pickle file generated by extract_spike_data.py
"""

from joblib import Parallel, delayed
from docopt import docopt
import matplotlib.pyplot as plt
import pickle

def convert_spike_data(spike):
    """Select spike data"""
    return [int(spike[0]), int(spike[1]), spike[2], int(spike[3]), int(spike[4]), int(spike[5])]


if __name__ == '__main__':
    args = docopt(
        __doc__, version='Dragon Board spike study software v.1.0'
    )

    f = open(args['<inputfile>'], "rb")
    spike_data = []
    while 1:
        try:
            spike_data.append(pickle.load(f))
        except EOFError:
            break

    spikes = list(spike for spikes in spike_data for spike in spikes)

    min_adc = int(args['-m']) if args['-m'] else 0
    spikes = list(spike for spike in spikes if spike[0] >= min_adc)

    with Parallel(int(args['-n']), verbose=int(args['-v'])) as pool:

        spikes = list(
            pool(
                delayed(convert_spike_data)(spike) for spike in spikes
            )
        )

    double_spikes = []
    triple_spikes = []
    for i in range(len(spikes) - 2):
        if spikes[i][1:3] == spikes[i + 1][1:3]:
            if (spikes[i + 1][4] - spikes[i][4]) == 1:
                double_spikes.append(spikes[i])
                double_spikes.append(spikes[i + 1])
                if spikes[i][1:3] == spikes[i + 1][1:3] == spikes[i + 2][1:3]:
                    if (spikes[i + 2][4] - spikes[i + 1][4]) == 1:
                        triple_spikes.append(spikes[i])
                        triple_spikes.append(spikes[i + 1])
                        triple_spikes.append(spikes[i + 2])

    # spikes = double_spikes
    # double_spikes = list(double_spike for double_spike in double_spikes if double_spike not in triple_spikes)
    # single_spikes = list(single_spike for single_spike in spikes if single_spike not in double_spikes)
    # print(len(double_spikes))
    # print(len(triple_spikes))
    # print(len(spikes))
    spikes = triple_spikes
    # spikes = double_spikes
    # spikes = single_spikes

    adc = list(spike[0] for spike in spikes)
    pixel = list(spike[1] for spike in spikes)
    channel = list(spike[2] for spike in spikes)
    event_id = list(spike[3] for spike in spikes)
    cell_id = list(spike[4] for spike in spikes)
    sample_id = list(spike[5] for spike in spikes)

    # double_spikes = list(spikes[i] for i in double_spikes)

    # print(double_spikes)
    # print(len(double_spikes))

    # hist = dict((gain, channel.count(gain)) for gain in channel)
    # print(hist)
    # [adc, pixel, channel, event_id, cell_id, sample_id]
    # [23,2,"high",502,19] links correctly to the following @skip=500
    # plt.plot(calib_events[1].data[2]["high"][5:36])

    plt.style.use('ggplot')
    # plt.figure()
    # plt.hist(adc, bins=200)
    # plt.hist(cell_id, bins=4096)
    # plt.hist(sample_id, bins=120)
    # plt.hist(pixel, bins=50)
    plt.scatter(event_id, adc)
    # plt.scatter(cell_id, adc)
    # plt.scatter(sample_id, cell_id)
    # plt.title("Histogram: ADC > {} ({} triple-or-higher spikes)".format(min_adc, len(spikes)))
    # plt.xlabel("ADC Counts")
    # plt.ylabel("Occurence")

    plt.title("Histogram: Sample_id ADC > {} ({} triple-or-higher spikes)".format(min_adc, len(spikes)))
    plt.xlabel("Sample_id")
    plt.ylabel("Occurence")

    plt.show()
