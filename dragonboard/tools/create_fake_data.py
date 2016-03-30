'''
Create dragonboard data files with random noise

Usage:
    create_fake_pedestals.py <outputfile> [options]

Options:
    -v <v>, --version=<v>      file version for the output file,
                               v5_1_05 or v5_1_0B [default: v5_1_0B]

    -n <N>, --num-events=<N>   Number of events to produce [default: 100]
'''
import numpy as np
import struct
from docopt import docopt


def write_header_v5_1_0B(
        f,
        pps_counter,
        event_counter,
        trigger_counter,
        counter_10MHz,
        counter_133MHz,
        ):
    header_bytes = struct.pack(
        '!HHIIIQQ16s',
        0xaaaa,
        pps_counter,
        counter_10MHz,
        event_counter,
        trigger_counter,
        counter_133MHz,
        0xdddddddddddddddd,
        b'\xf0\x01\xf0\x03\xf0\x01\xf0\x00\xf0\x00\xf0\x00\xf0\x03\xf0\x01'
    )

    f.write(header_bytes)


def write_header_v5_1_05(
        f,
        event_counter,
        trigger_counter,
        clock,
        ):
    header_bytes = struct.pack(
        '!IIQ16s',
        event_counter,
        trigger_counter,
        clock,
        b'\xf0\x02\xf0\x02\xf0\x02\xf0\x02\xf0\x02\xf0\x02\xf0\x02\xf0\x02'
    )

    f.write(header_bytes)


def write_adc_data(f, data):
    data = data.astype('>i2')
    data.tofile(f)


def write_stop_cells(f, stop_cells):
    stop_cells = stop_cells.astype('<i2')
    stop_cells.tofile(f)


def create_noise_file(filename, version='v5_1_0B', num_events=100, mean=100, std=5):

    assert version in ('v5_1_0B', 'v5_1_05'), 'Unsupported Version: {}'.format(version)

    with open(filename, 'wb') as f:
        for event_counter in range(num_events):

            if version == 'v5_1_0B':
                write_header_v5_1_0B(
                    f,
                    pps_counter=0,
                    event_counter=event_counter,
                    trigger_counter=0,
                    counter_10MHz=event_counter * 1000,
                    counter_133MHz=event_counter * 13300,
                )
            elif version == 'v5_1_05':
                write_header_v5_1_05(
                    f,
                    event_counter=event_counter,
                    trigger_counter=0,
                    clock=event_counter * 1000,
                )

            write_stop_cells(f, np.zeros(8))
            write_adc_data(f, np.random.normal(mean, std, 16 * 1024))


def main():
    args = docopt(__doc__)

    create_noise_file(
        args['<outputfile>'],
        version=args['--version'],
        num_events=int(args['--num-events']),
    )

if __name__ == '__main__':
    main()
