''' Read Dragon Board Data '''

from __future__ import division, print_function, absolute_import
import struct
import numpy as np
from collections import namedtuple

EventHeader = namedtuple('EventHeader', [
    'event_counter',
    'trigger_counter',
    'counter_133MHz',
    'counter_10MHz',
    'pps_counter',
    'stop_cells',
    'flag',
])

Event = namedtuple('Event', ['header', 'roi', 'data'])

max_roi = 4096
gaintypes = ["low", "high"]
stop_cell_dtype = np.dtype('uint16').newbyteorder('>')
stop_cell_size = 8 * stop_cell_dtype.itemsize
expected_relative_address_of_flag = 16
timestamp_conversion_to_s = 7.5e-9
num_channels = 8
num_gains = 2
adc_word_size = 2

def get_event_size(roi):
    ''' return event_size in bytes, based on roi in samples.
    '''
    header_size = 4 * 16
    body_size = roi * adc_word_size * num_gains * num_channels 
    return header_size + body_size


def get_roi(event_size):
    ''' return roi in samples, based on event_size in bytes.
    '''
    header_size = 4 * 16
    body_size = event_size - header_size
    roi = body_size/(adc_word_size * num_gains * num_channels)
    assert roi.is_integer()
    return int(roi)


def read_header(f):
    ''' return EventHeader from file f
    '''
    (
        header_aaaa, # should be constant 0xaaaa
        pps_counter, # pps: pulse per second = 1Hz
        counter_10MHz,
        event_counter,
        trigger_counter,
        counter_133MHz,
        data_header_all_ds, # should be constant 8 times 0xdd
        flags,
    ) = struct.unpack('!HHIIIQQ16s', f.read(struct.calcsize('!HHIIIQQ16s')))

    assert header_aaaa == 0xaaaa, \
        "Header is not 0xaaaa but {!r}".format(header_aaaa)
    assert data_header_all_ds == 0xdddddddddddddddd, \
        "data_header is not 8x 0xdd but {!r}".format(data_header_all_ds)

    stop_cells_for_user = np.empty(
        num_channels, dtype=[('low', 'i2'), ('high', 'i2')]
    )
    stop_cells__in_drs4_chip_order = np.frombuffer(
        f.read(stop_cell_size), dtype=stop_cell_dtype)

    stop_cells_for_user["high"][0] = stop_cells__in_drs4_chip_order[0]
    stop_cells_for_user["high"][1] = stop_cells__in_drs4_chip_order[0]

    stop_cells_for_user["high"][2] = stop_cells__in_drs4_chip_order[2]
    stop_cells_for_user["high"][3] = stop_cells__in_drs4_chip_order[2]

    stop_cells_for_user["high"][4] = stop_cells__in_drs4_chip_order[4]
    stop_cells_for_user["high"][5] = stop_cells__in_drs4_chip_order[4]

    stop_cells_for_user["high"][6] = stop_cells__in_drs4_chip_order[6]
    stop_cells_for_user["high"][7] = stop_cells__in_drs4_chip_order[6]

    stop_cells_for_user["low"][0] = stop_cells__in_drs4_chip_order[1]
    stop_cells_for_user["low"][1] = stop_cells__in_drs4_chip_order[1]

    stop_cells_for_user["low"][2] = stop_cells__in_drs4_chip_order[3]
    stop_cells_for_user["low"][3] = stop_cells__in_drs4_chip_order[3]

    stop_cells_for_user["low"][4] = stop_cells__in_drs4_chip_order[5]
    stop_cells_for_user["low"][5] = stop_cells__in_drs4_chip_order[5]

    stop_cells_for_user["low"][6] = stop_cells__in_drs4_chip_order[7]
    stop_cells_for_user["low"][7] = stop_cells__in_drs4_chip_order[7]

    return EventHeader(
        event_counter, 
        trigger_counter, 
        counter_133MHz, 
        counter_10MHz, 
        pps_counter,
        stop_cells_for_user,
        flags
    )


def read_data(f, roi):
    ''' return array of raw ADC data, shape:(16, roi)

    The ADC data, is just a contiguous bunch of 16bit integers
    So its easy to read.
    However the assignment of integers to the 16 channels
    is not soo easy. I hope I did it correctly.
    '''
    d = np.fromfile(f, '>i2', num_gains * num_channels * roi)

    N = num_gains * num_channels * roi
    roi_dtype = '{}>i2'.format(roi)
    array = np.empty(
        num_channels, dtype=[('low', roi_dtype), ('high', roi_dtype)]
    )
    data_odd = d[N / 2:]
    data_even = d[:N / 2]
    for channel in range(0, num_channels, 2):
        array['high'][channel] = data_even[channel::8]
        array['low'][channel] = data_even[channel + 1::8]
        array['high'][channel + 1] = data_odd[channel::8]
        array['low'][channel + 1] = data_odd[channel + 1::8]

    return array


def guess_event_size(f):
    ''' try to find out the event size for this file.

    Each even header contains a flag.
    The distance between two flags is just the event size.
    '''
    current_position = f.tell()
    f.seek(0)

    max_event_size = get_event_size(roi=max_roi)
    # I don't believe myself, so I add 50% here
    chunk_size = int(max_event_size * 1.5)
    chunk = f.read(chunk_size)

    flag = b'\xdd'*8
    first_flag = chunk.find(flag)
    second_flag = chunk.find(flag, first_flag + 1)

    event_size = second_flag - first_flag
    f.seek(current_position)

    return event_size


def read(path, max_events=None):
    ''' return list of Events in file path '''
    with open(path, 'rb') as f:
        return list(EventGenerator(f, max_events=None))


class EventGenerator(object):
    def __init__(self, file_descriptor, max_events=None):
        self.file_descriptor = file_descriptor
        self.event_size = guess_event_size(file_descriptor)
        self.max_events = max_events

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def previous(self):
        try:
            self.file_descriptor.seek(- 2 * self.event_size, 1)
        except OSError:
            raise ValueError('Already at first event')
        return self.next()

    def next(self):
        try:
            event_header = read_header(self.file_descriptor)
            event_size = guess_event_size(self.file_descriptor)
            roi = get_roi(event_size)
            self.roi = roi
            data = read_data(self.file_descriptor, roi)

            if self.max_events is not None:
                if event_header.event_counter > self.max_events:
                    raise StopIteration

            return Event(event_header, roi, data)

        except struct.error:
            raise StopIteration
