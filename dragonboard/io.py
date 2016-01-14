''' Read Dragon Board Data '''

from __future__ import division, print_function, absolute_import
import struct
import numpy as np
from collections import namedtuple

EventHeader = namedtuple('EventHeader', [
    'event_counter',
    'trigger_counter',
    'timestamp',
    'stop_cells',
    'flag',
])

Event = namedtuple('Event', ['header', 'roi', 'data'])

max_roi = 4096
header_size_in_bytes = 32
stop_cell_dtype = np.dtype('uint16').newbyteorder('>')
stop_cell_size = 8 * stop_cell_dtype.itemsize
expected_relative_address_of_flag = 16
timestamp_conversion_to_s = 7.5e-9
num_channels = 8
num_gains = 2


def get_event_size(roi):
    ''' return event_size in bytes, based on roi in samples.
    '''
    return 16 * (2 * roi + 3)


def get_roi(event_size):
    ''' return roi in samples, based on event_size in bytes.
    '''

    roi = ((event_size / 16) - 3) / 2
    assert roi.is_integer()
    return int(roi)


def read_header(f, flag=None):
    ''' return EventHeader from file f

    if a *flag* is provided, we can check if the header
    looks correct. If not, we can't check anything.
    '''
    chunk = f.read(header_size_in_bytes)
    # the format string:
    #   ! -> network endian order
    #   I -> integer
    #   Q -> unsingned long
    #   s -> char
    #   H -> unsigned short
    (
        event_id,
        trigger_id,
        clock,
        found_flag,
    ) = struct.unpack('!IIQ16s', chunk)
    stop_cells = np.frombuffer(f.read(stop_cell_size), dtype=stop_cell_dtype)
    timestamp_in_s = clock * timestamp_conversion_to_s

    if flag is not None:
        msg = ('event header looks wrong: '
               'flag is not at the right position\n'
               'found: {}, expected: {}'.format(found_flag, flag))

        assert chunk.find(flag) == expected_relative_address_of_flag, msg

    return EventHeader(
        event_id, trigger_id, timestamp_in_s, stop_cells, found_flag
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
    data_odd = d[N/2:]
    data_even = d[:N/2]
    for channel in range(0, num_channels, 2):
        array['high'][channel] = data_even[channel::8]
        array['low'][channel] = data_even[channel+1::8]
        array['high'][channel + 1] = data_odd[channel::8]
        array['low'][channel + 1] = data_odd[channel+1::8]

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

    # the flag should be found two times in the chunk:
    #  1.) in the very first 48 bytes as part of the first event header
    #  2.) somewhere later, as part of the second header.
    # the distance is simply the event size:
    #
    # Note! At first i though the flag is always this: flag = b'\xf0\x02' * 8
    # But then I found files, where this is not true,
    # Now I make another assumption about the flag.
    # I assume: The flag is the the bytestring from address 16 to 32 of the
    # file:

    flag = chunk[16:32]
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
