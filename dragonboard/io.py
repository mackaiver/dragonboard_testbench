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
int2gain = {0: "high", 1: "low"}
gain2int = {"high": 0, "low": 1}
gains = ["high", "low"]


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
    stop_cells_for_user = np.empty(
        num_channels, dtype=[('low', 'i2'), ('high', 'i2')]
    )
    stop_cells__in_drs4_chip_order = np.frombuffer(
        f.read(stop_cell_size), dtype=stop_cell_dtype)

    for gain in gains:
        for channel in range(num_channels):
            stop_cells_for_user[gain][channel] = stop_cells__in_drs4_chip_order[
                2 * (channel // 2) + gain2int[gain]]

    timestamp_in_s = clock * timestamp_conversion_to_s

    if flag is not None:
        msg = ('event header looks wrong: '
               'flag is not at the right position\n'
               'found: {}, expected: {}'.format(found_flag, flag))

        assert chunk.find(flag) == expected_relative_address_of_flag, msg

    return EventHeader(
        event_id, trigger_id, timestamp_in_s, stop_cells_for_user, found_flag
    )


def read_data(f, roi):
    ''' return array of raw ADC data

    shape: (num_pixel, "high"|"low", roi)

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


def read_data_3d(f, roi):
    ''' return array of raw ADC data

    shape: (num_pixel, num_gains, roi)

    1st dimension: pixels, 0..6 for single dragon Board
    2nd dimension: gains, [0:high, 1:low]
    3rd dimension: samples, relative to stop_cell
    '''
    d = np.fromfile(f, '>i2', num_gains * num_channels * roi)

    d = d.reshape(2, roi, num_channels // 2, num_gains
                  ).swapaxes(0, 3
                             ).reshape(num_gains, roi, num_channels
                                       ).swapaxes(1, 2
                                                  ).swapaxes(0, 1)

    # d.shape = (pixel, hi/lo, roi)
    return d


def read_header_3d(f, flag=None):
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
    stop_cells__in_drs4_chip_order = np.frombuffer(
        f.read(stop_cell_size), dtype=stop_cell_dtype)
    stop_cells_for_user = np.empty((num_channels, num_gains),
                                   dtype=stop_cells__in_drs4_chip_order.dtype)

    for gain in gains:
        for channel in range(num_channels):
            stop_cells_for_user[channel, gain2int[gain]] = stop_cells__in_drs4_chip_order[
                2 * (channel // 2) + gain2int[gain]]

    timestamp_in_s = clock * timestamp_conversion_to_s

    if flag is not None:
        msg = ('event header looks wrong: '
               'flag is not at the right position\n'
               'found: {}, expected: {}'.format(found_flag, flag))

        assert chunk.find(flag) == expected_relative_address_of_flag, msg

    return EventHeader(
        event_id, trigger_id, timestamp_in_s, stop_cells_for_user, found_flag
    )


def measure_event_size(f):
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


def get_file_size(f):
    ''' return file_size in bytes '''

    # By using seek. we also can measure the true size of
    # a zipped file. os.path.getsize in contrast, will not return the length
    # of the unzipped data.
    current_position = f.tell()
    file_size = f.seek(0, 2)
    f.seek(current_position)
    return file_size


class File(object):

    def __init__(self, path, max_events=None, return_structured_array=True):
        self.path = path
        self.file_descriptor = open(self.path, "rb")
        self.event_size = measure_event_size(self.file_descriptor)
        self.roi = get_roi(self.event_size)
        num_events = get_file_size(self.file_descriptor) / self.event_size
        assert num_events.is_integer()
        self.num_events = int(num_events)

        self.max_events = max_events

        self.__current_event_pointer = 0

        if return_structured_array:
            self.read_header = read_header
            self.read_data = read_data
        else:
            self.read_header = read_header_3d
            self.read_data = read_data_3d

    def __getitem__(self, index_or_slice):
        if isinstance(index_or_slice, slice):
            event_ids = range(*index_or_slice.indices(self.num_events))
            events = []
            for event_id in event_ids:
                self.seek_event(event_id)
                events.append(next(self))
            return events
        else:
            if index_or_slice >= 0:
                if index_or_slice >= self.num_events:
                    raise IndexError(
                        "index {} is out of range for num_events={}".format(
                            index_or_slice, self.num_events))
                self.seek_event(index_or_slice)
            else:
                if index_or_slice < -self.num_events:
                    raise IndexError(
                        "index {} is out of range for num_events={}".format(
                            index_or_slice, self.num_events))
                self.seek_event(self.num_events + index_or_slice)
            return next(self)

    def __iter__(self):
        return self

    def next(self):
        return next(self)

    def previous(self):
        try:
            self.file_descriptor.seek(- 2 * self.event_size, 1)
        except OSError:
            raise ValueError('Already at first event')
        return next(self)

    def __next__(self):
        try:
            event_header = self.read_header(self.file_descriptor)
            data = self.read_data(self.file_descriptor, self.roi)

            if self.max_events is not None:
                if event_header.event_counter > self.max_events:
                    raise StopIteration

            self.__current_event_pointer += 1
            return Event(event_header, self.roi, data)

        except struct.error:
            raise StopIteration

    def seek_event(self, needed_event_id):
        # we start with a stupid implementation:
        self.file_descriptor.seek(needed_event_id * self.event_size)
