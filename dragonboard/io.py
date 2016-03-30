''' Read Dragon Board Data '''

from __future__ import division, print_function, absolute_import
import struct
import numpy as np
from collections import namedtuple
import os.path
import warnings

stop_cell_map = {
    ("high", 0): 0,
    ("high", 1): 0,
    ("high", 2): 2,
    ("high", 3): 2,
    ("high", 4): 4,
    ("high", 5): 4,
    ("high", 6): 6,
    ("high", 7): 6,
    ("low", 0): 1,
    ("low", 1): 1,
    ("low", 2): 3,
    ("low", 3): 3,
    ("low", 4): 5,
    ("low", 5): 5,
    ("low", 6): 7,
    ("low", 7): 7,
}

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


def read(path, max_events=None):
    ''' return list of Events in file path '''
    return list(EventGenerator(path, max_events=None))


class AbstractEventGenerator(object):
    header_size = None

    def __init__(self, path, max_events=None):
        self.path = os.path.realpath(path)

        self.file_descriptor = open(self.path, "rb")

        self.event_size = self.guess_event_size()
        self.roi = self.calc_roi()
        self.num_events = self.calc_num_events()
        if max_events is None or max_events > len(self):
            self.max_events = len(self)
        else:
            self.max_events = max_events

        self.event_counter = 0

    def __repr__(self):
        return(
            "{name}(\n"
            "path={S.path!r}, \n"
            "max_events={S.max_events})\n"
            "roi ......: {S.roi}\n"
            "event_size: {S.event_size} byte\n"
            "#events ..: {N}"
        ).format(
            name=self.__class__.__name__,
            S=self,
            N=len(self),
        )

    def calc_num_events(self):
        f = self.file_descriptor
        current_position = f.tell()
        f.seek(0)
        filesize = f.seek(0, 2)
        f.seek(current_position)
        num_events = filesize / self.event_size
        if not num_events.is_integer():
            warnings.warn(("\n"
                "File:\n"
                "{0}\n"
                "might be broken.\n"
                "Number of events "
                "is not integer but {1:.2f}.").format(
                    self.path,
                    num_events
                ))
        return int(num_events)

    def __len__(self):
        return self.num_events

    def guess_event_size(self):
        raise NotImplementedError

    def read_header(self):
        raise NotImplementedError

    def calc_roi(self):
        raise NotImplementedError

    def read_chunk(self):
        N = self.header_size + max_roi * adc_word_size * num_gains * num_channels
        return self.file_descriptor.read(int(N * 1.5))

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def previous(self):
        try:
            self.file_descriptor.seek(- 2 * self.event_size, 1)
            self.event_counter -= 2
        except OSError:
            raise ValueError('Already at first event')
        return self.next()

    def next(self):
        if self.event_counter >= self.max_events:
            raise StopIteration

        try:
            event_header = self.read_header()
            data = self.read_adc_data()

            self.event_counter += 1
            return Event(event_header, self.roi, data)

        except struct.error:
            raise StopIteration

    def read_adc_data(self):
        ''' return array of raw ADC data, shape:(16, roi)

        The ADC data, is just a contiguous bunch of 16bit integers
        So its easy to read.
        However the assignment of integers to the 16 channels
        is not soo easy. I hope I did it correctly.
        '''
        f = self.file_descriptor

        d = np.fromfile(f, '>i2', num_gains * num_channels * self.roi)

        N = num_gains * num_channels * self.roi
        roi_dtype = '{}>i2'.format(self.roi)
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


class EventGenerator_v5_1_05(AbstractEventGenerator):
    header_size = 3 * 16

    EventHeader = namedtuple('EventHeader', [
        'event_counter',
        'trigger_counter',
        'timestamp',
        'stop_cells',
        'flag',
    ])

    def read_header(self):
        ''' return EventHeader from file f

        if a *flag* is provided, we can check if the header
        looks correct. If not, we can't check anything.
        '''
        # the format string:
        #   ! -> network endian order
        #   I -> integer
        #   Q -> unsingned long
        #   s -> char
        #   H -> unsigned short
        f = self.file_descriptor

        (
            event_id,
            trigger_id,
            clock,
            found_flag,
        ) = struct.unpack('!IIQ16s', f.read(struct.calcsize('!IIQ16s')))
        stop_cells_for_user = np.empty(
            num_channels, dtype=[('low', 'i2'), ('high', 'i2')]
        )
        stop_cells__in_drs4_chip_order = np.frombuffer(
            f.read(stop_cell_size), dtype=stop_cell_dtype)

        for g, p in stop_cell_map:
            chip = stop_cell_map[(g, p)]
            stop_cells_for_user[g][p] = stop_cells__in_drs4_chip_order[chip]

        timestamp_in_s = clock * timestamp_conversion_to_s

        return self.EventHeader(
            event_id, trigger_id, timestamp_in_s, stop_cells_for_user, found_flag
        )

    def calc_roi(self):
        body_size = self.event_size - self.header_size
        roi = body_size/(adc_word_size * num_gains * num_channels)
        assert roi.is_integer()
        return int(roi)

    def guess_event_size(self):
        ''' try to find out the event size for this file.

        Each even header contains a flag.
        The distance between two flags is just the event size.
        '''
        f = self.file_descriptor
        current_position = f.tell()
        f.seek(0)

        chunk = self.read_chunk()

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


class EventGenerator_v5_1_0B(AbstractEventGenerator):
    header_size = 4 * 16

    EventHeader = namedtuple('EventHeader', [
        'event_counter',
        'trigger_counter',
        'counter_133MHz',
        'counter_10MHz',
        'pps_counter',
        'stop_cells',
        'flag',
    ])

    def calc_roi(self):
        body_size = self.event_size - self.header_size
        roi = body_size / (adc_word_size * num_gains * num_channels)
        assert roi.is_integer()
        return int(roi)

    def read_header(self):
        ''' return EventHeader from file f
        '''
        f = self.file_descriptor

        (
            header_aaaa,         # should be constant 0xaaaa
            pps_counter,         # pps: pulse per second = 1Hz
            counter_10MHz,
            event_counter,
            trigger_counter,
            counter_133MHz,
            data_header_all_ds,  # should be constant 8 times 0xdd
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

        for g, p in stop_cell_map:
            chip = stop_cell_map[(g, p)]
            stop_cells_for_user[g][p] = stop_cells__in_drs4_chip_order[chip]

        return self.EventHeader(
            event_counter,
            trigger_counter,
            counter_133MHz,
            counter_10MHz,
            pps_counter,
            stop_cells_for_user,
            flags
        )

    def guess_event_size(self):
        ''' try to find out the event size for this file.

        Each even header contains a data_header
        The distance between two data_headers is just the event size.
        '''
        f = self.file_descriptor
        current_position = f.tell()
        f.seek(0)

        chunk = self.read_chunk()

        data_header = b'\xdd'*8
        first_data_header = chunk.find(data_header)
        second_data_header = chunk.find(data_header, first_data_header + 1)

        event_size = second_data_header - first_data_header
        f.seek(current_position)

        return event_size


version_map = {
    "v5_1_05": EventGenerator_v5_1_05,
    "v5_1_0B": EventGenerator_v5_1_0B,
}


def EventGenerator(path, max_events=None, version=None):
    if version is None:
        return version_map[guess_version(path)](path, max_events)
    else:
        return version_map[version](path, max_events)


def guess_version(path):
    for version_name in version_map:
        EG = version_map[version_name]
        try:
            EG(path)
            return version_name
        except:
            pass
    raise IOError('File version could not be determined for file {}'.format(path))
