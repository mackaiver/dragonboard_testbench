''' Read Dragon Board Data '''

import struct
import numpy as np
from collections import namedtuple

EventHeader = namedtuple('EventHeader', [
    'event_counter', 'trigger_counter', 'timestamp', 'stop_cells', 'flag'
])

Event = namedtuple('Event', ['header', 'roi', 'data'])

max_read_depth = 4096
header_size_in_bytes = 32
stop_cell_dtype = np.dtype('uint16').newbyteorder('>')
stop_cell_size = 8 * stop_cell_dtype.itemsize
expected_relative_address_of_flag = 16
timestamp_conversion_to_s = 7.5e-9


def get_event_size(read_depth):
    ''' return event_size in bytes, based on read_depth in samples.
    '''
    return 16 * (2 * read_depth + 3)


def get_read_depth(event_size):
    ''' return read_depth in samples, based on event_size in bytes.
    '''

    read_depth = ((event_size / 16) - 3) / 2
    assert read_depth.is_integer()
    return int(read_depth)


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


def read_data(f, read_depth):
    ''' return array of raw ADC data, shape:(16, read_depth)

    The ADC data, is just a contiguous bunch of 16bit integers
    So its easy to read.
    However the assignment of integers to the 16 channels
    is not soo easy. I hope I did it correctly.
    '''
    d = np.fromfile(f, '>i2', 2 * 8 * read_depth)
    N = 8 * read_depth

    d1, d2 = d[:N], d[N:]

    d1 = d1.reshape(read_depth, 8).T
    d2 = d2.reshape(read_depth, 8).T

    d = np.vstack((d1, d2))
    return d


def guess_event_size(f):
    ''' try to find out the event size for this file.

    Each even header contains a flag.
    The distance between two flags is just the event size.
    '''
    current_position = f.tell()
    f.seek(0)

    max_event_size = get_event_size(read_depth=max_read_depth)
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


def read(path):
    ''' return list of Events in file path '''
    with open(path, 'rb') as f:
        return list(event_generator(f))


def event_generator(file_descriptor):
    f = file_descriptor
    while True:
        try:
            event_header = read_header(f)
            event_size = guess_event_size(f)
            read_depth = get_read_depth(event_size)
            data = read_data(f, read_depth)
            yield Event(event_header, read_depth, data)

        except struct.error:
            raise StopIteration
