
def test_reading_v5_1_05():
    from ..io import EventGenerator_v5_1_05

    eg = EventGenerator_v5_1_05(
        'data/fake_pedestal_v5_1_05.dat',
        max_events=10,
    )

    repr(eg)
    assert eg.event_size == 32816

    event = next(eg)
    assert event.roi == 1024
