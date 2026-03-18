import time

from mc1.clock import LogicalClock, current_clock


def test_logical_clock_seconds_advances_while_idle():
    clock = LogicalClock()
    start = clock.seconds

    time.sleep(0.03)

    assert clock.seconds >= start + 0.02


def test_logical_clock_keeps_advancing_across_idle_gap_between_tasks():
    clock = LogicalClock()
    observed = []

    def snapshot():
        observed.append(current_clock().seconds)
        if False:
            yield 0.0

    first = clock.play(snapshot())
    assert first.wait(1.0)
    assert first.exception() is None

    time.sleep(0.03)

    second = clock.play(snapshot())
    assert second.wait(1.0)
    assert second.exception() is None

    assert observed[1] >= observed[0] + 0.02
