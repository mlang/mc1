import asyncio
import time as pytime

import pytest

import loop


def test_run_advances_logical_time_across_yields():
    observed = []

    def routine(time):
        observed.append(time.seconds)
        time = yield 0.01
        observed.append(time.seconds)
        time = yield 0.01
        observed.append(time.seconds)

    asyncio.run(loop.run(routine))

    assert observed == pytest.approx([0.0, 0.01, 0.02])


def test_run_returns_routine_final_value():
    def routine(time):
        assert time.seconds == pytest.approx(0.0)
        time = yield 0.01
        assert time.seconds == pytest.approx(0.01)
        return "done"

    result = asyncio.run(loop.run(routine))

    assert result == "done"


def test_run_reuses_same_logical_time_instance_and_projects_wallclock(monkeypatch):
    monkeypatch.setattr(loop.pytime, "time", lambda: 123.0)
    observed_ids = []
    observed_wallclock = []

    def routine(time):
        observed_ids.append(id(time))
        observed_wallclock.append(time.wallclock_time)
        time = yield 0.01
        observed_ids.append(id(time))
        observed_wallclock.append(time.wallclock_time)
        time = yield 0.02
        observed_ids.append(id(time))
        observed_wallclock.append(time.wallclock_time)

    asyncio.run(loop.run(routine))

    assert observed_ids[0] == observed_ids[1] == observed_ids[2]
    assert observed_wallclock == pytest.approx([123.0, 123.01, 123.03])


def test_run_advances_beat_time_across_yields(monkeypatch):
    monkeypatch.setattr(loop.pytime, "time", lambda: 200.0)
    observed_beats = []
    observed_wallclock = []

    def routine(time):
        observed_beats.append(time.beats)
        observed_wallclock.append(time.wallclock_time)
        time = yield 1.0
        observed_beats.append(time.beats)
        observed_wallclock.append(time.wallclock_time)
        time = yield 1.0
        observed_beats.append(time.beats)
        observed_wallclock.append(time.wallclock_time)

    asyncio.run(loop.run(routine, time=loop.BeatTime(tempo=6000.0)))

    assert observed_beats == pytest.approx([0.0, 1.0, 2.0])
    assert observed_wallclock == pytest.approx([200.0, 200.01, 200.02])


def test_run_uses_updated_beat_tempo_for_future_steps(monkeypatch):
    monkeypatch.setattr(loop.pytime, "time", lambda: 300.0)
    checkpoints = []
    observed_wallclock = []
    observed_tempo = []

    def routine(time):
        checkpoints.append(pytime.monotonic())
        observed_wallclock.append(time.wallclock_time)
        observed_tempo.append(time.tempo)
        time = yield 1.0
        checkpoints.append(pytime.monotonic())
        observed_wallclock.append(time.wallclock_time)
        observed_tempo.append(time.tempo)
        time.tempo = 6000.0
        time = yield 1.0
        checkpoints.append(pytime.monotonic())
        observed_wallclock.append(time.wallclock_time)
        observed_tempo.append(time.tempo)

    asyncio.run(loop.run(routine, time=loop.BeatTime(tempo=600.0)))

    first_gap = checkpoints[1] - checkpoints[0]
    second_gap = checkpoints[2] - checkpoints[1]

    assert observed_wallclock == pytest.approx([300.0, 300.1, 300.11])
    assert observed_tempo == pytest.approx([600.0, 600.0, 6000.0])
    assert second_gap < first_gap * 0.5


def test_time_classes_project_wallclock_from_base_seconds(monkeypatch):
    monkeypatch.setattr(loop.pytime, "time", lambda: 50.0)

    logical_time = loop.LogicalTime()
    logical_time += 0.25
    assert logical_time.wallclock_time == pytest.approx(50.25)

    beat_time = loop.BeatTime(tempo=120.0)
    assert isinstance(beat_time, loop.LogicalTime)
    beat_time += 1.0
    assert beat_time.seconds == pytest.approx(0.5)
    assert beat_time.wallclock_time == pytest.approx(50.5)


def test_beat_time_rejects_invalid_tempo():
    with pytest.raises(ValueError):
        loop.BeatTime(tempo=0.0)

    beat_time = loop.BeatTime()

    with pytest.raises(ValueError):
        beat_time.tempo = 0.0


def test_run_preserves_supplied_time_position_and_reanchors_wallclock(monkeypatch):
    monkeypatch.setattr(loop.pytime, "time", lambda: 500.0)
    current_time = loop.LogicalTime(1.5, wallclock_time=10.0)
    observed = []

    def routine(time):
        observed.append(time.seconds)
        observed.append(time.wallclock_time)
        return "done"
        yield 0.0

    result = asyncio.run(loop.run(routine, time=current_time))

    assert result == "done"
    assert observed == pytest.approx([1.5, 500.0])
    assert current_time.wallclock_time == pytest.approx(500.0)
