import pytest

from mc1 import default_event


class FakeDSP:
    def __init__(self):
        self.calls = []
        self.next_synth_id = 1

    def append(self, instrument, **controls):
        synth_id = self.next_synth_id
        self.next_synth_id += 1
        self.calls.append(("append", instrument, controls, synth_id))
        return synth_id

    def set(self, synth_id, **controls):
        self.calls.append(("set", synth_id, controls))


def test_default_event_duration_uses_tempo():
    event = default_event(bpm=72, beats=2)

    assert event.duration == pytest.approx(5 / 3)
    assert event.sustain == pytest.approx((5 / 3) * 0.8)


def test_default_event_freq_tracks_midi_note():
    event = default_event(midi_note=69)

    assert event.freq == pytest.approx(440.0)


def test_default_event_reports_rests():
    event = default_event(midi_note=None)

    assert event.is_rest is True
    assert event.freq is None


def test_default_event_preserves_kwargs_bag():
    kwargs = {"amp": 0.2}
    event = default_event(kwargs=kwargs)

    assert event.kwargs == kwargs


def test_default_event_call_returns_event_copy():
    event = default_event(bpm=72)

    assert type(event) is type(default_event)
    assert event is not default_event
    assert event.bpm == 72
    assert default_event.bpm == 120.0


def test_default_event_supports_with_blocks():
    with default_event(bpm=90) as event:
        assert event.bpm == 90
        assert event is not default_event

    assert default_event.bpm == 120.0


def test_default_event_play_returns_routine_for_non_rest():
    kwargs = {"amp": 0.2, "freq": 123.0, "gate": 99}
    event = default_event(midi_note=69, beats=2, kwargs=kwargs)
    dsp = FakeDSP()

    routine = event.play(dsp)

    assert next(routine) == pytest.approx(event.sustain)
    assert dsp.calls == [
        ("append", "default", {"amp": 0.2, "freq": pytest.approx(440.0), "gate": 1}, 1)
    ]
    assert kwargs == {"amp": 0.2, "freq": 123.0, "gate": 99}

    assert next(routine) == pytest.approx(event.duration - event.sustain)
    assert dsp.calls[-1] == ("set", 1, {"gate": 0})

    with pytest.raises(StopIteration):
        next(routine)


def test_default_event_play_returns_duration_only_for_rests():
    event = default_event(midi_note=None, beats=2)
    dsp = FakeDSP()

    routine = event.play(dsp)

    assert next(routine) == pytest.approx(event.duration)
    with pytest.raises(StopIteration):
        next(routine)
    assert dsp.calls == []
