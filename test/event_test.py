import pytest

from mc1 import default_event


def test_default_event_duration_uses_tempo():
    event = default_event.with_(tempo=72, beats=2)

    assert event.duration == pytest.approx(5 / 3)
    assert event.sustain == pytest.approx((5 / 3) * 0.8)


def test_default_event_freq_tracks_midi_note():
    event = default_event.with_(midi_note=69)

    assert event.freq == pytest.approx(440.0)


def test_default_event_reports_rests():
    event = default_event.with_(midi_note=None)

    assert event.is_rest is True
    assert event.freq is None


def test_default_event_preserves_kwargs_bag():
    kwargs = {"amp": 0.2}
    event = default_event.with_(kwargs=kwargs)

    assert event.kwargs == kwargs
