import math

import pytest

import mc1.timetag
from mc1.timetag import NTP_EPOCH_OFFSET_SECONDS, from_unix, here


def test_from_unix_converts_whole_seconds():
    assert from_unix(0.0) == NTP_EPOCH_OFFSET_SECONDS << 32


def test_from_unix_converts_fractional_seconds():
    assert from_unix(1.25) == ((NTP_EPOCH_OFFSET_SECONDS + 1) << 32) | (1 << 30)


@pytest.mark.parametrize(("unix_seconds", "expected"), [
    (0.1, (NTP_EPOCH_OFFSET_SECONDS << 32) | 429_496_729),
    (1.25, ((NTP_EPOCH_OFFSET_SECONDS + 1) << 32) | (1 << 30)),
    (1_742_608_800.5, ((NTP_EPOCH_OFFSET_SECONDS + 1_742_608_800) << 32) | (1 << 31)),
])
def test_from_unix_converts_expected_ntp_values(unix_seconds, expected):
    assert from_unix(unix_seconds) == expected


@pytest.mark.parametrize("unix_seconds", [math.nan, math.inf, -math.inf])
def test_from_unix_rejects_non_finite_seconds(unix_seconds):
    with pytest.raises(ValueError, match="seconds must be finite"):
        from_unix(unix_seconds)


def test_from_unix_rejects_negative_seconds():
    with pytest.raises(ValueError, match="unix_seconds must be non-negative"):
        from_unix(-0.1)


def test_from_unix_rejects_values_beyond_ntp_range():
    max_unix_seconds = (1 << 32) - NTP_EPOCH_OFFSET_SECONDS

    with pytest.raises(OverflowError, match="resulting timetag is out of range"):
        from_unix(float(max_unix_seconds))


def test_here_uses_current_clock_time_with_default_latency(monkeypatch):
    monkeypatch.setattr(
        mc1.timetag,
        "current_clock",
        lambda: type("FakeClock", (), {"time": 123.25})(),
    )

    assert here() == from_unix(123.35)


def test_here_uses_custom_latency(monkeypatch):
    monkeypatch.setattr(
        mc1.timetag,
        "current_clock",
        lambda: type("FakeClock", (), {"time": 123.25})(),
    )

    assert here(0.5) == from_unix(123.75)
