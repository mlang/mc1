import math

import pytest

from mc1.timetag import NTP_EPOCH_OFFSET_SECONDS, from_unix, from_unix_ns


def test_from_unix_converts_whole_seconds():
    assert from_unix(0.0) == NTP_EPOCH_OFFSET_SECONDS << 32


def test_from_unix_converts_fractional_seconds():
    assert from_unix(1.25) == ((NTP_EPOCH_OFFSET_SECONDS + 1) << 32) | (1 << 30)


@pytest.mark.parametrize(
    "unix_seconds",
    [0.0, 0.1, 1.25, 1_742_608_800.125, 1_742_608_800.999999],
)
def test_from_unix_matches_from_unix_ns(unix_seconds):
    assert from_unix(unix_seconds) == from_unix_ns(int(unix_seconds * 1_000_000_000))


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
