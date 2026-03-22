from __future__ import annotations

import math

from mc1.clock import current_clock


IMMEDIATE = 1
NTP_EPOCH_OFFSET_SECONDS = 2_208_988_800
_NANOSECONDS_PER_SECOND = 1_000_000_000
_NTP_FRACTION_SCALE = 1 << 32
_MAX_TIMETAG = 0xFFFFFFFFFFFFFFFF
_MAX_NTP_SECONDS = _MAX_TIMETAG >> 32


def _validate_seconds(seconds: float) -> float:
    seconds = float(seconds)
    if not math.isfinite(seconds):
        raise ValueError("seconds must be finite")
    return seconds


def from_unix(unix_seconds: float) -> int:
    unix_seconds = _validate_seconds(unix_seconds)
    if unix_seconds < 0:
        raise ValueError("unix_seconds must be non-negative")
    if unix_seconds >= (_MAX_NTP_SECONDS - NTP_EPOCH_OFFSET_SECONDS + 1):
        raise OverflowError("resulting timetag is out of range")
    unix_nanoseconds = int(unix_seconds * _NANOSECONDS_PER_SECOND)
    unix_whole_seconds, remainder_ns = divmod(unix_nanoseconds, _NANOSECONDS_PER_SECOND)
    ntp_seconds = unix_whole_seconds + NTP_EPOCH_OFFSET_SECONDS
    ntp_fraction = (remainder_ns * _NTP_FRACTION_SCALE) // _NANOSECONDS_PER_SECOND
    return (ntp_seconds << 32) | ntp_fraction


def here(latency: float = 0.1) -> int:
    return from_unix(current_clock().time + _validate_seconds(latency))
