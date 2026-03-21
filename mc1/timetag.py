from __future__ import annotations

import math
import time


IMMEDIATE = 1
DEFAULT_LATENCY = 0.1
NTP_EPOCH_OFFSET_SECONDS = 2_208_988_800
_NANOSECONDS_PER_SECOND = 1_000_000_000
_NTP_FRACTION_SCALE = 1 << 32
_MAX_TIMETAG = 0xFFFFFFFFFFFFFFFF
_MAX_NTP_SECONDS = _MAX_TIMETAG >> 32


def _validate_timetag(time_tag: int) -> int:
    if isinstance(time_tag, bool) or not isinstance(time_tag, int):
        raise TypeError("time_tag must be an OSC timetag integer")
    if time_tag < 0 or time_tag > _MAX_TIMETAG:
        raise ValueError("time_tag must fit in uint64")
    return time_tag


def _validate_seconds(seconds: float) -> float:
    seconds = float(seconds)
    if not math.isfinite(seconds):
        raise ValueError("seconds must be finite")
    return seconds


def from_unix_ns(unix_ns: int) -> int:
    unix_ns = _validate_timetag(unix_ns)
    unix_seconds, remainder_ns = divmod(unix_ns, _NANOSECONDS_PER_SECOND)
    ntp_seconds = unix_seconds + NTP_EPOCH_OFFSET_SECONDS
    ntp_fraction = (remainder_ns * _NTP_FRACTION_SCALE) // _NANOSECONDS_PER_SECOND
    return (ntp_seconds << 32) | ntp_fraction


def from_unix(unix_seconds: float) -> int:
    unix_seconds = _validate_seconds(unix_seconds)
    if unix_seconds < 0:
        raise ValueError("unix_seconds must be non-negative")
    if unix_seconds >= (_MAX_NTP_SECONDS - NTP_EPOCH_OFFSET_SECONDS + 1):
        raise OverflowError("resulting timetag is out of range")
    return from_unix_ns(int(unix_seconds * _NANOSECONDS_PER_SECOND))


def now() -> int:
    return from_unix_ns(time.time_ns())


def add(time_tag: int, seconds: float) -> int:
    base = now() if time_tag == IMMEDIATE else _validate_timetag(time_tag)
    delta_ticks = int(round(_validate_seconds(seconds) * _NTP_FRACTION_SCALE))
    result = base + delta_ticks
    if result < 0 or result > _MAX_TIMETAG:
        raise OverflowError("resulting timetag is out of range")
    return result


def after(seconds: float, *, base: int | None = None) -> int:
    return add(now() if base is None else base, seconds)


def latency(*, base: int | None = None, seconds: float = DEFAULT_LATENCY) -> int:
    return after(seconds, base=base)
