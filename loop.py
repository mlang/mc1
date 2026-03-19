from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator
from math import isfinite
from numbers import Real
import time as pytime
from typing import TypeVar


def _coerce_offset(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")

    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _coerce_tempo(value: float) -> float:
    tempo = _coerce_offset(value, name="tempo")
    if tempo <= 0:
        raise ValueError("tempo must be a finite positive number")
    return tempo


class LogicalTime:
    __slots__ = ("_seconds", "_wallclock_time")

    def __init__(self, seconds: float = 0.0, *, wallclock_time: float | None = None) -> None:
        self._seconds = _coerce_offset(seconds, name="seconds")
        if wallclock_time is None:
            wallclock_time = pytime.time()
        self._wallclock_time = _coerce_offset(wallclock_time, name="wallclock_time")

    @property
    def seconds(self) -> float:
        return self._seconds

    @property
    def wallclock_time(self) -> float:
        return self._wallclock_time + self._seconds

    @wallclock_time.setter
    def wallclock_time(self, value: float) -> None:
        self._wallclock_time = _coerce_offset(value, name="wallclock_time") - self._seconds

    def _delay_seconds(self, delay: float) -> float:
        return delay

    def __iadd__(self, delay: float) -> LogicalTime:
        delay = _coerce_delay(delay)
        self._seconds += self._delay_seconds(delay)
        return self


class BeatTime(LogicalTime):
    __slots__ = ("_beats", "_tempo")

    def __init__(
        self,
        beats: float = 0.0,
        *,
        tempo: float = 120.0,
        wallclock_time: float | None = None,
    ) -> None:
        self._beats = _coerce_offset(beats, name="beats")
        self._tempo = _coerce_tempo(tempo)
        super().__init__(self._delay_seconds(self._beats), wallclock_time=wallclock_time)

    @property
    def beats(self) -> float:
        return self._beats

    @property
    def tempo(self) -> float:
        return self._tempo

    @tempo.setter
    def tempo(self, value: float) -> None:
        self._tempo = _coerce_tempo(value)

    def _delay_seconds(self, delay: float) -> float:
        return delay * 60.0 / self._tempo

    def __iadd__(self, delay: float) -> BeatTime:
        delay = _coerce_delay(delay)
        self._beats += delay
        super().__iadd__(delay)
        return self


type Time = LogicalTime | BeatTime
ReturnT = TypeVar("ReturnT")
Routine = Generator[float, Time, ReturnT]
RoutineFactory = Callable[[Time], Routine[ReturnT]]


def doit(time: Time) -> Routine:
    print(time.wallclock_time)
    yield 0.5
    print(time.wallclock_time)
    yield 0.5
    print(time.wallclock_time)
    time.tempo=60
    yield 0.5
    print(time.wallclock_time)
    yield 0.5
    print(time.wallclock_time)



def _coerce_delay(delay: float) -> float:
    if isinstance(delay, bool) or not isinstance(delay, Real):
        raise TypeError("routine must yield a real-number delay")

    delay = float(delay)
    if delay < 0 or not isfinite(delay):
        raise ValueError("routine delay must be a finite non-negative number")
    return delay


async def run(gen: RoutineFactory[ReturnT], *, time: Time | None = None) -> ReturnT:
    loop = asyncio.get_running_loop()
    current_time = LogicalTime() if time is None else time
    if not isinstance(current_time, LogicalTime):
        raise TypeError("time must be a LogicalTime or BeatTime instance")
    current_time.wallclock_time = pytime.time()

    deadline = loop.time()
    routine = gen(current_time)
    done: asyncio.Future[ReturnT] = loop.create_future()
    pending_handle: asyncio.TimerHandle | None = None

    def clear_pending() -> None:
        nonlocal pending_handle
        pending_handle = None

    def finish_exception(exc: BaseException) -> None:
        if done.done():
            return
        done.set_exception(exc)

    def finish_result(value: ReturnT) -> None:
        if done.done():
            return
        done.set_result(value)

    def step(*, initial: bool = False) -> None:
        nonlocal current_time, deadline, pending_handle
        clear_pending()

        if done.cancelled():
            return

        try:
            delay = next(routine) if initial else routine.send(current_time)
            delay = _coerce_delay(delay)
            wait_seconds = current_time._delay_seconds(delay)
        except StopIteration as stop:
            finish_result(stop.value)
            return
        except Exception as exc:
            finish_exception(exc)
            return

        deadline += wait_seconds
        current_time += delay
        pending_handle = loop.call_at(deadline, step)

    def on_done(fut: asyncio.Future[None]) -> None:
        if pending_handle is not None:
            pending_handle.cancel()
        if fut.cancelled():
            routine.close()

    done.add_done_callback(on_done)
    step(initial=True)
    try:
        return await done
    except asyncio.CancelledError:
        done.cancel()
        raise


async def main() -> None:
    await run(doit)


if __name__ == "__main__":
    asyncio.run(main())
