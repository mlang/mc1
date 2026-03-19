from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextvars import ContextVar
from dataclasses import dataclass, field
import heapq
from inspect import isgenerator
from itertools import count
from math import isfinite
from numbers import Real
import time as pytime


def _coerce_offset(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")

    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


Routine = Generator[float, None, None]
_current_clock: ContextVar["LogicalClock"] = ContextVar("current_clock")


def current_clock() -> "LogicalClock":
    try:
        return _current_clock.get()
    except LookupError as exc:
        raise RuntimeError("current_clock() is only available while a routine is running") from exc


def doit() -> Routine:
    clock = current_clock()
    print(clock.time)
    yield 0.5
    print(clock.time)
    yield 0.5
    print(clock.time)
    yield 0.5
    print(clock.time)
    yield 0.5
    print(clock.time)


def _coerce_delay(delay: float) -> float:
    if isinstance(delay, bool) or not isinstance(delay, Real):
        raise TypeError("routine must yield a real-number delay")

    delay = float(delay)
    if delay < 0 or not isfinite(delay):
        raise ValueError("routine delay must be a finite non-negative number")
    return delay


class LogicalClock:
    __slots__ = ('_origin', '_sequence', '_queue', '_seconds')

    def __init__(self, seconds: float = 0) -> None:
        self._seconds = seconds
        self._queue: list[_ScheduledRoutine] = []
        self._sequence = count()
        self._origin = pytime.time()

    @property
    def seconds(self) -> float:
        return self._seconds

    @property
    def time(self) -> float:
        return self._origin + self.seconds

    def play(
        self,
        routine: Routine,
        *,
        relative: float | None = None,
        absolute: float | None = None,
    ) -> None:
        if relative is not None and absolute is not None:
            raise ValueError("play() accepts at most one of relative= or absolute=")

        if not isgenerator(routine):
            raise TypeError("play() expects a routine factory returning a generator")

        if relative is not None:
            due_seconds = self.seconds + _coerce_delay(relative)
        elif absolute is not None:
            due_seconds = max(self.seconds, _coerce_offset(absolute, name="absolute"))
        else:
            due_seconds = self.seconds

        heapq.heappush(
            self._queue,
            _ScheduledRoutine(
                due_seconds=due_seconds,
                sequence=next(self._sequence),
                routine=routine
            )
        )

    def step(self) -> bool:
        if not self._queue:
            return False

        entry = heapq.heappop(self._queue)
        self._seconds = entry.due_seconds
        clock_token = _current_clock.set(self)

        try:
            delay = next(entry.routine)
        except StopIteration:
            return True
        finally:
            _current_clock.reset(clock_token)

        entry.due_seconds += _coerce_delay(delay)
        entry.sequence = next(self._sequence)
        heapq.heappush(self._queue, entry)
        return True

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        wallclock_origin = loop.time() - self.seconds

        while self._queue:
            next_due = self._queue[0].due_seconds
            delay = wallclock_origin + next_due - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)

            while self._queue and self._queue[0].due_seconds <= next_due:
                self.step()


async def main() -> None:
    clock = LogicalClock()
    clock.play(doit())
    await clock.run()


@dataclass(order=True, slots=True)
class _ScheduledRoutine:
    due_seconds: float
    sequence: int
    routine: Routine = field(compare=False)


if __name__ == "__main__":
    asyncio.run(main())
