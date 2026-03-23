"""Logical-time scheduling for generator-driven routines.

This module provides an asyncio-backed scheduler that advances a logical
seconds timeline independently of routine execution cost. Routines cooperate by
yielding finite non-negative delays, which are accumulated onto their next due
time and ordered in a priority queue. Dispatch is synchronized to monotonic
wallclock time so future work is released near its scheduled logical instant
without accumulating processing drift.

The scheduler also exposes a wallclock projection of the current logical
position. When no work is queued, logical time continues to advance from the
last settled value, so newly scheduled relative work is anchored to the current
logical instant rather than to the time when the queue last drained.
"""

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


Routine = Generator[float, None, None]
_current_clock: ContextVar["LogicalClock"] = ContextVar("current_clock")


def current_clock() -> "LogicalClock":
    try:
        return _current_clock.get()
    except LookupError as exc:
        raise RuntimeError("current_clock() is only available while a routine is running") from exc


def merge(*routines: Routine) -> Routine:
    routines = _coerce_routines(routines, caller="merge()")

    def merged() -> Routine:
        elapsed = 0.0
        pending: list[tuple[Routine, float]] = [(routine, 0.0) for routine in routines]

        def advance_ready() -> None:
            nonlocal pending

            while True:
                advanced = False
                next_pending: list[tuple[Routine, float]] = []

                for routine, due in pending:
                    if due > elapsed:
                        next_pending.append((routine, due))
                        continue

                    advanced = True
                    try:
                        delta = next(routine)
                    except StopIteration:
                        continue

                    next_pending.append((routine, elapsed + _coerce_delay(delta)))

                pending = next_pending
                if not advanced:
                    return

        while pending:
            advance_ready()
            if not pending:
                return

            due = min(due for _, due in pending)
            delta = due - elapsed
            yield delta
            elapsed = due

    return merged()


class LogicalClock:
    __slots__ = (
        '_wallclock_origin',
        '_sequence',
        '_queue',
        '_seconds',
        '_idle_started_at',
        '_routine_finished',
        '_state_changed',
        '_task'
    )

    def __init__(self, seconds: float = 0) -> None:
        self._seconds = seconds
        self._queue: list[_ScheduledRoutine] = []
        self._sequence = count()
        self._idle_started_at: float | None = None
        self._routine_finished = asyncio.Event()
        self._state_changed: asyncio.Event | None = None
        self._task = None
        self._wallclock_origin = pytime.time()

    def __repr__(self):
        return f"<{self.__class__.__name__} seconds={self.seconds:.5f}>"

    @property
    def seconds(self) -> float:
        if self._idle_started_at is not None:
            return self._seconds + (pytime.monotonic() - self._idle_started_at)

        return self._seconds

    @property
    def time(self) -> float:
        return self._wallclock_origin + self.seconds

    def schedule(
        self,
        *routines: Routine,
        delay: float | None = None,
        at: float | None = None
    ) -> "RoutineHandle":
        if delay is not None and at is not None:
            raise ValueError("schedule() accepts at most one of delay= or at=")

        routines = _coerce_routines(routines, caller="schedule()")
        if len(routines) == 1:
            routine = routines[0]
        else:
            routine = merge(*routines)

        current_seconds = self.seconds
        if self._idle_started_at is not None:
            self._seconds = current_seconds
            self._idle_started_at = None

        if delay is not None:
            due_seconds = current_seconds + _coerce_delay(delay)
        elif at is not None:
            due_seconds = max(current_seconds, _coerce_offset(at, name="at"))
        else:
            due_seconds = current_seconds

        scheduled = _ScheduledRoutine(
            due_seconds=due_seconds,
            sequence=next(self._sequence),
            routine=routine
        )
        heapq.heappush(self._queue, scheduled)
        if self._state_changed is not None:
            self._state_changed.set()
        return RoutineHandle(self, scheduled)

    def _step(self) -> None:
        entry = heapq.heappop(self._queue)
        self._seconds = entry.due_seconds
        clock_token = _current_clock.set(self)

        try:
            delay = next(entry.routine)
        except StopIteration:
            self._routine_finished.set()
            return
        finally:
            _current_clock.reset(clock_token)

        entry.due_seconds += _coerce_delay(delay)
        entry.sequence = next(self._sequence)
        heapq.heappush(self._queue, entry)

    async def _run(self) -> None:
        if self._state_changed is not None:
            raise RuntimeError("LogicalClock._run() is already running")

        self._state_changed = asyncio.Event()
        active_origin: float | None = None

        try:
            while True:
                if not self._queue:
                    active_origin = None
                    if self._idle_started_at is None:
                        self._idle_started_at = pytime.monotonic()
                    await self._state_changed.wait()
                    self._state_changed.clear()
                    continue

                self._idle_started_at = None
                if active_origin is None:
                    active_origin = pytime.monotonic() - self._seconds
                next_due = self._queue[0].due_seconds
                delay = active_origin + next_due - pytime.monotonic()
                if delay > 0:
                    try:
                        await asyncio.wait_for(self._state_changed.wait(), timeout=delay)
                    except TimeoutError:
                        pass
                    else:
                        self._state_changed.clear()
                        continue

                while self._queue and self._queue[0].due_seconds <= next_due:
                    self._step()
        finally:
            self._seconds = self.seconds
            self._idle_started_at = None
            self._state_changed = None

    def start(self) -> bool:
        if self._task is None:
            self._task = asyncio.create_task(self._run())
            return True
        return False

    async def wait_for_idle(self) -> None:
        while self._queue:
            await self._routine_finished.wait()
            self._routine_finished.clear()

    def stop(self) -> bool:
        if self._task is not None:
            self._task.cancel()
            self._task = None
            return True
        return False


class RoutineHandle:
    __slots__ = ('_clock', '_scheduled')

    def __init__(self, clock, scheduled):
        self._clock = clock
        self._scheduled = scheduled

    def cancel(self):
        if self._scheduled not in self._clock._queue:
            return
        self._clock._queue.remove(self._scheduled)
        if self._clock._queue:
            heapq.heapify(self._clock._queue)
        self._clock._routine_finished.set()

    def __await__(self):
        async def wait():
            while self._scheduled in self._clock._queue:
                await self._clock._routine_finished.wait()
                self._clock._routine_finished.clear()
        return wait().__await__()

@dataclass(order=True, slots=True)
class _ScheduledRoutine:
    due_seconds: float
    sequence: int
    routine: Routine = field(compare=False)


def _coerce_routines(routines: tuple[Routine, ...], *, caller: str) -> tuple[Routine, ...]:
    if not routines:
        raise ValueError(f"{caller} requires at least one routine")

    for routine in routines:
        if not isgenerator(routine):
            raise TypeError(f"{caller} expects generator routines")

    return routines


def _coerce_delay(delay: float) -> float:
    if isinstance(delay, bool) or not isinstance(delay, Real):
        raise TypeError("routine must yield a real-number delay")

    delay = float(delay)
    if delay < 0 or not isfinite(delay):
        raise ValueError("routine delay must be a finite non-negative number")
    return delay


def _coerce_offset(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")

    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value
