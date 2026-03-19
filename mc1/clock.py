"""Wall-clock scheduling for small musical or realtime control routines.

This module solves the gap between immediate Python execution and timed
musical processes that need to advance in logical time. In that setting, a
routine should be able to say "run me again 250 ms later" without manually
sleeping, tracking deadlines, or accumulating drift from work done between
events.

The approach here is cooperative scheduling: a task is written as a generator
that yields its next delay in seconds. The clock records task progress in its
own timeline, projects those deadlines onto the system wall clock using its
start epoch, and resumes tasks from a shared background executor thread when
their deadlines arrive. This keeps the task code simple while letting callers
derive real-world timestamps directly from the same clock state.

Each clock carries its own notion of current time relative to its start epoch.
Tasks can inspect the clock they are executing on, and outside code can query
that same timeline directly. The design is intentionally smaller than a full
sequencer: it is a foundation for "process-style" scheduling where timing is
driven by routines themselves rather than by a precomputed event list.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from enum import Enum, auto
from inspect import isgenerator
from itertools import count
from math import isfinite
from numbers import Real
from typing import NamedTuple
import heapq
import threading
import time


_scheduler_state = threading.local()
type TaskGenerator = Generator[float, None, None]


def current_clock() -> "LogicalClock":
    clock = getattr(_scheduler_state, "clock", None)
    if clock is None:
        raise RuntimeError("current_clock() is only available while a task is running")
    return clock


class _TaskState(Enum):
    PENDING = auto()
    RUNNING = auto()
    CANCELLING = auto()
    CANCELLED = auto()
    FINISHED = auto()
    FAILED = auto()

    @property
    def is_terminal(self) -> bool:
        return self in {self.CANCELLED, self.FINISHED, self.FAILED}


@dataclass(slots=True)
class TaskHandle:
    _done_callback: Callable[[], None] | None = field(default=None, repr=False)
    _event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _state: _TaskState = field(default=_TaskState.PENDING, init=False, repr=False)
    _exception: BaseException | None = field(default=None, init=False, repr=False)

    def cancel(self) -> bool:
        with self._lock:
            if self._state.is_terminal:
                return False

            callback = None
            if self._state is _TaskState.PENDING:
                callback = self._finish_locked(_TaskState.CANCELLED)
            elif self._state is _TaskState.RUNNING:
                self._state = _TaskState.CANCELLING

        if callback is not None: callback()
        return True

    def done(self) -> bool: return self._event.is_set()

    def cancelled(self) -> bool:
        with self._lock:
            return self._state is _TaskState.CANCELLED

    def exception(self) -> BaseException | None:
        with self._lock:
            return self._exception

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def _begin_step(self) -> bool:
        with self._lock:
            if self._state is not _TaskState.PENDING:
                return False

            self._state = _TaskState.RUNNING
            return True

    def _complete_step(self) -> bool:
        with self._lock:
            if self._state.is_terminal:
                return False

            callback = None
            if self._state is _TaskState.CANCELLING:
                callback = self._finish_locked(_TaskState.CANCELLED)
            elif self._state is _TaskState.RUNNING:
                self._state = _TaskState.PENDING
                return True

        if callback is not None: callback()
        return False

    def _finish(self, *, exception: BaseException | None = None, cancelled: bool = False) -> None:
        with self._lock:
            if cancelled:
                terminal_state = _TaskState.CANCELLED
            elif exception is None:
                terminal_state = _TaskState.FINISHED
            else:
                terminal_state = _TaskState.FAILED
            callback = self._finish_locked(terminal_state, exception=exception)
        if callback is not None: callback()

    def _finish_locked(
        self,
        terminal_state: _TaskState,
        *,
        exception: BaseException | None = None,
    ) -> Callable[[], None] | None:
        if self._state.is_terminal:
            return None

        self._state = terminal_state
        self._exception = exception
        self._event.set()
        callback = self._done_callback
        self._done_callback = None
        return callback

class LogicalClock:
    _executor: "_SharedExecutor | None" = None
    _executor_lock = threading.Lock()

    def __init__(self) -> None:
        self._epoch = time.time()
        self._seconds = 0.0
        self._running_steps = 0
        self._lock = threading.Lock()

    @property
    def epoch(self) -> float:
        return self._epoch

    @property
    def seconds(self) -> float:
        with self._lock:
            return self._current_seconds_locked()

    @property
    def time(self) -> float:
        return self.epoch + self.seconds

    def play(self, gen: TaskGenerator) -> TaskHandle:
        if not isgenerator(gen):
            raise TypeError("play() expects a generator instance")

        with self._lock:
            due_seconds = self._current_seconds_locked()

        handle = TaskHandle()
        self._shared_executor().submit(_ScheduledTask( clock=self, generator=gen, handle=handle, due_seconds=due_seconds))
        return handle

    def _begin_step(self, seconds: float) -> None:
        with self._lock:
            self._running_steps += 1
            self._seconds = seconds

    def _end_step(self) -> None:
        with self._lock:
            self._running_steps -= 1

    def _current_seconds_locked(self) -> float:
        if self._running_steps > 0:
            return self._seconds
        return time.time() - self._epoch

    def _due_at(self, due_seconds: float) -> float:
        with self._lock:
            return self._epoch + due_seconds

    @classmethod
    def _shared_executor(cls) -> "_SharedExecutor":
        with cls._executor_lock:
            if cls._executor is None:
                cls._executor = _SharedExecutor()
            return cls._executor


@dataclass(slots=True)
class _ScheduledTask:
    clock: LogicalClock
    generator: TaskGenerator
    handle: TaskHandle
    due_seconds: float


class _QueuedTask(NamedTuple):
    due_at: float
    sequence: int
    task: _ScheduledTask


class _SharedExecutor:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._queue: list[_QueuedTask] = []
        self._sequence = count()
        self._thread: threading.Thread | None = None

    def submit(self, task: _ScheduledTask) -> None:
        with self._condition:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run,
                    name="LogicalClockExecutor",
                    daemon=True,
                )
                self._thread.start()
            heapq.heappush(
                self._queue,
                _QueuedTask(
                    due_at=task.clock._due_at(task.due_seconds),
                    sequence=next(self._sequence),
                    task=task,
                )
            )
            self._condition.notify()

    def _run(self) -> None:
        while task := self._next_task():
            if not task.handle._begin_step(): continue

            should_resubmit = False
            try:
                _scheduler_state.clock = task.clock
                task.clock._begin_step(task.due_seconds)
                delay = _coerce_delay(next(task.generator))
                if task.handle._complete_step():
                    task.due_seconds += delay
                    should_resubmit = True
            except StopIteration:
                task.handle._finish()
            except BaseException as exc:
                task.handle._finish(exception=exc)
            finally:
                task.clock._end_step()
                _scheduler_state.clock = None

            if should_resubmit: self.submit(task)

    def _next_task(self) -> _ScheduledTask:
        with self._condition:
            while True:
                while self._queue and self._queue[0].task.handle.done():
                    heapq.heappop(self._queue)

                if not self._queue:
                    self._condition.wait()
                    continue

                queued_task = self._queue[0]
                delay = queued_task.due_at - time.time()
                if delay > 0:
                    self._condition.wait(timeout=delay)
                    continue

                heapq.heappop(self._queue)
                return queued_task.task


def _coerce_delay(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"tasks must yield a real delay, got {value!r}")
    delay = float(value)
    if not isfinite(delay) or delay < 0.0:
        raise ValueError(f"tasks must yield a finite non-negative delay, got {value!r}")
    return delay
