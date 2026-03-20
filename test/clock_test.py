from __future__ import annotations

import asyncio
import time as pytime

import pytest

from mc1.clock import LogicalClock, current_clock


def _instant_routine(callback):
    def routine():
        callback(current_clock().seconds)
        return
        yield 0.0

    return routine()


def test_start_waits_for_work_until_cancelled():
    async def exercise():
        clock = LogicalClock()
        assert clock.start() is True
        task = clock._task
        assert task is not None

        await asyncio.sleep(0.02)

        assert not task.done()
        elapsed = clock.seconds

        await asyncio.sleep(0.02)

        assert clock.seconds > elapsed

        assert clock.stop() is True
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())


def test_relative_schedule_after_idle_uses_current_time():
    async def exercise():
        clock = LogicalClock()
        fired = asyncio.Event()
        observed: list[float] = []
        assert clock.start() is True
        task = clock._task
        assert task is not None

        def record(seconds: float) -> None:
            observed.append(seconds)
            fired.set()

        await asyncio.sleep(0.03)
        scheduled_at = clock.seconds
        clock.schedule(_instant_routine(record), relative=0.08)

        await asyncio.sleep(0.05)
        assert not fired.is_set()

        await asyncio.wait_for(fired.wait(), timeout=0.2)
        assert observed == [pytest.approx(scheduled_at + 0.08, abs=0.03)]

        assert clock.stop() is True
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())


def test_schedule_wakes_runner_while_waiting_for_later_event():
    async def exercise():
        clock = LogicalClock()
        early_fired = asyncio.Event()
        long_fired = asyncio.Event()
        observed: dict[str, float] = {}
        assert clock.start() is True
        task = clock._task
        assert task is not None

        def record(name: str, seconds: float) -> None:
            observed[name] = seconds
            if name == "early":
                early_fired.set()
            else:
                long_fired.set()

        clock.schedule(_instant_routine(lambda seconds: record("long", seconds)), relative=0.20)

        await asyncio.sleep(0.05)
        clock.schedule(_instant_routine(lambda seconds: record("early", seconds)), absolute=0.06)

        await asyncio.wait_for(early_fired.wait(), timeout=0.2)
        assert observed["early"] == pytest.approx(0.06, abs=0.03)
        assert not long_fired.is_set()

        await asyncio.wait_for(long_fired.wait(), timeout=0.2)
        assert observed["long"] == pytest.approx(0.20, abs=0.03)

        assert clock.stop() is True
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())


def test_scheduleing_later_event_does_not_delay_current_head():
    async def exercise():
        clock = LogicalClock()
        early_fired = asyncio.Event()
        later_fired = asyncio.Event()
        observed: dict[str, float] = {}
        assert clock.start() is True
        task = clock._task
        assert task is not None

        def record(name: str, seconds: float) -> None:
            observed[name] = seconds
            if name == "early":
                early_fired.set()
            else:
                later_fired.set()

        clock.schedule(_instant_routine(lambda seconds: record("early", seconds)), absolute=0.06)

        await asyncio.sleep(0.02)
        clock.schedule(_instant_routine(lambda seconds: record("later", seconds)), absolute=0.20)

        await asyncio.wait_for(early_fired.wait(), timeout=0.2)
        assert observed["early"] == pytest.approx(0.06, abs=0.03)
        assert not later_fired.is_set()

        await asyncio.wait_for(later_fired.wait(), timeout=0.2)
        assert observed["later"] == pytest.approx(0.20, abs=0.03)

        assert clock.stop() is True
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())


def test_start_compensates_for_step_processing_drift():
    async def exercise():
        clock = LogicalClock()
        completed = asyncio.Event()
        assert clock.start() is True
        task = clock._task
        assert task is not None

        def routine():
            for _ in range(5):
                pytime.sleep(0.03)
                yield 0.01

            pytime.sleep(0.03)
            completed.set()
            return
            yield 0.0

        started_at = pytime.monotonic()
        clock.schedule(routine())

        await asyncio.wait_for(completed.wait(), timeout=0.5)
        elapsed = pytime.monotonic() - started_at
        assert elapsed < 0.21

        assert clock.stop() is True
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())


def test_wait_for_idle_returns_when_queue_drains():
    async def exercise():
        clock = LogicalClock()
        fired: list[str] = []

        def routine(name: str, delay: float):
            yield delay
            fired.append(name)

        clock.schedule(routine("later", 0.04))
        clock.schedule(routine("sooner", 0.01))

        assert clock.start() is True
        task = clock._task
        assert task is not None

        await asyncio.wait_for(clock.wait_for_idle(), timeout=0.2)
        assert fired == ["sooner", "later"]
        assert clock._queue == []

        assert clock.stop() is True
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())


def test_wait_for_idle_returns_immediately_when_queue_is_empty():
    async def exercise():
        clock = LogicalClock()

        await asyncio.wait_for(clock.wait_for_idle(), timeout=0.01)

    asyncio.run(exercise())


def test_wait_for_idle_stays_pending_while_clock_is_stopped():
    async def exercise():
        clock = LogicalClock()

        def routine():
            yield 0.05

        clock.schedule(routine())
        assert clock.start() is True
        first_task = clock._task
        assert first_task is not None

        waiter = asyncio.create_task(clock.wait_for_idle())
        await asyncio.sleep(0.01)

        assert clock.stop() is True
        with pytest.raises(asyncio.CancelledError):
            await first_task

        await asyncio.sleep(0.06)
        assert not waiter.done()

        assert clock.start() is True
        second_task = clock._task
        assert second_task is not None

        await asyncio.wait_for(waiter, timeout=0.2)

        assert clock.stop() is True
        with pytest.raises(asyncio.CancelledError):
            await second_task

    asyncio.run(exercise())
