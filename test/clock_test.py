from __future__ import annotations

import asyncio
import time as pytime

import pytest

from mc1.clock import LogicalClock, current_clock, merge


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


def test_delay_schedule_after_idle_uses_current_time():
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
        clock.schedule(_instant_routine(record), delay=0.08)

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

        clock.schedule(_instant_routine(lambda seconds: record("long", seconds)), delay=0.20)

        await asyncio.sleep(0.05)
        clock.schedule(_instant_routine(lambda seconds: record("early", seconds)), at=0.06)

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

        clock.schedule(_instant_routine(lambda seconds: record("early", seconds)), at=0.06)

        await asyncio.sleep(0.02)
        clock.schedule(_instant_routine(lambda seconds: record("later", seconds)), at=0.20)

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


def test_merge_interleaves_routines_by_next_due_time():
    observed: list[tuple[str, float]] = []

    def routine(name: str, *delays: float):
        observed.append((name, 0.0))
        for delay in delays:
            yield delay
            observed.append((name, delay))

    merged = merge(routine("a", 0.10, 0.20), routine("b", 0.05))

    assert next(merged) == pytest.approx(0.05)
    assert next(merged) == pytest.approx(0.05)
    assert next(merged) == pytest.approx(0.20)
    with pytest.raises(StopIteration):
        next(merged)

    assert observed == [
        ("a", 0.0),
        ("b", 0.0),
        ("b", 0.05),
        ("a", 0.10),
        ("a", 0.20),
    ]


def test_merge_prefers_input_order_for_equal_due_times():
    observed: list[str] = []

    def routine_a():
        observed.append("a-start")
        yield 0.05
        observed.append("a-mid")
        yield 0.05
        observed.append("a-end")

    def routine_b():
        observed.append("b-start")
        yield 0.10
        observed.append("b-end")

    merged = merge(routine_a(), routine_b())

    assert next(merged) == pytest.approx(0.05)
    assert observed == ["a-start", "b-start"]

    assert next(merged) == pytest.approx(0.05)
    assert observed == ["a-start", "b-start", "a-mid"]

    with pytest.raises(StopIteration):
        next(merged)

    assert observed == ["a-start", "b-start", "a-mid", "a-end", "b-end"]


def test_merge_rejects_empty_input():
    with pytest.raises(ValueError):
        merge()


def test_schedule_accepts_multiple_routines_with_one_handle():
    async def exercise():
        clock = LogicalClock()
        observed: list[tuple[str, float]] = []
        assert clock.start() is True
        task = clock._task
        assert task is not None

        def routine(name: str, *, tail_delay: float | None = None):
            observed.append((f"{name}-start", current_clock().seconds))
            if tail_delay is None:
                return
                yield 0.0

            yield tail_delay
            observed.append((f"{name}-end", current_clock().seconds))

        handle = clock.schedule(routine("short"), routine("long", tail_delay=0.03), at=0.04)

        await asyncio.wait_for(handle, timeout=0.2)
        assert observed[0][0] == "short-start"
        assert observed[1][0] == "long-start"
        assert observed[0][1] == pytest.approx(0.04, abs=0.03)
        assert observed[1][1] == pytest.approx(0.04, abs=0.03)
        assert observed[2][0] == "long-end"
        assert observed[2][1] == pytest.approx(0.07, abs=0.03)
        assert clock._queue == []

        assert clock.stop() is True
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())


def test_cancelling_grouped_schedule_cancels_all_routines():
    async def exercise():
        clock = LogicalClock()
        fired: list[str] = []
        assert clock.start() is True
        task = clock._task
        assert task is not None

        def routine(name: str):
            fired.append(name)
            return
            yield 0.0

        handle = clock.schedule(routine("first"), routine("second"), delay=0.05)
        handle.cancel()

        await asyncio.sleep(0.08)
        assert fired == []
        assert clock._queue == []

        assert clock.stop() is True
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
