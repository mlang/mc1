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


def test_run_waits_for_work_until_cancelled():
    async def exercise():
        clock = LogicalClock()
        task = asyncio.create_task(clock.run())

        await asyncio.sleep(0.02)

        assert not task.done()
        elapsed = clock.seconds

        await asyncio.sleep(0.02)

        assert clock.seconds > elapsed

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())


def test_relative_play_after_idle_uses_current_time():
    async def exercise():
        clock = LogicalClock()
        fired = asyncio.Event()
        observed: list[float] = []
        task = asyncio.create_task(clock.run())

        def record(seconds: float) -> None:
            observed.append(seconds)
            fired.set()

        await asyncio.sleep(0.03)
        scheduled_at = clock.seconds
        clock.play(_instant_routine(record), relative=0.08)

        await asyncio.sleep(0.05)
        assert not fired.is_set()

        await asyncio.wait_for(fired.wait(), timeout=0.2)
        assert observed == [pytest.approx(scheduled_at + 0.08, abs=0.03)]

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())


def test_play_wakes_runner_while_waiting_for_later_event():
    async def exercise():
        clock = LogicalClock()
        early_fired = asyncio.Event()
        long_fired = asyncio.Event()
        observed: dict[str, float] = {}
        task = asyncio.create_task(clock.run())

        def record(name: str, seconds: float) -> None:
            observed[name] = seconds
            if name == "early":
                early_fired.set()
            else:
                long_fired.set()

        clock.play(_instant_routine(lambda seconds: record("long", seconds)), relative=0.20)

        await asyncio.sleep(0.05)
        clock.play(_instant_routine(lambda seconds: record("early", seconds)), absolute=0.06)

        await asyncio.wait_for(early_fired.wait(), timeout=0.2)
        assert observed["early"] == pytest.approx(0.06, abs=0.03)
        assert not long_fired.is_set()

        await asyncio.wait_for(long_fired.wait(), timeout=0.2)
        assert observed["long"] == pytest.approx(0.20, abs=0.03)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())


def test_playing_later_event_does_not_delay_current_head():
    async def exercise():
        clock = LogicalClock()
        early_fired = asyncio.Event()
        later_fired = asyncio.Event()
        observed: dict[str, float] = {}
        task = asyncio.create_task(clock.run())

        def record(name: str, seconds: float) -> None:
            observed[name] = seconds
            if name == "early":
                early_fired.set()
            else:
                later_fired.set()

        clock.play(_instant_routine(lambda seconds: record("early", seconds)), absolute=0.06)

        await asyncio.sleep(0.02)
        clock.play(_instant_routine(lambda seconds: record("later", seconds)), absolute=0.20)

        await asyncio.wait_for(early_fired.wait(), timeout=0.2)
        assert observed["early"] == pytest.approx(0.06, abs=0.03)
        assert not later_fired.is_set()

        await asyncio.wait_for(later_fired.wait(), timeout=0.2)
        assert observed["later"] == pytest.approx(0.20, abs=0.03)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())


def test_run_compensates_for_step_processing_drift():
    async def exercise():
        clock = LogicalClock()
        completed = asyncio.Event()
        task = asyncio.create_task(clock.run())

        def routine():
            for _ in range(5):
                pytime.sleep(0.03)
                yield 0.01

            pytime.sleep(0.03)
            completed.set()
            return
            yield 0.0

        started_at = pytime.monotonic()
        clock.play(routine())

        await asyncio.wait_for(completed.wait(), timeout=0.5)
        elapsed = pytime.monotonic() - started_at
        assert elapsed < 0.21

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
