from __future__ import annotations

import math
from numbers import Integral

from .dag import DAG

import mc1._core
from mc1.clock import current_clock


_MAX_TIMETAG = 0xFFFFFFFFFFFFFFFF


def _validate_latency(latency: float) -> float:
    if isinstance(latency, bool):
        raise ValueError("latency must be a finite non-negative number")

    try:
        latency = float(latency)
    except (TypeError, ValueError) as exc:
        raise ValueError("latency must be a finite non-negative number") from exc

    if not math.isfinite(latency) or latency < 0:
        raise ValueError("latency must be a finite non-negative number")
    return latency


class _ScheduledDSP:
    __slots__ = ("_dsp", "_time_tag")

    def __init__(self, dsp: "DSP", time_tag: float) -> None:
        self._dsp = dsp
        self._time_tag = time_tag

    def append(self, synth_name, **controls):
        return mc1._core.DSP.append(self._dsp, self._time_tag, synth_name, **controls)

    def prepend(self, synth_name, **controls):
        return mc1._core.DSP.prepend(self._dsp, self._time_tag, synth_name, **controls)

    def insert_before(self, synth_id, synth_name, **controls):
        return mc1._core.DSP.insert_before(
            self._dsp, self._time_tag, synth_id, synth_name, **controls
        )

    def insert_after(self, synth_id, synth_name, **controls):
        return mc1._core.DSP.insert_after(
            self._dsp, self._time_tag, synth_id, synth_name, **controls
        )

    def set(self, synth_id, **controls):
        return mc1._core.DSP.set(self._dsp, self._time_tag, synth_id, **controls)

    def remove(self, synth_id):
        return mc1._core.DSP.remove(self._dsp, self._time_tag, synth_id)


class DSP(mc1._core.DSP):
    def __init__(self, *args, latency: float = 0.05, **kwargs) -> None:
        latency = _validate_latency(latency)
        super().__init__(*args, **kwargs)
        self.latency = latency

    def compile(self, func):
        dag = DAG(func) if not isinstance(func, DAG) else func
        super().compile(bytes(dag))
        return dag

    def __getitem__(self, when: float) -> _ScheduledDSP:
        return _ScheduledDSP(self, when)

    def _scheduled(self) -> _ScheduledDSP:
        return self[current_clock().time + self.latency]

    def append(self, synth_name, **controls):
        return self._scheduled().append(synth_name, **controls)

    def prepend(self, synth_name, **controls):
        return self._scheduled().prepend(synth_name, **controls)

    def insert_before(self, synth_id, synth_name, **controls):
        return self._scheduled().insert_before(synth_id, synth_name, **controls)

    def insert_after(self, synth_id, synth_name, **controls):
        return self._scheduled().insert_after(synth_id, synth_name, **controls)

    def set(self, synth_id, **controls):
        return self._scheduled().set(synth_id, **controls)

    def remove(self, synth_id):
        return self._scheduled().remove(synth_id)
