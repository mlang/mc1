"""MiniCollider's Python front end for graph building and DSP execution.

`mc1` has two layers:

- Python graph construction via `@DAG` and opcode classes such as `SinOsc`,
  `ADSR`, `Pan`, and `Out`.
- Native compilation/runtime support in `mc1._core`, exposed here as `DSP`
  and `perft`.

Decorating a Python function with `@DAG` turns its default-valued parameters
into named controls and records every opcode allocated while the function runs.
Calling `bytes(...)` on the resulting `DAG` object serializes that graph for
the native compiler/runtime.

Simple example:

    >>> from mc1 import DAG, Out, Pan, SinOsc
    >>> @DAG
    ... def beep(freq=440, amp=0.1):
    ...     Out.ar(0, Pan(SinOsc.ar(freq) * amp))
    ...
    >>> graph_bytes = bytes(beep)
    >>> len(graph_bytes) > 0
    True

With the native extension built, the same graph can be compiled and scheduled as a synth:

    >>> from mc1 import DSP, IMMEDIATE
    >>> dsp = DSP()
    >>> dsp.compile(graph_bytes)
    >>> _ = dsp[IMMEDIATE].append("beep", freq=660)

Use `perft(...)` when you want to hand serialized graph bytes directly to the
native performance/compiler test entry point.
"""

import mc1._core
from mc1.clock import *
from mc1.dag import *
from mc1.dsp import DSP
from mc1.graphs import default, drone, klang_cloud, rhodey, rhodey_chorus, tube_bell
from mc1.pitch import *
from mc1.timetag import IMMEDIATE, here

__all__ = (
    "ADSR",
    "DAG",
    "DSP",
    "In",
    "LogicalClock",
    "Out",
    "Pan",
    "SinOsc",
    "Trigger",
    "current_clock",
    "default",
    "drone",
    "here",
    "IMMEDIATE",
    "klang_cloud",
    "midi2cps",
    "perft",
    "rhodey",
    "rhodey_chorus",
    "tube_bell",
)


def perft(dag):
    """Run the native performance/compiler test entry point on a graph.

    `dag` may be either a `DAG` instance or already-serialized `bytes`.

    Example:

        >>> perft(default)
        ...
    """
    if isinstance(dag, DAG):
        dag = bytes(dag)
    return mc1._core.perft(dag)
