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

With the native extension built, the same graph can be compiled and played:

    >>> from mc1 import DSP
    >>> dsp = DSP()
    >>> dsp.compile(graph_bytes)
    >>> _ = dsp.append("beep", freq=660)

Use `perft(...)` when you want to hand serialized graph bytes directly to the
native performance/compiler test entry point.
"""

import mc1._core
from mc1._core import DSP
from mc1.dag import *
from mc1.graphs import drone

__all__ = (
    "ADSR",
    "DAG",
    "DSP",
    "In",
    "Out",
    "Pan",
    "SinOsc",
    "Trigger",
    "default",
    "drone",
    "perft",
)

def default(
    freq=440,
    amp=0.2,
    index=5.0,
    carrier_ratio=1.0,
    mod_ratio=3.0,
    gate=1,
    attack=0.01,
    decay=0.60,
    sustain=0.45,
    release=0.35,
    mod_attack=0.0,
    mod_decay=0.20,
    mod_sustain=0.18,
    mod_release=0.15,
    done_action=0,
):
    """A compact FM voice used as the package's default synth graph.

    The graph exposes pitch, amplitude, FM index/ratios, and separate carrier
    and modulation envelopes. It is a good reference for how a non-trivial
    `@DAG` graph is assembled from Python expressions.
    """
    amp_env = ADSR.ar(gate, attack, decay, sustain, release, done_action)
    mod_env = ADSR.ar(gate, mod_attack, mod_decay, mod_sustain, mod_release)
    mod = SinOsc.ar(freq * mod_ratio, 0) * (freq * index * mod_env)
    carrier = SinOsc.ar(freq * carrier_ratio + mod, 0)
    Out.ar(0, Pan(carrier * amp * amp_env))


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
