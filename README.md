# mc1

`mc1` is a proof-of-concept re-architecture of SuperCollider with Python as the host language and `gccjit` as the compiler backend for DSP kernels.

The current repo is aimed at developers working on the compiler, runtime, and graph model. It is intentionally exploratory code: useful for iterating on design ideas, not a stable end-user package.

## What This Repo Is Trying To Prove

- Python can stay the user-facing language; no separate DSL is required.
- SynthDefs can be compiled into efficient runtime process functions.
- A mixed Python/C++ stack can keep iteration speed high while moving the performance-critical DSP path into native code.

## Current State

- Early-stage PoC with active API and implementation churn.
- The core path exists: define a graph in Python, compile it through the native layer, and run it via the DSP runtime.
- Tests currently focus on DSP runtime behavior, graph/compiler behavior, and golden-output checks for generated code.
- The repo is not production-ready and should be treated as a research vehicle.

## Developer Setup

### Prerequisites

- Python 3.12+
- `uv`
- CMake
- A C++23-capable compiler
- System development packages for `gccjit` and Boost

The Python package is built with `scikit-build-core` and `pybind11`, and the native extension is linked against `gccjit`.

### Common Commands

```bash
uv sync --dev
uv run pytest -q
uv run pytest -q test/dsp_test.py
uv run python -m mc1
```

`uv run python -m mc1` starts the current MiniCollider REPL. The first build of the native modules happens as part of the normal Python package build flow.

## Demo

Run the bundled default-synth arrangement example with:

```bash
uv run python -m mc1 examples/joy.py
```

## Repository Layout

- `mc1/`: Python API surface, DAG helpers, graph construction, and the REPL entry point.
- `c++/`: JIT compiler, DAG/runtime/audio bindings, and native test support.
- `c++/mlang/`: small C++ utility headers used by the native implementation.
- `test/`: Python regression tests and the compiler golden-output fixture.
- `poc/`: side experiments and older prototypes that inform the main design but are not the primary integration path.
- `.build/`: local generated build artifacts from scikit-build/CMake.

## Working Assumptions

- Expect interfaces to change while the graph model and runtime settle.
- Optimize for readable, concise code and quick iteration over broad feature coverage.
- When changing compiler output, review both runtime tests and golden-output diffs deliberately.

## Python Graph Building

Graphs are built directly with Python expressions. Arithmetic operators create graph nodes, so ordinary control flow and helpers like `sum` compose naturally.

```python
from mc1 import DAG, Out, Pan, SinOsc


@DAG
def harmonics(freq=110, amp=0.12):
    partials = sum(
        SinOsc.ar(freq * harmonic) * (1 / harmonic)
        for harmonic in range(1, 6)
    )
    Out.ar(0, Pan(partials * amp))
```

Builtins and folds work the way you would expect on graph nodes, which is useful once you start combining operators programmatically.

```python
from functools import reduce
import operator

from mc1 import DAG, Out, Pan, SinOsc


@DAG
def folded_modulation(freq=220, amp=0.15):
    signal = reduce(
        operator.mul,
        (
            SinOsc.ar(freq),
            SinOsc.ar(2) * 0.25 + 0.75,
            amp,
        ),
        1,
    )
    Out.ar(0, Pan(signal))
```

Multi-channel expansion is also driven from plain Python values. Passing a `list`, `tuple`, or `range` into a graph constructor returns a tuple of nodes. Scalar inputs are broadcast, and shorter sequences wrap to match the longest input.

```python
from mc1 import Pan, SinOsc


voices = SinOsc.ar((220, 330, 440), phase=(0.0, 0.5))
stereo = Pan(voices, pan=(-0.6, 0.6))

# `voices` contains three oscillators with phases 0.0, 0.5, 0.0.
# `stereo` contains three panners with pans -0.6, 0.6, -0.6.
```

That expansion happens entirely on the Python side before the graph is serialized, which makes it easy to build small channelized structures with ordinary tuples, lists, and ranges.

## Near-Term Focus

- Correctness of generated DSP kernels
- Clarity of the Python graph model and control handling
- Runtime ergonomics for compiling, appending, and controlling synth instances
- Keeping the C++ layer modern, compact, and easy to reason about
