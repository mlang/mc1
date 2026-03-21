# mc1

`mc1` is a proof-of-concept re-architecture of SuperCollider with Python as the host language and `gccjit` as the compiler backend for DSP kernels.

The repo is aimed at developers iterating on the graph model, compiler, runtime, and host execution model. It is useful for evaluating ideas, not a stable end-user package.

## What This Repo Is Trying To Prove

- Python can stay the user-facing language; no separate DSL is required.
- Graphs built in Python can be serialized and compiled into efficient native DSP kernels.
- A mixed Python/C++ stack can keep iteration speed high while moving the DSP hot path into native code.
- Host-side scheduling and runtime control can stay lightweight enough to prototype musical structure directly in Python.

## Current State

- Early-stage PoC with active API and implementation churn.
- The core path works today: build a graph in Python, compile it through the native layer, instantiate it in the DSP runtime, and control it over time.
- The backend runtime is already a primary evaluation target, not just the graph serializer. Current tests cover compile/append/set/remove behavior, runtime ordering, timed commands, and idle waiting.
- Examples and helper APIs are still settling. Treat the repo as a research vehicle, not as a polished library surface.

## Developer Setup

### Prerequisites

- Python 3.12+
- `uv`
- CMake
- A C++23-capable compiler
- System development packages for `gccjit` and Boost

The Python package is built with `scikit-build-core` and `pybind11`, and the native extension links against `gccjit`.

### Common Commands

```bash
uv sync --dev
uv run pytest -q
uv run pytest -q test/dsp_test.py
uv run python -m mc1
uv run python -m mc1 examples/joy.py
```

`uv run python -m mc1` starts the current MiniCollider host/REPL. The first native build happens as part of the normal Python package build flow.

## Host Execution Model

`python -m mc1` configures a `DSP` instance, compiles the built-in `default` synth, and starts a `LogicalClock`.

In interactive mode, it exposes a namespace that includes:

- `dsp` for runtime operations
- `clock` for scheduling routines
- graph helpers and timing helpers from `mc1`

In script mode, the host executes the target script inside that same namespace. This is the intended model for backend examples such as `examples/joy.py`.

When a script returns, the host waits for the logical clock queue to drain before exiting. A script can schedule work and return without adding explicit shutdown code just to keep its routines alive.

## `examples/joy.py`

`examples/joy.py` is the current working backend example.

It is not a standalone importable library example. It is a host script meant to be run with:

```bash
uv run python -m mc1 examples/joy.py
```

The script models a score as Python data:

- `Event` values encode note or rest durations in beats plus optional per-note controls.
- `Part` values combine an event sequence with a fixed voice-level control set.
- `voice(...)` is a generator routine that converts beats to seconds, schedules note-on commands with `dsp.append(...)`, yields the event duration to the logical clock, then sends note-off with `dsp.set(..., gate=0)`.

This makes `joy.py` a useful reference for the current backend shape:

- musical structure stays in plain Python data and generators
- absolute timing is provided by the host clock
- synth instances are controlled through runtime commands rather than through a separate language layer

## Runtime Surface Being Exercised

The current repo is already testing and relying on these backend operations:

- `DSP.compile(bytes(dag))`
- `dsp.append(...)`
- `dsp.prepend(...)`
- `dsp.insert_before(...)`
- `dsp.insert_after(...)`
- `dsp.set(...)`
- `dsp.remove(...)`
- `dsp.start()` / `dsp.stop()`
- `dsp.wait_until_idle(...)`

For time tags, the main helpers are:

- `IMMEDIATE` for immediate commands
- `after(...)`, `add(...)`, `now()`, and `latency(...)`
- `mc1.timetag` helpers when a script needs explicit conversions

## Async Scheduling

The host clock is `asyncio`-backed. In the REPL, top-level `await` is enabled, so scheduled routines can be awaited directly from the prompt.

```python
await clock.schedule(tune())
```

`clock.schedule(...)` returns an awaitable handle. Awaiting that handle waits until the scheduled routine or routine group has left the clock queue.

Inside a running routine, `current_clock()` exposes the active logical clock. That is the mechanism `examples/joy.py` uses when it translates logical time into runtime time tags for `dsp.append(...)` and `dsp.set(...)`.

## Python Graph Building

Graphs are built directly with Python expressions. Arithmetic operators create graph nodes, so normal control flow and helpers like `sum` compose naturally.

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

With the native extension built, that graph can be compiled and instantiated through the runtime:

```python
from mc1 import DSP, IMMEDIATE

dsp = DSP()
dsp.compile(bytes(harmonics))
dsp.append(IMMEDIATE, "harmonics", freq=220)
```

Multi-channel expansion is also driven from ordinary Python values before serialization.

```python
from mc1 import Pan, SinOsc


voices = SinOsc.ar((220, 330, 440), phase=(0.0, 0.5))
stereo = Pan(voices, pan=(-0.6, 0.6))
```

## Repository Layout

- `mc1/`: Python API surface, DAG helpers, graph construction, scheduling helpers, and the REPL/host entry point.
- `c++/`: JIT compiler, DAG/runtime/audio bindings, and native test support.
- `c++/mlang/`: small C++ utility headers used by the native implementation.
- `test/`: Python regression tests and the compiler golden-output fixture.
- `poc/`: side experiments and older prototypes that inform the main design but are not the primary integration path.
- `.build/`: local generated build artifacts from scikit-build/CMake.

## Working Assumptions

- Expect interfaces to change while the graph model and runtime settle.
- Prefer readable, concise code and quick iteration over broad feature coverage.
- Review runtime tests and golden-output diffs deliberately when changing compiler or scheduling behavior.

## Near-Term Focus

- Correctness of generated DSP kernels
- Clarity of the Python graph model and control handling
- Runtime ergonomics for compiling, instantiating, ordering, and controlling synth instances
- Keeping the C++ layer modern, compact, and easy to reason about
