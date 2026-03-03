# mc1

Experimental re-architecture of SuperCollider using Python as the host language and **gccjit** to compile SynthDefs into runtime process functions.

## Purpose
- Keep the user-facing API in Python (no new DSL).
- Model SynthDefs as graphs and JIT-compile them into efficient audio kernels.
- Explore stateful DSP nodes and scheduling without the SC server/client split.

## State
- Early prototype / research code.
- Core compilation path exists; API and graph model are still in flux.
- Not production-ready.

## Direction
- Python drives graph construction and orchestration.
- gccjit emits per-node kernels and block processors.
- Focus on correctness and performance of compiled DSP kernels.
