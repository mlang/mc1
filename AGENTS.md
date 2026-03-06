# Repository Guidelines

## Project Structure & Module Organization
- `mc1/` contains the core implementation: Python orchestration (`dag.py`, `engine.py`, `message.py`) plus C++ compilation/runtime pieces (`compiler.cpp`, `dag.cpp`, `engine.cpp`, headers).
- `mc1/mlang/` holds lower-level C++ support headers used by the compiler/runtime path.
- `test/` contains Python and C++ tests, including graph fixtures (`graphs.py`) and golden-output artifacts (`compile_synth.expected.txt`).
- `poc/` contains experimental throwaway prototypes; do not treat it as stable API.
- Build output is expected under `.build/default/` via CMake presets.

## Build, Test, and Development Commands
- `cmake --preset default`: Configure a Release build in `.build/default`.
- `cmake --build --preset default`: Build all targets (`engine`, `dag2json`, `compile_synth_test`).
- `ctest --test-dir .build/default --output-on-failure`: Run the configured test suite (`mypy`, `pytest`, and compile golden test).
- `pytest -q`: Run Python tests directly from repo root.
- `python -m mc1` or `python -m mc1 test/compile.py`: Start the interactive engine or run a script.

## Coding Style & Naming Conventions
- Python: follow PEP 8, 4-space indentation, `snake_case` for functions/variables, `CamelCase` for node classes (for example `SinOsc`, `Control`).
- C++: project is set to C++23; follow existing style in `mc1/*.cpp` (2-space indentation, brace-on-next-line, `snake_case` methods, `CamelCase` types).
- Keep modules focused: graph modeling in `dag.py`, transport/messages in `message.py`, runtime control in `engine.py`.

## Testing Guidelines
- Add Python tests in `test/*_test.py`; prefer deterministic assertions and `pytest.mark.parametrize` for graph variations.
- For compiler output changes, update the golden file with:
  `python -m test.compile_synth_test .build/default/compile_synth_test --regen`.
- Ensure `ctest` passes before opening a PR.

## Commit & Pull Request Guidelines
- History shows both short subjects and Conventional Commit prefixes (`feat:`, `fix:`, `refactor:`, `docs:`); prefer Conventional Commit style going forward.
- Keep commit subjects imperative and concise (for example `fix: handle Control arg indexing in DAG serializer`).
- PRs should include: purpose, behavioral impact, test evidence (`ctest`/`pytest` output), and any golden-file updates.
