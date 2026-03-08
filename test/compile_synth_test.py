"""Golden-output test."""

from __future__ import annotations
import difflib
import pathlib
from wurlitzer import pipes
from mc1 import perft, tone

def compile(dag) -> str:
    with pipes() as (out, err): perf = perft(bytes(dag))
    return err.read() + out.read()


expected_path = pathlib.Path(__file__).with_name("compile_synth.expected.txt")

def test_compile_synth_golden():
    actual = compile(tone)
    expected = expected_path.read_text()

    if actual != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile=str(expected_path),
                tofile="actual",
            )
        )
        raise AssertionError(diff)


if __name__ == "__main__":
    print("Regenerating golden output...")
    actual = compile(tone)
    expected_path.write_text(actual)
