"""Golden-output test."""

from __future__ import annotations
import difflib
import pathlib
from wurlitzer import pipes
from mc1 import ADSR, DAG, Out, SinOsc, default, perft


def compile(dag) -> str:
    with pipes() as (out, err): perf = perft(bytes(dag))
    return err.read() + out.read()


@DAG
def adsr_tone(
    freq=220,
    gate=0,
    attack=0.25,
    decay=0.25,
    sustain=0.25,
    release=0.25,
    done_action=0,
):
    Out.ar(0, SinOsc.ar(freq) * ADSR.ar(gate, attack, decay, sustain, release, done_action))


expected_path = pathlib.Path(__file__).with_name("compile_synth.expected.txt")

def test_compile_synth_golden():
    actual = compile(default)
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


def test_compile_adsr_uses_jit_kernel():
    actual = compile(adsr_tone)

    assert "void ADSR_bbbbbba" in actual
    assert "d9 = ADSR_bbbbbba" not in actual
    assert "mc1_adsr_process" not in actual


def test_compile_marks_abus_base_pointer_aligned():
    actual = compile(default)

    assert "__builtin_assume_aligned (abus, 64)" in actual


if __name__ == "__main__":
    print("Regenerating golden output...")
    actual = compile(default)
    expected_path.write_text(actual)
