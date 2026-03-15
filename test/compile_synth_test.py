"""Golden-output test."""

from __future__ import annotations
import base64
import difflib
import pathlib
import subprocess
import sys
from mc1 import ADSR, DAG, Out, SinOsc, default
from mc1.dag import EQ


def compile(dag) -> str:
    if isinstance(dag, DAG):
        dag = bytes(dag)

    encoded = base64.b64encode(dag).decode("ascii")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import base64, sys, mc1._core; mc1._core.perft(base64.b64decode(sys.argv[1]))",
            encoded,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stderr + result.stdout


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


@DAG
def ordered_compare(freq=1.0, threshold=0.0):
    Out.ar(0, SinOsc.ar(freq) >= threshold)


@DAG
def internal_eq(level=0.0, threshold=0.5):
    Out.ar(0, (SinOsc.ar(0) * 0) + EQ(level, threshold))


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


def test_compile_ordered_compare_uses_jit_kernel():
    actual = compile(ordered_compare)

    assert "void GE_aba" in actual
    assert "GE_aba (" in actual


def test_compile_internal_eq_opcode_succeeds():
    actual = compile(internal_eq)

    assert "internal_eq_process" in actual


if __name__ == "__main__":
    print("Regenerating golden output...")
    actual = compile(default)
    expected_path.write_text(actual)
