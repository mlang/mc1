"""Golden-output test for standalone compiler executable."""

from __future__ import annotations

import argparse
import difflib
import pathlib
import subprocess
import sys

from test.graphs import tone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "exe",
        nargs="?",
        default=".build/default/compile_synth_test",
        help="Path to compile_synth_test executable.",
    )
    parser.add_argument(
        "--regen",
        action="store_true",
        help="Regenerate expected output from current executable output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exe = pathlib.Path(args.exe)
    expected_path = pathlib.Path(__file__).with_name("compile_synth.expected.txt")

    proc = subprocess.run(
        [str(exe)],
        input=bytes(tone),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if proc.returncode != 0:
        sys.stderr.write(f"{exe} exited with {proc.returncode}\n")
        if proc.stderr:
            sys.stderr.write(proc.stderr.decode(errors="replace"))
        return 1

    stderr_text = proc.stderr.decode(errors="replace")
    stdout_text = proc.stdout.decode(errors="replace")
    actual = stderr_text + stdout_text

    if args.regen:
        expected_path.write_text(actual)
        return 0

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
        sys.stderr.write(diff)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
