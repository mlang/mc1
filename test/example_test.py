from __future__ import annotations

import pathlib
import runpy

import pytest


EXAMPLE_PATH = pathlib.Path(__file__).resolve().parents[1] / "examples" / "joy.py"


def add(time_tag, seconds):
    return time_tag + int(round(float(seconds) * (1 << 32)))


class FakeDSP:
    def __init__(self):
        self.calls = []
        self._next_module_id = 1

    def start(self):
        self.calls.append(("start",))

    def stop(self):
        self.calls.append(("stop",))

    def append(self, time_tag, synth_name, **controls):
        module_id = self._next_module_id
        self._next_module_id += 1
        self.calls.append(("append", time_tag, module_id, synth_name, controls))
        return module_id

    def set(self, time_tag, module_id, **controls):
        self.calls.append(("set", time_tag, module_id, controls))

    def wait_until_idle(self, timeout=None):
        self.calls.append(("wait_until_idle", timeout))
        return True


def load_example():
    return runpy.run_path(str(EXAMPLE_PATH), run_name="default_melody_test")


def test_midi2cps_maps_a440():
    example = load_example()
    assert example["midi2cps"](69) == pytest.approx(440.0)


def test_parts_cover_all_translated_midi_lines():
    example = load_example()

    assert [part.name for part in example["PARTS"]] == [
        "Treble",
        "Alto",
        "Tenor",
        "Bass",
        "Piano_1",
        "Piano_2",
        "Piano2_1",
        "Piano2_2",
        "Piano3_1",
        "Piano3_2",
        "Piano4_1",
        "Piano4_2",
    ]


def test_voice_schedules_note_and_rest_durations():
    example = load_example()
    dsp = FakeDSP()
    start_time = 1 << 32

    total = example["voice"](
        [
            example["note"](69, 1),
            example["rest"](0.5),
            example["note"](72, 0.5),
        ],
        bpm=120,
        dsp_instance=dsp,
        start_time=start_time,
        voice_controls={"amp": 0.1, "release": 0.25, "done_action": 1},
    )

    assert total == pytest.approx(1.0)
    assert dsp.calls[0][0] == "append"
    assert dsp.calls[0][1] == start_time
    assert dsp.calls[0][4]["freq"] == pytest.approx(440.0)
    assert dsp.calls[1] == ("set", add(start_time, 0.5), 1, {"gate": 0})
    assert dsp.calls[2][0] == "append"
    assert dsp.calls[2][1] == add(start_time, 0.75)
    assert dsp.calls[3] == ("set", add(start_time, 1.0), 2, {"gate": 0})


def test_play_schedules_parts_from_shared_start_time():
    example = load_example()
    dsp = FakeDSP()
    start_time = 5 << 32
    parts = [
        example["Part"](
            "Lead",
            (
                example["note"](69, 1),
                example["rest"](0.5),
            ),
            {"amp": 0.1, "release": 0.25, "done_action": 1},
        ),
        example["Part"](
            "Bass",
            (
                example["rest"](0.5),
                example["note"](57, 1),
            ),
            {"amp": 0.1, "release": 0.25, "done_action": 1},
        ),
    ]

    total = example["play"](
        parts,
        bpm=120,
        dsp_instance=dsp,
        start_time=start_time,
    )

    assert total == pytest.approx(0.75)
    assert [call[0] for call in dsp.calls] == ["append", "set", "append", "set"]
    assert dsp.calls[0][1:4] == (start_time, 1, "default")
    assert dsp.calls[0][4]["freq"] == pytest.approx(440.0)
    assert dsp.calls[0][4]["gate"] == 1
    assert dsp.calls[1] == ("set", add(start_time, 0.5), 1, {"gate": 0})
    assert dsp.calls[2][1:4] == (add(start_time, 0.25), 2, "default")
    assert dsp.calls[2][4]["freq"] == pytest.approx(220.0)
    assert dsp.calls[2][4]["gate"] == 1
    assert dsp.calls[3] == ("set", add(start_time, 0.75), 2, {"gate": 0})


def test_main_requires_explicit_dsp_outside_mc1_host():
    example = load_example()

    with pytest.raises(RuntimeError, match="python -m mc1 examples/default_melody.py"):
        example["main"]()


def test_main_schedules_all_parts_and_waits_until_idle():
    example = load_example()
    dsp = FakeDSP()
    now_calls = 0

    def fake_now():
        nonlocal now_calls
        now_calls += 1
        return 7 << 32

    example["main"].__globals__["now"] = fake_now
    example["main"].__globals__["latency"] = lambda *, base, seconds=0.1: add(base, seconds)

    example["main"](dsp)

    assert now_calls == 1
    assert [call[0] for call in dsp.calls].count("start") == 1
    assert [call[0] for call in dsp.calls].count("wait_until_idle") == 1
    assert [call[0] for call in dsp.calls].count("stop") == 1

    append_calls = [call for call in dsp.calls if call[0] == "append"]
    set_calls = [call for call in dsp.calls if call[0] == "set"]

    expected_notes = sum(
        event.midi_note is not None
        for part in example["PARTS"]
        for event in part.events
    )
    assert len(append_calls) == expected_notes
    assert len(set_calls) == expected_notes
    assert {call[4]["carrier_ratio"] for call in append_calls} == {0.5, 0.75, 1.0}

    appended_ids = {call[2] for call in append_calls}
    released_ids = {call[2] for call in set_calls if call[3] == {"gate": 0}}
    assert released_ids == appended_ids

    start_time = add(7 << 32, 0.1)
    assert min(call[1] for call in append_calls) >= start_time
    assert min(call[1] for call in set_calls) >= start_time
    assert dsp.calls[-2] == ("wait_until_idle", None)
    assert dsp.calls[-1] == ("stop",)
