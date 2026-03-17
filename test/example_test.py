from __future__ import annotations

import pathlib
import runpy

import pytest


EXAMPLE_PATH = pathlib.Path(__file__).resolve().parents[1] / "examples" / "joy.py"


class FakeDSP:
    def __init__(self):
        self.calls = []
        self._next_module_id = 1

    def start(self):
        self.calls.append(("start",))

    def stop(self):
        self.calls.append(("stop",))

    def append(self, synth_name, **controls):
        module_id = self._next_module_id
        self._next_module_id += 1
        self.calls.append(("append", module_id, synth_name, controls))
        return module_id

    def set(self, module_id, **controls):
        self.calls.append(("set", module_id, controls))


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


def test_voice_yields_note_and_rest_durations():
    example = load_example()
    dsp = FakeDSP()

    player = example["voice"](
        [
            example["note"](69, 1),
            example["rest"](0.5),
            example["note"](72, 0.5),
        ],
        bpm=120,
        dsp_instance=dsp,
        voice_controls={"amp": 0.1, "release": 0.25, "done_action": 1},
    )

    assert next(player) == pytest.approx(0.5)
    assert dsp.calls[0][0] == "append"
    assert dsp.calls[0][3]["freq"] == pytest.approx(440.0)

    assert next(player) == pytest.approx(0.25)
    assert dsp.calls[1] == ("set", 1, {"gate": 0})

    assert next(player) == pytest.approx(0.25)
    assert dsp.calls[2][0] == "append"

    with pytest.raises(StopIteration):
        next(player)

    assert dsp.calls[3] == ("set", 2, {"gate": 0})


def test_play_interleaves_generators_by_shortest_pending_delay():
    example = load_example()
    events = []
    sleeps = []

    def generator(name, waits):
        for wait in waits:
            events.append((name, "tick"))
            yield wait
        events.append((name, "done"))

    example["play"](
        [
            generator("lead", [1.0, 1.0]),
            generator("bass", [0.5, 1.5]),
        ],
        sleep=sleeps.append,
    )

    assert sleeps == pytest.approx([0.5, 0.5, 1.0])
    assert events == [
        ("lead", "tick"),
        ("bass", "tick"),
        ("bass", "tick"),
        ("lead", "tick"),
        ("lead", "done"),
        ("bass", "done"),
    ]


def test_main_requires_explicit_dsp_outside_mc1_host():
    example = load_example()

    with pytest.raises(RuntimeError, match="python -m mc1 examples/default_melody.py"):
        example["main"](sleep=lambda delay: None)


def test_main_plays_all_parts_and_releases_every_note():
    example = load_example()
    dsp = FakeDSP()
    sleeps = []

    example["main"](dsp, sleep=sleeps.append)

    assert [call[0] for call in dsp.calls].count("start") == 1
    assert [call[0] for call in dsp.calls].count("stop") == 1

    append_calls = [call for call in dsp.calls if call[0] == "append"]
    set_calls = [call for call in dsp.calls if call[0] == "set"]

    expected_notes = sum(
        event.midi_note is not None
        for part in example["PARTS"]
        for event in part.events
    )
    assert len(append_calls) == expected_notes
    assert {call[3]["carrier_ratio"] for call in append_calls} == {0.5, 0.75, 1.0}

    appended_ids = {call[1] for call in append_calls}
    released_ids = {
        call[1]
        for call in set_calls
        if call[2] == {"gate": 0}
    }
    assert released_ids == appended_ids

    expected_total = (
        max(
            sum(event.beats for event in part.events)
            for part in example["PARTS"]
        )
        * 60.0
        / example["BPM"]
    ) + max(
        part.voice_controls["release"]
        for part in example["PARTS"]
    )
    assert sum(sleeps) == pytest.approx(expected_total)
