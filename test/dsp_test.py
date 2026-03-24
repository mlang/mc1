import time

import pytest

import mc1._core
import mc1.dsp
import mc1._test
import mc1.timetag
from mc1.graphs import default as graph_default
from mc1 import (
    ADSR,
    DAG,
    DSP,
    IMMEDIATE,
    Out,
    SinOsc,
    Trigger,
    default,
    here,
    klang_cloud,
    rhodey,
    rhodey_chorus,
    tube_bell,
)
from mc1.timetag import from_unix


def wait_for(condition, timeout=0.5, interval=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(interval)
    assert condition()


SHOWCASE_GRAPHS = (klang_cloud, tube_bell, rhodey, rhodey_chorus)


def test_dsp_defaults():
    dsp = DSP()
    assert dsp.sample_rate == 44100
    assert dsp.block_size == 32
    assert dsp.input_channels == 0
    assert dsp.output_channels == 2
    assert dsp.latency == 0.05


def test_dsp_keyword_args():
    dsp = DSP(sample_rate=48000, block_size=64, input_channels=1, output_channels=2, latency=0.1)
    assert dsp.sample_rate == 48000
    assert dsp.block_size == 64
    assert dsp.input_channels == 1
    assert dsp.output_channels == 2
    assert dsp.latency == 0.1


def test_aligned_abus_storage_is_64_byte_aligned():
    assert mc1._test._aligned_abus_modulo(block_size=1, channels=2) == 0
    assert mc1._test._aligned_abus_modulo(block_size=64, channels=2) == 0


def test_dsp_positional_args():
    dsp = DSP(96000, 128, 2, 4)
    assert dsp.sample_rate == 96000
    assert dsp.block_size == 128
    assert dsp.input_channels == 2
    assert dsp.output_channels == 4


def test_public_dsp_is_python_wrapper():
    dsp = DSP()
    assert isinstance(dsp, mc1._core.DSP)
    assert type(dsp) is DSP


def test_dsp_repr():
    dsp = DSP(sample_rate=48000, block_size=64, input_channels=1, output_channels=2)
    assert repr(dsp) == (
        "DSP(sample_rate=48000, block_size=64, input_channels=1, output_channels=2)"
    )


def test_dsp_rejects_non_positive_values():
    with pytest.raises(ValueError):
        DSP(sample_rate=0, block_size=32)
    with pytest.raises(ValueError):
        DSP(sample_rate=44100, block_size=0)
    with pytest.raises(ValueError):
        DSP(sample_rate=-1, block_size=32)
    with pytest.raises(ValueError):
        DSP(sample_rate=44100, block_size=-1)


def test_dsp_rejects_invalid_channel_values():
    with pytest.raises(ValueError):
        DSP(input_channels=-1)
    with pytest.raises(ValueError):
        DSP(output_channels=-1)
    with pytest.raises(ValueError):
        DSP(input_channels=0, output_channels=0)


def test_dsp_rejects_invalid_latency():
    with pytest.raises(ValueError, match="latency must be a finite non-negative number"):
        DSP(latency=-0.01)
    with pytest.raises(ValueError, match="latency must be a finite non-negative number"):
        DSP(latency=float("inf"))


def test_dsp_compile_and_append():
    dsp = DSP()
    dsp.compile(default)
    dsp[IMMEDIATE].append("default")
    dsp[IMMEDIATE].append("default", freq=220)


def test_default_is_reexported_from_graphs():
    assert graph_default is default


def test_dsp_compile_and_append_showcase_graphs():
    dsp = DSP()

    for graph in SHOWCASE_GRAPHS:
        dsp.compile(graph)

    for graph in SHOWCASE_GRAPHS:
        dsp[IMMEDIATE].append(graph.name)


def test_rhodey_chorus_graph_is_larger_than_single_voice():
    assert len(rhodey_chorus.operations) > len(rhodey.operations)


def test_dsp_start_and_stop():
    dsp = DSP()
    dsp.stop()
    dsp.start()
    dsp.start()
    dsp.stop()
    dsp.stop()


def test_dsp_append_unknown_synth_raises():
    dsp = DSP()
    with pytest.raises(ValueError):
        dsp[IMMEDIATE].append("missing")


def test_dsp_compile_same_name_overwrites_and_appends_multiple():
    dsp = DSP()
    dsp.compile(default)
    dsp.compile(default)
    dsp[IMMEDIATE].append("default")
    dsp[IMMEDIATE].append("default")


def test_dsp_append_unknown_control_raises():
    dsp = DSP()
    dsp.compile(default)
    with pytest.raises(ValueError, match="unknown control"):
        dsp[IMMEDIATE].append("default", unknown=1.0)


def test_dsp_append_rejects_non_numeric_scalar_control():
    dsp = DSP()
    dsp.compile(default)
    with pytest.raises(ValueError, match="must be a number"):
        dsp[IMMEDIATE].append("default", freq="nope")


@DAG
def _multi_control(freq=(440, 442)):
    Out.ar(0, SinOsc.ar(220))


@DAG
def _trigger_control(trig: Trigger = 0):
    Out.ar(0, SinOsc.ar(220) * trig)


@DAG
def _annotated_value_control(freq: float = 440):
    Out.ar(0, SinOsc.ar(freq))


@DAG
def _trigger_default_pulse(trig: Trigger = 1):
    Out.ar(0, SinOsc.ar(220) * trig)


@DAG
def _adsr_env(gate=0, attack=0.25, decay=0.25, sustain=0.25, release=0.25, done_action=0):
    Out.ar(0, ADSR.ar(gate, attack, decay, sustain, release, done_action))


@DAG
def _block_comparison(level=0.0, threshold=0.5):
    Out.ar(0, (SinOsc.ar(0) * 0) + (level > threshold))


@DAG
def _audio_comparison(freq=1.0):
    Out.ar(0, SinOsc.ar(freq) < -0.5)


@DAG
def _double_out_constant():
    zero = SinOsc.ar(0) * 0
    Out.ar(0, zero + 0.25)
    Out.ar(0, zero + 0.5)


@DAG
def _constant_quarter():
    Out.ar(0, (SinOsc.ar(0) * 0) + 0.25)


def test_dsp_append_accepts_multi_width_control_sequence():
    dsp = DSP()
    dsp.compile(_multi_control)
    dsp[IMMEDIATE].append("_multi_control", freq=[220, 330])


def test_dsp_compile_accepts_trigger_control_annotation():
    dsp = DSP()
    dsp.compile(_trigger_control)
    dsp[IMMEDIATE].append("_trigger_control", trig=1)


def test_dsp_compile_ignores_non_trigger_annotations():
    dsp = DSP()
    dsp.compile(_annotated_value_control)
    dsp[IMMEDIATE].append("_annotated_value_control", freq=220)


def test_trigger_control_rejects_sequence_default():
    with pytest.raises(ValueError, match="must be a scalar default value"):
        @DAG
        def bad_trigger(trig: Trigger = (0, 1)):
            Out.ar(0, SinOsc.ar(220) * trig)


def test_dsp_append_rejects_multi_width_control_wrong_length():
    dsp = DSP()
    dsp.compile(_multi_control)
    with pytest.raises(ValueError, match="expects 2 values"):
        dsp[IMMEDIATE].append("_multi_control", freq=[220])


def test_dsp_remove_added_synth():
    dsp = DSP()
    dsp.compile(default)
    synth_id = dsp[IMMEDIATE].append("default")
    dsp[IMMEDIATE].remove(synth_id)


def test_dsp_set_added_synth_control():
    dsp = DSP()
    dsp.compile(default)
    synth_id = dsp[IMMEDIATE].append("default")
    dsp[IMMEDIATE].set(synth_id, freq=220)


def test_dsp_set_accepts_multi_width_control_sequence():
    dsp = DSP()
    dsp.compile(_multi_control)
    synth_id = dsp[IMMEDIATE].append("_multi_control")
    dsp[IMMEDIATE].set(synth_id, freq=[220, 330])


def test_dsp_append_rejects_sequence_value_for_trigger_control():
    dsp = DSP()
    dsp.compile(_trigger_control)
    with pytest.raises(ValueError, match="must be a number"):
        dsp[IMMEDIATE].append("_trigger_control", trig=[1, 0])


def test_dsp_set_rejects_sequence_value_for_trigger_control():
    dsp = DSP()
    dsp.compile(_trigger_control)
    synth_id = dsp[IMMEDIATE].append("_trigger_control")
    with pytest.raises(ValueError, match="must be a number"):
        dsp[IMMEDIATE].set(synth_id, trig=[1, 0])


def test_dsp_set_unknown_control_raises():
    dsp = DSP()
    dsp.compile(default)
    synth_id = dsp[IMMEDIATE].append("default")
    with pytest.raises(ValueError, match="unknown control"):
        dsp[IMMEDIATE].set(synth_id, unknown=1.0)


def test_dsp_set_rejects_non_numeric_scalar_control():
    dsp = DSP()
    dsp.compile(default)
    synth_id = dsp[IMMEDIATE].append("default")
    with pytest.raises(ValueError, match="must be a number"):
        dsp[IMMEDIATE].set(synth_id, freq="nope")


def test_dsp_set_rejects_multi_width_control_wrong_length():
    dsp = DSP()
    dsp.compile(_multi_control)
    synth_id = dsp[IMMEDIATE].append("_multi_control")
    with pytest.raises(ValueError, match="expects 2 values"):
        dsp[IMMEDIATE].set(synth_id, freq=[220])


def test_dsp_remove_unknown_synth_id_raises():
    dsp = DSP()
    with pytest.raises(ValueError, match="unknown synth_id"):
        dsp[IMMEDIATE].remove(1)


def test_dsp_set_unknown_synth_id_raises():
    dsp = DSP()
    with pytest.raises(ValueError, match="unknown synth_id"):
        dsp[IMMEDIATE].set(1, freq=220)


def test_dsp_add_is_not_available():
    assert not hasattr(DSP, "add")


def test_dsp_synth_ids_reflect_runtime_order():
    dsp = DSP()
    dsp.compile(default)
    first = dsp[IMMEDIATE].append("default")
    second = dsp[IMMEDIATE].append("default")
    head = dsp[IMMEDIATE].prepend("default")
    before_second = dsp[IMMEDIATE].insert_before(second, "default")
    after_head = dsp[IMMEDIATE].insert_after(head, "default")

    assert dsp.synth_ids == [head, after_head, first, before_second, second]


def test_dsp_remove_preserves_survivor_order():
    dsp = DSP()
    dsp.compile(default)
    first = dsp[IMMEDIATE].append("default")
    second = dsp[IMMEDIATE].append("default")
    third = dsp[IMMEDIATE].append("default")
    fourth = dsp[IMMEDIATE].append("default")

    assert dsp.synth_ids == [first, second, third, fourth]

    dsp[IMMEDIATE].remove(second)
    assert dsp.synth_ids == [first, third, fourth]

    dsp[IMMEDIATE].remove(fourth)
    assert dsp.synth_ids == [first, third]

    dsp[IMMEDIATE].remove(first)
    assert dsp.synth_ids == [third]


def test_dsp_insert_before_unknown_synth_id_raises():
    dsp = DSP()
    dsp.compile(default)
    with pytest.raises(ValueError, match="unknown synth_id"):
        dsp[IMMEDIATE].insert_before(1, "default")


def test_dsp_insert_after_unknown_synth_id_raises():
    dsp = DSP()
    dsp.compile(default)
    with pytest.raises(ValueError, match="unknown synth_id"):
        dsp[IMMEDIATE].insert_after(1, "default")


def test_dsp_prepend_unknown_control_raises():
    dsp = DSP()
    dsp.compile(default)
    with pytest.raises(ValueError, match="unknown control"):
        dsp[IMMEDIATE].prepend("default", unknown=1.0)


def test_dsp_insert_after_rejects_non_numeric_scalar_control():
    dsp = DSP()
    dsp.compile(default)
    anchor = dsp[IMMEDIATE].append("default")
    with pytest.raises(ValueError, match="must be a number"):
        dsp[IMMEDIATE].insert_after(anchor, "default", freq="nope")


def test_dsp_insert_before_rejects_multi_width_control_wrong_length():
    dsp = DSP()
    dsp.compile(_multi_control)
    anchor = dsp[IMMEDIATE].append("_multi_control")
    with pytest.raises(ValueError, match="expects 2 values"):
        dsp[IMMEDIATE].insert_before(anchor, "_multi_control", freq=[220])


def test_compiled_controls_preserve_trigger_kind():
    controls = mc1._test._compiled_controls(bytes(_trigger_control))

    assert controls == [
        {
            "name": "trig",
            "index": 0,
            "width": 1,
            "kind": "trigger",
        }
    ]


def test_default_compiled_controls_include_mod_env():
    controls = mc1._test._compiled_controls(bytes(default))

    assert [control["name"] for control in controls] == [
        "freq",
        "amp",
        "index",
        "carrier_ratio",
        "mod_ratio",
        "gate",
        "env_attack",
        "env_decay",
        "env_sustain",
        "env_release",
        "mod_attack",
        "mod_decay",
        "mod_sustain",
        "mod_release",
        "done_action",
    ]
    assert all(control["width"] == 1 for control in controls)
    assert all(control["kind"] == "value" for control in controls)


def test_default_control_defaults_are_fm_oriented():
    assert [name for name, _, _ in default.controlNames] == [
        "freq",
        "amp",
        "index",
        "carrier_ratio",
        "mod_ratio",
        "gate",
        "env_attack",
        "env_decay",
        "env_sustain",
        "env_release",
        "mod_attack",
        "mod_decay",
        "mod_sustain",
        "mod_release",
        "done_action",
    ]


def test_trigger_default_resets_after_first_block():
    first_block, second_block = mc1._test._render_blocks(bytes(_trigger_default_pulse), blocks=2)

    assert any(sample != 0.0 for sample in first_block)
    assert all(sample == 0.0 for sample in second_block)


def test_adsr_renders_attack_decay_sustain_release():
    rendered = mc1._test._render_control_blocks(
        bytes(_adsr_env),
        [
            {"gate": 1, "attack": 0.25, "decay": 0.25, "sustain": 0.25, "release": 0.25},
            {"gate": 1},
            {"gate": 0},
        ],
        sample_rate=8,
        block_size=4,
        output_channels=1,
    )

    assert rendered["blocks"][0] == pytest.approx([0.5, 1.0, 0.625, 0.25])
    assert rendered["blocks"][1] == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert rendered["blocks"][2] == pytest.approx([0.125, 0.0, 0.0, 0.0])
    assert rendered["done_actions"] == [0, 0, 0]


def test_adsr_zero_times_and_done_action_fire_immediately():
    rendered = mc1._test._render_control_blocks(
        bytes(_adsr_env),
        [
            {"gate": 1, "attack": 0, "decay": 0, "sustain": 0.4, "release": 0, "done_action": 1},
            {"gate": 0},
        ],
        sample_rate=8,
        block_size=1,
        output_channels=1,
    )

    assert rendered["blocks"][0] == pytest.approx([0.4])
    assert rendered["blocks"][1] == pytest.approx([0.0])
    assert rendered["done_actions"] == [0, 1]


def test_block_rate_comparison_renders_bipolar_mask():
    rendered = mc1._test._render_control_blocks(
        bytes(_block_comparison),
        [
            {"level": 0.25, "threshold": 0.5},
            {"level": 0.75, "threshold": 0.5},
        ],
        sample_rate=8,
        block_size=4,
        output_channels=1,
    )

    assert rendered["blocks"][0] == pytest.approx([-1.0, -1.0, -1.0, -1.0])
    assert rendered["blocks"][1] == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_audio_rate_comparison_renders_bipolar_mask():
    rendered = mc1._test._render_blocks(
        bytes(_audio_comparison),
        blocks=1,
        sample_rate=8,
        block_size=8,
        output_channels=1,
    )

    assert rendered[0] == pytest.approx([-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0])


def test_multiple_out_nodes_mix_on_same_bus():
    rendered = mc1._test._render_blocks(
        bytes(_double_out_constant),
        blocks=1,
        sample_rate=8,
        block_size=4,
        output_channels=1,
    )

    assert rendered[0] == pytest.approx([0.75, 0.75, 0.75, 0.75])


def test_runtime_mixes_multiple_synths_on_same_bus():
    rendered = mc1._test._runtime_render_blocks(
        bytes(_constant_quarter),
        synth_count=2,
        blocks=1,
        sample_rate=8,
        block_size=4,
        output_channels=1,
    )

    assert rendered[0] == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_adsr_done_action_removes_synth_from_runtime():
    synth_ids = mc1._test._runtime_synth_ids_per_block(
        bytes(_adsr_env),
        [
            {"gate": 1, "attack": 0, "decay": 0, "sustain": 1, "release": 0, "done_action": 1},
            {"gate": 0},
            {"gate": 0},
        ],
        sample_rate=8,
        block_size=1,
        output_channels=1,
    )

    assert synth_ids == [[1], [], []]


def test_adsr_done_action_zero_keeps_synth_in_runtime():
    synth_ids = mc1._test._runtime_synth_ids_per_block(
        bytes(_adsr_env),
        [
            {"gate": 1, "attack": 0, "decay": 0, "sustain": 1, "release": 0, "done_action": 0},
            {"gate": 0},
            {"gate": 0},
        ],
        sample_rate=8,
        block_size=1,
        output_channels=1,
    )

    assert synth_ids == [[1], [1], [1]]


def test_default_mod_env_decays_timbre_faster_than_loudness():
    rendered = mc1._test._render_control_blocks(
        bytes(default),
        [{"amp": 1, "env_attack": 0, "env_decay": 0, "env_sustain": 1, "env_release": 0}] + ([{}] * 11),
        sample_rate=44100,
        block_size=1024,
        output_channels=2,
    )

    def left_channel(block):
        return block[::2]

    def mean_abs_second_difference(samples):
        return sum(
            abs(samples[index + 2] - (2 * samples[index + 1]) + samples[index])
            for index in range(len(samples) - 2)
        ) / (len(samples) - 2)

    first_block = left_channel(rendered["blocks"][0])
    sustain_block = left_channel(rendered["blocks"][-1])
    sustain_level = sum(abs(sample) for sample in sustain_block) / len(sustain_block)
    first_curvature = mean_abs_second_difference(first_block)
    sustain_curvature = mean_abs_second_difference(sustain_block)

    assert sustain_level > 0.1
    assert first_curvature > sustain_curvature * 2


def test_dsp_stale_anchor_insert_is_dropped():
    dsp = DSP()
    dsp.compile(default)
    anchor = dsp[IMMEDIATE].append("default")
    survivor = dsp[IMMEDIATE].append("default")

    dsp[IMMEDIATE].remove(anchor)
    dropped = dsp[IMMEDIATE].insert_after(anchor, "default")

    assert dsp.synth_ids == [survivor]

    with pytest.raises(ValueError, match="unknown synth_id"):
        dsp[IMMEDIATE].set(dropped, freq=220)


def test_dsp_getitem_rejects_invalid_timetag():
    dsp = DSP()
    with pytest.raises(ValueError, match="time_tag must be an OSC timetag integer"):
        dsp[1.5]
    with pytest.raises(ValueError, match="time_tag must be an OSC timetag integer"):
        dsp[True]


def test_dsp_append_uses_current_clock_and_latency(monkeypatch):
    dsp = DSP(latency=0.05)
    dsp.compile(default)
    observed = []

    monkeypatch.setattr(
        mc1.dsp,
        "current_clock",
        lambda: type("FakeClock", (), {"time": 42.0})(),
    )
    monkeypatch.setattr(
        mc1.dsp,
        "from_unix",
        lambda seconds: observed.append(seconds) or IMMEDIATE,
    )

    synth_id = dsp.append("default")

    assert observed == pytest.approx([42.05])
    wait_for(lambda: dsp.synth_ids == [synth_id])


def test_dsp_set_uses_current_clock_and_latency(monkeypatch):
    dsp = DSP(latency=0.05)
    dsp.compile(default)
    synth_id = dsp[IMMEDIATE].append("default")
    observed = {"clock": 0, "seconds": []}

    def fake_current_clock():
        observed["clock"] += 1
        return type("FakeClock", (), {"time": 42.0})()

    monkeypatch.setattr(mc1.dsp, "current_clock", fake_current_clock)
    monkeypatch.setattr(
        mc1.dsp,
        "from_unix",
        lambda seconds: observed["seconds"].append(seconds) or IMMEDIATE,
    )

    dsp.set(synth_id, freq=220)

    assert observed["clock"] == 1
    assert observed["seconds"] == pytest.approx([42.05])


def test_dsp_remove_requires_running_clock_for_implicit_schedule():
    dsp = DSP()
    with pytest.raises(RuntimeError, match="current_clock\\(\\) is only available"):
        dsp.remove(1)


def test_timetag_here_schedules_current_clock_wallclock_values(monkeypatch):
    monkeypatch.setattr(
        mc1.timetag,
        "current_clock",
        lambda: type("FakeClock", (), {"time": 42.0})(),
    )

    assert here(0) == from_unix(42.0)
    assert here() == from_unix(42.1)
    assert here(0.05) > here(0)


def test_dsp_scheduled_append_remains_pending_until_due():
    dsp = DSP()
    dsp.compile(default)

    scheduled = from_unix(time.time() + 0.05)
    synth_id = dsp[scheduled].append("default")

    assert dsp.synth_ids == []

    wait_for(lambda: dsp.synth_ids == [synth_id])


def test_dsp_same_timetag_commands_preserve_fifo_order():
    dsp = DSP()
    dsp.compile(default)

    scheduled = from_unix(time.time() + 0.05)
    first = dsp[scheduled].append("default")
    second = dsp[scheduled].append("default")
    before_second = dsp[scheduled].insert_before(second, "default")

    assert dsp.synth_ids == []

    wait_for(lambda: dsp.synth_ids == [first, before_second, second])


def test_dsp_scheduled_remove_applies_when_due():
    dsp = DSP()
    dsp.compile(default)

    first = dsp[IMMEDIATE].append("default")
    second = dsp[IMMEDIATE].append("default")
    scheduled = from_unix(time.time() + 0.05)
    dsp[scheduled].remove(first)

    assert dsp.synth_ids == [first, second]

    wait_for(lambda: dsp.synth_ids == [second])


def test_dsp_scheduled_insert_after_missing_anchor_is_dropped():
    dsp = DSP()
    dsp.compile(default)

    anchor = dsp[IMMEDIATE].append("default")
    survivor = dsp[IMMEDIATE].append("default")
    remove_at_unix = time.time() + 0.03
    insert_at_unix = remove_at_unix + 0.02
    remove_at = from_unix(remove_at_unix)
    insert_at = from_unix(insert_at_unix)
    dropped = dsp[insert_at].insert_after(anchor, "default")
    dsp[remove_at].remove(anchor)

    wait_for(lambda: dsp.synth_ids == [survivor])
    wait_for(lambda: time.time() >= insert_at_unix and dsp.synth_ids == [survivor])

    with pytest.raises(ValueError, match="unknown synth_id"):
        dsp[IMMEDIATE].set(dropped, freq=220)


def test_dsp_wait_until_idle_requires_started_runtime():
    dsp = DSP()
    with pytest.raises(ValueError, match="wait_until_idle is only available while DSP is started"):
        dsp.wait_until_idle()
