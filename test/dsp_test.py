import pytest

import mc1._core
import mc1._test
from mc1 import ADSR, DAG, DSP, Out, SinOsc, Trigger, default


def test_dsp_defaults():
    dsp = DSP()
    assert dsp.sample_rate == 44100
    assert dsp.block_size == 32
    assert dsp.input_channels == 0
    assert dsp.output_channels == 2


def test_dsp_keyword_args():
    dsp = DSP(sample_rate=48000, block_size=64, input_channels=1, output_channels=2)
    assert dsp.sample_rate == 48000
    assert dsp.block_size == 64
    assert dsp.input_channels == 1
    assert dsp.output_channels == 2


def test_aligned_abus_storage_is_64_byte_aligned():
    assert mc1._test._aligned_abus_modulo(block_size=1, channels=2) == 0
    assert mc1._test._aligned_abus_modulo(block_size=64, channels=2) == 0


def test_dsp_positional_args():
    dsp = DSP(96000, 128, 2, 4)
    assert dsp.sample_rate == 96000
    assert dsp.block_size == 128
    assert dsp.input_channels == 2
    assert dsp.output_channels == 4


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


def test_dsp_compile_and_append():
    dsp = DSP()
    dsp.compile(bytes(default))
    dsp.append("default")
    dsp.append("default", freq=220)


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
        dsp.append("missing")


def test_dsp_compile_invalid_bytes_raises():
    dsp = DSP()
    with pytest.raises(ValueError):
        dsp.compile(b"not a dag")


def test_dsp_compile_same_name_overwrites_and_appends_multiple():
    dsp = DSP()
    dsp.compile(bytes(default))
    dsp.compile(bytes(default))
    dsp.append("default")
    dsp.append("default")


def test_dsp_append_unknown_control_raises():
    dsp = DSP()
    dsp.compile(bytes(default))
    with pytest.raises(ValueError, match="unknown control"):
        dsp.append("default", unknown=1.0)


def test_dsp_append_rejects_non_numeric_scalar_control():
    dsp = DSP()
    dsp.compile(bytes(default))
    with pytest.raises(ValueError, match="must be a number"):
        dsp.append("default", freq="nope")


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


def test_dsp_append_accepts_multi_width_control_sequence():
    dsp = DSP()
    dsp.compile(bytes(_multi_control))
    dsp.append("_multi_control", freq=[220, 330])


def test_dsp_compile_accepts_trigger_control_annotation():
    dsp = DSP()
    dsp.compile(bytes(_trigger_control))
    dsp.append("_trigger_control", trig=1)


def test_dsp_compile_ignores_non_trigger_annotations():
    dsp = DSP()
    dsp.compile(bytes(_annotated_value_control))
    dsp.append("_annotated_value_control", freq=220)


def test_trigger_control_rejects_sequence_default():
    with pytest.raises(ValueError, match="must be a scalar default value"):
        @DAG
        def bad_trigger(trig: Trigger = (0, 1)):
            Out.ar(0, SinOsc.ar(220) * trig)


def test_dsp_append_rejects_multi_width_control_wrong_length():
    dsp = DSP()
    dsp.compile(bytes(_multi_control))
    with pytest.raises(ValueError, match="expects 2 values"):
        dsp.append("_multi_control", freq=[220])


def test_dsp_remove_added_module():
    dsp = DSP()
    dsp.compile(bytes(default))
    module_id = dsp.append("default")
    dsp.remove(module_id)


def test_dsp_set_added_module_control():
    dsp = DSP()
    dsp.compile(bytes(default))
    module_id = dsp.append("default")
    dsp.set(module_id, freq=220)


def test_dsp_set_accepts_multi_width_control_sequence():
    dsp = DSP()
    dsp.compile(bytes(_multi_control))
    module_id = dsp.append("_multi_control")
    dsp.set(module_id, freq=[220, 330])


def test_dsp_append_rejects_sequence_value_for_trigger_control():
    dsp = DSP()
    dsp.compile(bytes(_trigger_control))
    with pytest.raises(ValueError, match="must be a number"):
        dsp.append("_trigger_control", trig=[1, 0])


def test_dsp_set_rejects_sequence_value_for_trigger_control():
    dsp = DSP()
    dsp.compile(bytes(_trigger_control))
    module_id = dsp.append("_trigger_control")
    with pytest.raises(ValueError, match="must be a number"):
        dsp.set(module_id, trig=[1, 0])


def test_dsp_set_unknown_control_raises():
    dsp = DSP()
    dsp.compile(bytes(default))
    module_id = dsp.append("default")
    with pytest.raises(ValueError, match="unknown control"):
        dsp.set(module_id, unknown=1.0)


def test_dsp_set_rejects_non_numeric_scalar_control():
    dsp = DSP()
    dsp.compile(bytes(default))
    module_id = dsp.append("default")
    with pytest.raises(ValueError, match="must be a number"):
        dsp.set(module_id, freq="nope")


def test_dsp_set_rejects_multi_width_control_wrong_length():
    dsp = DSP()
    dsp.compile(bytes(_multi_control))
    module_id = dsp.append("_multi_control")
    with pytest.raises(ValueError, match="expects 2 values"):
        dsp.set(module_id, freq=[220])


def test_dsp_remove_unknown_module_id_raises():
    dsp = DSP()
    with pytest.raises(ValueError, match="unknown module_id"):
        dsp.remove(1)


def test_dsp_set_unknown_module_id_raises():
    dsp = DSP()
    with pytest.raises(ValueError, match="unknown module_id"):
        dsp.set(1, freq=220)


def test_dsp_add_is_not_available():
    assert not hasattr(DSP, "add")


def test_dsp_module_ids_reflect_runtime_order():
    dsp = DSP()
    dsp.compile(bytes(default))
    first = dsp.append("default")
    second = dsp.append("default")
    head = dsp.prepend("default")
    before_second = dsp.insert_before(second, "default")
    after_head = dsp.insert_after(head, "default")

    assert dsp.module_ids == [head, after_head, first, before_second, second]


def test_dsp_remove_preserves_survivor_order():
    dsp = DSP()
    dsp.compile(bytes(default))
    first = dsp.append("default")
    second = dsp.append("default")
    third = dsp.append("default")
    fourth = dsp.append("default")

    assert dsp.module_ids == [first, second, third, fourth]

    dsp.remove(second)
    assert dsp.module_ids == [first, third, fourth]

    dsp.remove(fourth)
    assert dsp.module_ids == [first, third]

    dsp.remove(first)
    assert dsp.module_ids == [third]


def test_dsp_insert_before_unknown_module_id_raises():
    dsp = DSP()
    dsp.compile(bytes(default))
    with pytest.raises(ValueError, match="unknown module_id"):
        dsp.insert_before(1, "default")


def test_dsp_insert_after_unknown_module_id_raises():
    dsp = DSP()
    dsp.compile(bytes(default))
    with pytest.raises(ValueError, match="unknown module_id"):
        dsp.insert_after(1, "default")


def test_dsp_prepend_unknown_control_raises():
    dsp = DSP()
    dsp.compile(bytes(default))
    with pytest.raises(ValueError, match="unknown control"):
        dsp.prepend("default", unknown=1.0)


def test_dsp_insert_after_rejects_non_numeric_scalar_control():
    dsp = DSP()
    dsp.compile(bytes(default))
    anchor = dsp.append("default")
    with pytest.raises(ValueError, match="must be a number"):
        dsp.insert_after(anchor, "default", freq="nope")


def test_dsp_insert_before_rejects_multi_width_control_wrong_length():
    dsp = DSP()
    dsp.compile(bytes(_multi_control))
    anchor = dsp.append("_multi_control")
    with pytest.raises(ValueError, match="expects 2 values"):
        dsp.insert_before(anchor, "_multi_control", freq=[220])


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
        "attack",
        "decay",
        "sustain",
        "release",
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
        "attack",
        "decay",
        "sustain",
        "release",
        "mod_attack",
        "mod_decay",
        "mod_sustain",
        "mod_release",
        "done_action",
    ]
    assert default.controls == pytest.approx([
        440.0,
        0.2,
        5.0,
        1.0,
        3.0,
        1.0,
        0.01,
        0.60,
        0.45,
        0.35,
        0.0,
        0.20,
        0.18,
        0.15,
        0.0,
    ])


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


def test_adsr_done_action_removes_module_from_runtime():
    module_ids = mc1._test._runtime_module_ids_per_block(
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

    assert module_ids == [[1], [], []]


def test_adsr_done_action_zero_keeps_module_in_runtime():
    module_ids = mc1._test._runtime_module_ids_per_block(
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

    assert module_ids == [[1], [1], [1]]


def test_default_mod_env_decays_timbre_faster_than_loudness():
    rendered = mc1._test._render_control_blocks(
        bytes(default),
        [{"amp": 1, "attack": 0, "decay": 0, "sustain": 1, "release": 0}] + ([{}] * 11),
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
    dsp.compile(bytes(default))
    anchor = dsp.append("default")
    survivor = dsp.append("default")

    dsp.remove(anchor)
    dropped = dsp.insert_after(anchor, "default")

    assert dsp.module_ids == [survivor]

    with pytest.raises(ValueError, match="unknown module_id"):
        dsp.set(dropped, freq=220)
