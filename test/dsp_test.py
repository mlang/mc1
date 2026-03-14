import pytest

import mc1._core
import mc1._test
from mc1 import DAG, DSP, Out, SinOsc, Trigger, tone


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
    dsp.compile(bytes(tone))
    dsp.append("tone")
    dsp.append("tone", freq=220)


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
    dsp.compile(bytes(tone))
    dsp.compile(bytes(tone))
    dsp.append("tone")
    dsp.append("tone")


def test_dsp_append_unknown_control_raises():
    dsp = DSP()
    dsp.compile(bytes(tone))
    with pytest.raises(ValueError, match="unknown control"):
        dsp.append("tone", unknown=1.0)


def test_dsp_append_rejects_non_numeric_scalar_control():
    dsp = DSP()
    dsp.compile(bytes(tone))
    with pytest.raises(ValueError, match="must be a number"):
        dsp.append("tone", freq="nope")


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
    dsp.compile(bytes(tone))
    module_id = dsp.append("tone")
    dsp.remove(module_id)


def test_dsp_set_added_module_control():
    dsp = DSP()
    dsp.compile(bytes(tone))
    module_id = dsp.append("tone")
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
    dsp.compile(bytes(tone))
    module_id = dsp.append("tone")
    with pytest.raises(ValueError, match="unknown control"):
        dsp.set(module_id, unknown=1.0)


def test_dsp_set_rejects_non_numeric_scalar_control():
    dsp = DSP()
    dsp.compile(bytes(tone))
    module_id = dsp.append("tone")
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
    dsp.compile(bytes(tone))
    first = dsp.append("tone")
    second = dsp.append("tone")
    head = dsp.prepend("tone")
    before_second = dsp.insert_before(second, "tone")
    after_head = dsp.insert_after(head, "tone")

    assert dsp.module_ids == [head, after_head, first, before_second, second]


def test_dsp_remove_preserves_survivor_order():
    dsp = DSP()
    dsp.compile(bytes(tone))
    first = dsp.append("tone")
    second = dsp.append("tone")
    third = dsp.append("tone")
    fourth = dsp.append("tone")

    assert dsp.module_ids == [first, second, third, fourth]

    dsp.remove(second)
    assert dsp.module_ids == [first, third, fourth]

    dsp.remove(fourth)
    assert dsp.module_ids == [first, third]

    dsp.remove(first)
    assert dsp.module_ids == [third]


def test_dsp_insert_before_unknown_module_id_raises():
    dsp = DSP()
    dsp.compile(bytes(tone))
    with pytest.raises(ValueError, match="unknown module_id"):
        dsp.insert_before(1, "tone")


def test_dsp_insert_after_unknown_module_id_raises():
    dsp = DSP()
    dsp.compile(bytes(tone))
    with pytest.raises(ValueError, match="unknown module_id"):
        dsp.insert_after(1, "tone")


def test_dsp_prepend_unknown_control_raises():
    dsp = DSP()
    dsp.compile(bytes(tone))
    with pytest.raises(ValueError, match="unknown control"):
        dsp.prepend("tone", unknown=1.0)


def test_dsp_insert_after_rejects_non_numeric_scalar_control():
    dsp = DSP()
    dsp.compile(bytes(tone))
    anchor = dsp.append("tone")
    with pytest.raises(ValueError, match="must be a number"):
        dsp.insert_after(anchor, "tone", freq="nope")


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


def test_trigger_default_resets_after_first_block():
    first_block, second_block = mc1._test._render_blocks(bytes(_trigger_default_pulse), blocks=2)

    assert any(sample != 0.0 for sample in first_block)
    assert all(sample == 0.0 for sample in second_block)


def test_dsp_stale_anchor_insert_is_dropped():
    dsp = DSP()
    dsp.compile(bytes(tone))
    anchor = dsp.append("tone")
    survivor = dsp.append("tone")

    dsp.remove(anchor)
    dropped = dsp.insert_after(anchor, "tone")

    assert dsp.module_ids == [survivor]

    with pytest.raises(ValueError, match="unknown module_id"):
        dsp.set(dropped, freq=220)
