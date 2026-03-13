import pytest

from mc1 import DAG, DSP, Out, SinOsc, tone


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


def test_dsp_compile_and_add():
    dsp = DSP()
    dsp.compile(bytes(tone))
    dsp.add("tone")
    dsp.add("tone", freq=220)


def test_dsp_start_and_stop():
    dsp = DSP()
    dsp.stop()
    dsp.start()
    dsp.start()
    dsp.stop()
    dsp.stop()


def test_dsp_add_unknown_synth_raises():
    dsp = DSP()
    with pytest.raises(ValueError):
        dsp.add("missing")


def test_dsp_compile_invalid_bytes_raises():
    dsp = DSP()
    with pytest.raises(ValueError):
        dsp.compile(b"not a dag")


def test_dsp_compile_same_name_overwrites_and_adds_multiple():
    dsp = DSP()
    dsp.compile(bytes(tone))
    dsp.compile(bytes(tone))
    dsp.add("tone")
    dsp.add("tone")


def test_dsp_add_unknown_control_raises():
    dsp = DSP()
    dsp.compile(bytes(tone))
    with pytest.raises(ValueError, match="unknown control"):
        dsp.add("tone", unknown=1.0)


def test_dsp_add_rejects_non_numeric_scalar_control():
    dsp = DSP()
    dsp.compile(bytes(tone))
    with pytest.raises(ValueError, match="must be a number"):
        dsp.add("tone", freq="nope")


@DAG
def _multi_control(freq=(440, 442)):
    Out.ar(0, SinOsc.ar(220))


def test_dsp_add_accepts_multi_width_control_sequence():
    dsp = DSP()
    dsp.compile(bytes(_multi_control))
    dsp.add("_multi_control", freq=[220, 330])


def test_dsp_add_rejects_multi_width_control_wrong_length():
    dsp = DSP()
    dsp.compile(bytes(_multi_control))
    with pytest.raises(ValueError, match="expects 2 values"):
        dsp.add("_multi_control", freq=[220])


def test_dsp_remove_added_module():
    dsp = DSP()
    dsp.compile(bytes(tone))
    module_id = dsp.add("tone")
    dsp.remove(module_id)


def test_dsp_set_added_module_control():
    dsp = DSP()
    dsp.compile(bytes(tone))
    module_id = dsp.add("tone")
    dsp.set(module_id, freq=220)


def test_dsp_set_accepts_multi_width_control_sequence():
    dsp = DSP()
    dsp.compile(bytes(_multi_control))
    module_id = dsp.add("_multi_control")
    dsp.set(module_id, freq=[220, 330])


def test_dsp_set_unknown_control_raises():
    dsp = DSP()
    dsp.compile(bytes(tone))
    module_id = dsp.add("tone")
    with pytest.raises(ValueError, match="unknown control"):
        dsp.set(module_id, unknown=1.0)


def test_dsp_set_rejects_non_numeric_scalar_control():
    dsp = DSP()
    dsp.compile(bytes(tone))
    module_id = dsp.add("tone")
    with pytest.raises(ValueError, match="must be a number"):
        dsp.set(module_id, freq="nope")


def test_dsp_set_rejects_multi_width_control_wrong_length():
    dsp = DSP()
    dsp.compile(bytes(_multi_control))
    module_id = dsp.add("_multi_control")
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
