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


def test_dsp_compile_and_play():
    dsp = DSP()
    dsp.compile(bytes(tone))
    dsp.play("tone")
    dsp.play("tone", freq=220)


def test_dsp_play_unknown_synth_raises():
    dsp = DSP()
    with pytest.raises(ValueError):
        dsp.play("missing")


def test_dsp_compile_invalid_bytes_raises():
    dsp = DSP()
    with pytest.raises(ValueError):
        dsp.compile(b"not a dag")


def test_dsp_compile_same_name_overwrites_and_plays_multiple():
    dsp = DSP()
    dsp.compile(bytes(tone))
    dsp.compile(bytes(tone))
    dsp.play("tone")
    dsp.play("tone")


def test_dsp_play_unknown_control_raises():
    dsp = DSP()
    dsp.compile(bytes(tone))
    with pytest.raises(ValueError, match="unknown control"):
        dsp.play("tone", unknown=1.0)


def test_dsp_play_rejects_non_numeric_scalar_control():
    dsp = DSP()
    dsp.compile(bytes(tone))
    with pytest.raises(ValueError, match="must be a number"):
        dsp.play("tone", freq="nope")


@DAG
def _multi_control(freq=(440, 442)):
    Out.ar(0, SinOsc.ar(220))


def test_dsp_play_accepts_multi_width_control_sequence():
    dsp = DSP()
    dsp.compile(bytes(_multi_control))
    dsp.play("_multi_control", freq=[220, 330])


def test_dsp_play_rejects_multi_width_control_wrong_length():
    dsp = DSP()
    dsp.compile(bytes(_multi_control))
    with pytest.raises(ValueError, match="expects 2 values"):
        dsp.play("_multi_control", freq=[220])
