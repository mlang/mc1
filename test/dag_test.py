from __future__ import annotations

from functools import reduce
import operator

import pytest

from mc1 import ADSR, DAG, In, Out, Pan, SinOsc
from mc1.dag import Add, EQ, GE, GT, LE, LT, Mul, NE, Rate


@pytest.fixture(autouse=True)
def reset_dag_state():
    DAG._reset()
    yield
    DAG._reset()


def test_graph_args_new_returns_single_node_for_scalar_inputs():
    osc = SinOsc.ar(220, 0.5)

    assert isinstance(osc, SinOsc)
    assert osc.rate is Rate.AUDIO
    assert float(osc.args[0]) == 220.0
    assert float(osc.args[1]) == 0.5


def test_graph_args_new_expands_positional_sequences_to_longest_input():
    oscs = SinOsc.ar(range(220, 550, 110), (0.0, 0.5))

    assert isinstance(oscs, tuple)
    assert len(oscs) == 3
    assert [float(osc.args[0]) for osc in oscs] == [220.0, 330.0, 440.0]
    assert [float(osc.args[1]) for osc in oscs] == [0.0, 0.5, 0.0]


def test_graph_args_new_expands_keyword_sequences():
    signal = SinOsc.ar(220)
    pans = Pan(signal, pan=[-0.75, 0.75])

    assert isinstance(pans, tuple)
    assert len(pans) == 2
    assert [pan.args[0] for pan in pans] == [signal, signal]
    assert [float(pan.args[1]) for pan in pans] == [-0.75, 0.75]


def test_graph_args_new_can_expand_pan_inputs_and_cycle_shorter_sequences():
    voices = [SinOsc.ar(220), SinOsc.ar(330), SinOsc.ar(440)]
    stereo = Pan(voices, pan=(-0.5, 0.5))

    assert isinstance(stereo, tuple)
    assert len(stereo) == 3
    assert [pan.args[0] for pan in stereo] == voices
    assert [float(pan.args[1]) for pan in stereo] == [-0.5, 0.5, -0.5]

def test_graph_args_new_can_expand_out_indices():
    signal = SinOsc.ar(220)
    outs = Out.ar(range(0, 6, 2), signal)

    assert isinstance(outs, tuple)
    assert len(outs) == 3
    assert [float(out.args[0]) for out in outs] == [0.0, 2.0, 4.0]
    assert [out.args[1] for out in outs] == [signal, signal, signal]
    assert [out.num_out for out in outs] == [1, 1, 1]


def test_python_sum_builds_graph_additions():
    mix = sum(
        SinOsc.ar(freq) * weight
        for freq, weight in ((220, 1.0), (330, 0.5), (440, 0.25))
    )

    assert isinstance(mix, Add)
    assert mix.rate is Rate.AUDIO


def test_reduce_can_fold_graph_nodes_with_operators():
    folded = reduce(
        operator.mul,
        (
            SinOsc.ar(220),
            SinOsc.ar(2) * 0.25 + 0.75,
            0.1,
        ),
        1,
    )

    assert isinstance(folded, Mul)
    assert folded.rate is Rate.AUDIO


def test_seeded_bounds_cover_known_signal_shapes():
    osc = SinOsc.ar(220)
    env = ADSR.ar(1, 0.01, 0.2, 0.5, 0.4)
    gt = osc > 0.25

    assert osc.bounds == (-1.0, 1.0)
    assert osc.is_bipolar is True
    assert osc.is_unipolar is False
    assert env.bounds == (0.0, 1.0)
    assert env.is_unipolar is True
    assert env.is_bipolar is False
    assert gt.bounds == (-1.0, 1.0)


def test_arithmetic_bounds_are_inferred_with_interval_math():
    modulator = SinOsc.ar(2) * 0.25 + 0.75

    assert modulator.bounds == (0.5, 1.0)
    assert modulator.rate is Rate.AUDIO


def test_pan_and_out_forward_signal_bounds():
    signal = ADSR.ar(1, 0.01, 0.2, 0.5, 0.4)
    stereo = Pan(signal, pan=0.25)
    out = Out.ar(0, stereo)

    assert stereo.bounds == (0.0, 1.0)
    assert out.bounds == (0.0, 1.0)


def test_range_remaps_bipolar_signal_to_requested_bounds():
    signal = SinOsc.ar(220)
    remapped = signal.range(-0.5, 0.5)

    assert isinstance(remapped, Add)
    assert remapped.bounds == (-0.5, 0.5)


def test_range_supports_inverted_output_ranges():
    env = ADSR.ar(1, 0.01, 0.2, 0.5, 0.4)
    remapped = env.range(1.0, 0.0)

    assert remapped.bounds == (0.0, 1.0)


def test_range_requires_known_source_bounds():
    signal = In.ar(0)

    with pytest.raises(ValueError, match="known source bounds"):
        signal.range(-1.0, 1.0)


def test_range_rejects_zero_width_source_ranges():
    constant = SinOsc.ar(220).with_bounds(0.5, 0.5)

    with pytest.raises(ValueError, match="non-zero source range"):
        constant.range(0.0, 1.0)


def test_with_bounds_enables_range_for_unknown_inputs():
    signal = In.ar(0).with_bounds(0.0, 1.0)
    cutoff = signal.range(200.0, 4000.0)

    assert signal.bounds == (0.0, 1.0)
    assert cutoff.bounds == (200.0, 4000.0)


def test_ordering_comparisons_build_graph_nodes():
    signal = SinOsc.ar(220)

    lt = signal < 0.25
    le = signal <= 0.25
    gt = signal > 0.25
    ge = signal >= 0.25

    assert isinstance(lt, LT)
    assert isinstance(le, LE)
    assert isinstance(gt, GT)
    assert isinstance(ge, GE)
    assert lt.rate is Rate.AUDIO
    assert le.rate is Rate.AUDIO
    assert gt.rate is Rate.AUDIO
    assert ge.rate is Rate.AUDIO


def test_internal_equality_comparison_nodes_preserve_fastest_rate():
    signal = SinOsc.ar(220)

    eq = EQ(signal, 0.0)
    ne = NE(0.0, signal)

    assert isinstance(eq, EQ)
    assert isinstance(ne, NE)
    assert eq.rate is Rate.AUDIO
    assert ne.rate is Rate.AUDIO


def test_python_equality_remains_identity_based_boolean():
    signal = SinOsc.ar(220)
    other = SinOsc.ar(220)

    assert (signal == signal) is True
    assert (signal == other) is False
    assert (signal != signal) is False
    assert (signal != other) is True
