from __future__ import annotations

from functools import reduce
import operator

import pytest

from mc1 import DAG, Out, Pan, SinOsc
from mc1.dag import Add, Mul, Rate


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
