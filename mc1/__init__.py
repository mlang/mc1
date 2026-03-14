import mc1._core
from mc1._core import DSP
from mc1.dag import *
from mc1.graphs import drone


@DAG
def default(
    freq=440,
    amp=0.2,
    index=1.0,
    carrier_ratio=1.0,
    mod_ratio=2.0,
    gate=1,
    attack=0.25,
    decay=0.25,
    sustain=0.25,
    release=0.25,
    done_action=0,
):
    env = ADSR.ar(gate, attack, decay, sustain, release, done_action)
    mod = SinOsc.ar(freq * mod_ratio, 0) * (freq * index * env)
    carrier = SinOsc.ar(freq * carrier_ratio + mod, 0)
    Out.ar(0, Pan(carrier * amp * env))


def perft(dag):
    if isinstance(dag, DAG):
        dag = bytes(dag)
    return mc1._core.perft(dag)
