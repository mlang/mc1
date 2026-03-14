import mc1._core
from mc1._core import DSP
from mc1.dag import *
from mc1.graphs import drone


@DAG
def default(
    freq=440,
    amp=0.2,
    index=5.0,
    carrier_ratio=1.0,
    mod_ratio=3.0,
    gate=1,
    attack=0.01,
    decay=0.60,
    sustain=0.45,
    release=0.35,
    mod_attack=0.0,
    mod_decay=0.20,
    mod_sustain=0.18,
    mod_release=0.15,
    done_action=0,
):
    amp_env = ADSR.ar(gate, attack, decay, sustain, release, done_action)
    mod_env = ADSR.ar(gate, mod_attack, mod_decay, mod_sustain, mod_release)
    mod = SinOsc.ar(freq * mod_ratio, 0) * (freq * index * mod_env)
    carrier = SinOsc.ar(freq * carrier_ratio + mod, 0)
    Out.ar(0, Pan(carrier * amp * amp_env))


def perft(dag):
    if isinstance(dag, DAG):
        dag = bytes(dag)
    return mc1._core.perft(dag)
