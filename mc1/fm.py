"""FM synthesizer graphs for compiler tests."""

from mc1.dag import DAG, SinOsc


@DAG
def fm_two_op(
    base_freq=220.0,
    mod_ratio=2.0,
    index=1.0,
    amp=0.2,
):
    """Two-operator FM with musically useful whole-number ratios."""
    mod_freq = base_freq * mod_ratio
    modulation = SinOsc.ar(mod_freq, 0) * (base_freq * index)
    carrier = SinOsc.ar(base_freq + modulation, 0)
    return carrier * amp
