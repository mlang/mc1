from mc1.dag import ADSR, DAG, Out, Pan, SinOsc


KLANG_RATIOS = (0.56, 0.57, 0.92, 0.93, 1.19, 1.70, 2.00, 2.74, 3.00, 3.76, 4.07, 5.30)
KLANG_WEIGHTS = (1.00, 0.90, 1.00, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15)
CHORUS_DETUNE_SHAPE = (-1.0, -0.35, 0.25, 1.0)
CHORUS_PANS = (-0.75, -0.25, 0.25, 0.75)
_KLANG_WEIGHT_TOTAL = sum(KLANG_WEIGHTS)


def _fm_pair(carrier_freq, mod_freq, mod_amount):
    return SinOsc.ar(carrier_freq + SinOsc.ar(mod_freq) * mod_amount)


def _klang_bank(base_freq):
    return sum(
        SinOsc.ar(base_freq * ratio) * (weight / _KLANG_WEIGHT_TOTAL)
        for ratio, weight in zip(KLANG_RATIOS, KLANG_WEIGHTS, strict=True)
    )


def _rhodey_voice(freq, index, crossfade, body_env, tine_env):
    body = _fm_pair(freq, freq * 2.0, freq * index * body_env)
    tine = _fm_pair(freq * 4.0, freq * 7.0, freq * (index * 0.5) * tine_env)
    return body * (1.0 - crossfade) + tine * crossfade


@DAG
def default(
    freq=440,
    amp=0.2,
    index=5.0,
    carrier_ratio=1.0,
    mod_ratio=3.0,
    gate=1,
    env_attack=0.01,
    env_decay=0.20,
    env_sustain=0.45,
    env_release=0.45,
    mod_attack=0.0,
    mod_decay=0.20,
    mod_sustain=0.18,
    mod_release=0.15,
    done_action=1
):
    """A compact FM voice used as the package's default synth graph.

    The graph exposes pitch, amplitude, FM index/ratios, and separate carrier
    and modulation envelopes. It is a good reference for how a non-trivial
    `@DAG` graph is assembled from Python expressions.
    """
    amp_env = ADSR.ar(gate, env_attack, env_decay, env_sustain, env_release, done_action)
    mod_env = ADSR.ar(gate, mod_attack, mod_decay, mod_sustain, mod_release)
    mod = SinOsc.ar(freq * mod_ratio, 0) * (freq * index * mod_env)
    carrier = SinOsc.ar(freq * carrier_ratio + mod, 0)
    Out.ar(0, Pan(carrier * amp * amp_env))


@DAG
def tone(freq=440, amp=0.2):
    """A minimal centered sine tone graph."""
    Out.ar(0, Pan(SinOsc.ar(freq) * amp))


@DAG
def drone(freq=440, amp=0.2):
    """A bright additive drone built from the first 49 harmonics."""
    Out.ar(0, Pan(sum(SinOsc.ar(freq * i) * (1/i) for i in range(1, 50)) * amp))


@DAG
def klang_cloud(freq=110.0, amp=0.12, detune=0.003, spread=0.8):
    """A dense additive cloud inspired by SuperCollider's sine-bank UGens."""
    left = _klang_bank(freq * (1.0 - detune)) * (amp * 0.5)
    right = _klang_bank(freq * (1.0 + detune)) * (amp * 0.5)
    Out.ar(0, Pan(left, pan=spread * -0.5))
    Out.ar(0, Pan(right, pan=spread * 0.5))


@DAG
def tube_bell(
    freq=440.0,
    amp=0.2,
    index=2.0,
    crossfade=0.35,
    vib_rate=6.0,
    vib_depth=0.003,
    gate=1,
    attack=0.001,
    decay=4.0,
    sustain=0.0,
    release=0.25,
    mod_decay=1.2,
):
    """A sine-only tubular bell port of the STK/Csound algorithm-5 family."""
    body_env = ADSR.ar(gate, attack, decay, sustain, release)
    mod_env = ADSR.ar(gate, 0.001, mod_decay, 0.0, 0.08)
    vibrato = SinOsc.ar(vib_rate) * (freq * vib_depth)
    base_freq = freq + vibrato
    pair_a = _fm_pair(base_freq, base_freq * 1.414, base_freq * index * 2.0 * mod_env)
    pair_b = _fm_pair(base_freq * 2.0, base_freq * 2.82, base_freq * index * 1.25 * mod_env)
    signal = (pair_a * (1.0 - crossfade) + pair_b * crossfade) * amp * body_env
    Out.ar(0, Pan(signal))


@DAG
def rhodey(
    freq=220.0,
    amp=0.2,
    index=3.0,
    crossfade=0.2,
    vib_rate=4.0,
    vib_depth=0.002,
    gate=1,
    attack=0.001,
    decay=1.5,
    sustain=0.0,
    release=0.45,
    tine_decay=0.08,
):
    """A sine-only electric piano port of the STK/Csound Rhodey family."""
    body_env = ADSR.ar(gate, attack, decay, sustain, release)
    tine_env = ADSR.ar(gate, 0.001, tine_decay, 0.0, 0.06)
    vibrato = SinOsc.ar(vib_rate) * (freq * vib_depth)
    signal = _rhodey_voice(freq + vibrato, index, crossfade, body_env, tine_env) * amp
    Out.ar(0, Pan(signal))


@DAG
def rhodey_chorus(
    freq=220.0,
    amp=0.18,
    index=3.0,
    crossfade=0.2,
    vib_rate=4.0,
    vib_depth=0.002,
    detune=0.003,
    spread=1.0,
    gate=1,
    attack=0.001,
    decay=1.8,
    sustain=0.0,
    release=0.5,
    tine_decay=0.08,
):
    """A widened four-voice Rhodey stack intended as the heavy FM showcase."""
    body_env = ADSR.ar(gate, attack, decay, sustain, release)
    tine_env = ADSR.ar(gate, 0.001, tine_decay, 0.0, 0.06)
    vibrato = SinOsc.ar(vib_rate) * (freq * vib_depth)
    voice_freqs = tuple(
        (freq + vibrato) * (1.0 + detune * shape)
        for shape in CHORUS_DETUNE_SHAPE
    )
    voices = tuple(
        _rhodey_voice(voice_freq, index, crossfade, body_env, tine_env) * amp * 0.25
        for voice_freq in voice_freqs
    )
    pans = Pan(
        voices,
        pan=tuple(pan * spread for pan in CHORUS_PANS),
    )
    for signal in pans:
        Out.ar(0, signal)
