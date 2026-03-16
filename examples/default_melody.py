from __future__ import annotations

from dataclasses import dataclass
import time


BPM = 108
EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class Event:
    midi_note: int | None
    beats: float
    controls: dict[str, float]


def midi2cps(note):
    return 440.0 * 2 ** ((note - 69) / 12)


def note(midi_note, beats, **controls):
    beats = float(beats)
    if beats < 0:
        raise ValueError("beats must be >= 0")
    return Event(int(midi_note), beats, dict(controls))


def rest(beats):
    beats = float(beats)
    if beats < 0:
        raise ValueError("beats must be >= 0")
    return Event(None, beats, {})


LEAD_CONTROLS = {
    "amp": 0.14,
    "index": 4.5,
    "carrier_ratio": 1.0,
    "mod_ratio": 3.0,
    "attack": 0.01,
    "decay": 0.18,
    "sustain": 0.35,
    "release": 0.20,
    "done_action": 1,
}

BASS_CONTROLS = {
    "amp": 0.12,
    "index": 2.5,
    "carrier_ratio": 0.5,
    "mod_ratio": 2.0,
    "attack": 0.01,
    "decay": 0.12,
    "sustain": 0.45,
    "release": 0.28,
    "done_action": 1,
}

LEAD = [
    note(72, 1),
    note(74, 1),
    note(76, 1),
    note(79, 1),
    note(76, 1),
    note(74, 1),
    note(72, 1),
    note(67, 1),
    note(69, 1),
    note(71, 1),
    note(72, 1),
    note(74, 1),
    note(76, 1),
    note(74, 1),
    note(72, 2),
]

BASS = [
    note(48, 2),
    note(43, 2),
    note(45, 2),
    note(40, 2),
    note(41, 2),
    note(43, 2),
    note(47, 2),
    note(48, 2),
]


def voice(events, *, bpm, dsp_instance, synth_name="default", voice_controls):
    if bpm <= 0:
        raise ValueError("bpm must be > 0")

    seconds_per_beat = 60.0 / bpm
    active_module_id = None

    try:
        for event in events:
            if event.midi_note is None:
                yield event.beats * seconds_per_beat
                continue

            controls = dict(voice_controls)
            controls.update(event.controls)
            controls["freq"] = midi2cps(event.midi_note)
            controls["gate"] = 1

            active_module_id = dsp_instance.append(synth_name, **controls)
            yield event.beats * seconds_per_beat
            dsp_instance.set(active_module_id, gate=0)
            active_module_id = None
    finally:
        if active_module_id is not None:
            try:
                dsp_instance.set(active_module_id, gate=0)
            except Exception:
                pass


def _next_delay(generator):
    delay = float(next(generator))
    if delay < 0:
        raise ValueError("voice generators must yield non-negative delays")
    return delay


def play(generators, sleep=time.sleep):
    generators = list(generators)
    active = []

    try:
        for index, generator in enumerate(generators):
            try:
                active.append([index, generator, _next_delay(generator)])
            except StopIteration:
                pass

        while active:
            wait = min(state[2] for state in active)
            if wait > 0:
                sleep(wait)

            ready = []
            for state in active:
                state[2] -= wait
                if state[2] <= EPSILON:
                    ready.append(state)

            active = [state for state in active if state[2] > EPSILON]

            for index, generator, _ in sorted(ready, key=lambda state: state[0]):
                try:
                    active.append([index, generator, _next_delay(generator)])
                except StopIteration:
                    pass
    finally:
        for generator in generators:
            generator.close()


def release_tail(*parts):
    releases = []
    for events, voice_controls in parts:
        default_release = float(voice_controls.get("release", 0.0))
        releases.append(default_release)
        for event in events:
            if event.midi_note is None:
                continue
            releases.append(float(event.controls.get("release", default_release)))
    return max(releases, default=0.0)


def main(dsp_instance=None, sleep=time.sleep):
    generators = [
        voice(LEAD, bpm=BPM, dsp_instance=dsp, voice_controls=LEAD_CONTROLS),
        voice(BASS, bpm=BPM, dsp_instance=dsp, voice_controls=BASS_CONTROLS),
    ]

    dsp.start()
    try:
        play(generators, sleep=sleep)
        sleep(release_tail((LEAD, LEAD_CONTROLS), (BASS, BASS_CONTROLS)))
    finally:
        dsp.stop()


if __name__ == "__main__":
    main()
