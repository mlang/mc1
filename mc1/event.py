from functools import partial

from .graphs import default as default_instrument
from .pitch import midi2cps

__all__ = ("Event", "default_event")

_sentinel = object()


def _default_play(event, dsp):
    if event.is_rest:
        yield event.duration
        return

    synth_id = dsp.append(event.instrument.name, **event.controls)
    yield event.sustain
    if "gate" in map(lambda x: x[0], event.instrument.controlNames):
        dsp.set(synth_id, gate=0)
    yield event.duration - event.sustain


class Event:
    __slots__ = ("_data", "_stack")

    def __init__(self, mapping=None, **kvs):
        if isinstance(mapping, Event):
            self._data = dict(mapping._data)
        else:
            self._data = dict(mapping or {})
        self._data.update(kvs)
        self._stack = []

    # ----- core evaluation -----

    def raw(self, key, default=_sentinel):
        if key in self._data:
            return self._data[key]
        if default is _sentinel:
            raise KeyError(key)
        return default

    def eval(self, key, default=_sentinel):
        if key not in self._data:
            if default is _sentinel:
                raise KeyError(key)
            return default

        if key in self._stack:
            raise RuntimeError(f"Cyclic dependency while evaluating {key}: {self._stack + [key]}")
        self._stack.append(key)
        try:
            v = self._data[key]
            return v(self) if callable(v) else v
        finally:
            self._stack.pop()

    def get(self, key, default=None):
        return self.eval(key, default=default)

    # ----- mutation / functional updates -----

    def set(self, **kvs):
        self._data.update(kvs)
        return self

    # ----- copying / context manager -----

    def __call__(self, **overrides):
        """
        Usage:
            event = default_event(bpm=90)

            with default_event(bpm=90) as faster:
                ...

        Produces a new Event (does not mutate the original event).
        """
        return self.__class__(self._data, **overrides)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    # ----- dict / attribute sugar -----

    def __contains__(self, key): return key in self._data
    def __getitem__(self, key): return self.eval(key)
    def __setitem__(self, key, value): self._data[key] = value
    def __delitem__(self, key): del self._data[key]

    def __getattr__(self, name):
        try:
            return self.eval(name)
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        if name in Event.__slots__:
            return super().__setattr__(name, value)
        self._data[name] = value

    def __repr__(self):
        return f"Event({self._data!r})"


default_event = Event(
    instrument=default_instrument,
    bpm=120.0,
    tempo=lambda e: 60.0 / e.bpm,
    beats=1.0,
    duration=lambda e: e.beats * e.tempo,
    legato=0.8,
    sustain=lambda e: e.duration * e.legato,
    midi_note=60,
    freq=lambda e: None if e.midi_note is None else midi2cps(e.midi_note),
    gate=1,
    controls=lambda e: {name: e[name] for name, _, _ in e.instrument.controlNames
                        if e.raw(name, None) is not None},
    is_rest=lambda e: e.midi_note is None,
    play=lambda e: partial(_default_play, e),
)
