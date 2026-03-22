from contextlib import contextmanager

from .pitch import midi2cps

__all__ = ("Event", "default_event")

_sentinel = object()


class Event:
    __slots__ = ("_data", "_stack")

    def __init__(self, mapping=None, **kvs):
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

    def with_(self, **kvs):
        return self.__class__(self._data, **kvs)

    # ----- context manager via __call__ -----

    @contextmanager
    def __call__(self, **overrides):
        """
        Usage:
            with default_event(tempo=90) as faster:
                ...
        Produces a new Event (does not mutate the original event).
        """
        yield self.with_(**overrides)

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
    instrument="default",
    tempo=120.0,
    beats=1.0,
    beat_duration=lambda e: 60.0 / e.tempo,
    duration=lambda e: e.beats * e.beat_duration,
    legato=0.8,
    sustain=lambda e: e.duration * e.legato,
    midi_note=60,
    freq=lambda e: None if e.midi_note is None else midi2cps(e.midi_note),
    is_rest=lambda e: e.midi_note is None,
)
