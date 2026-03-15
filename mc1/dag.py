"""Core Python graph objects used to build and serialize MiniCollider DAGs.

Most users work with the public classes exported from `mc1`: decorate a
function with `@DAG`, create opcode nodes such as `SinOsc.ar(...)`, combine
them with ordinary Python operators, and finally serialize the captured graph
with `bytes(...)`.
"""

import enum
import inspect
import io
import struct


__all__ = ('ADSR', 'In', 'Out', 'Pan', 'SinOsc', 'Trigger', 'DAG')


class Trigger:
    """Annotation marker for trigger-style controls in a `@DAG` function.

    A parameter annotated as `Trigger` is compiled as a scalar trigger control
    instead of a regular value control.

    Example:

        >>> @DAG
        ... def ping(trig: Trigger = 0):
        ...     Out.ar(0, SinOsc.ar(220) * trig)
    """

    pass


class ControlKind(enum.Enum):
    VALUE = 0
    TRIGGER = 1


class Rate(enum.Enum):
    AUDIO = b'a'
    BLOCK = b'b'

def fastest_rate(*args):
    for a in args:
        r = a if isinstance(a, Rate) else a.rate
        if r is Rate.AUDIO:
            return Rate.AUDIO
    return Rate.BLOCK


def _normalize_bounds(lo, hi):
    lo = float(lo)
    hi = float(hi)
    if lo > hi:
        raise ValueError("bounds require lo <= hi")
    return (lo, hi)


def _add_bounds(left, right):
    if left is None or right is None:
        return None
    return (left[0] + right[0], left[1] + right[1])


def _sub_bounds(left, right):
    if left is None or right is None:
        return None
    return (left[0] - right[1], left[1] - right[0])


def _mul_bounds(left, right):
    if left is None or right is None:
        return None
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    return (min(products), max(products))


def _div_bounds(left, right):
    if left is None or right is None:
        return None
    if right[0] <= 0 <= right[1]:
        return None
    quotients = (
        left[0] / right[0],
        left[0] / right[1],
        left[1] / right[0],
        left[1] / right[1],
    )
    return (min(quotients), max(quotients))

class _Node:
    """Base class for Python graph nodes and opcode instances."""
    __slots__ = ('rate', 'num_out', 'args', '_index', '_bounds')

    def __init__(self, rate, num_out, *args, bounds=None):
        self.rate = rate
        self.num_out = num_out
        self.args = args
        self._bounds = bounds
        self._index = len(DAG._operations)
        DAG._operations.append(self)

    def __add__(self, other):  return Add(self, other)
    def __radd__(self, other): return Add(other, self)
    def __mul__(self, other):  return Mul(self, other)
    def __rmul__(self, other): return Mul(other, self)
    def __sub__(self, other):  return Sub(self, other)
    def __rsub__(self, other): return Sub(other, self)
    def __lt__(self, other):  return LT(self, other)
    def __le__(self, other):  return LE(self, other)
    def __gt__(self, other):  return GT(self, other)
    def __ge__(self, other):  return GE(self, other)

    @property
    def bounds(self):
        """Known `(lo, hi)` interval for this node, or `None` if unknown."""
        return self._bounds

    @property
    def is_unipolar(self):
        """Whether this node is known to stay within the range `(0.0, 1.0)`."""
        return self.bounds == (0.0, 1.0)

    @property
    def is_bipolar(self):
        """Whether this node is known to stay within the range `(-1.0, 1.0)`."""
        return self.bounds == (-1.0, 1.0)

    def with_bounds(self, lo, hi):
        """Attach explicit bounds to a node and return the same node.

        This is mainly useful for nodes such as `In.ar(...)` whose range is not
        known automatically but should participate in later `range(...)` or
        `linlin(...)` mappings.
        """
        self._bounds = _normalize_bounds(lo, hi)
        return self

    def linlin(self, in_lo, in_hi, out_lo, out_hi):
        """Linearly remap a value from one range into another.

        The result is expressed as ordinary graph math, so it can be combined
        with other nodes just like any manually-written expression.
        """
        in_lo = float(in_lo)
        in_hi = float(in_hi)
        out_lo = float(out_lo)
        out_hi = float(out_hi)
        if in_lo == in_hi:
            raise ValueError("linlin requires a non-zero source range")

        scale = (out_hi - out_lo) / (in_hi - in_lo)
        mapped = (self - in_lo) * scale + out_lo
        mapped._bounds = (min(out_lo, out_hi), max(out_lo, out_hi))
        return mapped

    def range(self, out_lo, out_hi):
        """Remap a node from its known bounds into a new output range."""
        if self.bounds is None:
            raise ValueError("range requires known source bounds")
        return self.linlin(*self.bounds, out_lo, out_hi)

    def __bytes__(self):
        buf = io.BytesIO()
        pack = lambda fmt, *args: buf.write(struct.pack(fmt, *args))

        name = bytes(self.__class__.__name__, 'utf-8')
        pack(f'{len(name)+1}p', name)

        pack('c', self.rate.value)

        pack('N', self.num_out)

        pack('N', len(self.args))
        for arg in self.args:
            pack('N', arg if isinstance(arg, int) else arg._index)

        return buf.getvalue()

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.args!r}>"


class Const(_Node):
    """Constant scalar embedded in a graph.

    Python numbers are converted to `Const` nodes automatically, so this class
    is mostly useful when inspecting or debugging the graph builder.
    """

    def __init__(self, value):
        value = float(value)
        try:
            cindex = DAG._constants.index(value)
        except ValueError:
            cindex = len(DAG._constants)
            DAG._constants.append(value)
        super().__init__(Rate.BLOCK, 1, cindex, bounds=(value, value))

    def __float__(self) -> float: return DAG._constants[self.args[0]]

    def __repr__(self):
        return f"<{self.__class__.__name__} {float(self)}>"


class Control(_Node):
    """Named control input created from a `@DAG` function parameter.

    Controls are synthesized automatically from the decorated function's
    default-valued parameters; users normally do not instantiate this class
    directly.
    """

    __slots__ = ('name', 'kind')

    def __init__(self, name, *values, kind=ControlKind.VALUE):
        self.name = name
        self.kind = kind
        cindex = len(DAG._controls)
        DAG._controls.extend(values)
        DAG._controlNames.append((name, cindex, kind))
        super().__init__(Rate.BLOCK, len(values), cindex)

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.name}' {self.kind.name.lower()}>"


class _GraphArgs(_Node):
    """Node base class that normalizes Python values into graph arguments.

    It also implements the Python-side sequence expansion used by opcodes such
    as `SinOsc`, `Pan`, and `Out`.
    """

    def __new__(cls, *args, **kwargs):
        """Build one node, or a tuple of nodes for list/tuple/range inputs.

        Scalars are broadcast, and shorter sequences wrap to the longest input.
        """
        sequence_types = (list, range, tuple)
        nonexpanding_kwargs = {'bounds'}
        lengths = [len(arg) for arg in args if isinstance(arg, sequence_types)]
        lengths.extend(len(v)
            for k, v in kwargs.items()
            if k not in nonexpanding_kwargs and isinstance(v, sequence_types)
        )
        if not lengths:
            return super().__new__(cls)

        def item(arg, index: int, expand=True):
            if not expand:
                return arg
            return arg[index % len(arg)] if isinstance(arg, sequence_types) else arg

        return tuple(
            cls(
                *(item(arg, index) for arg in args),
                **{k: item(v, index, k not in nonexpanding_kwargs) for k, v in kwargs.items()}
            ) for index in range(max(lengths))
        )

    def __init__(self, rate, num_out, *args, bounds=None):
        super().__init__(rate, num_out, *map(_convert, args), bounds=bounds)


class _BinOp(_GraphArgs):
    """Base class for binary arithmetic opcode nodes."""

    _bounds_op = None

    def __init__(self, left, right):
        left = _convert(left)
        right = _convert(right)
        bounds = self._bounds_op(left.bounds, right.bounds)
        super().__init__(fastest_rate(left, right), 1, left, right, bounds=bounds)

class Add(_BinOp):
    """Addition node, usually created by `left + right`."""

    _bounds_op = staticmethod(_add_bounds)


class Div(_BinOp):
    """Division node, usually created by `left / right`."""

    _bounds_op = staticmethod(_div_bounds)


class Mul(_BinOp):
    """Multiplication node, usually created by `left * right`."""

    _bounds_op = staticmethod(_mul_bounds)


class Sub(_BinOp):
    """Subtraction node, usually created by `left - right`."""

    _bounds_op = staticmethod(_sub_bounds)


class _CmpOp(_GraphArgs):
    """Base class for comparison opcode nodes."""

    def __init__(self, left, right):
        left = _convert(left)
        right = _convert(right)
        super().__init__(fastest_rate(left, right), 1, left, right, bounds=(-1.0, 1.0))


class LT(_CmpOp):
    """Less-than comparison node, usually created by `left < right`."""


class LE(_CmpOp):
    """Less-than-or-equal comparison node, usually created by `left <= right`."""


class GT(_CmpOp):
    """Greater-than comparison node, usually created by `left > right`."""


class GE(_CmpOp):
    """Greater-than-or-equal node, usually created by `left >= right`."""


class EQ(_CmpOp):
    """Equality comparison node.

    Python `==` is intentionally left as normal object equality, so graph
    equality tests are spelled explicitly as `EQ(left, right)`.
    """


class NE(_CmpOp):
    """Inequality comparison node.

    Python `!=` is intentionally left as normal object inequality, so graph
    inequality tests are spelled explicitly as `NE(left, right)`.
    """


class SinOsc(_GraphArgs):
    """Audio-rate sine oscillator.

    Create instances with `SinOsc.ar(freq, phase=0)`. As with other graph
    constructors, passing a `list`, `tuple`, or `range` performs Python-side
    multi-channel expansion and returns a tuple of oscillators.

    Example:

        >>> osc = SinOsc.ar(440)
        >>> voices = SinOsc.ar((220, 330, 440), phase=(0.0, 0.5))
    """

    @classmethod
    def ar(cls, freq, phase=0):
        """Build an audio-rate sine oscillator node."""
        return cls(Rate.AUDIO, 1, freq, phase, bounds=(-1.0, 1.0))


class ADSR(_GraphArgs):
    """Attack/decay/sustain/release envelope generator.

    `gate` starts the envelope and later releases it when driven back to zero.
    The result is known to be unipolar, so its bounds are seeded to
    `(0.0, 1.0)`.

    Example:

        >>> env = ADSR.ar(1, 0.01, 0.2, 0.5, 0.4)
    """

    @classmethod
    def ar(cls, gate, attack, decay, sustain, release, done_action=0):
        """Build an audio-rate ADSR envelope node."""
        return cls(Rate.AUDIO, 1, gate, attack, decay, sustain, release, done_action, bounds=(0.0, 1.0))


class In(_GraphArgs):
    """Audio-rate input bus reader.

    Inputs do not have inferred bounds by default. Use `with_bounds(...)` when
    you want later interval-based helpers such as `range(...)` to reason about
    the expected signal range.

    Example:

        >>> cutoff_cv = In.ar(0).with_bounds(0.0, 1.0)
        >>> cutoff = cutoff_cv.range(200.0, 4000.0)
    """

    @classmethod
    def ar(cls, index=0):
        """Build an audio-rate input reader for the given bus index."""
        return cls(Rate.AUDIO, 1, index)


class Out(_GraphArgs):
    """Audio-rate output bus writer.

    `Out.ar(index, signal)` routes `signal` to the target output bus. If `index`
    is a sequence, standard Python-side graph expansion returns a tuple of
    writers.

    Example:

        >>> Out.ar(0, Pan(SinOsc.ar(440) * 0.1))
    """

    @classmethod
    def ar(cls, index, signal):
        """Build an audio-rate output writer node."""
        signal = _convert(signal)
        return cls(Rate.AUDIO, signal.num_out, index, signal, bounds=signal.bounds)


class Pan(_GraphArgs):
    """Stereo panner.

    `Pan(signal, pan=0)` converts a mono signal into a 2-channel signal.
    Sequence inputs participate in the same Python-side expansion rules as other
    graph constructors.

    Example:

        >>> stereo = Pan(SinOsc.ar(220), pan=(-0.5, 0.5))
    """

    def __init__(self, signal, pan=0):
        signal = _convert(signal)
        super().__init__(signal.rate, 2, signal, pan, bounds=signal.bounds)


def _convert(x):
    return x if isinstance(x, _Node) else Const(x)


class DAG:
    """Capture a graph-building function as a serializable synth definition.

    `@DAG` executes the decorated function once with synthetic `Control` nodes
    standing in for its default-valued parameters. Every graph node allocated
    during that call is recorded and can later be serialized with `bytes(...)`.

    Every parameter must have a default value. Annotating a parameter as
    `Trigger` produces a trigger control instead of a regular value control.

    Example:

        >>> @DAG
        ... def beep(freq=440, amp=0.1):
        ...     Out.ar(0, Pan(SinOsc.ar(freq) * amp))
        ...
        >>> payload = bytes(beep)
    """

    _constants: list[float] = []
    _controls: list[float] = []
    _controlNames: list[tuple[str, int, ControlKind]] = []
    _operations: list[_Node] = []
    _graph_slots = ('constants', 'controls', 'controlNames', 'operations')
    __slots__ = ('name', *_graph_slots)

    def __init__(self, func):
        self._reset()
        self.name = func.__name__

        def param(name, value, annotation):
            kind = ControlKind.TRIGGER if annotation is Trigger else ControlKind.VALUE
            if kind is ControlKind.TRIGGER:
                if isinstance(value, (list, range, tuple)):
                    raise ValueError(f"Trigger control '{name}' must be a scalar default value")
                value = (float(value),)
            elif not isinstance(value, (list, range, tuple)):
                value = (value,)
            return Control(name, *value, kind=kind)
        func = _WrapDefaults(func, param)
        func()
        for slot in self._graph_slots:
            setattr(self, slot, getattr(self, f'_{slot}'))

    @classmethod
    def _reset(cls):
        for slot in cls._graph_slots:
            setattr(cls, f'_{slot}', [])

    def __bytes__(self):
        buf = io.BytesIO()
        pack = lambda fmt, *args: buf.write(struct.pack(fmt, *args))

        name = bytes(self.name, 'utf-8')
        pack(f'{len(name)+1}p', name)

        pack('N'+'f'*len(self.constants), len(self.constants), *self.constants)
        pack('N'+'f'*len(self.controls), len(self.controls), *self.controls)
        pack('N', len(self.controlNames))
        for cname, cindex, ckind in self.controlNames:
            name = bytes(cname, 'utf-8')
            pack(f'{len(name)+1}p', name)
            pack('N', cindex)
            pack('B', ckind.value)

        pack('N', len(self.operations))
        for op in self.operations: buf.write(bytes(op))

        return buf.getvalue()


class _WrapDefaults:
    __slots__ = ('func', 'args', 'kwargs')

    def __init__(self, func, wrap=None):
        if not wrap:
            wrap = lambda name, value, annotation: value

        sig = inspect.signature(func)
        annotations = inspect.get_annotations(func, eval_str=True)
        args = []
        kwargs = {}
    
        for param in sig.parameters.values():
            if param.default is param.empty:
                raise ValueError(f"Parameter '{param.name}' has no default value")
            annotation = annotations.get(param.name, param.empty)
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.POSITIONAL_ONLY):
                args.append(wrap(param.name, param.default, annotation))
            elif param.kind == param.KEYWORD_ONLY:
                kwargs[param.name] = wrap(param.name, param.default, annotation)
    
        self.func = func
        self.args = tuple(args)
        self.kwargs = kwargs

    def __call__(self):
        return self.func(*self.args, **self.kwargs)


@DAG
def foo(freq=440, amp=0.1):
    sig = SinOsc.ar([freq, freq+1], 0)
    (SinOsc.ar(sig + freq, 0)[0] + SinOsc.ar(sig + freq, 0)[1]) * amp
