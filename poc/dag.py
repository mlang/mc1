"""A DSL to describe and serialize signal graphs."""

import enum
import inspect
import io
import struct


__all__ = ('SinOsc', 'DAG')


class Rate(enum.Enum):
    AUDIO = b'a'
    BLOCK = b'b'

class _Node:
    __slots__ = ('rate', 'num_out', 'args', '_index')

    def __init__(self, rate, num_out, *args):
        self.rate = rate
        self.num_out = num_out
        self.args = args
        self._index = len(DAG._operations)
        DAG._operations.append(self)

    def __add__(self, other):  return Add(self, other)
    def __radd__(self, other): return Add(other, self)
    def __mul__(self, other):  return Mul(self, other)
    def __rmul__(self, other): return Mul(other, self)
    def __sub__(self, other):  return Sub(self, other)
    def __rsub__(self, other): return Sub(other, self)

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
    def __init__(self, value):
        value = float(value)
        try:
            cindex = DAG._constants.index(value)
        except ValueError:
            cindex = len(DAG._constants)
            DAG._constants.append(value)
        super().__init__(Rate.BLOCK, 1, cindex)

    def __float__(self) -> float: return DAG._constants[self.args[0]]

    def __repr__(self):
        return f"<{self.__class__.__name__} {float(self)}>"


class Control(_Node):
    __slots__ = ('name')

    def __init__(self, name, *values):
        self.name = name
        cindex = len(DAG._controls)
        DAG._controls.extend(values)
        DAG._controlNames.append((name, cindex))
        super().__init__(Rate.BLOCK, len(values), cindex)

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.name}'>"


class _GraphArgs(_Node):
    def __new__(cls, *args, **kwargs):
        sequence_types = (list, range, tuple)
        lengths = [len(arg) for arg in args if isinstance(arg, sequence_types)]
        lengths.extend(len(v)
            for v in kwargs.values() if isinstance(v, sequence_types)
        )
        if not lengths:
            return super().__new__(cls)

        def item(arg, index: int):
            return arg[index % len(arg)] if isinstance(arg, sequence_types) else arg

        return tuple(
            cls(
                *(item(arg, index) for arg in args),
                **{k: item(v, index) for k, v in kwargs.items()}
            ) for index in range(max(lengths))
        )

    def __init__(self, rate, num_out, *args):
        super().__init__(rate, num_out, *map(_convert, args))


class _BinOp(_GraphArgs):
    def __init__(self, left, right):
        super().__init__(Rate.AUDIO, 1, left, right)


class Add(_BinOp): pass
class Div(_BinOp): pass
class Mul(_BinOp): pass
class Sub(_BinOp): pass


class SinOsc(_GraphArgs):
    @classmethod
    def ar(cls, freq, phase=0):
        return cls(Rate.AUDIO, 1, freq, phase)


def _convert(x):
    if x is None: return None
    return x if isinstance(x, _Node) else Const(x)


class DAG:
    _constants: list[float] = []
    _controls: list[float] = []
    _controlNames: list[tuple[str, int]] = []
    _operations: list[_Node] = []
    __slots__ = ('constants', 'controls', 'controlNames', 'operations')

    def __init__(self, func):
        self._reset()

        def param(name, value):
            if not isinstance(value, (list, range, tuple)):
                value = (value,)
            return Control(name, *value)
        func = _WrapDefaults(func, param)
        _convert(func())
        for slot in self.__slots__:
            setattr(self, slot, getattr(self, f'_{slot}'))

    @classmethod
    def _reset(cls):
        for slot in cls.__slots__:
            setattr(cls, f'_{slot}', [])

    def __bytes__(self):
        buf = io.BytesIO()
        def pack(fmt, *args):
            return buf.write(struct.pack(fmt, *args))

        pack('N', len(self.constants))
        pack('f'*len(self.constants), *self.constants)

        pack('N', len(self.controls))
        pack('f'*len(self.controls), *self.controls)

        pack('N', len(self.operations))
        for op in self.operations: buf.write(bytes(op))

        return buf.getvalue()


class _WrapDefaults:
    __slots__ = ('func', 'args', 'kwargs')

    def __init__(self, func, wrap=None):
        if not wrap:
            wrap = lambda name, value: value

        sig = inspect.signature(func)
        args = []
        kwargs = {}
    
        for param in sig.parameters.values():
            if param.default is param.empty:
                raise ValueError(f"Parameter '{name}' has no default value")
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.POSITIONAL_ONLY):
                args.append(wrap(param.name, param.default))
            elif param.kind == param.KEYWORD_ONLY:
                kwargs[param.name] = wrap(param.name, param.default)
    
        self.func = func
        self.args = tuple(args)
        self.kwargs = kwargs

    def __call__(self):
        return self.func(*self.args, **self.kwargs)


@DAG
def foo(freq=440, amp=0.1):
    sig = SinOsc.ar([freq, freq+1], 0)
    (SinOsc.ar(sig + freq, 0)[0] + SinOsc.ar(sig + freq, 0)[1]) * amp
