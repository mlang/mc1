"""A DSL to describe and serialize signal graphs."""

import abc
import enum
import inspect
import io
import struct


__all__ = ('SinOsc', 'DAG')


class _Node(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def address(self) -> int: pass

    def __add__(self, other):  return Add(self, other)
    def __radd__(self, other): return Add(other, self)
    def __mul__(self, other):  return Mul(self, other)
    def __rmul__(self, other): return Mul(other, self)
    def __sub__(self, other):  return Sub(self, other)
    def __rsub__(self, other): return Sub(other, self)


class OpBase(_Node):
    __slots__ = ('rate', 'args', '_index')

    def __init__(self, rate, *args):
        self.rate = rate
        self.args = tuple(map(_convert, args))
        self._index = _append(DAG._operations, self)

    def address(self): return self._index

    def __bytes__(self):
        buf = io.BytesIO()
        def pack(fmt, *args):
            return buf.write(struct.pack(fmt, *args))
        name = bytes(self.__class__.__name__, 'utf-8')
        pack(f'{len(name)+1}p', name)

        pack('c', self.rate.value)

        pack('H', len(self.args))
        for arg in self.args: pack('H', arg.address())

        return buf.getvalue()


class Op(OpBase):
    __slots__ = ('inputRates')

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

    def __init__(self, rate, *args):
        super().__init__(rate, *args)
        self.inputRates = b''.join(arg.rate.value for arg in self.args)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.rate} {list(arg.address() for arg in self.args)}>"


class Rate(enum.Enum):
    AUDIO = b'a'
    BLOCK = b'b'
    CONST = b'c'


class Const(_Node):
    __slots__ = ('_index')
    rate: Rate = Rate.CONST

    def __init__(self, value):
        value = float(value)
        try:
            self._index = DAG._constants.index(value)
        except ValueError:
            self._index = _append(DAG._constants, value)

    def __float__(self) -> float: return DAG._constants[self._index]
    def address(self) -> int: return 0x8000 | self._index


class Param(OpBase):
    __slots__ = ('name', '_ctrlindex')

    def __init__(self, name, *values):
        super().__init__(Rate.BLOCK)
        self.name = name
        self._ctrlindex = _extend(DAG._controls, values)
        DAG._controlNames.append((name, self._ctrlindex))

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.name}'>"


class _BinOp(Op):
    def __init__(self, left, right):
        super().__init__(Rate.AUDIO, left, right)


class Add(_BinOp): pass
class Div(_BinOp): pass
class Mul(_BinOp): pass
class Sub(_BinOp): pass


class SinOsc(Op):
    @classmethod
    def ar(cls, freq, phase=0):
        return cls(Rate.AUDIO, freq, phase)


def _convert(x):
    if isinstance(x, _Node):
        return x
    return Const(x)


class DAG:
    _constants: list[float] = []
    _controls: list[float] = []
    _controlNames: list[tuple[str, int]] = []
    _operations: list[OpBase] = []
    __slots__ = ('constants', 'controls', 'controlNames', 'operations')

    def __init__(self, func):
        self._reset()

        def param(name, value):
            if not isinstance(value, (list, range, tuple)):
                value = (value,)
            return Param(name, *value)
        func = _WrapDefaults(func, param)
        func()
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

        pack('H', len(self.constants))
        pack('f'*len(self.constants), *self.constants)

        pack('H', len(self.controls))
        pack('f'*len(self.controls), *self.controls)

        pack('H', len(self.operations))
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


def _append(lst: list, item) -> int:
    index = len(lst)
    lst.append(item)
    return index


def _extend(lst: list, items: list) -> int:
    index = len(lst)
    lst.extend(items)
    return index


@DAG
def foo(freq=440, amp=0.1):
    sig = SinOsc.ar([freq, freq+1], 0)
    (SinOsc.ar(sig + freq, 0)[0] + SinOsc.ar(sig + freq, 0)[1]) * amp
