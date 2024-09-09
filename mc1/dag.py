"""A DSL to describe and serialize signal graphs."""

# {{{ Imports
import abc
import enum
import inspect
import io
import struct
# }}}

__all__ = ('SinOsc', 'DAG')


class _BroadcastInputs(metaclass=abc.ABCMeta):
    """Expand list arguments to tuples of instances."""

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


class _Node(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def ref(self): pass

    def __add__(self, other):  return Add(self, other)
    def __radd__(self, other): return Add(other, self)
    def __mul__(self, other):  return Mul(self, other)
    def __rmul__(self, other): return Mul(other, self)
    def __sub__(self, other):  return Sub(self, other)
    def __rsub__(self, other): return Sub(other, self)


class Op(_BroadcastInputs, _Node):
    __slots__ = ('rate', 'inputRates', 'args', '_index')

    def __init__(self, rate, *args):
        self.rate = rate
        args = tuple(map(_convert, args))
        self.inputRates = b''.join(arg.rate.value for arg in args)
        self.args = [arg.ref() for arg in args]
        self._index = _append(DAG._operations, self)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.rate} {self.args!r}>"

    def ref(self):
        return {'node': self._index}


class Rate(enum.Enum):
    AUDIO = b'a'
    BLOCK = b'b'
    CONST = b'c'


class Const(_Node):
    __slots__ = ('_index')
    rate = Rate.CONST

    def __init__(self, value):
        value = float(value)
        try:
            self._index = DAG._constants.index(value)
        except ValueError:
            self._index = _append(DAG._constants, value)

    def __float__(self):
        return _constants[self._index]

    def ref(self):
        return {'const': self._index}


class Param(_Node):
    __slots__ = ('name', '_index', '_ctrlindex')
    args = []
    rate = Rate.BLOCK

    def __init__(self, name, *values):
        self.name = name
        self._index = _append(DAG._operations, self)
        self._ctrlindex = DAG._controls.extend(values)
        DAG._controlNames.append((name, self._ctrlindex))

    def ref(self):
        return {'node': self._index}


class _BinOp(Op):
    def __init__(self, left, right):
        super().__init__(Rate.AUDIO, left, right)


class Add(_BinOp): pass
class Div(_BinOp): pass
class Mul(_BinOp): pass
class Sub(_BinOp): pass


class SinOsc(Op):
    @classmethod
    def ar(cls, freq, phase):
        return cls(Rate.AUDIO, freq, phase)


def _convert(x):
    if isinstance(x, _Node):
        return x
    return Const(x)


class DAG:
    _constants = []
    _controls = []
    _controlNames = []
    _operations = []
    __slots__ = ('constants', 'controls', 'controlNames', 'operations')

    def __init__(self, func):
        def param(name, value):
            if not isinstance(value, (list, range, tuple)):
                value = (value,)
            return Param(name, *value)
        self._reset()
        func = _WrapDefaults(func, param)
        func()
        for slot in self.__slots__:
            setattr(self, slot, getattr(self, f'_{slot}'))

    @classmethod
    def _reset(cls):
        cls._constants = []
        cls._controls = []
        cls._controlNames = []
        cls._operations = []

    def __bytes__(self):
        buf = io.BytesIO()
        def pack(fmt, *args):
            return buf.write(struct.pack(fmt, *args))

        pack('H', len(self.constants))
        pack('f'*len(self.constants), *self.constants)

        pack('H', len(self.controls))
        pack('f'*len(self.controls), *self.controls)

        pack('H', len(self.operations))
        for op in self.operations:
            name = bytes(op.__class__.__name__, 'utf-8')
            pack(f'{len(name)+1}p', name)

            pack('c', op.rate.value)

            pack('H', len(op.args))
            for arg in op.args:
                id = arg['node'] if 'node' in arg else arg['const'] + 0x8000
                pack('H', id)

        return buf.getvalue()


class _WrapDefaults:
    __slots__ = ('func', 'args', 'kwargs')

    def __init__(self, func, wrap=None):
        if not wrap:
            def wrap(name, value):
                return value

        sig = inspect.signature(func)
        args = []
        kwargs = {}
    
        for name, param in sig.parameters.items():
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


def _append(lst, item) -> int:
    index = len(lst)
    lst.append(item)
    return index


@DAG
def foo(freq=440, amp=0.1):
    sig = SinOsc.ar([freq, freq+1], 0)
    return (SinOsc.ar(sig + freq, 0)[0] + SinOsc.ar(sig + freq, 0)[1]) * amp
