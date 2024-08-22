"""A DSL to describe and serialize signal graphs."""

import abc
import enum
import io
import struct


from mc1.abc import ArgExpander


_consts = []
_ops = []


class Op(ArgExpander, metaclass=abc.ABCMeta):
    def __init__(self, rate, *args):
        self.rate = rate
        self.args = [_convert(arg).ref() for arg in args]
        self._index = len(_ops)
        _ops.append(self)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.rate} {self.args!r}>"

    def ref(self):
        return {'node': self._index}

    def __add__(self, other):  return Add(self, other)
    def __radd__(self, other): return Add(other, self)
    def __mul__(self, other):  return Mul(self, other)
    def __rmul__(self, other): return Mul(other, self)
    def __sub__(self, other):  return Sub(self, other)
    def __rsub__(self, other): return Sub(other, self)


class Const(Op):
    def __init__(self, value):
        value = float(value)
        self.rate = 'scalar'
        try:
            self._index = _consts.index(value)
        except ValueError:
            self._index = len(_consts)
            _consts.append(value)

    def __float__(self):
        return _consts[self._index]

    def ref(self):
        return {'const': self._index}


class BinOp(Op, metaclass=abc.ABCMeta):
    def __init__(self, left, right):
        super().__init__(Rate.AUDIO, left, right)


class Add(BinOp): pass
class Div(BinOp): pass
class Mul(BinOp): pass
class Sub(BinOp): pass


class SinOsc(Op):
    @classmethod
    def ar(cls, freq, phase):
        return cls(Rate.AUDIO, freq, phase)


def _convert(x):
    if isinstance(x, Op):
        return x
    return Const(x)


class Rate(enum.Enum):
    AUDIO = b'a'
    BLOCK = b'b'
    CONST = b'c'


class DAG:
    def __init__(self, func):
        _reset()

        func()

        self.constants = _consts
        self.operations = _ops

    def __bytes__(self):
        buf = io.BytesIO()
        def pack(fmt, *args):
            return buf.write(struct.pack(fmt, *args))

        pack('H', len(self.constants))
        pack('f'*len(self.constants), *self.constants)

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


def _reset():
    global _ops, _consts

    _ops = []
    _consts = []


@DAG
def foo():
    sig = SinOsc.ar([440, 441], 0)
    return (SinOsc.ar(sig, 0)[0] + SinOsc.ar(sig, 0)[1]) * 0.1
