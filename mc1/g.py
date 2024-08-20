import abc

from mc1.abc import ArgExpander


_ops = []
_consts = []


class Op(ArgExpander, metaclass=abc.ABCMeta):
    def __init__(self, rate, *args):
        self.rate = rate
        self.args = [_convert(arg).ref() for arg in args]
        self.index = len(_ops)
        _ops.append(self)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.rate} {self.args!r}>"

    def ref(self):
        return {'node': self.index}

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)


class Const(Op):
    def __init__(self, value):
        value = float(value)
        self.rate = 'scalar'
        try:
            self.index = _consts.index(value)
        except ValueError:
            self.index = len(_consts)
            _consts.append(value)

    def __float__(self):
        return _consts[self.index]

    def ref(self):
        return {'const': self.index}


class BinOp(Op, metaclass=abc.ABCMeta):
    def __init__(self, left, right):
        super().__init__('audio', left, right)


class Add(BinOp): pass
class Mul(BinOp): pass


class SinOsc(Op):
    @classmethod
    def ar(cls, freq, phase):
        return cls('audio', freq, phase)


def _convert(x):
    if isinstance(x, Op):
        return x
    return Const(x)


def dag(func):
    global _ops, _consts

    _ops = []
    _consts = []

    func()

    return (_consts, _ops)


@dag
def foo():
    sig = SinOsc.ar([440, 441], 0)
    return SinOsc.ar(sig, 0)[0] * 0.1
