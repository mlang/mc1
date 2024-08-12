import abc

class BaseUGen(metaclass=abc.ABCMeta):
    def __add__(self, other):
        return Add(self, asUGen(other))

    def __radd__(self, other):
        return Add(asUGen(other), self)

    def __mul__(self, other):
        return Mul(self, asUGen(other))

    def __rmul__(self, other):
        return Mul(asUGen(other), self)


class Constant(BaseUGen):
    def __init__(self, value):
        self.value = value


class Nary(BaseUGen):
    def __init__(self, *inputs):
        self.inputs = [asUGen(input) for input in inputs]


def asUGen(value):
    if isinstance(value, BaseUGen):
        return value
    return Constant(value)

class Add(Nary):
    def __init__(self, a, b):
        super().__init__(a, b)

class Mul(Nary):
    def __init__(self, a, b):
        super().__init__(a, b)

class SinOsc(Nary):
    def __init__(self, freq, phase):
        super().__init__(freq, phase)


# The magic!
def dag(expr):
    constants = []
    nodes = []

    def find_hash(h):
        for index, node in enumerate(nodes):
            if node['hash'] == h:
                return index

        return None

    def recurse(node):
        if isinstance(node, Constant):
            if node.value not in constants:
                constants.append(node.value)
            return {"constant": constants.index(node.value) }
        else:
            ref = find_hash(hash(node))
            if ref is None:
                nodes.append({
                    "name": node.__class__.__name__,
                    "inputs": [recurse(input) for input in node.inputs],
                    "hash": hash(node)
                })
            return { "node": find_hash(hash(node)) }

    recurse(expr)

    for node in nodes:
        del node["hash"]

    return { "constants": constants, "nodes": nodes }
