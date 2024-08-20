import abc
import inspect


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

class NamedControl(BaseUGen):
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.inputs = []


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
    arguments = []
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

def synth(func):
    signature = inspect.signature(func)
    args = {}
    for param_name, param in signature.parameters.items():
        if param.default is inspect.Parameter.empty:
            raise ValueError(f"Argument '{param_name}' has no default value.")
        args[param_name] = NamedControl(param_name, param.default)
    return dag(func(**args))
