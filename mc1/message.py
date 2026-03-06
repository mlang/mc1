from mc1.dag import DAG
from mc1 import osc

DONE_ADDRESS = "/mc1/done"


class Message:
    __slots__ = ("_args",)

    def __init_subclass__(cls, /, address, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.address = address

    def __init__(self, *args):
        if self.__class__ is Message:
            raise TypeError("message class cannot be instantiated directly.")
        self._args = args

    def __bytes__(self):
        return osc.encode_message(self.address, *self._args)


class Quit(Message, address="/mc1/quit"): pass

class Compile(Message, address="/mc1/compile"):
    def __init__(self, payload):
        data = bytes(DAG(payload) if callable(payload) else payload)
        super().__init__(data)


class Sync(Message, address="/mc1/sync"): pass
