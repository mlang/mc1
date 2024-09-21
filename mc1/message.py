import struct

from mc1.dag import DAG


class Message:
    __slots__ = ('_payload',)

    def __init_subclass__(cls, /, identifier, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.identifier = identifier

    def __init__(self, payload=b''):
        if self.__class__ is Message:
            raise TypeError("message class cannot be instantiated directly.")
        self._payload = payload

    def __bytes__(self):
        return struct.pack('H', self.identifier) + bytes(self._payload)


class Quit(Message, identifier=0): pass

class Compile(Message, identifier=1):
    def __init__(self, payload):
        super().__init__(DAG(payload) if callable(payload) else payload)
