import code
import struct

from mc1.dag import *
from mc1.engine import Engine


def msg(id: int, content):
    return struct.pack('H', id) + bytes(content)


Engine.build()
dsp = Engine()
dsp.run()
code.interact(local=locals(), banner="", exitmsg="")
