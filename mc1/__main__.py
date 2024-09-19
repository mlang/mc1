import code
import time

from mc1.dag import *
from mc1.engine import Engine
from mc1.message import quit, compile


Engine.build()
dsp = Engine()
dsp.run()
time.sleep(0.1)
code.interact(
    local=locals(),
    banner="""MiniCollider

Example:
    dsp.send(compile(lambda freq=440: SinOsc.ar(freq) * 0.1))""",
    exitmsg=""
)
