import code
import time

from mc1.dag import *
from mc1.engine import Engine
from mc1.message import Quit, Compile


Engine.build()
dsp = Engine()
dsp.run()
time.sleep(0.1)
code.interact(
    local=locals(),
    banner="""MiniCollider

Example:
    dsp.send(Compile(lambda freq=440: SinOsc.ar(freq) * 0.1))""",
    exitmsg=""
)
