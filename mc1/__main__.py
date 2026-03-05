import code
import runpy
import sys
import time

from mc1.dag import *
from mc1.engine import Engine
from mc1.message import Compile, Quit


def main():
    script = sys.argv[1] if len(sys.argv) > 1 else None
    script_args = sys.argv[2:] if len(sys.argv) > 2 else []

    Engine.build()
    dsp = Engine()
    dsp.run()
    time.sleep(0.1)

    ns = {k: v for k, v in globals().items() if not k.startswith("__")}
    ns["dsp"] = dsp

    if script is not None:
        sys.argv = [script, *script_args]
        runpy.run_path(script, init_globals=ns, run_name="__main__")
        return

    code.interact(
        local=ns,
        banner="""MiniCollider

Example:
    dsp.send(Compile(lambda freq=440: SinOsc.ar(freq) * 0.1))""",
        exitmsg="",
    )


if __name__ == "__main__":
    main()
