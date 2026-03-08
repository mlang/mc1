import code
import runpy
import sys

from mc1 import *


if __name__ == "__main__":
    script = sys.argv[1] if len(sys.argv) > 1 else None
    script_args = sys.argv[2:] if len(sys.argv) > 2 else []

    ns = {k: v for k, v in globals().items() if not k.startswith("__")}
    ns['dsp'] = DSP()

    if script is not None:
        sys.argv = [script, *script_args]
        runpy.run_path(script, init_globals=ns, run_name="__main__")
        sys.exit()

    try:
        import readline
        import rlcompleter

        # Match CPython's libedit/readline TAB binding behavior.
        if "libedit" in (readline.__doc__ or ""):
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")
        readline.set_completer(rlcompleter.Completer(ns).complete)
    except ImportError:
        pass

    code.interact(
        local=ns,
        banner="""MiniCollider

Example:
    perft(bytes(lambda freq=440: SinOsc.ar(freq) * 0.1))""",
        exitmsg="",
    )
