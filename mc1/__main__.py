import argparse
import code
import runpy
import sys

from mc1 import *

dsp = None


def compile(func):
    dag = DAG(func) if not isinstance(func, DAG) else func
    if dsp is None:
        raise RuntimeError("DSP is not initialized")
    dsp.compile(bytes(dag))
    return dag


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m mc1")
    parser.add_argument("-b", "--block-size", dest="block_size", type=int)
    parser.add_argument("-r", "--sample-rate", dest="sample_rate", type=int)
    parser.add_argument("-i", "--input-channels", dest="input_channels", type=int)
    parser.add_argument("-o", "--output-channels", dest="output_channels", type=int)
    parser.add_argument("script", nargs="?")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def configure_dsp(args):
    global dsp

    dsp_kwargs = {}
    for name in ("block_size", "sample_rate", "input_channels", "output_channels"):
        value = getattr(args, name)
        if value is not None:
            dsp_kwargs[name] = value

    dsp = DSP(**dsp_kwargs)
    compile(default)


def build_namespace():
    excluded = {
        "argparse",
        "build_namespace",
        "code",
        "configure_dsp",
        "main",
        "parse_args",
        "runpy",
        "sys",
    }
    return {
        name: value
        for name, value in globals().items()
        if not name.startswith("__") and name not in excluded
    }


def main(argv=None):
    args = parse_args(argv)
    configure_dsp(args)

    ns = build_namespace()

    if args.script is not None:
        sys.argv = [args.script, *args.script_args]
        runpy.run_path(args.script, init_globals=ns, run_name="__main__")
        return 0

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
    perft(drone)
    dsp.append(IMMEDIATE, "default", freq=440)""",
        exitmsg="",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
