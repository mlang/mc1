import argparse
import asyncio
import builtins
import inspect
import sys

from mc1 import *
from mc1.async_repl import interact as async_interact
from mc1.clock import LogicalClock

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
        "async_interact",
        "asyncio",
        "build_namespace",
        "builtins",
        "configure_dsp",
        "init_namespace",
        "inspect",
        "LogicalClock",
        "main",
        "parse_args",
        "run_script",
        "shutdown_clock",
        "sys",
    }
    return {
        name: value
        for name, value in globals().items()
        if not name.startswith("__") and name not in excluded
    }


async def shutdown_clock(clock):
    task = clock._task
    if task is None:
        return

    if clock.stop():
        try:
            await task
        except asyncio.CancelledError:
            pass


async def init_namespace():
    ns = build_namespace()
    clock = LogicalClock()
    clock.start()
    ns["clock"] = clock
    return ns


async def run_script(path, script_args):
    ns = await init_namespace()
    clock = ns["clock"]
    original_argv = sys.argv[:]

    try:
        sys.argv = [path, *script_args]
        ns.update(
            __name__="__main__",
            __file__=path,
            __package__=None,
            __cached__=None,
            __spec__=None,
        )
        with builtins.open(path, "rb") as handle:
            source = handle.read()
        exec(builtins.compile(source, path, "exec"), ns)

        script_main = ns.get("main")
        if inspect.iscoroutinefunction(script_main):
            await script_main()
        return ns
    finally:
        sys.argv = original_argv
        await shutdown_clock(clock)


def main(argv=None):
    args = parse_args(argv)
    configure_dsp(args)

    if args.script is not None:
        asyncio.run(run_script(args.script, args.script_args))
        return 0

    async_interact(
        banner="""MiniCollider

Example:
    perft(drone)
    dsp.append(IMMEDIATE, "default", freq=440)""",
        locals=init_namespace,
        exitmsg="",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
