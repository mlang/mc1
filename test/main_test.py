import asyncio
import importlib
import sys
import textwrap

import mc1
import mc1.__main__


def test_parse_args_preserves_script_args_after_script():
    main = importlib.reload(mc1.__main__)

    args = main.parse_args(
        ["-b", "64", "-r", "48000", "-i", "1", "-o", "2", "demo.py", "--flag", "value"]
    )

    assert args.block_size == 64
    assert args.sample_rate == 48000
    assert args.input_channels == 1
    assert args.output_channels == 2
    assert args.script == "demo.py"
    assert args.script_args == ["--flag", "value"]


def test_configure_dsp_uses_cli_values():
    main = importlib.reload(mc1.__main__)

    args = main.parse_args(["-b", "64", "-r", "48000", "-i", "1", "-o", "2"])
    main.configure_dsp(args)

    assert main.dsp.sample_rate == 48000
    assert main.dsp.block_size == 64
    assert main.dsp.input_channels == 1
    assert main.dsp.output_channels == 2


def test_init_namespace_adds_running_clock():
    main = importlib.reload(mc1.__main__)
    main.configure_dsp(main.parse_args([]))

    ns = asyncio.run(main.init_namespace())

    assert ns["dsp"] is main.dsp
    assert ns["clock"]._task is not None

    asyncio.run(main.drain_scheduler(ns["clock"]))
    assert ns["clock"]._task is None


def test_mc1_exports_midi2cps():
    assert "midi2cps" in mc1.__all__
    assert mc1.midi2cps(69) == 440.0


def test_run_script_executes_async_main_and_restores_sys_argv(tmp_path):
    main = importlib.reload(mc1.__main__)
    main.configure_dsp(main.parse_args([]))

    script = tmp_path / "async_script.py"
    script.write_text(
        textwrap.dedent(
            """
            import asyncio
            import sys

            argv = list(sys.argv)
            ran = False
            saw_dsp = False
            saw_running_clock = False

            async def main():
                global ran, saw_dsp, saw_running_clock

                await asyncio.sleep(0)
                ran = True
                saw_dsp = dsp is not None
                saw_running_clock = clock._task is not None
            """
        )
    )

    original_argv = sys.argv[:]
    script_ns = asyncio.run(main.run_script(str(script), ["--flag", "value"]))

    assert sys.argv == original_argv
    assert script_ns["argv"] == [str(script), "--flag", "value"]
    assert script_ns["ran"] is True
    assert script_ns["saw_dsp"] is True
    assert script_ns["saw_running_clock"] is True
    assert script_ns["clock"]._task is None


def test_run_script_exposes_midi2cps_from_mc1_namespace(tmp_path):
    main = importlib.reload(mc1.__main__)
    main.configure_dsp(main.parse_args([]))

    script = tmp_path / "pitch_script.py"
    script.write_text(
        textwrap.dedent(
            """
            saw_midi2cps = midi2cps(69)
            """
        )
    )

    script_ns = asyncio.run(main.run_script(str(script), []))

    assert script_ns["saw_midi2cps"] == 440.0
    assert script_ns["clock"]._task is None


def test_run_script_does_not_auto_call_sync_main(tmp_path):
    main = importlib.reload(mc1.__main__)
    main.configure_dsp(main.parse_args([]))

    script = tmp_path / "sync_script.py"
    script.write_text(
        textwrap.dedent(
            """
            ran = False

            def main():
                global ran
                ran = True
            """
        )
    )

    script_ns = asyncio.run(main.run_script(str(script), []))

    assert script_ns["ran"] is False
    assert script_ns["clock"]._task is None


def test_main_uses_async_repl_for_interactive_session(monkeypatch):
    main = importlib.reload(mc1.__main__)
    calls = []

    def fake_interact(*, banner, locals, exitmsg):
        calls.append((banner, locals, exitmsg))

    monkeypatch.setattr(main, "async_interact", fake_interact)

    assert main.main([]) == 0
    assert calls == [
        (
            """MiniCollider

Example:
    perft(drone)
    dsp.append(IMMEDIATE, "default", freq=440)""",
            main.init_namespace,
            "",
        )
    ]
