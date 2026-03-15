import importlib

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
