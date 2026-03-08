import pytest

from mc1 import osc
from mc1.message import Compile, Quit, Sync


def test_message_roundtrip_core_types():
    packet = osc.encode_message("/mc1/test", 42, 0.25, "abc", b"\x01\x02")
    decoded = osc.decode_packet(packet)
    assert isinstance(decoded, osc.Message)
    assert decoded.address == "/mc1/test"
    assert decoded.args[0] == 42
    assert decoded.args[1] == pytest.approx(0.25, rel=1e-6)
    assert decoded.args[2] == "abc"
    assert decoded.args[3] == b"\x01\x02"


def test_bundle_roundtrip():
    inner = osc.Message("/mc1/test", (7,))
    bundle = osc.Bundle(timetag=1, elements=(inner,))
    decoded = osc.decode_packet(osc.encode_packet(bundle))
    assert isinstance(decoded, osc.Bundle)
    assert decoded.timetag == 1
    assert len(decoded.elements) == 1
    assert isinstance(decoded.elements[0], osc.Message)
    assert decoded.elements[0].address == "/mc1/test"
    assert decoded.elements[0].args == (7,)


@pytest.mark.parametrize("packet", [
    b"/bad\0\0\0",
    b"/ok\0\0\0,\0\0\0\x00\x00\x00\x01",
])
def test_decode_invalid_packet(packet):
    with pytest.raises(ValueError):
        osc.decode_packet(packet)


def test_message_classes_emit_osc_addresses():
    quit_msg = osc.decode_packet(bytes(Quit()))
    sync_msg = osc.decode_packet(bytes(Sync()))
    assert isinstance(quit_msg, osc.Message)
    assert isinstance(sync_msg, osc.Message)
    assert quit_msg.address == "/mc1/quit"
    assert sync_msg.address == "/mc1/sync"


def test_compile_payload_is_blob():
    msg = osc.decode_packet(bytes(Compile(lambda freq=440: freq)))
    assert isinstance(msg, osc.Message)
    assert msg.address == "/mc1/compile"
    assert len(msg.args) == 1
    assert isinstance(msg.args[0], bytes)
