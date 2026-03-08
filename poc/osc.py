"""Minimal OSC codec supporting messages and bundles."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import TypeAlias


OscAtom: TypeAlias = int | float | str | bytes


@dataclass(frozen=True, slots=True)
class Message:
    address: str
    args: tuple[OscAtom, ...] = ()


@dataclass(frozen=True, slots=True)
class Bundle:
    timetag: int
    elements: tuple["Packet", ...]


Packet: TypeAlias = Message | Bundle


def _pad4(size: int) -> int:
    return (size + 3) & ~0x03


def _pack_i32(value: int) -> bytes:
    return struct.pack(">i", value)


def _unpack_i32(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 4 > len(data):
        raise ValueError("truncated int32")
    return struct.unpack(">i", data[offset : offset + 4])[0], offset + 4


def _pack_u64(value: int) -> bytes:
    if value < 0 or value > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("timetag out of range")
    return struct.pack(">Q", value)


def _unpack_u64(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 8 > len(data):
        raise ValueError("truncated timetag")
    return struct.unpack(">Q", data[offset : offset + 8])[0], offset + 8


def _pack_f32(value: float) -> bytes:
    return struct.pack(">f", value)


def _unpack_f32(data: bytes, offset: int) -> tuple[float, int]:
    if offset + 4 > len(data):
        raise ValueError("truncated float32")
    return struct.unpack(">f", data[offset : offset + 4])[0], offset + 4


def _pack_string(text: str) -> bytes:
    raw = text.encode("utf-8") + b"\0"
    return raw + (b"\0" * (_pad4(len(raw)) - len(raw)))


def _unpack_string(data: bytes, offset: int) -> tuple[str, int]:
    try:
        nul = data.index(0, offset)
    except ValueError as exc:
        raise ValueError("unterminated string") from exc
    end = _pad4(nul + 1)
    if end > len(data):
        raise ValueError("truncated padded string")
    if any(data[i] != 0 for i in range(nul + 1, end)):
        raise ValueError("invalid string padding")
    return data[offset:nul].decode("utf-8"), end


def _pack_blob(blob: bytes) -> bytes:
    raw = bytes(blob)
    head = _pack_i32(len(raw))
    pad_len = _pad4(len(raw)) - len(raw)
    return head + raw + (b"\0" * pad_len)


def _unpack_blob(data: bytes, offset: int) -> tuple[bytes, int]:
    size, offset = _unpack_i32(data, offset)
    if size < 0:
        raise ValueError("negative blob size")
    end_raw = offset + size
    end = _pad4(end_raw)
    if end > len(data):
        raise ValueError("truncated blob")
    if any(data[i] != 0 for i in range(end_raw, end)):
        raise ValueError("invalid blob padding")
    return data[offset:end_raw], end


def encode_message(address: str, *args: OscAtom) -> bytes:
    if not address.startswith("/"):
        raise ValueError("OSC address must start with '/'")

    tags = [","]
    payload = bytearray()
    for arg in args:
        if isinstance(arg, bool):
            raise TypeError("bool is not a supported OSC atom in this subset")
        if isinstance(arg, int):
            tags.append("i")
            payload.extend(_pack_i32(arg))
        elif isinstance(arg, float):
            tags.append("f")
            payload.extend(_pack_f32(arg))
        elif isinstance(arg, str):
            tags.append("s")
            payload.extend(_pack_string(arg))
        elif isinstance(arg, (bytes, bytearray, memoryview)):
            tags.append("b")
            payload.extend(_pack_blob(bytes(arg)))
        else:
            raise TypeError(f"unsupported OSC argument type: {type(arg)!r}")

    return _pack_string(address) + _pack_string("".join(tags)) + bytes(payload)


def decode_message(packet: bytes) -> Message:
    address, offset = _unpack_string(packet, 0)
    if not address.startswith("/"):
        raise ValueError("OSC address must start with '/'")
    tags, offset = _unpack_string(packet, offset)
    if not tags.startswith(","):
        raise ValueError("OSC typetag string must start with ','")

    args: list[OscAtom] = []
    for tag in tags[1:]:
        if tag == "i":
            int_value, offset = _unpack_i32(packet, offset)
            args.append(int_value)
        elif tag == "f":
            float_value, offset = _unpack_f32(packet, offset)
            args.append(float_value)
        elif tag == "s":
            str_value, offset = _unpack_string(packet, offset)
            args.append(str_value)
        elif tag == "b":
            blob_value, offset = _unpack_blob(packet, offset)
            args.append(blob_value)
        else:
            raise ValueError(f"unsupported OSC typetag: {tag!r}")

    if offset != len(packet):
        raise ValueError("trailing bytes in OSC message")
    return Message(address=address, args=tuple(args))


def encode_bundle(elements: tuple[Packet, ...], timetag: int = 1) -> bytes:
    payload = bytearray()
    for element in elements:
        encoded = encode_packet(element)
        payload.extend(_pack_i32(len(encoded)))
        payload.extend(encoded)
    return _pack_string("#bundle") + _pack_u64(timetag) + bytes(payload)


def decode_bundle(packet: bytes) -> Bundle:
    marker, offset = _unpack_string(packet, 0)
    if marker != "#bundle":
        raise ValueError("OSC bundle marker mismatch")
    timetag, offset = _unpack_u64(packet, offset)

    elements: list[Packet] = []
    while offset < len(packet):
        size, offset = _unpack_i32(packet, offset)
        if size <= 0:
            raise ValueError("invalid OSC bundle element size")
        end = offset + size
        if end > len(packet):
            raise ValueError("truncated OSC bundle element")
        elements.append(decode_packet(packet[offset:end]))
        offset = end
    return Bundle(timetag=timetag, elements=tuple(elements))


def encode_packet(packet: Packet) -> bytes:
    if isinstance(packet, Message):
        return encode_message(packet.address, *packet.args)
    if isinstance(packet, Bundle):
        return encode_bundle(packet.elements, timetag=packet.timetag)
    raise TypeError(f"unsupported OSC packet type: {type(packet)!r}")


def decode_packet(packet: bytes) -> Packet:
    marker, _ = _unpack_string(packet, 0)
    if marker == "#bundle":
        return decode_bundle(packet)
    return decode_message(packet)
