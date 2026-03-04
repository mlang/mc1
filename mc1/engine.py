import atexit
import pathlib
import socket
import struct
import subprocess

from mc1.message import DONE_IDENTIFIER, Sync


class Engine:
    build_dir: pathlib.Path = pathlib.Path(".build/default")

    @classmethod
    def build(cls):
        if not cls.build_dir.is_dir():
            subprocess.run(["cmake", "--preset", "default"], check=True)
        subprocess.run(["cmake", "--build", "--preset", "default"], check=True)

    def __init__(self, port=5555):
        self.port = int(port)
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.bind(("127.0.0.1", 0))

    def run(self):
        self.process = subprocess.Popen(
            map(str, (self.build_dir / "engine", self.port))
        )
        atexit.register(self.stop)

    def send(self, data):
        return self.udp.sendto(bytes(data), ("localhost", self.port))

    def sync(self, timeout=5.0):
        self.send(Sync())
        old_timeout = self.udp.gettimeout()
        self.udp.settimeout(timeout)
        try:
            while True:
                data, _sender = self.udp.recvfrom(1024)
                if len(data) < 2:
                    continue
                msg_id = struct.unpack("H", data[:2])[0]
                if msg_id == DONE_IDENTIFIER:
                    return
        finally:
            self.udp.settimeout(old_timeout)

    def stop(self):
        if is_running(self.process):
            self.process.terminate()
            self.process.wait()


def is_running(process):
    return process.poll() is None
