import atexit
import pathlib
import socket
import subprocess

from mc1 import osc
from mc1.message import DONE_ADDRESS, Sync


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

    def sync(self, timeout=30.0):
        self.send(Sync())
        old_timeout = self.udp.gettimeout()
        self.udp.settimeout(timeout)
        try:
            while True:
                data, _sender = self.udp.recvfrom(1024)
                try:
                    packet = osc.decode_packet(data)
                except ValueError:
                    continue
                if isinstance(packet, osc.Message) and packet.address == DONE_ADDRESS:
                    return
        finally:
            self.udp.settimeout(old_timeout)

    def stop(self):
        if is_running(self.process):
            self.process.terminate()
            self.process.wait()


def is_running(process):
    return process.poll() is None
