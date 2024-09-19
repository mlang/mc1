import atexit
import pathlib
import socket
import subprocess


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

    def run(self):
        self.process = subprocess.Popen(
            map(str, (self.build_dir / "engine", self.port))
        )
        atexit.register(self.stop)

    def send(self, data):
        return self.udp.sendto(bytes(data), ("localhost", self.port))

    def stop(self):
        if is_running(self.process):
            self.process.terminate()
            self.process.wait()


def is_running(process):
    return process.poll() is None
