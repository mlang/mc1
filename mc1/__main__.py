import atexit
import code
import os
import socket
import struct
import subprocess

from mc1.dag import *

class Engine:
    def __init__(self, port=5555):
        self.port = port
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    @classmethod
    def build(cls):
        if not os.path.isdir(".build/default"):
            subprocess.run(["cmake", "--preset", "default"], check=True)
        subprocess.run(["cmake", "--build", "--preset", "default"], check=True)

    def run(self):
        self.process = subprocess.Popen([".build/default/engine", str(self.port)])
        atexit.register(self.stop)

    def send(self, data):
        return self.udp.sendto(bytes(data), ("localhost", self.port))

    def stop(self):
        if is_running(self.process):
            self.process.terminate()
            self.process.wait()


def msg(id: int, content):
    return struct.pack('H', id) + bytes(content)

def is_running(process):
    return process.poll() is None


Engine.build()
dsp = Engine()
dsp.run()

code.interact(local=locals(), banner="", exitmsg="")
