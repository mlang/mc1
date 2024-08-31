import atexit
import code
import os
import socket
import struct
import subprocess

from mc1.dag import *

if not os.path.isdir("build/default"):
    subprocess.run(["cmake", "--preset", "default"], check=True)
subprocess.run(["cmake", "--build", "--preset", "default"], check=True)

port = 5555
engine = subprocess.Popen(["build/default/engine", str(port)])

def cleanup_subprocess(process):
    if is_running(process):
        process.terminate()
        process.wait()

atexit.register(cleanup_subprocess, engine)

udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send(data):
    return udp.sendto(bytes(data), ("localhost", port))

def msg(id: int, content):
    return struct.pack('H', id) + bytes(content)

def is_running(process):
    return process.poll() is None


code.interact(local=locals(), banner="", exitmsg="Bye!")
