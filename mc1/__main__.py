import atexit
import os
import socket
import struct
import subprocess

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

def send(data: bytes):
    return udp.sendto(data, ("localhost", port))

def is_running(process):
    return process.poll() is None
