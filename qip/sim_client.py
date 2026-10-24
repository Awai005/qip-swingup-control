"""Minimal client for ./simulator/base (TCP, localhost:65432). One client at a time."""
import socket
import numpy as np


class SimClient:
    def __init__(self, host="localhost", port=65432, timeout=5.0):
        self.addr, self.timeout, self.sock = (host, port), timeout, None

    def send(self, command):
        for attempt in range(2):
            try:
                if self.sock is None:
                    self.sock = socket.create_connection(self.addr, timeout=self.timeout)
                    self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock.sendall(command.encode())
                data = self.sock.recv(4096).decode().strip()
                if not data:
                    raise ConnectionError("empty reply")
                return data
            except (OSError, ConnectionError):
                self.close()
                if attempt == 1:
                    raise
        return ""

    def state(self, command):
        """'reset' or 'step <force>' -> state as a numpy array."""
        return np.array([float(v) for v in self.send(command).split()])

    def ctrl_ms(self):
        return int(float(self.send("get_ctrl_ms").split()[0]))

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None
