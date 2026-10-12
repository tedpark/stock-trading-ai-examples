"""Block TCP/UDP connections in verification processes. Activate via PYTHONPATH."""
import socket


def blocked(*args, **kwargs):
    raise RuntimeError('Offline verification: network connections forbidden')


socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked
