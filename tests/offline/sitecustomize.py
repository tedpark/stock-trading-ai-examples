"""검증 프로세스에서 TCP/UDP 연결을 차단한다. PYTHONPATH로 활성화."""
import socket


def blocked(*args, **kwargs):
    raise RuntimeError('Offline verification: network connections forbidden')


socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked
