import io
import json

from slai_mi.devices.iphone.client import IPhonePoseClient


def test_socket_client_receives_one_packet():
    packet = {"format_version": 1, "sequence": 1, "timestamp_s": 1, "sent_at_unix_s": 2, "tracking": "normal", "world_from_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}

    class Socket:
        def settimeout(self, _value): pass
        def makefile(self, _mode): return io.BytesIO(json.dumps(packet).encode() + b"\n")
        def close(self): pass

    client = IPhonePoseClient("localhost", socket_factory=lambda *_args, **_kwargs: Socket())
    client.connect()
    assert client.receive().sequence == 1
    client.close()
