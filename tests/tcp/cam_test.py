import json
import socket
import struct

def recv_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("중계 연결이 끊겼습니다")
        data.extend(chunk)
    return bytes(data)

with socket.create_connection(("127.0.0.1", 47092), timeout=10) as sock:
    received = set()
    while received != {"link", "gripper", "fixed"}:
        magic, length = struct.unpack(">4sI", recv_exact(sock, 8))
        if magic != b"ETV1":
            raise ValueError(f"잘못된 프레임 형식: {magic!r}")

        body = recv_exact(sock, length)
        json_length = struct.unpack_from(">I", body)[0]
        metadata = json.loads(body[4:4 + json_length])
        jpeg = body[4 + json_length:]

        channel = metadata["channel"]
        print(
            channel,
            "seq=", metadata["image_seq"],
            "jpeg_bytes=", len(jpeg),
            "jpeg_signature_ok=", jpeg.startswith(b"\\xff\\xd8") and jpeg.endswith(b"\\xff\\xd9"),
        )
        received.add(channel)