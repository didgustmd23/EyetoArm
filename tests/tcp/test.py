import socket

with socket.create_connection(("127.0.0.1", 47090), timeout=3) as sock:
    sock.sendall(b'{"version":"1.1","type":"GET_STATUS"}\n')
    response = b""
    while b"\n" not in response:
        chunk = sock.recv(4096)
        if not chunk:
            break
        response += chunk

print(response.decode("utf-8").strip())