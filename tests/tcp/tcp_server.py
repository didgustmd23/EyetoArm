import json
import socket

from protocol import encode_response, handle_request

HOST = "127.0.0.1"
PORT = 47090
MAX_LINE_BYTES = 4096


def handle_client(conn: socket.socket) -> None:
    """클라이언트 연결 하나를 처리합니다."""
    buffer = b""

    with conn:
        while True:
            chunk = conn.recv(1024)

            if not chunk:
                # 연결이 끊기면 LF 없는 미완성 메시지는 버립니다.
                break

            buffer += chunk

            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)

                # 명세서에 따라 줄 끝의 CR 하나는 제거할 수 있습니다.
                if line.endswith(b"\r"):
                    line = line[:-1]

                if len(line) > MAX_LINE_BYTES:
                    response = {"error": "BAD_FORMAT"}
                    conn.sendall(encode_response(response))
                    continue

                try:
                    request = json.loads(line.decode("utf-8"))
                    if not isinstance(request, dict):
                        raise ValueError("JSON 최상위 값이 객체가 아닙니다")
                    response = handle_request(request)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    response = {
                        "version": "1.1",
                        "type": "STATUS",
                        "code": "BAD_FORMAT",
                        "robot_ready": False,
                        "not_ready_reasons": ["RECOVERING"],
                        "active_command_id": None,
                        "server_time_ms": 0,
                    }

                conn.sendall(encode_response(response))


def main() -> None:
    """TCP 서버를 실행합니다."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen()

        print(f"TCP 서버 대기 중: {HOST}:{PORT}")

        while True:
            conn, address = server.accept()
            print(f"클라이언트 접속: {address}")
            handle_client(conn)


if __name__ == "__main__":
    main()