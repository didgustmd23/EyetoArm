#!/usr/bin/env python3
"""Day7 WSL mock-C peer: the UNCHANGED C TcpServer with a not-ready mock readiness, GET_STATUS only.

Runs inside an already running WSL distro with the distro's own python3 (standard library only; nothing is
installed). It is started by tests/integration/test_d_day7_wsl.py and never by itself on import.

- Imports robot_server_package.tcp_server from the repository (read via /mnt/<drive>/...); no C file is changed.
- Binds 127.0.0.1 port 0 (never the fixed 47090), robot_ready=False, no move executor, no ROS/Gazebo/robot.
- stdout, one JSON object per line: {"event": "ready", ...} then {"event": "stopped", ...}.
- Stops on a stdin line "STOP", on stdin EOF, or by itself after --deadline seconds (default 30).
- Its temporary folder (command log) is created with mkdtemp and removed before "stopped" is printed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

TEMP_PREFIX = "eyetoarm-d7-wsl-"


def emit(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}), flush=True)


async def serve(repo: Path, deadline_s: float) -> dict:
    sys.path.insert(0, str(repo / "robot_ws/src/robot_server_package"))
    from robot_server_package.tcp_server import TcpServer   # unchanged C source

    temp = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX))
    seen: dict[str, int] = {}
    stop = asyncio.Event()
    reason = "deadline"
    listener = None
    server = None
    loop = asyncio.get_running_loop()
    try:
        log_path = temp / "c-command.jsonl"
        log_path.touch(exist_ok=False)   # initialized-empty command log, as in the Windows fixture

        def readiness():
            return {"robot_ready": False, "not_ready_reasons": ["SIM_PAUSED"], "active_command_id": None}

        server = TcpServer(readiness_provider=readiness, command_log_path=log_path)   # no executor: never moves
        original = server._handle_line

        async def counting_handle_line(line, session, previous):
            try:
                kind = json.loads(line).get("type", "?")
            except (ValueError, AttributeError):
                kind = "INVALID"
            seen[str(kind)] = seen.get(str(kind), 0) + 1
            await original(line, session, previous)

        server._handle_line = counting_handle_line   # observe only; C still answers every line itself
        listener = await asyncio.start_server(server._handle_client, "127.0.0.1", 0)
        port = listener.sockets[0].getsockname()[1]

        def on_stdin():
            nonlocal reason
            data = os.read(sys.stdin.fileno(), 1024)
            if not data:
                reason = "stdin EOF"
            elif b"STOP" in data:
                reason = "stdin STOP"
            else:
                return
            loop.remove_reader(sys.stdin.fileno())
            stop.set()

        loop.add_reader(sys.stdin.fileno(), on_stdin)   # no blocking readline thread that could hang shutdown
        emit("ready", pid=os.getpid(), ppid=os.getppid(), executable=sys.executable, temp=str(temp),
             host="127.0.0.1", port=port, created_ms=int(time.time() * 1000), deadline_s=deadline_s)
        try:
            await asyncio.wait_for(stop.wait(), deadline_s)
        except asyncio.TimeoutError:
            reason = "deadline"
    finally:
        try:
            loop.remove_reader(sys.stdin.fileno())
        except (ValueError, OSError):
            pass
        if listener is not None:
            listener.close()
            await listener.wait_closed()
        if server is not None:
            for session in list(server._sessions.values()):
                session.writer.close()
            await server.close()
        current = asyncio.current_task()
        tasks = [task for task in asyncio.all_tasks() if task is not current]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if temp.parent == Path(tempfile.gettempdir()) and temp.name.startswith(TEMP_PREFIX):
            shutil.rmtree(temp, ignore_errors=True)
    return {"reason": reason, "requests": seen, "temp": str(temp), "temp_removed": not temp.exists()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="repository root as a Linux path, e.g. /mnt/c/.../EyetoArm")
    parser.add_argument("--deadline", type=float, default=30.0)
    args = parser.parse_args(argv)
    if not 0 < args.deadline <= 120:
        parser.error("--deadline must be within (0, 120] seconds")
    result = asyncio.run(serve(Path(args.repo), args.deadline))
    emit("stopped", **result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
