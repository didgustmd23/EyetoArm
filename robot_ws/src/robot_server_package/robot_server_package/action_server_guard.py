"""Prevent competing local and ROS-graph action servers from owning the same action."""
from __future__ import annotations

import fcntl
import os
import re
import time


class ActionServerConflict(RuntimeError):
    """Raised when another process already owns the guarded action name."""


class ActionServerGuard:
    """Hold a per-domain local lock for the lifetime of one action server."""

    def __init__(self, fd: int, action_name: str) -> None:
        self._fd = fd
        self.action_name = action_name

    def close(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


def acquire_action_server_guard(node, action_name: str, *, discovery_s: float = 1.0) -> ActionServerGuard:
    """Acquire the local lock, then refuse startup if ROS already sees this action.

    The lock serializes starts on one WSL host. The graph check also catches a server
    running on another host in the same ROS domain. Keep the returned guard alive until
    the ActionServer has been destroyed.
    """
    normalized = "/" + "/".join(part for part in str(action_name).split("/") if part)
    domain = re.sub(r"[^A-Za-z0-9_-]", "_", os.environ.get("ROS_DOMAIN_ID", "0"))
    lock_path = f"/tmp/eyetoarm_action_{domain}_{normalized.strip('/').replace('/', '_')}.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ActionServerConflict(f"{normalized} already has a local server lock ({lock_path})") from None

        deadline = time.monotonic() + max(0.0, discovery_s)
        server_services = {
            f"{normalized}/_action/send_goal",
            f"{normalized}/_action/get_result",
            f"{normalized}/_action/cancel_goal",
        }
        while True:
            for node_name, namespace in node.get_node_names_and_namespaces():
                offered = {name for name, _types in node.get_service_names_and_types_by_node(node_name, namespace)}
                if server_services.issubset(offered):
                    raise ActionServerConflict(f"ROS graph already contains an action server for {normalized}")
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        return ActionServerGuard(fd, normalized)
    except Exception:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        raise
