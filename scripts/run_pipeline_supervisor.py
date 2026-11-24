"""Restart the live A/B pipeline after its SIM_TIME reset boundary.

The Gazebo process and C relay stay alive.  Only run_pipeline (which contains
the B list consumer) is restarted when it exits with reset_detected (3).

Restarts are limited (``--max-restarts`` within ``--restart-window`` seconds) so
a reset storm or a pipeline that keeps exiting 3 does not loop forever.

Stop signals (SIGINT, SIGTERM, SIGHUP) are forwarded once to the running
pipeline, which then finishes its own cleanup.  The supervisor waits for it,
escalates to SIGTERM and then SIGKILL only after ``--stop-wait`` and
``--term-wait`` seconds, and never restarts after a stop signal.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
RESET_EXIT = 3


def _next_path(value: str, generation: int) -> str:
    path = Path(value)
    return str(path.with_name(f"{path.stem}.generation{generation}{path.suffix}"))


def _generation_args(command: list[str], generation: int) -> list[str]:
    result = list(command)
    for option in ("--log", "--live-observation-out"):
        try:
            index = result.index(option) + 1
        except ValueError:
            continue
        if index < len(result):
            result[index] = _next_path(result[index], generation)
    return result


class _Stop:
    """Stop request state shared by the signal handler and the wait loop."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.signum: int | None = None
        self.forwarded = False
        self.extra = 0              # later stop signals (not forwarded again)
        self.child: subprocess.Popen | None = None

    def on_signal(self, signum, frame) -> None:
        """First stop signal: remember it and forward it once to the running pipeline."""
        if self.event.is_set():
            self.extra += 1
            return

        self.signum = signum
        self.event.set()
        self._forward()

    def _forward(self) -> None:
        child = self.child

        if child is None or self.forwarded or child.poll() is not None:
            return

        self.forwarded = True

        try:
            child.send_signal(self.signum)
        except ProcessLookupError:   # the pipeline has just exited
            pass

    def attach(self, child: subprocess.Popen) -> None:
        """Track a new pipeline; forward a stop that arrived while it was starting."""
        self.child, self.forwarded = child, False

        if self.event.is_set():
            self._forward()


def _wait_child(child: subprocess.Popen, stop: _Stop, *, stop_wait: float, term_wait: float, kill_wait: float,
                poll: float = 0.1, log=print) -> int:
    """Wait for the pipeline. After a stop signal: wait stop_wait, then SIGTERM, wait term_wait, then SIGKILL."""
    while child.poll() is None and not stop.event.is_set():
        stop.event.wait(poll)

    if child.poll() is not None:
        return child.returncode

    for sig, wait_s in ((None, stop_wait), (signal.SIGTERM, term_wait), (signal.SIGKILL, kill_wait)):
        if sig is not None and child.poll() is None:
            log(f"[pipeline supervisor] pipeline still running after stop; sending {signal.Signals(sig).name}")

            try:
                child.send_signal(sig)
            except ProcessLookupError:
                pass

        try:
            return child.wait(wait_s)
        except subprocess.TimeoutExpired:
            continue

    return child.wait()


def _exit_status(returncode: int) -> int:
    """Child return code as a process exit status (killed by signal N -> 128 + N, like the shell)."""
    return returncode if returncode >= 0 else 128 - returncode


def _restart_allowed(history: deque, now: float, *, max_restarts: int, window: float) -> bool:
    """Sliding-window restart limit: at most max_restarts restarts within window seconds."""
    while history and now - history[0] > window:
        history.popleft()

    return len(history) < max_restarts


def main(argv=None) -> int:
    """Run the pipeline command, restart it on exit 3 within the limit, and forward stop signals."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--restart-delay", type=float, default=0.5)
    parser.add_argument("--max-restarts", type=int, default=3,
                        help="restarts allowed within --restart-window (default 3)")
    parser.add_argument("--restart-window", type=float, default=600.0,
                        help="restart limit window in seconds (default 600)")
    parser.add_argument("--stop-wait", type=float, default=15.0,
                        help="seconds to wait for the pipeline after forwarding a stop signal (default 15)")
    parser.add_argument("--term-wait", type=float, default=5.0,
                        help="seconds to wait after SIGTERM before SIGKILL (default 5)")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a run_pipeline command is required after --")
    if args.max_restarts < 0 or args.restart_window <= 0 or args.stop_wait <= 0 or args.term_wait <= 0:
        parser.error("--max-restarts must be >= 0 and --restart-window/--stop-wait/--term-wait > 0")

    stop = _Stop()
    old = {sig: signal.signal(sig, stop.on_signal) for sig in STOP_SIGNALS}
    history: deque = deque()
    generation = 0

    try:
        while True:
            child = subprocess.Popen(_generation_args(command, generation))
            stop.attach(child)
            returncode = _wait_child(child, stop, stop_wait=args.stop_wait, term_wait=args.term_wait, kill_wait=2.0)

            if stop.event.is_set():
                print(f"[pipeline supervisor] stopped by {signal.Signals(stop.signum).name}; "
                      f"pipeline exit {returncode}; no restart", flush=True)
                return _exit_status(returncode)

            if returncode != RESET_EXIT:
                return _exit_status(returncode)

            now = time.monotonic()

            if not _restart_allowed(history, now, max_restarts=args.max_restarts, window=args.restart_window):
                print(f"[pipeline supervisor] reset_detected but restart limit reached "
                      f"({args.max_restarts} within {args.restart_window:g} s); not restarting", flush=True)
                return returncode

            history.append(now)
            generation += 1
            print(f"[pipeline supervisor] reset_detected; restarting generation {generation} "
                  f"({len(history)}/{args.max_restarts} within {args.restart_window:g} s)", flush=True)

            if stop.event.wait(max(0.0, args.restart_delay)):
                print(f"[pipeline supervisor] stopped by {signal.Signals(stop.signum).name} before restart",
                      flush=True)
                return returncode
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
