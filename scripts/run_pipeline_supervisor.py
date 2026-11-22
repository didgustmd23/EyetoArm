"""Restart the live A/B pipeline after its SIM_TIME reset boundary.

The Gazebo process and C relay stay alive.  Only run_pipeline (which contains
the B list consumer) is restarted when it exits with reset_detected (3).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart-delay", type=float, default=0.5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a run_pipeline command is required after --")

    generation = 0
    while True:
        process = subprocess.run(_generation_args(command, generation))
        if process.returncode != 3:
            return process.returncode
        generation += 1
        print(f"[pipeline supervisor] reset_detected; restarting generation {generation}", flush=True)
        time.sleep(max(0.0, args.restart_delay))


if __name__ == "__main__":
    raise SystemExit(main())
