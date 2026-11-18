"""B → C 연결 시험 — 시험값(`config/robot/b_c_test_vectors.json`)을 B 클라이언트로 `/sort_part`에 보낸다.

지금은 가짜 서버(`scripts/b_fake_sort_part_server.py`), C 서버가 나오면 C 서버에 대고 그대로 돌린다.
경우마다 결과를 기다린 뒤 다음 것을 보낸다(클라이언트는 같은 종류가 결과 대기 중이면 안 보냄).

    source /opt/ros/jazzy/setup.bash && source robot_ws/install/setup.bash
    PYTHONPATH=src:$PYTHONPATH python3 scripts/b_send_test_goal.py [--case bolt_a_along_x] [--wait-server 5]
"""
import argparse
import json
import time
import uuid
from pathlib import Path

import rclpy

from eyetoarm.transforms.sort_part_client import SortPartClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", default=None, help="이름 하나만 (기본: 전부)")
    ap.add_argument("--wait-server", type=float, default=5.0, help="서버 기다리는 시간(s)")
    ap.add_argument("--result-timeout", type=float, default=30.0, help="결과 기다리는 시간(s)")
    a = ap.parse_args()

    cases = json.loads((ROOT / "config/robot/b_c_test_vectors.json").read_text(encoding="utf-8"))["cases"]
    cases = [c for c in cases if a.case in (None, c["name"])]
    if not cases:
        print(f"시험값에 없는 이름: {a.case}")
        return 1

    rclpy.init()
    node = rclpy.create_node("b_send_test_goal")
    client = SortPartClient(node, stage=None)
    try:
        if not client.client.wait_for_server(timeout_sec=a.wait_server):
            print(f"/sort_part 서버 없음 ({a.wait_server}s)")
            return 1

        for c in cases:
            goal = dict(c["expected"], goal_id=str(uuid.uuid4()), spec=c["input"]["spec"], source_id="b_test",
                        capture_stamp_ns=node.get_clock().now().nanoseconds or 1, calibration_id="cal-test")
            sent = client.send(goal)
            print(f"[{c['name']}] {goal['kind']} 보냄={sent}")
            t0 = time.monotonic()
            while sent and client.inflight and time.monotonic() - t0 < a.result_timeout:
                rclpy.spin_once(node, timeout_sec=0.1)
            if sent and client.inflight:
                print(f"[{c['name']}] 결과 없음 ({a.result_timeout}s)")

        print("통계:", dict(client.stats))
        return 0 if not client.inflight else 2
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
