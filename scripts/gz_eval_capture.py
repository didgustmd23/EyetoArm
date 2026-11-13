#!/usr/bin/env python3
"""Gazebo 평가 세트 수집(실행·캡처·라벨 생성).

**`--yes` 없이는 Gazebo를 띄우지 않는다.** 근거: A 상세설계 §7.1 「Gazebo 평가 세트」(A 제안).

입력: `scripts/gz_eval_scene.py`가 만든 폴더(`plan.json`, `world_bXXXX.sdf`).
출력: `data/derived/gz_eval_v1/`(Git 제외) — `eval_size_recall.py`가 읽는 YOLO 분할 형식
  `images/test/*.png`, `labels/test/*.txt`, `dataset.yaml`(2클래스 bolt·nut), `manifest.jsonl`(표본별 세부 클래스·
  월드 좌표·카메라 거리·영상 짧은 변 px·시드·Gazebo 버전).

수집 방식(A 제안): 묶음(=월드 사본 하나, 카메라·조명 고정)마다 `gz sim`을 한 번 띄우고, 표본마다
`/world/<world>/set_pose_vector` 서비스(`gz service` CLI)로 18개 슬롯 pose를 한 번에 바꾼 뒤,
`ros_gz_bridge`로 받은 RGB·분할 라벨 영상 중 **pose 변경 뒤 시뮬레이션 시각 + settle** 이후 프레임을 저장한다.
python `gz.transport13` 바인딩이 없어서 서비스는 CLI, 영상은 ROS 브리지(rclpy)로 받는다.

안전 장치(다른 세션 보호, 실행 시 강제):
- 고유 `GZ_PARTITION`(기본 `eyetoarm_gz_eval_<pid>`)과 `ROS_DOMAIN_ID`(기본 61)를 자식 프로세스와 이 프로세스에 설정.
- `setsid`(start_new_session)로 띄우고 **자기 프로세스 그룹만** `os.killpg`로 종료. `pkill`·`killall`을 쓰지 않는다.
- 시작 전 `MemAvailable`(기본 ≥ 6 GiB)과 GPU 사용 메모리(기본 ≤ 8000 MiB)를 확인해 기준 밖이면 시작 거부.
- 다른 `gz sim` 프로세스가 있으면 경고만 출력하고 그대로 둔다.
- 실행 중에는 표본마다 `MemAvailable`을 확인해 `--stop-mem-gib`(기본 2 GiB) 아래면 자기 그룹만 종료하고 멈춘다
  (사용자 결정 2026-11-12: 실행 중 종료 조건은 메모리 하나).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gz_eval_scene as scene  # noqa: E402

DEFAULT_DOMAIN_ID = 61
DEFAULT_MIN_MEM_GIB = 6.0
DEFAULT_MAX_GPU_USED_MIB = 8000
RGB_TOPIC = "/eval_cam/image"
SEG_TOPIC = "/eval_cam/seg/labels_map"
CLASS_NAMES = ["bolt", "nut"]


# ---------------------------------------------------------------- 안전 장치(순수 함수)
def default_partition(pid=None) -> str:
    """기본 `GZ_PARTITION` 이름(`eyetoarm_gz_eval_<pid>`, pid 생략 시 현재 프로세스)."""
    return f"eyetoarm_gz_eval_{pid if pid is not None else os.getpid()}"


def build_env(base: dict, partition: str, domain_id: int, resource_path: str) -> dict:
    """`base` 사본에 GZ_PARTITION·ROS_DOMAIN_ID·GZ_SIM_RESOURCE_PATH를 넣어 돌려준다. 파티션이 비면 ValueError."""
    if not partition:
        raise ValueError("GZ_PARTITION이 비어 있음")

    env = dict(base)
    env["GZ_PARTITION"] = partition
    env["ROS_DOMAIN_ID"] = str(int(domain_id))
    env["GZ_SIM_RESOURCE_PATH"] = resource_path
    return env


def parse_meminfo_available_kib(text: str) -> int:
    """`/proc/meminfo` 내용에서 MemAvailable(KiB). 줄이 없으면 ValueError."""
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])

    raise ValueError("MemAvailable 없음")


def parse_gpu_used_mib(nvidia_smi_csv: str) -> int:
    """`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` 출력의 최댓값."""
    vals = [int(x.strip()) for x in nvidia_smi_csv.strip().splitlines() if x.strip()]

    if not vals:
        raise ValueError("GPU 사용량 없음")

    return max(vals)


def check_resources(mem_avail_kib: int, gpu_used_mib, min_mem_gib=DEFAULT_MIN_MEM_GIB,
                    max_gpu_used_mib=DEFAULT_MAX_GPU_USED_MIB) -> list:
    """시작 거부 사유 목록(비면 시작 가능). GPU 값을 못 읽으면(None) 거부."""
    reasons = []

    if mem_avail_kib < min_mem_gib * 1024 * 1024:
        reasons.append(f"MemAvailable {mem_avail_kib / 1048576:.1f} GiB < {min_mem_gib} GiB")

    if gpu_used_mib is None:
        reasons.append("GPU 사용량을 읽지 못함")
    elif gpu_used_mib > max_gpu_used_mib:
        reasons.append(f"GPU 사용 {gpu_used_mib} MiB > {max_gpu_used_mib} MiB")

    return reasons


def memory_low(mem_avail_kib: int, stop_mem_gib: float) -> bool:
    """실행 중 종료 조건(사용자 결정: 메모리 하나)."""
    return mem_avail_kib < stop_mem_gib * 1024 * 1024


def reset_latest(latest: dict) -> None:
    """수신 상태 dict의 값을 모두 None으로 비운다(키는 유지)."""
    for k in latest:
        latest[k] = None


def other_gz_processes(ps_lines: list, own_pids: set) -> list:
    """`ps -eo pid,args` 줄에서 남의 `gz sim` 프로세스(경고용)."""
    out = []

    for line in ps_lines:
        parts = line.strip().split(None, 1)

        if len(parts) != 2 or not parts[0].isdigit():
            continue

        pid, args = int(parts[0]), parts[1]

        if pid in own_pids:
            continue

        if "gz sim" in args or "gz-sim" in args.split(" ")[0]:
            out.append(line.strip())

    return out


def launch_group(cmd: list, env: dict, log_path: Path) -> subprocess.Popen:
    """새 세션(=새 프로세스 그룹)으로 실행. 반환된 Popen의 pid가 곧 PGID.

    로그 파일은 자식이 복제한 디스크립터로 계속 쓰며, 부모 쪽 핸들은 Popen 직후(실패해도) 닫는다.
    """
    with open(log_path, "w") as log:
        return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def stop_group(proc: subprocess.Popen, grace_s=5.0) -> bool:
    """자기가 띄운 프로세스 그룹만 종료하고 True를 돌려준다.

    이미 끝난 프로세스면 바로 True. PGID가 proc.pid와 다르거나 이 프로세스의 그룹이면
    종료하지 않고 RuntimeError를 낸다. SIGTERM 뒤 `grace_s` 안에 끝나지 않으면 SIGKILL.
    """
    if proc.poll() is not None:
        return True

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return True

    if pgid != proc.pid or pgid == os.getpgrp():
        raise RuntimeError(f"자기 그룹이 아님(pid {proc.pid}, pgid {pgid}) — 종료하지 않음")

    os.killpg(pgid, signal.SIGTERM)

    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGKILL)
        proc.wait(timeout=grace_s)

    return True


# ---------------------------------------------------------------- 서비스 요청
def pose_vector_request(parts: list) -> str:
    """18개 슬롯 전부의 pose(protobuf 텍스트). 표본에 없는 슬롯은 카메라 뒤 바닥(PARK)에 줄지어 둔다."""
    placed = {p["slot"]: p for p in parts}
    items = []

    for i, slot in enumerate(scene.slot_table()):
        if slot in placed:
            x, y, z = placed[slot]["xyz"]
            r, pch, yw = placed[slot]["rpy"]
        else:
            x, y, z = scene.PARK[0], scene.PARK[1] + 0.1 * i, scene.PARK[2]
            r = pch = yw = 0.0

        qx, qy, qz, qw = scene.rpy_to_quat(r, pch, yw)
        items.append(f'pose {{ name: "{slot}" position {{ x: {x:.6f} y: {y:.6f} z: {z:.6f} }} '
                     f'orientation {{ x: {qx:.8f} y: {qy:.8f} z: {qz:.8f} w: {qw:.8f} }} }}')

    return " ".join(items)


def set_pose_cmd(req: str, world=scene.WORLD_NAME, timeout_ms=3000) -> list:
    """`set_pose_vector` 서비스를 부르는 `gz service` 명령 인자 목록."""
    return ["gz", "service", "-s", f"/world/{world}/set_pose_vector", "--reqtype", "gz.msgs.Pose_V",
            "--reptype", "gz.msgs.Boolean", "--timeout", str(timeout_ms), "--req", req]


# ---------------------------------------------------------------- 라벨 영상 → YOLO
def label_channel(seg: np.ndarray):
    """(라벨 2차원 배열, 모드). semantic이면 세 채널이 같고, panoptic이면 셋째 채널이 라벨."""
    if seg.ndim == 2:
        return seg, "single"

    if seg.shape[2] >= 3 and np.array_equal(seg[:, :, 0], seg[:, :, 2]) and np.array_equal(seg[:, :, 1], seg[:, :, 2]):
        return seg[:, :, 0], "semantic"

    return seg[:, :, 2], "panoptic"


def part_polygon(labels: np.ndarray, label: int, min_area_px=4.0):
    """라벨 값의 가장 큰 외곽 윤곽 다각형(px)과 정보. 보이지 않으면 polygon=None."""
    mask = (labels == label).astype(np.uint8)
    area = int(mask.sum())

    if area == 0:
        return None, {"visible": False, "area_px": 0, "fragments": 0}

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big = max(contours, key=cv2.contourArea)
    info = {"visible": True, "area_px": area, "fragments": len(contours)}

    if cv2.contourArea(big) < min_area_px or len(big) < 3:
        info["too_small"] = True
        return None, info

    poly = cv2.approxPolyDP(big, 0.7, True).reshape(-1, 2).astype(float)

    if len(poly) < 3:
        poly = big.reshape(-1, 2).astype(float)

    (_, _), (w, h), _ = cv2.minAreaRect(poly.astype(np.float32))
    info["short_px"] = round(float(min(w, h)), 2)
    info["long_px"] = round(float(max(w, h)), 2)
    return poly, info


def yolo_line(cls: int, poly_px: np.ndarray, width: int, height: int) -> str:
    """화소 다각형을 YOLO 분할 라벨 한 줄로 바꾼다([0, 1]로 자름)."""
    norm = np.clip(np.asarray(poly_px, float) / [width, height], 0.0, 1.0)
    return f"{cls} " + " ".join(f"{v:.6f}" for v in norm.ravel())


def labels_for_sample(seg: np.ndarray, parts: list, min_area_px=4.0):
    """(YOLO 줄 목록, 부품별 manifest 정보, 모드, 계획에 없는 라벨 값 목록)."""
    labels, mode = label_channel(seg)
    h, w = labels.shape[:2]
    lines, infos = [], []

    for p in parts:
        poly, info = part_polygon(labels, p["label"], min_area_px)
        rec = {k: p[k] for k in ("slot", "label", "cls", "yolo_cls", "xyz", "rpy", "cam_depth_m", "expected_px")}
        rec.update(info)

        if poly is not None:
            lines.append(yolo_line(p["yolo_cls"], poly, w, h))
            rec["labeled"] = True
        else:
            rec["labeled"] = False

        infos.append(rec)

    unknown = sorted(set(np.unique(labels).tolist()) - {0} - {p["label"] for p in parts})
    return lines, infos, mode, unknown


def dataset_yaml() -> str:
    """평가 세트용 dataset.yaml 내용(train·val·test 모두 images/test)."""
    return ("# Gazebo 평가 세트(평가 전용, 학습에 쓰지 않음). gz_eval_capture.py가 생성\n"
            "train: images/test\nval: images/test\ntest: images/test\n"
            "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))


# ---------------------------------------------------------------- 실행
def _gz_version(env) -> str:
    try:
        return subprocess.run(["gz", "sim", "--versions"], env=env, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


def preflight(min_mem_gib, max_gpu_used_mib) -> list:
    """메모리·GPU를 확인해 시작 거부 사유 목록을 돌려준다. 다른 gz sim 프로세스는 stderr 경고만 한다."""
    mem = parse_meminfo_available_kib(Path("/proc/meminfo").read_text())

    try:
        gpu = parse_gpu_used_mib(subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout)
    except Exception:  # noqa: BLE001
        gpu = None

    reasons = check_resources(mem, gpu, min_mem_gib, max_gpu_used_mib)
    ps = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()[1:]

    for line in other_gz_processes(ps, {os.getpid()}):
        print(f"[경고] 다른 gz sim 프로세스가 실행 중(건드리지 않음): {line}", file=sys.stderr)

    return reasons


def run(a) -> int:
    """계획의 묶음마다 Gazebo·브리지를 띄워 표본을 수집한다.

    종료 코드: 0 완료, 2 사전 점검 거부, 3 /clock 수신 없음, 4 set_pose_vector 실패, 5 새 프레임 없음,
    6 실행 중 메모리 부족. 어느 경우든 자기가 띄운 프로세스 그룹은 종료한다.
    """
    plan_dir, out = Path(a.plan_dir), Path(a.out)
    plans = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
    partition = a.partition or default_partition()
    repo = Path(__file__).resolve().parents[1]
    env = build_env(os.environ, partition, a.domain_id, str(repo / "simulation" / "models"))
    os.environ.update({"GZ_PARTITION": env["GZ_PARTITION"], "ROS_DOMAIN_ID": env["ROS_DOMAIN_ID"]})

    for d in ("images/test", "labels/test", "logs"):
        (out / d).mkdir(parents=True, exist_ok=True)

    (out / "dataset.yaml").write_text(dataset_yaml(), encoding="utf-8")
    version = _gz_version(env)

    import rclpy  # 지연 import(시험은 ROS 없이)
    from rclpy.node import Node
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import Image

    rclpy.init()
    latest = {"rgb": None, "seg": None, "clock": None}
    node = None

    def new_node(batch_no):
        """묶음마다 구독 노드를 새로 만든다.

        이전 묶음 브리지가 남긴 수신 큐의 메시지를 버리기 위해서다. 2026-11-13 수집에서 이전 /clock이
        큐에서 늦게 처리돼 기준 시각이 높아지는 문제가 있었다.
        """
        n = Node(f"gz_eval_capture_b{batch_no:04d}")
        n.create_subscription(Image, RGB_TOPIC, lambda m: latest.__setitem__("rgb", m), 2)
        n.create_subscription(Image, SEG_TOPIC, lambda m: latest.__setitem__("seg", m), 2)
        n.create_subscription(Clock, "/clock", lambda m: latest.__setitem__("clock", m), 10)
        return n

    def stamp_s(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def to_array(msg):
        ch = 3 if msg.encoding in ("rgb8", "bgr8") else 1
        return np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, ch)

    manifest = open(out / "manifest.jsonl", "a", encoding="utf-8")

    try:
        for b in plans[a.first_batch:a.first_batch + a.max_batches if a.max_batches else None]:
            reasons = preflight(a.min_mem_gib, a.max_gpu_used_mib)

            if reasons:
                print(f"[거부] 묶음 {b['batch']} 시작하지 않음: {reasons}", file=sys.stderr)
                return 2

            world = plan_dir / f"world_b{b['batch']:04d}.sdf"
            gz = launch_group(["gz", "sim", "-s", "-r", "--headless-rendering", str(world)], env,
                              out / "logs" / f"gz_b{b['batch']:04d}.log")
            br = launch_group(["ros2", "run", "ros_gz_bridge", "parameter_bridge",
                               f"{RGB_TOPIC}@sensor_msgs/msg/Image[gz.msgs.Image",
                               f"{SEG_TOPIC}@sensor_msgs/msg/Image[gz.msgs.Image",
                               "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"], env,
                              out / "logs" / f"bridge_b{b['batch']:04d}.log")

            # 묶음마다 수신 상태를 비운다(새 월드는 시뮬레이션 시각이 0부터 다시 시작한다).
            # 비우지 않으면 이전 묶음의 /clock으로 시작 대기를 건너뛰고 기준 시각이 너무 높아진다
            # (2026-11-13 수집 첫 시도에서 실패한 원인).
            if node is not None:
                node.destroy_node()

            reset_latest(latest)
            node = new_node(b["batch"])

            try:
                t_end = time.monotonic() + a.startup_timeout_s

                while latest["clock"] is None and time.monotonic() < t_end:
                    rclpy.spin_once(node, timeout_sec=0.2)

                if latest["clock"] is None:
                    print(f"[실패] 묶음 {b['batch']}: /clock 수신 없음", file=sys.stderr)
                    return 3

                for s in b["samples"]:
                    if memory_low(parse_meminfo_available_kib(Path("/proc/meminfo").read_text()), a.stop_mem_gib):
                        print(f"[중지] MemAvailable < {a.stop_mem_gib} GiB — 자기 그룹만 종료", file=sys.stderr)
                        return 6

                    req = pose_vector_request(s["parts"])
                    r = subprocess.run(set_pose_cmd(req), env=env, capture_output=True, text=True, timeout=30)

                    if "data: true" not in r.stdout:
                        print(f"[실패] set_pose_vector: {r.stdout} {r.stderr}", file=sys.stderr)
                        return 4

                    clk = latest["clock"].clock
                    t0 = clk.sec + clk.nanosec * 1e-9 + a.settle_s
                    t_end = time.monotonic() + a.frame_timeout_s

                    while time.monotonic() < t_end:
                        rclpy.spin_once(node, timeout_sec=0.1)
                        if (latest["rgb"] is not None and latest["seg"] is not None
                                and stamp_s(latest["rgb"]) >= t0 and stamp_s(latest["seg"]) >= t0):
                            break
                    else:
                        st = {k: (None if latest[k] is None else (stamp_s(latest[k]) if k != "clock" else
                                  latest[k].clock.sec + latest[k].clock.nanosec * 1e-9)) for k in latest}
                        print(f"[실패] 묶음 {b['batch']} 표본 {s['index']}: 새 프레임 없음 (기준 {t0:.3f}, 마지막 {st})",
                              file=sys.stderr)
                        return 5

                    rgb, seg = to_array(latest["rgb"]), to_array(latest["seg"])
                    sid = f"gz_b{b['batch']:04d}_s{s['index']:03d}"
                    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if latest["rgb"].encoding == "rgb8" else rgb
                    cv2.imwrite(str(out / "images/test" / f"{sid}.png"), img)

                    if a.save_seg:  # 확인용 원본 라벨 영상(무손실)
                        (out / "seg_raw").mkdir(exist_ok=True)
                        np.save(out / "seg_raw" / f"{sid}.npy", seg)

                    lines, infos, mode, unknown = labels_for_sample(seg, s["parts"], a.min_area_px)
                    (out / "labels/test" / f"{sid}.txt").write_text("".join(x + "\n" for x in lines), encoding="utf-8")
                    manifest.write(json.dumps({
                        "sample_id": sid, "batch": b["batch"], "seed": b["seed"], "cam_xyz": b["cam_xyz"],
                        "cam_rpy": b["cam_rpy"], "light_diffuse": b["light_diffuse"],
                        "rgb_stamp_s": stamp_s(latest["rgb"]), "seg_stamp_s": stamp_s(latest["seg"]),
                        "seg_mode": mode, "unknown_labels": unknown, "gz_version": version,
                        "gz_partition": partition, "parts": infos}, ensure_ascii=False) + "\n")
                    manifest.flush()
            finally:
                stop_group(br)
                stop_group(gz)
    finally:
        manifest.close()

        if node is not None:
            node.destroy_node()

        rclpy.shutdown()

    return 0


def main(argv=None) -> int:
    """CLI 진입점. `--yes`가 없으면 사전 점검 결과만 출력하고 0, 있으면 `run`의 종료 코드."""
    p = argparse.ArgumentParser(description="Gazebo 평가 세트 수집(--yes 없이는 실행하지 않음)")
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out", default="data/derived/gz_eval_v1")
    p.add_argument("--partition", default=None, help="GZ_PARTITION(기본: eyetoarm_gz_eval_<pid>)")
    p.add_argument("--domain-id", type=int, default=DEFAULT_DOMAIN_ID)
    p.add_argument("--min-mem-gib", type=float, default=DEFAULT_MIN_MEM_GIB)
    p.add_argument("--max-gpu-used-mib", type=int, default=DEFAULT_MAX_GPU_USED_MIB)
    p.add_argument("--first-batch", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=0, help="0이면 전부")
    p.add_argument("--settle-s", type=float, default=0.5)
    p.add_argument("--startup-timeout-s", type=float, default=60.0)
    p.add_argument("--frame-timeout-s", type=float, default=20.0)
    p.add_argument("--min-area-px", type=float, default=4.0)
    p.add_argument("--stop-mem-gib", type=float, default=2.0, help="실행 중 MemAvailable이 이보다 작으면 중지")
    p.add_argument("--save-seg", action="store_true", help="원본 분할 라벨 영상을 seg_raw/*.npy로도 저장(확인용)")
    p.add_argument("--yes", action="store_true", help="사용자 확인 뒤에만 붙인다. 없으면 사전 점검만 한다")
    a = p.parse_args(argv)

    if not a.yes:
        reasons = preflight(a.min_mem_gib, a.max_gpu_used_mib)
        print(json.dumps({"would_run": False, "preflight_reasons": reasons,
                          "partition": a.partition or default_partition(), "domain_id": a.domain_id},
                         ensure_ascii=False))
        return 0

    return run(a)


if __name__ == "__main__":
    sys.exit(main())
