"""공식 사전학습 가중치 확인·준비. ultralytics 자동 내려받기를 쓰지 않는다.

있으면 sha256·크기를 대조하고(불일치면 덮어쓰지 않음), 없으면 `allow_download=True`일 때만 `.part`로 받아
맞을 때만 `os.replace`한다. 실제 내려받기는 부모가 전체 기한을 쥔 자식 프로세스에서 한다(소켓 timeout은
요청 전체 시간을 보장하지 않으므로). `_stream_download`는 가짜 응답 시험용이다.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass

from eyetoarm.training.config import Candidate, thaw

_CHUNK = 1 << 20


class WeightsError(RuntimeError):
    """가중치 확인·준비 실패. `child`에는 자식 내려받기 정리 정보(pid·kill·reaped·rc)가 있을 수 있다."""

    def __init__(self, message: str, child: dict | None = None) -> None:
        """메시지와 자식 정리 정보."""
        super().__init__(message)
        self.child = child


# 자식 내려받기 코드(표준 라이브러리만). 인자: url dest max_bytes io_timeout. 종료 코드 0 성공, 3 크기 초과
_CHILD_CODE = r"""
import sys, urllib.request
url, dest, max_bytes, io_timeout = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
total = 0
with urllib.request.urlopen(url, timeout=io_timeout) as r, open(dest, "xb") as f:
    n = r.headers.get("Content-Length")
    if n is not None and n.isdigit() and int(n) > max_bytes:
        sys.stderr.write(f"Content-Length {n} exceeds allowed {max_bytes} bytes"); sys.exit(3)
    while True:
        c = r.read(1 << 16)
        if not c:
            break
        total += len(c)
        if total > max_bytes:
            sys.stderr.write(f"download size {total} exceeds allowed {max_bytes} bytes"); sys.exit(3)
        f.write(c)
"""


def _kill_and_reap(p, info: dict, reap_s: float) -> None:
    """자식 그룹 SIGKILL → 상한 있는 회수 한 번. 결과를 info에 쓴다(예외를 내지 않음)."""
    try:
        os.killpg(p.pid, signal.SIGKILL)
        info["kill"] = "killpg"
    except OSError as exc:
        try:
            p.kill()
            info["kill"] = f"kill ({type(exc).__name__})"
        except OSError as exc2:
            info["kill"] = f"failed ({type(exc2).__name__})"
    try:
        p.communicate(timeout=reap_s)
        info["reaped"] = True
    except Exception:
        info["reaped"] = p.poll() is not None
        if p.stderr is not None:
            try:
                p.stderr.close()
            except OSError:
                pass
    info["rc"] = p.returncode


def download_with_deadline(url: str, dest: str, *, max_bytes: int, deadline_s: float, io_timeout_s: float,
                           reap_s: float = 5.0, child_argv: list | None = None, should_stop=None,
                           poll_s: float = 0.2, on_child=None) -> dict:
    """자식 프로세스로 받는다. 기한을 넘으면 그룹 SIGKILL → 상한 있는 회수 한 번. 실패는 WeightsError.

    `should_stop`이 참이면 자식을 정리하고 WeightsError, 그 밖의 예외도 자식을 정리한 뒤 다시 올린다.
    """
    argv = child_argv or [sys.executable, "-I", "-c", _CHILD_CODE, url, dest, str(max_bytes), str(io_timeout_s)]
    t0 = time.monotonic()
    p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                         start_new_session=True)
    info = {"pid": p.pid, "deadline_s": deadline_s, "io_timeout_s": io_timeout_s, "kill": None, "reaped": None,
            "rc": None}
    err = b""

    if on_child is not None:
        on_child(dict(info))

    try:
        while True:
            left = deadline_s - (time.monotonic() - t0)
            if left <= 0:
                _kill_and_reap(p, info, reap_s)
                info["elapsed_s"] = round(time.monotonic() - t0, 3)
                raise WeightsError(f"download deadline {deadline_s} s exceeded (child pid {p.pid} killed, "
                                   f"reaped={info['reaped']})", child=info)
            if should_stop is not None and should_stop():
                _kill_and_reap(p, info, reap_s)
                info["elapsed_s"] = round(time.monotonic() - t0, 3)
                raise WeightsError(f"download stop_requested (child pid {p.pid} killed, reaped={info['reaped']})",
                                   child=info)
            try:
                _, err = p.communicate(timeout=min(left, poll_s))
                break
            except subprocess.TimeoutExpired:
                continue
    except WeightsError:
        raise
    except BaseException as exc:   # KeyboardInterrupt 등: 자식을 정리하고 원래 예외를 다시 올린다
        if p.poll() is None or info["kill"] is None:
            _kill_and_reap(p, info, reap_s)
        info["elapsed_s"] = round(time.monotonic() - t0, 3)
        try:
            exc.child = info
        except AttributeError:
            pass
        raise

    info.update(rc=p.returncode, reaped=True, elapsed_s=round(time.monotonic() - t0, 3))

    if p.returncode != 0:
        tail = (err or b"").decode(errors="replace").strip().splitlines()
        raise WeightsError(f"download child failed rc={p.returncode}: {tail[-1] if tail else ''}", child=info)

    return info


@dataclass(frozen=True)
class WeightsCheck:
    """확인 결과: status ok·missing·mismatch."""

    status: str
    path: str
    actual_sha256: str | None
    actual_size: int | None


def _sha256(path: str) -> str:
    """파일 sha256."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def check_weights(cand: Candidate, weights_dir: str) -> WeightsCheck:
    """파일 존재와 sha256·크기 일치를 확인한다."""
    path = os.path.join(weights_dir, cand.weights_file)

    if not os.path.isfile(path):
        return WeightsCheck("missing", path, None, None)

    size, sha = os.path.getsize(path), _sha256(path)
    status = "ok" if (sha == cand.sha256 and size == cand.size_bytes) else "mismatch"

    return WeightsCheck(status, path, sha, size)


def _stream_download(url: str, dest: str, *, max_bytes: int, deadline_s: float, io_timeout_s: float,
                     opener=None, clock=time.monotonic) -> int:
    """응답을 조각으로 받아 dest에 쓴다. 크기·전체 기한을 넘으면 WeightsError. 받은 바이트 수를 돌려준다."""
    start = clock()
    total = 0
    open_url = opener or (lambda u, timeout: urllib.request.urlopen(u, timeout=timeout))

    with open_url(url, timeout=io_timeout_s) as r, open(dest, "xb") as f:
        length = (getattr(r, "headers", None) or {}).get("Content-Length")

        if length is not None and str(length).isdigit() and int(length) > max_bytes:
            raise WeightsError(f"Content-Length {length} exceeds allowed {max_bytes} bytes")

        while True:
            if clock() - start > deadline_s:
                raise WeightsError(f"download deadline {deadline_s} s exceeded after {total} bytes")

            chunk = r.read(_CHUNK)

            if not chunk:
                return total

            total += len(chunk)

            if total > max_bytes:
                raise WeightsError(f"download size {total} exceeds allowed {max_bytes} bytes")

            f.write(chunk)


def _write_provenance(weights_dir: str, name: str, record: dict) -> None:
    """provenance.json에 파일 항목을 원자 교체로 반영한다."""
    path = os.path.join(weights_dir, "provenance.json")
    doc = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    doc[name] = record
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)


def prepare_weights(cand: Candidate, weights_dir: str, *, allow_download: bool = False, downloader=None,
                    timeout_s: float = 300.0, io_timeout_s: float = 30.0, opener=None,
                    clock=time.monotonic, should_stop=None) -> dict:
    """가중치를 준비하고 provenance 항목을 돌려준다. 실패는 WeightsError."""
    os.makedirs(weights_dir, exist_ok=True)
    check = check_weights(cand, weights_dir)
    record = {"sha256": cand.sha256, "size_bytes": cand.size_bytes, "source": thaw(cand.source),
              "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    if check.status == "ok":
        record["method"] = "existing"
        _write_provenance(weights_dir, cand.weights_file, record)
        return record

    if check.status == "mismatch":
        raise WeightsError(f"{check.path} exists but does not match (sha256 {check.actual_sha256}, "
                           f"size {check.actual_size}); not overwriting")

    if not allow_download:
        raise WeightsError(f"{check.path} is missing; pass allow_download to fetch {cand.source['url']}")

    part = check.path + ".part"

    if os.path.exists(part):
        raise WeightsError(f"{part} already exists (previous attempt?); not overwriting")

    try:
        if downloader is not None:
            downloader(cand.source["url"], part, timeout_s)
        elif opener is not None:   # 시험용 가짜 응답(같은 프로세스)
            _stream_download(cand.source["url"], part, max_bytes=cand.size_bytes, deadline_s=timeout_s,
                             io_timeout_s=min(io_timeout_s, timeout_s), opener=opener, clock=clock)
        else:                      # 실제 네트워크: 부모가 전체 기한을 쥔 자식 프로세스
            download_with_deadline(cand.source["url"], part, max_bytes=cand.size_bytes, deadline_s=timeout_s,
                                   io_timeout_s=min(io_timeout_s, timeout_s), should_stop=should_stop)
        sha, size = _sha256(part), os.path.getsize(part)
    except BaseException as exc:   # KeyboardInterrupt 포함: 자기 .part를 지우고 원래 예외를 올린다
        if os.path.exists(part):
            os.remove(part)
        if isinstance(exc, WeightsError) or not isinstance(exc, Exception):
            raise
        raise WeightsError(f"download failed: {type(exc).__name__}: {exc}") from exc

    if sha != cand.sha256 or size != cand.size_bytes:
        os.remove(part)
        raise WeightsError(f"downloaded file mismatch: sha256 {sha} size {size}")

    os.replace(part, check.path)
    record["method"] = "downloaded"
    _write_provenance(weights_dir, cand.weights_file, record)

    return record


__all__ = ["WeightsCheck", "WeightsError", "check_weights", "download_with_deadline", "prepare_weights"]
