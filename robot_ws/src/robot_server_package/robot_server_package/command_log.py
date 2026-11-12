"""TCP 명령의 추가 전용 JSONL 기록과 저장 확정을 담당합니다."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


class CommandLogError(RuntimeError):
    """명령 로그를 신뢰하거나 저장할 수 없습니다."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"중복 JSON 키: {key}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(f"허용되지 않는 JSON 숫자: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("유한하지 않은 실수")
    return parsed


class CommandLog:
    """TCP 1.1 명령 상태를 JSONL + fsync로 기록합니다."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if self.path.as_posix().startswith("/mnt/"):
            raise CommandLogError("명령 로그는 WSL 리눅스 파일 시스템에 있어야 합니다")
        self._seq = 0
        self._confirmed_offset = 0
        self._last_record_offset = 0
        self._needs_rollback = False

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    @property
    def last_record_offset(self) -> int:
        """마지막으로 저장 확정된 로그 레코드의 시작 오프셋입니다."""
        return self._last_record_offset

    @staticmethod
    def initialize(path: str | Path) -> None:
        """명시적 최초 초기화용. 기존 파일은 덮어쓰지 않습니다."""
        target = Path(path).expanduser().resolve()
        if target.as_posix().startswith("/mnt/"):
            raise CommandLogError("명령 로그는 WSL 리눅스 파일 시스템에 있어야 합니다")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise CommandLogError(f"로그 파일이 이미 있습니다: {target}") from error
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        dir_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def load(self) -> list[dict[str, Any]]:
        """전체 JSONL을 점검하고 순번이 연속인 기록만 반환합니다."""
        return [record for _, record in self.load_with_offsets()]

    def load_with_offsets(self) -> list[tuple[int, dict[str, Any]]]:
        """전체 로그를 검증하고 각 JSONL 레코드의 시작 오프셋과 함께 반환합니다."""
        if not self.path.is_file():
            raise CommandLogError("명령 로그 파일이 없습니다. 명시적 초기화가 필요합니다")
        try:
            data = self.path.read_bytes()
        except OSError as error:
            raise CommandLogError(f"명령 로그를 읽을 수 없습니다: {error}") from error
        if data and not data.endswith(b"\n"):
            raise CommandLogError("명령 로그 마지막 줄이 LF로 끝나지 않습니다")

        records: list[tuple[int, dict[str, Any]]] = []
        expected_seq = 1
        offset = 0
        lines = data.split(b"\n") if data else []
        if data.endswith(b"\n"):
            lines.pop()
        for line_number, raw_line in enumerate(lines, start=1):
            try:
                record = json.loads(
                    raw_line.decode("utf-8"),
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_reject_non_finite_constant,
                    parse_float=_parse_finite_float,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                raise CommandLogError(f"명령 로그 {line_number}행 JSON 오류") from error
            if not isinstance(record, dict):
                raise CommandLogError(f"명령 로그 {line_number}행은 객체가 아닙니다")
            if type(record.get("seq")) is not int or record["seq"] != expected_seq:
                raise CommandLogError(f"명령 로그 seq 누락 또는 중복: {line_number}행")
            if not isinstance(record.get("event"), str) or not isinstance(record.get("command_id"), str):
                raise CommandLogError(f"명령 로그 필수 필드 누락: {line_number}행")
            if record["event"] == "REGISTER":
                request = record.get("request")
                status = record.get("status")
                if not isinstance(request, dict) or not isinstance(status, dict):
                    raise CommandLogError(f"REGISTER 요청 또는 상태 기록 누락: {line_number}행")
                if request.get("command_id") != record["command_id"]:
                    raise CommandLogError(f"REGISTER command_id 불일치: {line_number}행")
                if status.get("command_id") != record["command_id"] or status.get("state") != "ACK":
                    raise CommandLogError(f"REGISTER ACK 상태 불일치: {line_number}행")
            if record["event"] == "STATE":
                status = record.get("status")
                if not isinstance(status, dict):
                    raise CommandLogError(f"STATE 응답 기록 누락: {line_number}행")
                if status.get("command_id") != record["command_id"]:
                    raise CommandLogError(f"STATE command_id 불일치: {line_number}행")
                if status.get("state") not in {"PLANNING", "MOVING", "REACHED", "FAILED"}:
                    raise CommandLogError(f"STATE state 값이 올바르지 않습니다: {line_number}행")
                if status.get("state") == "MOVING":
                    goal_id = record.get("goal_id")
                    if not isinstance(goal_id, str) or not goal_id:
                        raise CommandLogError(f"MOVING goal_id 누락: {line_number}행")
            if record["event"] not in {"REGISTER", "STATE"}:
                raise CommandLogError(f"알 수 없는 명령 로그 event: {line_number}행")
            records.append((offset, record))
            offset += len(raw_line) + 1
            expected_seq += 1

        self._seq = expected_seq - 1
        self._confirmed_offset = len(data)
        self._last_record_offset = records[-1][0] if records else 0
        return records

    def read_record_at(self, offset: int) -> dict[str, Any]:
        """색인 오프셋에 있는 한 레코드를 읽습니다."""
        if type(offset) is not int or offset < 0:
            raise CommandLogError("로그 오프셋이 올바르지 않습니다")
        try:
            with self.path.open("rb") as stream:
                stream.seek(offset)
                line = stream.readline()
            record = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_non_finite_constant,
                parse_float=_parse_finite_float,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise CommandLogError(f"색인된 로그 레코드를 읽을 수 없습니다: {error}") from error
        if not line.endswith(b"\n") or not isinstance(record, dict):
            raise CommandLogError("색인된 로그 레코드가 손상되었습니다")
        return record

    def _verify_confirmed_size(self) -> None:
        try:
            actual_size = self.path.stat().st_size
        except OSError as error:
            raise CommandLogError(f"명령 로그 파일을 확인할 수 없습니다: {error}") from error
        if actual_size != self._confirmed_offset:
            raise CommandLogError(
                f"명령 로그 크기가 저장 확정 위치와 다릅니다: {actual_size} != {self._confirmed_offset}"
            )

    def append(self, event: str, command_id: str, **fields: Any) -> dict[str, Any]:
        """한 레코드를 단일 append 후 fsync해 저장 확정합니다."""
        if not self.path.is_file():
            raise CommandLogError("명령 로그 파일이 초기화되지 않았습니다")
        record = {"seq": self._seq + 1, "event": event, "command_id": command_id, **fields}
        try:
            encoded = (
                json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise CommandLogError(f"로그 레코드 직렬화 실패: {error}") from error

        if self._needs_rollback:
            self._rollback_unconfirmed_tail()
        self._verify_confirmed_size()
        try:
            record_offset = self._confirmed_offset
            with self.path.open("ab") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
                self._confirmed_offset = stream.tell()
            self._seq += 1
            self._last_record_offset = record_offset
        except OSError as error:
            self._needs_rollback = True
            self._rollback_unconfirmed_tail()
            raise CommandLogError(f"명령 로그 저장 확정 실패: {error}") from error
        return record

    def probe(self) -> None:
        """파일 권한과 같은 파일시스템의 실제 쓰기·fsync 가능 여부를 확인합니다."""
        if not self.path.is_file():
            raise CommandLogError("명령 로그 파일이 초기화되지 않았습니다")
        if self._needs_rollback:
            self._rollback_unconfirmed_tail()
        self._verify_confirmed_size()
        fd: int | None = None
        probe_path: str | None = None
        try:
            # 대상 로그 파일을 append 모드로 열어 파일 권한을 확인합니다.
            with self.path.open("ab") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            # 임시 옆 파일에 실제 바이트를 기록해 디스크 공간과 디렉터리를 확인합니다.
            fd, probe_path = tempfile.mkstemp(prefix=".command_log_probe_", dir=self.path.parent)
            os.write(fd, b"\0")
            os.fsync(fd)
        except OSError as error:
            raise CommandLogError(f"명령 로그 시험 쓰기 실패: {error}") from error
        finally:
            if fd is not None:
                os.close(fd)
            if probe_path is not None:
                try:
                    os.unlink(probe_path)
                    dir_fd = os.open(self.path.parent, os.O_RDONLY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                except OSError as error:
                    raise CommandLogError(f"명령 로그 시험 파일 정리 실패: {error}") from error
    def _rollback_unconfirmed_tail(self) -> None:
        try:
            size = self.path.stat().st_size
            if size < self._confirmed_offset:
                raise CommandLogError("저장 확정된 로그 앞부분이 외부에서 사라졌습니다")
            with self.path.open("r+b") as stream:
                stream.truncate(self._confirmed_offset)
                stream.flush()
                os.fsync(stream.fileno())
            self._needs_rollback = False
        except OSError as error:
            self._needs_rollback = True
            raise CommandLogError(f"미확정 로그 조각 정리 실패: {error}") from error
