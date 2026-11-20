"""D 내부 표시 예제 전용. C 전송 규격이나 A/B 공통 자료형이 아니다."""

import copy
import math
import time

from PySide6.QtCore import QObject, Signal


class DisplayModel(QObject):
    """호출자가 공급한 식별자로만 화면과 결과를 맞춘다. 시각은 수신 경과 감시에만 쓴다."""

    changed = Signal()

    def __init__(self, parent=None, *, stale_seconds=3.0, clock=time.monotonic):
        super().__init__(parent)
        self.clock, self.stale_seconds = clock, stale_seconds
        self.connected = False
        self.frames = {}
        self.frame_marks = {}
        self.origins = {}
        self.result = None
        self.high_water = {}
        self.result_keys = {}
        self.invalid_jobs = {}
        self.latest_jobs = {}
        self.blocked_sources = set()
        self.run_id = None
        self.link_source_id = None
        self.generations = {}
        self.active_configs = {}
        self.mode = "raw"
        self.reason = "결과 미수신"

    def set_connected(self, connected):
        self.connected = bool(connected)
        if not connected:
            self.frames.clear()
            self.result = None
            self.reason = "표시 연결 끊김"
        self.changed.emit()

    def set_mode(self, mode):
        if mode not in ("raw", "detect", "center"):
            raise ValueError("unknown link mode")
        self.mode = mode
        self.changed.emit()

    def begin_session(self, run_id, link_source_id):
        """호출자가 명시한 새 세션만 순서 기록을 초기화한다. 재접속은 초기화하지 않는다."""
        if not isinstance(run_id, str) or not run_id or not isinstance(link_source_id, str) or not link_source_id or run_id == self.run_id:
            return False
        self.run_id = run_id
        self.link_source_id = link_source_id
        self._clear_session("새 세션 결과 대기")
        return True

    def reset_session(self, reason):
        """다른 서버로 옮길 때만 부른다. 이전 서버의 run_id·순서·결과 기록을 모두 버린다(같은 주소 재접속은 부르지 않는다)."""
        self.run_id = self.link_source_id = None
        self._clear_session(reason)

    def reset_source(self, source_id, reason):
        """한 소스의 순서·세대·설정·프레임·결과만 버린다. 링크 표시 종류를 실제로 바꿀 때만 부른다.

        raw·detect·center 는 같은 source_id 에 서로 다른 순번·세대·설정을 쓰므로, 전환 뒤 새 종류가 이전 기록에 막히지 않게 한다.
        다른 소스와 세션(run_id)은 그대로다."""
        for table in (self.generations, self.active_configs, self.high_water, self.result_keys,
                      self.invalid_jobs, self.latest_jobs, self.origins):
            table.pop(source_id, None)
        self.blocked_sources.discard(source_id)
        self.frames = {k: v for k, v in self.frames.items() if v["source_id"] != source_id}
        self.frame_marks = {k: v for k, v in self.frame_marks.items() if v["source_id"] != source_id}
        if self.result and self.result["source_id"] == source_id:
            self.result = None
        self.reason = reason
        self.changed.emit()

    def _clear_session(self, reason):
        self.generations.clear()
        self.active_configs.clear()
        self.high_water.clear()
        self.result_keys.clear()
        self.invalid_jobs.clear()
        self.latest_jobs.clear()
        self.blocked_sources.clear()
        self.frames.clear()
        self.frame_marks.clear()
        self.origins.clear()
        self.result = None
        self.reason = reason
        self.changed.emit()

    def _config_conflict(self, source, generation, config):
        """같은 source·세대의 설정은 고정한다. 설정 변경은 세대 증가로만 한다."""
        active = self.active_configs.get(source)
        return self.generations.get(source) == generation and active is not None and active != config

    def _generation(self, source, generation):
        old = self.generations.get(source)
        if old is not None and generation < old:
            return False
        if old is None or generation > old:
            self.generations[source] = generation
            self.active_configs.pop(source, None)
            self.high_water.pop(source, None)
            self.result_keys.pop(source, None)
            self.invalid_jobs.pop(source, None)
            self.latest_jobs.pop(source, None)
            self.blocked_sources.discard(source)
            self.frames = {k: v for k, v in self.frames.items() if v["source_id"] != source}
            self.frame_marks = {k: v for k, v in self.frame_marks.items() if v["source_id"] != source}
            if self.result and self.result["source_id"] == source:
                self.result = None
            self.reason = "새 세대 결과 대기"
        return True

    def advance_generation(self, source_id, *, run_id, generation, order, detector_config_id):
        """알려진 소스의 명시적 세대 전환만 수락한다. 소스 오류와 구분한다."""
        if (not self.connected or not self.run_id or run_id != self.run_id or
                not isinstance(source_id, str) or source_id not in self.generations or
                type(generation) is not int or generation <= self.generations[source_id] or
                type(order) is not int or order < 0 or
                not isinstance(detector_config_id, str) or not detector_config_id):
            return False
        self._generation(source_id, generation)
        self.active_configs[source_id] = detector_config_id
        self.high_water[source_id] = order
        self.changed.emit()
        return True

    def ingest_frame(self, channel, image, *, source_id, image_seq, generation, calibration_id, run_id, capture_generation, detector_config_id, job_seq, variant="raw", confirmed=True):
        """confirmed=False: 연결 직후 캐시일 수 있는 첫 프레임. 보관·표시는 하되 신선(fresh)·결과 일치에는 넣지 않는다."""
        if not self.connected or channel not in ("link", "gripper", "fixed") or image is None or image.isNull():
            return False
        if not isinstance(source_id, str) or not source_id or type(image_seq) is not int or type(generation) is not int:
            return False
        if channel == "link" and source_id != self.link_source_id:
            return False
        if not self.run_id or run_id != self.run_id or not isinstance(detector_config_id, str) or not detector_config_id or any(type(x) is not int or x < 0 for x in (capture_generation, job_seq, image_seq, generation)):
            return False
        if calibration_id is not None and (not isinstance(calibration_id, str) or not calibration_id):
            return False
        if channel == "link" and variant not in ("raw", "detect", "center"):
            return False
        if (channel == "link" and variant != "raw" and self.generations.get(source_id) == generation and
                (source_id in self.blocked_sources or job_seq <= self.invalid_jobs.get(source_id, -1))):
            return False
        origin = self.origins.get(source_id)
        physical = (capture_generation, image_seq)
        if origin and physical < origin["physical"]:
            return False
        if self._config_conflict(source_id, generation, detector_config_id):
            return False
        if not self._generation(source_id, generation):
            return False
        self.active_configs.setdefault(source_id, detector_config_id)
        if origin is None or physical > origin["physical"]:
            origin = {"physical": physical, "received": self.clock()}
            self.origins[source_id] = origin
        key = (channel, variant if channel == "link" else "raw")
        old = self.frame_marks.get(key)
        if old and (capture_generation, image_seq, job_seq) <= (old["capture_generation"], old["image_seq"], old["job_seq"]):
            return False
        received = origin["received"]
        self.frames[key] = dict(image=image.copy(), source_id=source_id, image_seq=image_seq,
                                generation=generation, calibration_id=calibration_id, run_id=run_id,
                                capture_generation=capture_generation, detector_config_id=detector_config_id,
                                job_seq=job_seq, received=received, confirmed=confirmed is True)
        self.frame_marks[key] = {k: v for k, v in self.frames[key].items() if k != "image"}
        self.changed.emit()
        return True

    def ingest_result(self, result, *, generation, order):
        """result는 D 전용 사전. B의 mm 출력은 그대로 사용하며 수신 순서 order는 호출자가 제공한다."""
        if not self.connected or not isinstance(result, dict) or type(generation) is not int or generation < 0 or type(order) is not int or order < 0:
            return False
        source = result.get("source_id")
        if source != self.link_source_id or type(result.get("image_seq")) is not int:
            return False
        if result.get("run_id") != self.run_id or not self.run_id or not isinstance(result.get("detector_config_id"), str) or not result["detector_config_id"] or any(type(result.get(x)) is not int or result[x] < 0 for x in ("capture_generation", "job_seq", "image_seq")):
            return False
        if type(result.get("valid")) is not bool:
            return False
        # B는 무효 결과에 [None, None] 좌표를 낸다. 유효 결과의 좌표는 유한 수만 받는다.
        allowed = (int, float) if result["valid"] else (int, float, type(None))
        for key in ("center_px", "center_w_mm"):
            value = result.get(key)
            if value is not None and (not isinstance(value, (tuple, list)) or len(value) != 2 or
                                      any(type(x) not in allowed or (x is not None and not math.isfinite(x)) for x in value)):
                return False
        theta = result.get("theta_deg")
        if theta is not None and (type(theta) not in (int, float) or not math.isfinite(theta)):
            return False
        if (type(result.get("theta_valid", False)) is not bool or
                (theta is None) != (result.get("theta_valid", False) is False)):
            return False
        calibration = result.get("calibration_id")
        if calibration is not None and (not isinstance(calibration, str) or not calibration):
            return False
        if self._config_conflict(source, generation, result["detector_config_id"]):
            return False
        if not self._generation(source, generation):
            return False
        self.active_configs.setdefault(source, result["detector_config_id"])
        if source in self.blocked_sources or result["job_seq"] <= self.invalid_jobs.get(source, -1):
            return False
        if result["job_seq"] <= self.latest_jobs.get(source, -1):
            return False
        if order <= self.high_water.get(source, -1):
            return False
        identity = tuple(result[k] for k in ("run_id", "capture_generation", "image_seq", "detector_config_id", "job_seq"))
        if self.result_keys.get(source) == identity:
            return False
        self.high_water[source] = order
        self.result_keys[source] = identity
        self.latest_jobs[source] = result["job_seq"]
        self.result = dict(copy.deepcopy(result), generation=generation, received=self.clock())
        self.reason = "프레임·보정 불일치 또는 노후" if result["valid"] else str(result.get("reason") or "결과 무효")
        self.changed.emit()
        return True

    def invalidate(self, source_id, *, run_id, generation, order, reason, detector_config_id, job_seq=None):
        """프레임 없는 무효화도 순서를 보존하여 늦은 결과의 부활을 막는다."""
        if not self.connected or not self.run_id or run_id != self.run_id or type(generation) is not int or generation < 0 or type(order) is not int or order < 0 or not isinstance(source_id, str) or not source_id or not isinstance(detector_config_id, str) or not detector_config_id or (job_seq is not None and (type(job_seq) is not int or job_seq < 0)):
            return False
        # 같은 세대의 활성 설정은 결과 유무·단절과 무관하게 유지한다. 새 세대는 새 설정을 연다.
        if self._config_conflict(source_id, generation, detector_config_id):
            return False
        if not self._generation(source_id, generation):
            return False
        self.active_configs.setdefault(source_id, detector_config_id)
        if order <= self.high_water.get(source_id, -1):
            return False
        self.high_water[source_id] = order
        if job_seq is None:
            self.blocked_sources.add(source_id)
        else:
            self.invalid_jobs[source_id] = max(job_seq, self.invalid_jobs.get(source_id, -1))
        self.frames = {key: frame for key, frame in self.frames.items()
                       if not (key[0] == "link" and key[1] != "raw" and frame["source_id"] == source_id and
                               (job_seq is None or frame["job_seq"] <= job_seq))}
        if self.result and self.result.get("source_id") == source_id and (job_seq is None or self.result["job_seq"] <= job_seq):
            self.result = None
        self.reason = str(reason) if reason else "무효화"
        self.changed.emit()
        return True

    def frame(self, channel):
        key = (channel, self.mode if channel == "link" else "raw")
        frame = self.frames.get(key)
        origin = self.origins.get(frame["source_id"]) if frame else None
        if frame and origin and (frame["capture_generation"], frame["image_seq"]) == origin["physical"]:
            return frame
        return None

    def fresh(self, item):
        return bool(self.connected and item and item.get("confirmed", True)
                    and 0 <= self.clock() - item["received"] <= self.stale_seconds)

    def matched_result(self):
        frame, result = self.frame("link"), self.result
        if not self.fresh(frame) or not self.fresh(result) or result.get("valid") is not True:
            return None
        origin = self.origins.get(frame["source_id"])
        if not origin or (frame["capture_generation"], frame["image_seq"]) != origin["physical"]:
            return None
        if not frame["calibration_id"] or frame["calibration_id"] != result.get("calibration_id"):
            return None
        if any(frame[k] != result.get(k) for k in ("source_id", "image_seq", "generation", "run_id", "capture_generation", "detector_config_id", "job_seq")):
            return None
        return result

    def has_matched_position(self):
        """표시용 위치의 일치 여부이며 MOVE 허가 판단이 아니다."""
        result = self.matched_result()
        return bool(result and result.get("center_w_mm") is not None)
