"""Socket-free classification state simulator; values are internal demo data."""

from uuid import uuid4

from PySide6.QtCore import QObject, QTimer, Signal


class DemoSortController(QObject):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.connected = False
        self.synchronized = False
        self._snapshot = None
        self._epoch_floor = 0
        self._timer = QTimer(self, singleShot=True, interval=400)
        self._timer.timeout.connect(self._finish)

    @property
    def snapshot(self):
        return None if self._snapshot is None else self._snapshot.copy()

    def set_connected(self, connected):
        connected = bool(connected)
        if self.connected == connected:
            return
        self.connected = connected
        if not connected:
            self._timer.stop()
            self.synchronized = False
        self.changed.emit()

    def apply_snapshot(self, snapshot, *, resync=False):
        if not self.connected or not isinstance(snapshot, dict):
            return False
        required = {"epoch", "seq", "state", "holding", "job_id", "detail"}
        if (set(snapshot) != required or type(snapshot["epoch"]) is not int
                or snapshot["epoch"] < 0 or type(snapshot["seq"]) is not int or snapshot["seq"] < 0
                or snapshot["state"] not in ("IDLE", "STARTING", "RUNNING", "STOPPING", "STOPPED", "RECOVERY_REQUIRED", "RECOVERING")
                or snapshot["holding"] not in ("DETACHED", "HELD", "HOLD_UNKNOWN")
                or not (snapshot["job_id"] is None or isinstance(snapshot["job_id"], str))
                or not isinstance(snapshot["detail"], str)):
            return False
        previous = self._snapshot
        epoch, seq = snapshot["epoch"], snapshot["seq"]
        if epoch < self._epoch_floor:
            return False
        self._epoch_floor = epoch
        if previous is not None and epoch > previous["epoch"] and not resync:
            self._timer.stop()
            self.synchronized = False
            self.changed.emit()
            return False
        if not self.synchronized and not resync:
            return False
        if previous is not None:
            if epoch == previous["epoch"] and (seq < previous["seq"] or (seq == previous["seq"] and not resync)):
                return False
        if previous is not None and snapshot["epoch"] == previous["epoch"] and snapshot["seq"] == previous["seq"] and snapshot != previous:
            return False
        if self.synchronized and resync and snapshot == previous:
            return True
        # Injected snapshots await their next snapshot; only local requests simulate completion.
        self._timer.stop()
        self._snapshot = snapshot.copy()
        self.synchronized = True
        self.changed.emit()
        return True

    def allowed(self, action):
        if not self.connected or not self.synchronized or self._snapshot is None:
            return False, "연결과 동기화가 필요합니다."
        state, holding = self._snapshot["state"], self._snapshot["holding"]
        if action == "START" and state in {"IDLE", "STOPPED"} and holding == "DETACHED":
            return True, ""
        if action == "STOP" and state in {"STARTING", "RUNNING"}:
            return True, ""
        if action == "RECOVER" and state == "RECOVERY_REQUIRED":
            return True, ""
        return False, "현재 상태에서 실행할 수 없습니다."

    def request(self, action, confirmed=False):
        if not self.allowed(action)[0] or (action == "RECOVER" and not confirmed):
            return False
        self._timer.stop()
        if action == "START":
            self._snapshot.update(state="STARTING", job_id=str(uuid4()), detail="모의 시작 접수")
        elif action == "STOP":
            self._snapshot.update(state="STOPPING", detail="모의 정지 접수")
        else:
            self._snapshot.update(state="RECOVERING", detail="모의 복구 중")
        self._snapshot["seq"] += 1
        self._timer.start()
        self.changed.emit()
        return True

    def _finish(self):
        if not self.connected or not self.synchronized:
            return
        state = self._snapshot["state"]
        if state == "STARTING":
            self._snapshot.update(state="RUNNING", detail="모의 분류 중")
        elif state == "STOPPING":
            next_state = "RECOVERY_REQUIRED" if self._snapshot["holding"] == "HOLD_UNKNOWN" else "STOPPED"
            self._snapshot.update(state=next_state, detail="모의 정지 완료")
        elif state == "RECOVERING":
            self._snapshot.update(state="STOPPED", holding="DETACHED", detail="모의 복구 완료")
        else:
            return
        self._snapshot["seq"] += 1
        self.changed.emit()

    def close(self):
        self._timer.stop()
        self.connected = self.synchronized = False


# ── 5일차 모의 공정·집계 (Qt 비의존) ─────────────────────────────────────────
# 아래 사건·스냅샷 형식은 D 내부 모의 전용이다. C 계약 필드가 아니며 실제 모드에 연결하지 않는다.

SPECS = ("BOLT_A", "BOLT_B", "NUT", "UNRESOLVED")
BINS = ("BOLT_A", "BOLT_B", "NUT")
BOLT_STAGES = ("이동 중 집기", "운반", "도착", "복귀")
NUT_STAGES = ("통과", "도착")
CATEGORIES = ("BOLT_A", "BOLT_B", "NUT", "QUARANTINE", "WRONG_BIN", "UNRESOLVED")
DISPLAY_FIELDS = ("recognition", "robot", "belt", "gripper")
_EVENT_KEYS = {
    "PROGRESS": ({"spec", "stage"}, {"note", *DISPLAY_FIELDS}),
    "ARRIVED": ({"spec", "bin", "verdict"}, set()),
    "QUARANTINED": (set(), {"reason"}),
    "FAILED": ({"reason"}, set()),
}
_BASE_KEYS = {"kind", "scope", "epoch", "seq", "part"}


class Outcome:
    """사건·스냅샷 처분. code: 반영/지연 반영/지연/중복/불일치/미동기화/옛 세대/다른 범위/낮은 경계/형식오류."""

    __slots__ = ("code", "counted", "note")

    def __init__(self, code, counted=False, note=""):
        self.code, self.counted, self.note = code, counted, note

    def __repr__(self):
        return f"Outcome({self.code!r}, {self.counted}, {self.note!r})"


def _int(value, low):
    return type(value) is int and value >= low


def stages_of(spec):
    return NUT_STAGES if spec == "NUT" else BOLT_STAGES


def final_category(event):
    """ARRIVED/QUARANTINED 사건의 부품별 최종 범주."""
    if event["kind"] == "QUARANTINED":
        return "QUARANTINE"
    return event["bin"] if event["verdict"] == "OK" else event["verdict"]


def event_error(event):
    if not isinstance(event, dict) or not isinstance(event.get("kind"), str) or event["kind"] not in _EVENT_KEYS:
        return "종류 오류"
    required, optional = _EVENT_KEYS[event["kind"]]
    keys = set(event)
    if not (_BASE_KEYS | required) <= keys or keys - _BASE_KEYS - required - optional:
        return "키 오류"
    if not isinstance(event["scope"], str) or not event["scope"] or not _int(event["epoch"], 0) or not _int(event["seq"], 1):
        return "범위·순번 오류"
    if not isinstance(event["part"], str) or not event["part"].strip():
        return "부품 ID 오류"
    if any(not isinstance(event[key], str) for key in keys & {"note", "reason", *DISPLAY_FIELDS}):
        return "문자열 필드 오류"
    if "spec" in event and event["spec"] not in SPECS:
        return "규격 오류"
    if event["kind"] == "PROGRESS" and (not _int(event["stage"], 0) or event["stage"] >= len(stages_of(event["spec"]))):
        return "단계 오류"
    if event["kind"] == "PROGRESS" and event["stage"] == stages_of(event["spec"]).index("도착"):
        return "도착은 ARRIVED 사건으로만"
    if event["kind"] == "ARRIVED":
        spec, bin_, verdict = event["spec"], event["bin"], event["verdict"]
        if bin_ not in BINS or verdict not in ("OK", "WRONG_BIN", "UNRESOLVED"):
            return "상자·판정 오류"
        if (verdict == "OK") != (spec == bin_) or (verdict == "UNRESOLVED") != (spec == "UNRESOLVED"):
            return "규격·상자·판정 불일치"
    if event["kind"] == "FAILED" and not event["reason"].strip():
        return "실패 사유 없음"
    return ""


def counts_of(finals):
    counts = dict.fromkeys(CATEGORIES, 0)
    for category in finals.values():
        counts[category] += 1
    return counts


def snapshot_error(snapshot):
    if not isinstance(snapshot, dict) or set(snapshot) != {"scope", "epoch", "boundary", "finals", "failures", "counts"}:
        return "키 오류"
    if not isinstance(snapshot["scope"], str) or not snapshot["scope"] or not _int(snapshot["epoch"], 0) or not _int(snapshot["boundary"], 0):
        return "범위·경계 오류"
    finals, failures = snapshot["finals"], snapshot["failures"]
    if not isinstance(finals, dict) or any(not isinstance(part, str) or not part.strip() or category not in CATEGORIES
                                           for part, category in finals.items()):
        return "최종 결과 집합 오류"
    if not isinstance(failures, (list, tuple)) or any(
            not isinstance(item, (list, tuple)) or len(item) != 2 or not all(isinstance(text, str) and text.strip() for text in item)
            for item in failures):
        return "실패 키 집합 오류"
    counts = snapshot["counts"]
    if (not isinstance(counts, dict) or set(counts) != set(CATEGORIES)
            or any(not _int(value, 0) for value in counts.values()) or counts != counts_of(finals)):
        return "누적값과 집합 불일치"
    return ""


class SortTally:
    """모의 공정 표시와 수량 집계. 조작 상태(6키 스냅샷)는 저장하지 않는다.

    - 정상 상자·격리·예외 도착은 부품별 최종 범주 하나로만 센다. 키는 (집계 범위, part_track_id).
    - 사건 순번(seq)은 집계 범위 안의 순서이며 조작 seq와 별개다. 표시 순서와 중복 판정을 분리한다.
    - 화면 기록을 지워도 이 객체는 바뀌지 않는다. 새 범위는 reset_session으로만 시작한다.
    """

    def __init__(self, scope, epoch=0):
        self.reset_session(scope, epoch)
        self.synchronized = False

    def reset_session(self, scope, epoch):
        """명시적 새 모의 세션. 수량·집합·표시를 비우고 동기화된 빈 상태로 시작한다."""
        self.scope, self.epoch = scope, epoch
        self.boundary = 0
        self.finals, self.failures, self._seen = {}, set(), {}
        self.conflict = False
        self.synchronized = True
        self.display_seq = {"bolt": 0, "nut": 0}   # 종류별 표시 순서 수평선(스냅샷 경계 이상)
        self.display = {"bolt": None, "nut": None}
        self.fields = dict.fromkeys(DISPLAY_FIELDS)
        self.last_failure = None
        self.failure_seq = 0
        self.stage_known = False

    def desync(self):
        """단절·세대 전환·동기화 상실. 마지막 수량은 남기고 단계는 미확인으로 둔다."""
        self.synchronized = False
        self.stage_known = False

    def mark_stage_unknown(self):
        self.stage_known = False

    @property
    def horizon(self):
        return max([self.boundary, *self._seen])

    def counts(self):
        return counts_of(self.finals)

    def failure_counts(self):
        result = {}
        for _part, reason in self.failures:
            result[reason] = result.get(reason, 0) + 1
        return result

    def process(self, event):
        error = event_error(event)
        if error:
            return Outcome("형식오류", note=error)
        if event["scope"] != self.scope:
            return Outcome("다른 범위", note=f"현재 범위 {self.scope}")
        if event["epoch"] < self.epoch:
            return Outcome("옛 세대", note=f"현재 세대 {self.epoch}")
        if not self.synchronized or event["epoch"] > self.epoch:
            return Outcome("미동기화", note="집계 스냅샷 동기화 전 · 기록만")
        seq, kind, part = event["seq"], event["kind"], event["part"]
        if seq in self._seen:
            if self._seen[seq] != event:
                self.conflict = True
                return Outcome("불일치", note="같은 사건 순번에 다른 내용 · 집계 미확인")
            return Outcome("중복", note="같은 사건 순번")
        within = seq <= self.boundary
        # 표시 순서는 종류(볼트·너트)별로 따로 본다. 실패·격리는 지금 표시 중인 같은 부품의 종류만
        group = self._group(event)
        late = within or (group is not None and seq < self.display_seq[group])
        if not within:
            self._seen[seq] = event.copy()
        if kind == "PROGRESS":
            if late:
                return Outcome("지연", note="최신 단계 유지 · 수량 +0")
            self._show(event, event["stage"])
            for key in DISPLAY_FIELDS:
                if key in event:
                    self.fields[key] = event[key]
            return Outcome("반영", note="표시만 · 수량 +0" + (f" · {event['note']}" if event.get("note") else ""))
        if kind == "FAILED":
            key = (part, event["reason"])
            if key in self.failures:
                return Outcome("중복", note="같은 실패 키")
            if within:
                self.conflict = True
                return Outcome("불일치", note="스냅샷 경계 이하인데 실패 집합에 없음 · 집계 미확인")
            self.failures.add(key)
            if not late:
                self._mark(part, seq, f"실패 · {event['reason']}")
            if seq > self.failure_seq:
                self.last_failure = f"{part} · {event['reason']}"
                self.failure_seq = seq
            return Outcome("지연 반영" if late else "반영", True, f"실패 사유 {event['reason']} +1")
        category, current = final_category(event), self.finals.get(part)
        if current == category:
            return Outcome("중복", note=f"이미 {category} · +0")
        if current is not None:
            self.conflict = True
            return Outcome("불일치", note=f"{current} 뒤 {category} · 합산 안 함 · 집계 미확인")
        if within:
            self.conflict = True
            return Outcome("불일치", note="스냅샷 경계 이하인데 집합에 없음 · 집계 미확인")
        self.finals[part] = category
        kept = ""
        if kind == "ARRIVED" and not late:
            arrival = stages_of(event["spec"]).index("도착")
            shown = self.display["nut" if event["spec"] == "NUT" else "bolt"]
            if shown is not None and shown["part"] == part and shown["stage"] is not None and shown["stage"] > arrival:
                # 같은 부품의 복귀가 먼저 왔다. 도착 확인은 수량만 올리고 표시 단계를 되돌리지 않는다
                self.display_seq[group], kept = event["seq"], " · 복귀 표시 유지"
            else:
                self._show(event, arrival)
        elif kind == "QUARANTINED" and not late:
            self._mark(part, seq, "격리" + (f" · {event['reason']}" if event.get("reason") else ""))
        return Outcome("지연 반영" if late else "반영", True, f"{category} +1" + (" · 최신 단계 유지" if late else kept))

    def _group(self, event):
        if "spec" in event:
            return "nut" if event["spec"] == "NUT" else "bolt"
        return next((g for g, shown in self.display.items() if shown is not None and shown["part"] == event["part"]), None)

    def _show(self, event, stage):
        group = self._group(event)
        self.display_seq[group] = event["seq"]
        self.display[group] = {"part": event["part"], "spec": event["spec"], "stage": stage}
        self.stage_known = True

    def _mark(self, part, seq, outcome):
        """지금 표시 중인 같은 부품만 실패·격리로 바꾼다(단계 미확인). 다른 부품 표시는 건드리지 않는다.

        모의 의미: 같은 부품의 더 새 PROGRESS가 오면 그 단계로 다시 표시한다. C의 실패 후 재시도 정책을 정한 것이 아니다.
        """
        for group, shown in self.display.items():
            if shown is not None and shown["part"] == part:
                self.display[group] = dict(shown, stage=None, outcome=outcome)
                self.display_seq[group] = seq

    def apply_snapshot(self, snapshot):
        """원자적 집계 스냅샷. 같은 범위만, 현재 수평선보다 낮은 경계는 거부. 수락하면 집합을 교체한다."""
        error = snapshot_error(snapshot)
        if error:
            return Outcome("형식오류", note=error)
        if snapshot["scope"] != self.scope:
            return Outcome("다른 범위", note="새 범위는 명시적 세션 초기화로만 시작")
        if snapshot["epoch"] < self.epoch:
            return Outcome("옛 세대", note=f"현재 세대 {self.epoch}")
        if snapshot["boundary"] < self.horizon:
            return Outcome("낮은 경계", note=f"경계 {snapshot['boundary']} < 현재 {self.horizon}")
        if snapshot["epoch"] == self.epoch and snapshot["boundary"] == self.boundary:
            same = (snapshot["finals"] == self.finals
                    and {tuple(item) for item in snapshot["failures"]} == self.failures)
            if same:
                self.synchronized = True
            if self.conflict or not same:
                self.conflict = True
                return Outcome("불일치", note="같은 경계의 다른 내용 또는 불일치 해소에는 더 최신 스냅샷 필요")
        before = self.counts()
        self.epoch, self.boundary = snapshot["epoch"], snapshot["boundary"]
        self.finals = dict(snapshot["finals"])
        self.failures = {tuple(item) for item in snapshot["failures"]}
        # 스냅샷의 실패 집합에는 발생 순서가 없으므로 최근 실패를 추정하지 않는다.
        self.last_failure, self.failure_seq = None, self.boundary
        self._seen = {}
        self.conflict = False
        self.synchronized = True
        self.display_seq = {group: max(seq, self.boundary) for group, seq in self.display_seq.items()}
        after = self.counts()
        lowered = [f"{key} {before[key]}→{after[key]}" for key in CATEGORIES if after[key] < before[key]]
        return Outcome("반영", True, ("감소·불일치 기록: " + ", ".join(lowered)) if lowered else f"경계 {self.boundary}")


class DemoProcessSupplier:
    """모의 공정 사건 공급자. 소비자(SortTally) 집합을 읽거나 복사하지 않고 자기가 만든 사건으로만 권위 집합을 만든다."""

    def __init__(self, scope="모의-세션-1"):
        self.scope = scope
        self.epoch = 0
        self.seq = 0
        self.finals, self.failures = {}, set()
        self._plan = []
        self._parts = 0

    @property
    def active(self):
        return bool(self._plan)

    def make(self, kind, part, epoch, **fields):
        """사건 하나를 만들고 권위 집합에 반영한다(첫 최종 결과 우선)."""
        self.seq += 1
        event = dict(kind=kind, scope=self.scope, epoch=epoch, seq=self.seq, part=part, **fields)
        if not event_error(event) and epoch >= self.epoch:
            self.epoch = epoch
            if kind in ("ARRIVED", "QUARANTINED"):
                self.finals.setdefault(part, final_category(event))
            elif kind == "FAILED":
                self.failures.add((part, fields["reason"]))
        return event

    def snapshot(self, epoch):
        self.epoch = max(self.epoch, epoch)
        return dict(scope=self.scope, epoch=epoch, boundary=self.seq, finals=dict(self.finals),
                    failures=sorted(self.failures), counts=counts_of(self.finals))

    def start_cycle(self):
        """볼트 A·볼트 B·너트 정상 1주기 계획. 사건은 next_event 때 하나씩 만든다."""
        if self._plan:
            return False
        for spec in ("BOLT_A", "BOLT_B", "NUT"):
            self._parts += 1
            part = f"P-{self._parts:04d}"
            if spec == "NUT":
                self._plan += [
                    ("PROGRESS", part, dict(spec=spec, stage=0, recognition="NUT 인식", robot="대기", belt="동작", gripper="열림")),
                    ("ARRIVED", part, dict(spec=spec, bin=spec, verdict="OK")),
                ]
                continue
            self._plan += [
                ("PROGRESS", part, dict(spec=spec, stage=0, recognition=f"{spec} 인식", robot="이동 중 집기", belt="동작", gripper="닫힘")),
                ("PROGRESS", part, dict(spec=spec, stage=1, robot="운반")),
                ("PROGRESS", part, dict(spec=spec, stage=1, gripper="열림", note="해제 요청")),
                ("ARRIVED", part, dict(spec=spec, bin=spec, verdict="OK")),
                ("PROGRESS", part, dict(spec=spec, stage=3, robot="복귀")),
            ]
        return True

    def next_event(self, epoch):
        if not self._plan:
            return None
        kind, part, fields = self._plan.pop(0)
        return self.make(kind, part, epoch, **fields)

    def abort(self):
        dropped = len(self._plan)
        self._plan.clear()
        return dropped
