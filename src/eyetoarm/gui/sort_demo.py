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
