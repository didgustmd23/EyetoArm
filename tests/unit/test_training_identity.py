"""학습기 작업 신원·자기 작업 판별·새 작업 허용·stale lock 해제 판정(`eyetoarm.training.identity`) 시험.
가짜 `/proc` 폴더를 주입하고 실제 신호는 보내지 않는다."""
import os
import tempfile
import unittest

from eyetoarm.training.identity import (TOKEN_ENV, ProcFS, blocking_states, new_job_token, own_processes,
                                        parent_alive, stale_lock_decision)


def _write(path, data, mode="w"):
    """파일을 쓰고 닫는다(시험 도우미)."""
    with open(path, mode) as f:
        f.write(data)


def _read(path, mode="r"):
    """파일을 읽고 닫는다(시험 도우미)."""
    with open(path, mode) as f:
        return f.read()


MY_UID = 1000
TOKEN = "tok-1111"
BOOT = "boot-aaaa"


class FakeProc:
    """가짜 /proc 트리를 만든다."""

    def __init__(self, root):
        """루트와 boot_id."""
        self.root = root
        os.makedirs(os.path.join(root, "sys", "kernel", "random"))
        _write(os.path.join(root, "sys", "kernel", "random", "boot_id"), BOOT + "\n", "w")

    def add(self, pid, uid=MY_UID, start=100, pgid=None, env=None, readable=True):
        """프로세스 하나. env는 dict, readable False면 environ 권한 0."""
        d = os.path.join(self.root, str(pid))
        os.makedirs(d)
        # 실제 stat에서 ")" 뒤 3번째가 pgrp, 20번째가 starttime이다
        fields = ["S", "1", str(pgid or pid)] + ["0"] * 16 + [str(start)] + ["0"] * 10
        _write(os.path.join(d, "stat"), f"{pid} (python x) " + " ".join(fields) + "\n", "w")
        _write(os.path.join(d, "status"), f"Name:\tpython\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n", "w")
        envp = os.path.join(d, "environ")
        _write(envp, b"\0".join(f"{k}={v}".encode() for k, v in (env or {}).items()) + b"\0", "wb")
        if not readable:
            os.chmod(envp, 0)


class IdentityTest(unittest.TestCase):
    """자기 작업 판별과 판정."""

    def setUp(self):
        """가짜 /proc."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fake = FakeProc(self.tmp.name)
        self.proc = ProcFS(self.tmp.name)

    def tearDown(self):
        """권한 0 파일을 정리 가능하게 되돌린다."""
        for dirpath, _, files in os.walk(self.tmp.name):
            for f in files:
                os.chmod(os.path.join(dirpath, f), 0o600)

    def test_procfs_reads(self):
        """ProcFS는 pid 목록·uid·시작 시각·pgid·environ·boot_id를 읽는다."""
        self.fake.add(10, start=555, pgid=10, env={"EYETOARM_TRAIN_JOB": TOKEN})

        self.assertEqual(self.proc.pids(), [10])
        self.assertEqual((self.proc.uid(10), self.proc.start_ticks(10), self.proc.pgid(10)), (MY_UID, 555, 10))
        self.assertEqual(self.proc.environ(10)["EYETOARM_TRAIN_JOB"], TOKEN)
        self.assertEqual(self.proc.boot_id(), BOOT)

    def test_own_processes_by_token_only(self):
        """토큰이 있는 자기 uid 프로세스(손자 포함)만 자기 작업이다(같은 pgid라도 토큰 없으면 아님)."""
        self.fake.add(10, pgid=10, env={"EYETOARM_TRAIN_JOB": TOKEN})
        self.fake.add(11, pgid=10, env={"EYETOARM_TRAIN_JOB": TOKEN})        # 데이터로더 worker
        self.fake.add(12, pgid=10, env={"EYETOARM_TRAIN_JOB": "other"})
        self.fake.add(13, pgid=10, env={})

        own, unknown = own_processes(TOKEN, self.proc, MY_UID)
        self.assertEqual((own, unknown), ([10, 11], []))

    def test_other_uid_unreadable_excluded_same_uid_unreadable_unknown(self):
        """타 uid의 읽기 불가 environ은 조사에서 빼고, 자기 uid인데 읽기 불가면 판별 불가다."""
        self.fake.add(20, uid=0, env={"EYETOARM_TRAIN_JOB": TOKEN}, readable=False)
        self.fake.add(21, uid=MY_UID, env={"EYETOARM_TRAIN_JOB": TOKEN}, readable=False)

        own, unknown = own_processes(TOKEN, self.proc, MY_UID)
        self.assertEqual((own, unknown), ([], [21]))

    def test_vanished_process_skipped(self):
        """조사 중 사라진 프로세스(파일 없음)는 건너뛴다."""
        self.fake.add(30, env={"EYETOARM_TRAIN_JOB": TOKEN})
        os.remove(os.path.join(self.tmp.name, "30", "environ"))

        self.assertEqual(own_processes(TOKEN, self.proc, MY_UID), ([], []))

    def test_blocking_states(self):
        """running·starting·cleanup_unconfirmed 실험이 있으면 새 작업 차단 목록에 나온다(lock과 무관)."""
        states = {"S1-01": {"state": "exited_cleaned"}, "S1-02": {"state": "cleanup_unconfirmed"},
                  "S1-03": {"state": "not_started"}, "S1-04": {"state": "starting"}}

        self.assertEqual(blocking_states(states), [("S1-02", "cleanup_unconfirmed"), ("S1-04", "starting")])
        self.assertEqual(blocking_states({"S1-01": {"state": "exited_cleaned"}}), [])

    def test_parent_alive_uses_start_time(self):
        """lock 부모 pid가 없거나 시작 시각이 다르면(재사용) 죽은 것으로 본다."""
        self.fake.add(40, start=700)

        self.assertTrue(parent_alive({"pid": 40, "start_ticks": 700}, self.proc))
        self.assertFalse(parent_alive({"pid": 40, "start_ticks": 1}, self.proc))
        self.assertFalse(parent_alive({"pid": 41, "start_ticks": 700}, self.proc))

    def lock(self, **over):
        """죽은 부모의 lock."""
        lk = {"pid": 99, "start_ticks": 5, "host": "h1", "boot_id": BOOT}
        lk.update(over)
        return lk

    def test_stale_lock_decision(self):
        """해제는 같은 호스트·부팅, 부모 사망, 미정리 실험의 자기 작업 0·판별 불가 0일 때만 허용."""
        states = {"S1-02": {"state": "running", "job_token": TOKEN}}

        ok, reasons = stale_lock_decision(self.lock(), states, self.proc, host="h1", uid=MY_UID)
        self.assertEqual((ok, reasons), (True, []))

        self.fake.add(50, env={"EYETOARM_TRAIN_JOB": TOKEN})                # 부모 사망 + 자식 생존
        ok, reasons = stale_lock_decision(self.lock(), states, self.proc, host="h1", uid=MY_UID)
        self.assertFalse(ok)
        self.assertTrue(any("50" in r for r in reasons))

    def test_stale_lock_other_host_boot_parent_unknown(self):
        """다른 호스트·다른 부팅 ID·부모 생존·자기 uid 판별 불가는 거부."""
        states = {"S1-02": {"state": "cleanup_unconfirmed", "job_token": TOKEN}}
        self.fake.add(60, start=5)

        for name, lk, host in (("host", self.lock(), "h2"), ("boot", self.lock(boot_id="boot-bbbb"), "h1"),
                               ("parent alive", self.lock(pid=60, start_ticks=5), "h1")):
            with self.subTest(case=name):
                ok, _ = stale_lock_decision(lk, states, self.proc, host=host, uid=MY_UID)
                self.assertFalse(ok)

        self.fake.add(61, env={"EYETOARM_TRAIN_JOB": TOKEN}, readable=False)
        ok, reasons = stale_lock_decision(self.lock(), states, self.proc, host="h1", uid=MY_UID)
        self.assertFalse(ok)
        self.assertTrue(any("unknown" in r for r in reasons))

    def test_job_token_unique(self):
        """job_token은 매번 새 값(uuid4 hex)이다."""
        a, b = new_job_token(), new_job_token()

        self.assertNotEqual(a, b)
        self.assertEqual(len(a), 32)


class FailClosedTest(unittest.TestCase):
    """살아 있으나 파싱에 실패한 /proc 항목은 소멸과 구분해 판별 불가로 본다."""

    def test_unparsable_status_is_unknown_missing_is_skipped(self):
        """status에 Uid 줄이 없는 항목은 unknown, 디렉터리 자체가 없어진 항목은 건너뜀."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fake = FakeProc(tmp.name)
        fake.add(70, env={"EYETOARM_TRAIN_JOB": TOKEN})
        _write(os.path.join(tmp.name, "70", "status"), "Name:\tpython\n", "w")

        own, unknown = own_processes(TOKEN, ProcFS(tmp.name), MY_UID)

        self.assertEqual((own, unknown), ([], [70]))


class StartFloorTest(unittest.TestCase):
    """작업 시작 전 프로세스 제외(읽기 불가 systemd --user 같은 프로세스가 복구를 막지 않게)."""

    def test_unreadable_older_than_job_excluded(self):
        """작업 전 시작한 읽기 불가 프로세스는 제외, 이후 시작한 것은 판별 불가, 토큰 자식은 자기 작업."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fake = FakeProc(tmp.name)
        fake.add(80, start=10, env={}, readable=False)  # 작업 전(systemd --user 모사)
        fake.add(81, start=500, env={"EYETOARM_TRAIN_JOB": TOKEN})           # 작업 자식
        fake.add(82, start=600, env={}, readable=False)                      # 작업 뒤 읽기 불가
        self.addCleanup(lambda: [os.chmod(os.path.join(tmp.name, str(n), "environ"), 0o600) for n in (80, 82)])

        self.assertEqual(own_processes(TOKEN, ProcFS(tmp.name), MY_UID, min_start_ticks=500), ([81], [82]))
        self.assertEqual(own_processes(TOKEN, ProcFS(tmp.name), MY_UID), ([81], [80, 82]))


class ZombieTest(unittest.TestCase):
    """좀비 항목 제외."""

    def test_zombie_not_counted(self):
        """상태 Z(좀비)인 토큰 프로세스는 자기 작업·판별 불가 어느 쪽에도 넣지 않는다."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fake = FakeProc(tmp.name)
        fake.add(90, env={"EYETOARM_TRAIN_JOB": TOKEN})
        stat = os.path.join(tmp.name, "90", "stat")
        with open(stat) as f:
            text = f.read()
        _write(stat, text.replace(") S ", ") Z ", 1))

        self.assertEqual(own_processes(TOKEN, ProcFS(tmp.name), MY_UID), ([], []))


class ScriptProc:
    """회차별 가짜 /proc. `pids()`를 부를 때마다 다음 회차로 넘어간다(조사 1회 = pids 1회).
    spec 키: start·state·env(dict | "EACCES")·then("gone" | "Z", environ 실패 직후 상태)·hidden.
    hidden이면 그 회차 목록에는 없지만 /proc/<pid>는 있다. 없는 pid는 사라진 것이다."""

    def __init__(self, rounds, uid=MY_UID):
        """회차 목록."""
        self.rounds, self.uid_value, self.i, self.failed = rounds, uid, -1, set()

    def cur(self):
        """현재 회차(마지막 뒤로는 마지막 유지)."""
        return self.rounds[min(max(self.i, 0), len(self.rounds) - 1)]

    def spec(self, pid):
        """현재 회차의 pid 명세(사라졌으면 FileNotFoundError)."""
        sp = self.cur().get(pid)
        if sp is None or (pid in self.failed and sp.get("then") == "gone"):
            raise FileNotFoundError(pid)
        return sp

    def pids(self):
        """다음 회차의 pid."""
        self.i += 1
        self.failed = set()
        # hidden: 목록 경합으로 빠진 pid
        return sorted(pid for pid, sp in self.cur().items() if not sp.get("hidden"))

    def uid(self, pid):
        """uid."""
        self.spec(pid)
        return self.uid_value

    def start_ticks(self, pid):
        """시작 시각."""
        return self.spec(pid)["start"]

    def state(self, pid):
        """상태(실패 뒤 then=Z면 Z)."""
        sp = self.spec(pid)
        return "Z" if pid in self.failed and sp.get("then") == "Z" else sp.get("state", "S")

    def environ(self, pid):
        """environ(EACCES면 PermissionError)."""
        sp = self.spec(pid)
        if sp["env"] == "EACCES":
            self.failed.add(pid)
            raise PermissionError(13, "Permission denied")
        return sp["env"]

    def boot_id(self):
        """부팅 ID."""
        return BOOT


def _ok(env=None, start=100):
    """읽히는 프로세스 명세."""
    return {"start": start, "env": env or {}}


def _bad(start=100, then=None):
    """읽기 불가 프로세스 명세."""
    return {"start": start, "env": "EACCES", **({"then": then} if then else {})}


class SettleTest(unittest.TestCase):
    """일시 전이와 지속 판별 불가의 구분(실패 뒤 재확인, 상한 있는 재조사, 최종 시점 기준)."""

    def settle(self, rounds, **kw):
        """settled_processes를 주입 대기로 부르고 (own, unknown, info, 대기 기록)."""
        from eyetoarm.training.identity import settled_processes
        waits = []
        own, unknown, info = settled_processes(TOKEN, ScriptProc(rounds), MY_UID, 50, sleep=waits.append, **kw)
        return own, unknown, info, waits

    def test_p1_vanish_or_zombie_right_after_failure_is_resolved(self):
        """environ 실패 직후 사라지거나 Z가 된 pid는 unknown이 아니다."""
        for then in ("gone", "Z"):
            with self.subTest(then=then):
                own, unknown = own_processes(TOKEN, ScriptProc([{7: _bad(then=then)}]), MY_UID, 50)
                self.assertEqual((own, unknown), ([], []))

    def test_unknown_then_readable_non_target_proceeds(self):
        """unknown → 다음 조사에서 정상 판독(토큰 없음) → 해소·진행, 조사 2회·대기 1회."""
        own, unknown, info, waits = self.settle([{7: _bad()}, {7: _ok()}])
        self.assertEqual((own, unknown, info["scans"], waits), ([], [], 2, [0.1]))

    def test_unknown_then_readable_own_token_counts_as_own(self):
        """unknown → 정상 판독으로 자기 토큰 확인 → own에 포함(정리 완료로 처리 금지)."""
        own, unknown, _, _ = self.settle([{7: _bad()}, {7: _ok({TOKEN_ENV: TOKEN})}])
        self.assertEqual((own, unknown), ([7], []))

    def test_persistent_unreadable_stays_unknown_with_cap(self):
        """모든 조사에서 살아 있는 읽기 불가 → unknown 유지, 조사 3회 상한·대기 2회(0.1 s)."""
        own, unknown, info, waits = self.settle([{7: _bad()}] * 5)
        self.assertEqual((own, unknown, info["scans"], waits), ([], [7], 3, [0.1, 0.1]))

    def test_moving_unknown_is_not_intersected_away(self):
        """{a}→{b}→{b}는 b가, 마지막 회차에 처음 나타난 c도 unknown으로 남는다(교집합 금지)."""
        _, unknown, _, _ = self.settle([{7: _bad()}, {8: _bad(start=120)}, {8: _bad(start=120)}])
        self.assertEqual(unknown, [8])
        _, unknown, _, _ = self.settle([{7: _bad()}, {7: _bad()}, {9: _bad(start=130)}])
        self.assertEqual(unknown, [9])

    def test_alternation_and_pid_reuse_keep_unresolved_identity(self):
        """읽힘/불가가 교대하거나 pid가 재사용돼 계속 불가면 unknown, 재사용 프로세스가 읽히면 reused로 해소."""
        _, unknown, info, _ = self.settle([{7: _bad(), 8: _bad()}, {7: _ok(), 8: _bad()}, {7: _bad(), 8: _ok()}])
        self.assertEqual((unknown, info["scans"], info["resolved"]), ([7], 3, {"8@100": "not_own"}))
        _, unknown, _, _ = self.settle([{7: _bad(start=100)}, {7: _bad(start=200)}, {7: _bad(start=200)}])
        self.assertEqual(unknown, [7])
        _, unknown, info, _ = self.settle([{7: _bad(start=100)}, {7: _ok(start=200)}])
        self.assertEqual((unknown, info["resolved"]), ([], {"7@100": "reused"}))

    def test_listing_race_needs_explicit_resolution(self):
        """목록에서 빠졌지만 살아 있는 이전 unknown은 명시적 판독으로만 해소한다(불가면 unknown, 토큰이면 own)."""
        _, unknown, _, _ = self.settle([{7: _bad()}, {7: {**_bad(), "hidden": True}}])
        self.assertEqual(unknown, [7])
        own, unknown, info, _ = self.settle([{7: _bad()}, {7: {**_ok({TOKEN_ENV: TOKEN}), "hidden": True}}])
        self.assertEqual((own, unknown, info["resolved"]), ([7], [], {"7@100": "own"}))

    def test_deadline_budget_limits_waits(self):
        """잔여 기한이 대기 1회분이면 조사 2회, 기한이 없으면 조사 1회·대기 0(unknown 유지)."""
        t = [0.0]

        def clock():
            """대기만큼 흐르는 시계."""
            return t[0]

        from eyetoarm.training.identity import settled_processes
        waits = []

        def sleep(s):
            """시계를 진행."""
            waits.append(s)
            t[0] += s

        _, unknown, info = settled_processes(TOKEN, ScriptProc([{7: _bad()}] * 5), MY_UID, 50, sleep=sleep,
                                             clock=clock, deadline=0.15)
        self.assertEqual((unknown, info["scans"], waits), ([7], 2, [0.1]))
        _, unknown, info = settled_processes(TOKEN, ScriptProc([{7: _bad()}] * 5), MY_UID, 50, sleep=sleep,
                                             clock=clock, deadline=0.0)
        self.assertEqual((unknown, info["scans"]), ([7], 1))

    def test_stale_decision_uses_settled_scan(self):
        """stale 판정: 일시 unknown(다음 조사에 사라짐)은 허용, 지속 unknown은 거부."""
        lock = {"pid": 4242, "start_ticks": 1, "host": "h", "boot_id": BOOT}
        js = {"E": {"state": "running", "job_token": TOKEN, "start_ticks": 50}}
        ok, _ = stale_lock_decision(lock, js, ScriptProc([{7: _bad()}, {}]), host="h", uid=MY_UID)
        self.assertTrue(ok)
        ok, reasons = stale_lock_decision(lock, js, ScriptProc([{7: _bad()}] * 5), host="h", uid=MY_UID)
        self.assertFalse(ok)
        self.assertTrue(any("unknown" in r for r in reasons))


if __name__ == "__main__":
    unittest.main()
