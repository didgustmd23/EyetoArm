"""scripts/check_opencv.py 단위시험.

세 묶음으로 나눈다.
1. 순수 계산 도우미: 기대값을 손으로 계산해 검사 (OpenCV 불필요)
2. 판정 연결: FAIL·UNVERIFIED가 올바른 종료 코드와 정상 실행 검증 결과로 이어지는지 (실패 주입)
3. 현재 환경의 정상 동작: 실제 OpenCV로 필수 시험 6개와 이미지 저장이 동작하는지
"""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_opencv.py"
_spec = importlib.util.spec_from_file_location("check_opencv", SCRIPT)
oc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oc)

REQUIRED = ("api_selection", "synthetic_detection", "blank_negative",
            "rotation_correspondence", "pnp_synthetic", "detect_to_pnp")
IMAGE_FILES = {"marker.png", "canvas_0deg.png", "overlay_000deg.png",
               "overlay_090deg.png", "overlay_180deg.png", "overlay_270deg.png"}


def assert_environment_check(report):
    """현재 환경에서 점검이 정상 동작했는지 판정한다.

    - 필수 항목에 FAIL이 하나라도 있으면 AssertionError (UNVERIFIED가 함께 있어도 skip으로 숨기지 않음)
    - FAIL 없이 UNVERIFIED만 있으면 사유와 종료 코드 2를 확인한 뒤 skip (기능 검증 통과로 세지 않음)
    - 전부 PASS이면 종료 코드 0과 핵심 결과를 확인
    """
    results = {r["name"]: r for r in report["results"]}

    if tuple(results) != REQUIRED:
        raise AssertionError(f"필수 시험 구성이 다름: {list(results)}")

    failed = [f"{n}: {r['reason']}" for n, r in results.items() if r["status"] == oc.FAIL]

    if failed:
        raise AssertionError("필수 시험 FAIL — " + " | ".join(failed))

    unverified = {n: r["reason"] for n, r in results.items() if r["status"] == oc.UNVERIFIED}

    if unverified:
        if not all(unverified.values()):
            raise AssertionError(f"UNVERIFIED 항목에 사유가 없음: {unverified}")

        if report["exit_code"] != 2:
            raise AssertionError(f"UNVERIFIED만 있는데 종료 코드가 {report['exit_code']}")

        pytest.skip("필수 시험 실행 불가(UNVERIFIED) — " + " | ".join(f"{n}: {r}" for n, r in unverified.items()))

    if any(r["status"] != oc.PASS for r in results.values()):
        raise AssertionError(f"알 수 없는 상태: {[r['status'] for r in results.values()]}")

    if report["exit_code"] != 0:
        raise AssertionError(f"필수 시험이 모두 PASS인데 종료 코드가 {report['exit_code']}")

    pnp = results["pnp_synthetic"]

    if not (pnp["candidate_count"] >= 1 and any(c["within_synthetic_tolerance"] for c in pnp["candidates"])):
        raise AssertionError("합성 PnP: 정답 허용치 안의 후보가 없음")

    if not all(c["in_front"] for c in pnp["candidates"]):
        raise AssertionError("합성 PnP: 카메라 뒤 후보가 PASS로 보고됨")

    rot = results["rotation_correspondence"]

    if [r["deg_ccw"] for r in rot["rotations"]] != [0, 90, 180, 270]:
        raise AssertionError("회전 대응: 0·90·180·270° 결과가 모두 있지 않음")


class _NoAruco:
    """cv2에서 aruco만 감춘 대리 객체. ArUco가 없는 OpenCV 빌드를 흉내 낸다."""

    def __init__(self, cv2):
        """감쌀 실제 cv2 모듈을 받아 둔다."""
        self._cv2 = cv2

    def __getattr__(self, name):
        """`aruco`는 AttributeError, 그 밖의 이름은 실제 cv2에서 찾아 돌려준다."""
        if name == "aruco":
            raise AttributeError(name)
        return getattr(self._cv2, name)


def _fail_pnp(*_a, **_k):
    """합성 PnP 시험 대신 넣는 함수. 항상 FAIL 결과를 돌려준다(실패 주입)."""
    return oc._result("pnp_synthetic", oc.FAIL, "실패 주입: 합성 PnP FAIL")


# ── 1. 순수 계산 도우미 ─────────────────────────────────────────────────────────

def test_marker_object_points_order_matches_ippe_square():
    """마커 꼭짓점 순서가 좌상·우상·우하·좌하이고 Z=0, 단위 m다(IPPE_SQUARE 순서, A 상세설계 §8)."""
    # A 상세설계 §8: 0 좌상 (-L/2,+L/2), 1 우상, 2 우하, 3 좌하, Z=0, 단위 m
    np.testing.assert_allclose(
        oc.marker_object_points(0.08),
        [[-0.04, 0.04, 0], [0.04, 0.04, 0], [0.04, -0.04, 0], [-0.04, -0.04, 0]],
    )


def test_project_pinhole_hand_values():
    """핀홀 투영이 손으로 계산한 화소 좌표와 같다(fx=fy=800, 주점 (320, 240))."""
    uv = oc.project_pinhole([[0, 0, 1], [0.1, -0.05, 1], [0.1, 0, 2]], 800, 800, 320, 240)
    np.testing.assert_allclose(uv, [[320, 240], [400, 200], [360, 240]])


def test_rotation_angle_deg():
    """두 회전행렬 사이 각도가 0°·150°·1e-6°에서 정확하다(작은 각도도 잃지 않음)."""
    eye = np.eye(3)

    assert oc.rotation_angle_deg(eye, eye) == pytest.approx(0.0, abs=1e-12)
    assert oc.rotation_angle_deg(eye, oc.rot_x(150.0)) == pytest.approx(150.0, abs=1e-9)
    assert oc.rotation_angle_deg(eye, oc.rot_x(1e-6)) == pytest.approx(1e-6, rel=1e-3)


def test_rot90_points_moves_top_left_to_bottom_left():
    """반시계 90° 회전에서 좌상 꼭짓점이 좌하로 가고, 4번 돌리면 제자리다."""
    origin = (220, 140)
    pts = np.array([[220.0, 140.0], [419.0, 140.0], [419.0, 339.0], [220.0, 339.0]])

    np.testing.assert_allclose(oc.rot90_points(pts, 1, 200, origin),
                               [[220, 339], [220, 140], [419, 140], [419, 339]])
    np.testing.assert_allclose(oc.rot90_points(pts, 4, 200, origin), pts)


def test_rot90_points_agrees_with_numpy_rot90():
    """점 회전 결과가 `np.rot90`으로 돌린 영상에서 찾은 화소 위치와 0~3회 모두 같다."""
    n = 7
    img = np.zeros((n, n), np.uint8)
    img[1, 5] = 1  # (x=5, y=1)

    for k in range(4):
        y, x = np.argwhere(np.rot90(img, k) == 1)[0]

        np.testing.assert_allclose(oc.rot90_points([[5, 1]], k, n, (0, 0)), [[x, y]])


@pytest.mark.parametrize("statuses,code", [
    (["PASS", "PASS"], 0),
    (["PASS", "FAIL"], 1),
    (["PASS", "UNVERIFIED"], 2),
    (["FAIL", "UNVERIFIED"], 1),
])
def test_overall_exit_code(statuses, code):
    """종합 종료 코드는 FAIL이 있으면 1, FAIL 없이 UNVERIFIED가 있으면 2, 모두 PASS면 0이다."""
    assert oc.overall_exit_code([{"status": s} for s in statuses]) == code


def test_all_finite_rejects_nan_inf():
    """`all_finite`는 NaN·Inf가 하나라도 있으면(중첩 목록 포함) False다."""
    assert oc.all_finite([1.0, 2.0])
    assert not oc.all_finite([1.0, math.nan])
    assert not oc.all_finite([[math.inf]])


# ── 2. 판정 연결 (실패 주입 회귀 시험) ──────────────────────────────────────────

def test_injected_pnp_fail_makes_environment_check_fail(monkeypatch):
    """합성 PnP에 FAIL을 주입하면 종료 코드 1이고, 환경 판정이 그 항목 이름으로 실패한다."""
    cv2 = pytest.importorskip("cv2")
    monkeypatch.setattr(oc, "test_pnp_synthetic", _fail_pnp)

    report = oc.run(cv2, np, oc.DEFAULT_CORNER_TOL_PX, None)

    assert report["exit_code"] == 1
    with pytest.raises(AssertionError, match="pnp_synthetic"):
        assert_environment_check(report)


def test_unverified_only_is_skipped_with_reason_and_exit_2():
    """aruco가 없으면 API 선택이 UNVERIFIED(FAIL 없음)·종료 코드 2이고, 환경 판정은 사유를 붙여 건너뛴다."""
    cv2 = pytest.importorskip("cv2")
    report = oc.run(_NoAruco(cv2), np, oc.DEFAULT_CORNER_TOL_PX, None)

    statuses = {r["name"]: r["status"] for r in report["results"]}

    assert statuses["api_selection"] == oc.UNVERIFIED
    assert oc.FAIL not in statuses.values()
    assert report["exit_code"] == 2
    with pytest.raises(pytest.skip.Exception, match="cv2.aruco 없음"):
        assert_environment_check(report)


def test_fail_with_unverified_fails_instead_of_skipping(monkeypatch):
    """UNVERIFIED와 FAIL이 함께 있으면 건너뛰지 않고 실패로 판정한다(종료 코드 1)."""
    cv2 = pytest.importorskip("cv2")
    monkeypatch.setattr(oc, "test_pnp_synthetic", _fail_pnp)

    report = oc.run(_NoAruco(cv2), np, oc.DEFAULT_CORNER_TOL_PX, None)

    assert report["exit_code"] == 1
    with pytest.raises(AssertionError, match="pnp_synthetic"):
        assert_environment_check(report)


def test_corner_mismatch_is_reported_as_fail_not_hidden():
    """어떤 차이도 통과할 수 없는 음수 허용치를 주면 회전 대응 시험이 FAIL로 보고되고 종료 코드 1이다."""
    cv2 = pytest.importorskip("cv2")
    report = oc.run(cv2, np, -0.5, None)  # 어떤 차이도 통과할 수 없는 허용치
    rot = next(r for r in report["results"] if r["name"] == "rotation_correspondence")

    if rot["status"] == oc.UNVERIFIED:
        pytest.skip(f"회전 시험 실행 불가: {rot['reason']}")

    assert rot["status"] == oc.FAIL
    assert report["exit_code"] == 1


def _required_all_pass(report):
    """필수 시험이 모두 PASS인지 돌려준다."""
    return all(r["status"] == oc.PASS for r in report["results"])


def test_save_imwrite_false_for_all(monkeypatch, tmp_path):
    """`imwrite`가 모두 False면 기존 파일이 있어도 저장 성공 0개·6개 모두 실패로 기록되고 종료 코드 1이다. 필수 시험 구성은 그대로다."""
    cv2 = pytest.importorskip("cv2")
    out = tmp_path / "imgs"
    out.mkdir()

    for f in IMAGE_FILES:  # 기존 파일이 있어도 이번 저장 성공으로 보지 않아야 함
        (out / f).write_bytes(b"old")

    monkeypatch.setattr(cv2, "imwrite", lambda *a, **k: False)
    report = oc.run(cv2, np, oc.DEFAULT_CORNER_TOL_PX, str(out))

    save = report["image_save"]

    assert save["status"] == oc.FAIL
    assert save["saved_files"] == [] and report["saved_files"] == []
    assert {Path(f["path"]).name for f in save["failed_files"]} == IMAGE_FILES
    assert all("False" in f["error"] for f in save["failed_files"])
    assert report["exit_code"] == 1
    assert [r["name"] for r in report["results"]] == list(REQUIRED)  # 필수 시험 구성은 그대로


def test_save_partial_success_lists_match_disk(monkeypatch, tmp_path):
    """저장이 번갈아 실패하면 성공·실패 목록이 실제 디스크 파일과 맞고(성공 3개), 판정은 FAIL·종료 코드 1이다."""
    cv2 = pytest.importorskip("cv2")
    real = cv2.imwrite
    calls = {"n": 0}

    def flaky(path, img, *a):
        """홀수 번째 호출은 실제로 저장하고 짝수 번째는 False를 돌려준다."""
        calls["n"] += 1
        return real(path, img, *a) if calls["n"] % 2 == 1 else False

    monkeypatch.setattr(cv2, "imwrite", flaky)
    out = tmp_path / "imgs"
    report = oc.run(cv2, np, oc.DEFAULT_CORNER_TOL_PX, str(out))
    save = report["image_save"]

    on_disk = {p.name for p in out.iterdir()}

    assert {Path(p).name for p in save["saved_files"]} == on_disk
    assert {Path(f["path"]).name for f in save["failed_files"]} == IMAGE_FILES - on_disk
    assert len(on_disk) == 3 and save["status"] == oc.FAIL and report["exit_code"] == 1


def test_save_exception_is_recorded_and_json_is_strict(monkeypatch, tmp_path, capsys):
    """`imwrite` 예외는 파일마다 예외 이름·메시지로 기록되고, `--json` 출력은 비유한 상수 없는 엄격한 JSON이다."""
    cv2 = pytest.importorskip("cv2")

    def boom(*_a, **_k):
        """항상 OSError를 낸다(디스크 오류 주입)."""
        raise OSError("디스크 오류 주입")

    monkeypatch.setattr(cv2, "imwrite", boom)
    out = tmp_path / "imgs"
    code = oc.main(["--json", "--output-dir", str(out)])
    data = json.loads(capsys.readouterr().out, parse_constant=lambda c: pytest.fail(f"비유한 상수 {c}"))

    save = data["image_save"]

    assert save["status"] == oc.FAIL and save["saved_files"] == []
    assert all("OSError" in f["error"] and "디스크 오류 주입" in f["error"] for f in save["failed_files"])
    assert {Path(f["path"]).name for f in save["failed_files"]} == IMAGE_FILES
    assert code == data["exit_code"] == 1


def test_save_mkdir_permission_error_is_unverified(monkeypatch, tmp_path):
    """출력 폴더를 만들 권한이 없으면 저장은 FAIL이 아니라 UNVERIFIED이고, 필수 시험이 모두 PASS면 종료 코드 2다."""
    cv2 = pytest.importorskip("cv2")

    def deny(self, *a, **k):
        """항상 PermissionError를 낸다(권한 거부 주입)."""
        raise PermissionError(13, "권한 거부 주입", str(self))

    monkeypatch.setattr(Path, "mkdir", deny)
    report = oc.run(cv2, np, oc.DEFAULT_CORNER_TOL_PX, str(tmp_path / "imgs"))

    save = report["image_save"]

    assert save["status"] == oc.UNVERIFIED and "PermissionError" in save["reason"]
    assert save["saved_files"] == [] and not (tmp_path / "imgs").exists()

    if _required_all_pass(report):
        assert report["exit_code"] == 2
    else:
        assert report["exit_code"] in (1, 2)


def test_save_failure_with_processing_fail_exit_1(monkeypatch, tmp_path):
    """저장 UNVERIFIED와 영상처리 FAIL이 함께 있으면 FAIL이 우선해 종료 코드 1이다."""
    cv2 = pytest.importorskip("cv2")
    monkeypatch.setattr(oc, "test_pnp_synthetic", _fail_pnp)
    monkeypatch.setattr(Path, "mkdir", lambda self, *a, **k: (_ for _ in ()).throw(PermissionError("주입")))

    report = oc.run(cv2, np, oc.DEFAULT_CORNER_TOL_PX, str(tmp_path / "imgs"))

    assert report["image_save"]["status"] == oc.UNVERIFIED
    assert report["exit_code"] == 1  # 영상처리 FAIL이 우선


# ── 3. 현재 환경의 정상 동작 ────────────────────────────────────────────────────

def test_environment_required_checks_pass():
    """현재 환경의 실제 OpenCV로 필수 시험 6개가 정상 동작한다(PASS, 또는 사유 있는 UNVERIFIED면 건너뜀)."""
    cv2 = pytest.importorskip("cv2")
    report = oc.run(cv2, np, oc.DEFAULT_CORNER_TOL_PX, None)
    assert_environment_check(report)


def test_no_output_dir_adds_no_save_result_and_writes_nothing(monkeypatch, tmp_path, capsys):
    """`--output-dir`이 없으면 보고서에 저장 결과가 없고 현재 폴더에 아무 파일도 쓰지 않는다."""
    pytest.importorskip("cv2")
    monkeypatch.chdir(tmp_path)
    code = oc.main(["--json"])

    data = json.loads(capsys.readouterr().out, parse_constant=lambda c: pytest.fail(f"비유한 상수 {c}"))

    assert "image_save" not in data and data["saved_files"] == []
    assert list(tmp_path.iterdir()) == []
    assert code == data["exit_code"]


def test_environment_saves_six_images(tmp_path):
    """출력 폴더를 주면 이미지 6개가 실제로 저장되고, 저장 목록이 디스크와 같으며 빈 파일이 없다."""
    cv2 = pytest.importorskip("cv2")
    out = tmp_path / "imgs"
    report = oc.run(cv2, np, oc.DEFAULT_CORNER_TOL_PX, str(out))
    save = report["image_save"]

    if save["status"] == oc.UNVERIFIED:
        pytest.skip(f"이미지 저장 실행 불가: {save['reason']}")

    assert save["status"] == oc.PASS, save

    on_disk = {p.name for p in out.iterdir()}

    assert on_disk == IMAGE_FILES
    assert {Path(p).name for p in save["saved_files"]} == on_disk
    assert all((out / f).stat().st_size > 0 for f in on_disk)

    assert_environment_check(report)
