"""기존 import 경로를 dashboard-client의 GUI 정본으로 연결한다."""
from pathlib import Path

# 저장소 경유 실행의 표시. client.py 는 이 값이 있을 때만 저장소 공용 config/gui/client.json 을 읽는다
_repository_root = Path(__file__).resolve().parents[3]
_gui = _repository_root / "dashboard-client" / "src" / "eyetoarm" / "gui"
if not _gui.is_dir():
    raise ImportError(f"대시보드 GUI를 찾을 수 없습니다: {_gui} (기존 src 경로는 같은 저장소의 dashboard-client 폴더가 필요)")
__path__ = [str(_gui)]
