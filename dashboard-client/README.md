# EyeToArm 대시보드 클라이언트 (Windows)

GUI 정본과 실행에 필요한 TCP 프로토콜·스키마·D 기본 설정을 이 폴더로 분류했습니다. 서버·학습 데이터·모델은 포함하지 않으며 공용 원본은 기존 위치에 유지합니다.

기존 eyetoarm 환경을 활성화하고 이 폴더에서 실행합니다. 환경 정의는 `src/eyetoarm/gui/environment.yml`을 사용합니다.

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m eyetoarm.gui.window --demo
python -m eyetoarm.gui --env-only --json
```

실제 연결은 `python -m eyetoarm.gui.window --host <서버 숫자 IP> --port 47090`을 사용합니다. 서버 수신·네트워크 준비는 별도로 필요합니다.

자세한 실행·기능 설명은 [GUI 안내](src/eyetoarm/gui/README.md)를 참고합니다. 기존 저장소의 `src/eyetoarm/gui` 경로는 호환 진입점으로 새 정본을 사용합니다. 독립 실행은 이 폴더의 config를, 저장소 호환 경로는 기존 공용 config를 읽습니다. 새 감사 기록은 `logs/gui/`에 저장하고 기존 기록은 유지합니다.
