"""기존 WSL 직접 실행 경로 호환."""
if __name__ == "__main__":
    import runpy
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    runpy.run_module("eyetoarm.gui.gz_view_relay", run_name="__main__")
