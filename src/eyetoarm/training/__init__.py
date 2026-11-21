"""AI 후보 학습기. 순수 로직만 두고, torch·ultralytics는 학습·평가 자식 프로세스 안에서만 import한다.

운영 비전 파이프라인(`eyetoarm.vision`)을 import하거나 바꾸지 않는다.
"""
