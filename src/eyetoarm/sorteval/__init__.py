"""분류 시험 평가 전용 패키지(#186). 공급 순서표(독립 정답)·실행 기록 어댑터·취합·AI/OpenCV 비교.

근거: 계획서 §2.6.2(공급기 — A 시험 순서표 작성·결과 취합)·§2.6.4(도착 확인 기반 집계)·§10.3(분류 평가 지표),
A 상세설계 §2(시험·취합)·§7.1(AI/OpenCV 동일 조건 비교)·§9(evaluation/ 보관), 코드 규약 §5(정답 격리).

- **정답을 읽는 곳은 이 패키지뿐이다.** 런타임 패키지(`vision`·`inputs`·`calibration`·`transforms`·`tcp`·`gui`·
  `common`·`commtest`)는 이 패키지와 `evaluation/` 경로를 참조하지 않는다(`tests/unit/test_ground_truth_isolation.py`).
- 이 패키지도 런타임 패키지를 import하지 않는다. 표준 라이브러리만 쓴다.
- 분류 대상은 볼트만(사용자 2026-11-19). 너트 행은 옵션(`include_nut`, 기본 꺼짐).
"""
