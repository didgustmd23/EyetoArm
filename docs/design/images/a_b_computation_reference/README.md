# A→B 계산 기준 후보 자료

모델 기반 정답 예시 12사례입니다. 실제 RGB 추출·실루엣 계산·운영 유효화 결과가 아닙니다. B가 계산에 필요한 항목·기준 높이·입력 형식을 선택하기 위한 자료이며, 후보가 채택됐다는 뜻은 아닙니다.

- [전체 모아보기](contact_sheet.png)
- [기준 정의](reference_definition.md)
- [기준 좌표 정본](reference_points_r1.jsonl)

각 사례는 왼쪽 원본, 가운데 기존 전체 물체 마스크(계산용 아님), 오른쪽 모델 기반 기준 표시 순서입니다. 가상점/내부 축은 투영 위치만 정답이며 실제 관측 가능성이 입증된 점이 아닙니다. 이미지와 함께 r1 기준 정의·좌표의 가시성 주석을 확인하세요.

| 사례 | 부품 | 이미지 |
|---|---|---|
| R01 | BOLT_A | [확대 보기](R01_BOLT_A.png) |
| R02 | BOLT_A | [확대 보기](R02_BOLT_A.png) |
| R03 | BOLT_A | [확대 보기](R03_BOLT_A.png) |
| R04 | BOLT_A | [확대 보기](R04_BOLT_A.png) |
| R05 | BOLT_B | [확대 보기](R05_BOLT_B.png) |
| R06 | BOLT_B | [확대 보기](R06_BOLT_B.png) |
| R07 | BOLT_B | [확대 보기](R07_BOLT_B.png) |
| R08 | BOLT_B | [확대 보기](R08_BOLT_B.png) |
| R09 | NUT | [확대 보기](R09_NUT.png) |
| R10 | NUT | [확대 보기](R10_NUT.png) |
| R11 | NUT | [확대 보기](R11_NUT.png) |
| R12 | NUT | [확대 보기](R12_NUT.png) |

12사례는 가까운 위치의 정적 배치 표본이며, 전체 위치·팔 자세·실물 조건의 정확도를 보장하지 않습니다. 원본 파일과 상세 작업 증거는 로컬 `_workspace/evidence/2026-11-17-computation-reference-pilot/`에 보존합니다.
