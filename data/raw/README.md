# 촬영 원본

스마트폰 등에서 얻은 원본 이미지·영상을 로컬에 보관합니다. 이 안내와 `.gitkeep`을 제외한 내용은 Git에서 제외됩니다. 공유용 소규모 예제는 `../samples/`에 보관합니다.

## 외부 데이터셋 출처·라이선스

이 폴더에 내려받은 외부 데이터셋의 출처와 라이선스를 기록합니다. 데이터·압축 파일·동봉 안내 파일은 이 README와 달리 Git에서 제외되며, 원본을 수정하지 않습니다.

### Bolt nut segmentation

| 항목 | 내용 |
|---|---|
| 이름 | Bolt nut segmentation |
| 제공자 | Bolt Nut (Roboflow Universe 게시명) |
| 버전 | v1 (2024-11-27) |
| 출처 | <https://universe.roboflow.com/bolt-nut-wikc6/bolt-nut-segmentation/dataset/1> |
| 라이선스 | CC BY 4.0 — <https://creativecommons.org/licenses/by/4.0/> (공식 한국어: <https://creativecommons.org/licenses/by/4.0/deed.ko>) |
| 형식·규모 | COCO Segmentation, 총 3,436장(학습 2,949 / 검증 315 / 시험 172) |
| 로컬 파일 | `Bolt nut segmentation.v1i.coco-segmentation.zip`, `train/`·`valid/`·`test/`, 동봉 안내 `README.dataset.txt`·`README.roboflow.txt` (모두 Git 제외) |
| 근거 | 동봉 `README.dataset.txt`(출처·`License: CC BY 4.0`), `README.roboflow.txt`(버전·이미지 수·형식) |

**현재 상태.** 원본 다운로드와 압축 해제를 마쳤습니다. 형식 변환과 라벨 보정은 하지 않았습니다. `train`의 Bolt 3장·Nut 2장 윤곽을 표본으로 검토했으며, 원본은 수정하지 않았습니다.

**재분할 목록(2026-11-12, v2 갱신).** 공개 분할은 같은 장면이 train과 valid·test에 섞여 있어, 장면 묶음 단위로 다시 나눈 목록을 `../splits/resplit_v2/`(`train.txt`·`valid.txt`·`test.txt`·`manifest.json`, 현재 목록)에 두었습니다. v2는 부품 크기 구간까지 분할 간에 맞춘 목록입니다. 첫 목록 `../splits/resplit_v1/`은 **대체됨**(크기 분포 쏠림: valid 너트 짧은 변 하위 5% 25 px)이며 비교용으로 남겨 두었습니다. 이 폴더 기준 상대 경로 목록이며 **원본은 수정·이동하지 않았습니다**. 만드는 방법은 `scripts/resplit_dataset.py`와 [A 상세설계 §12](../../docs/design/A담당_비전_상세설계.md) 데이터 분리 행을 따릅니다.

**증강 표기 불일치.** 동봉 `README.roboflow.txt`는 「No image augmentation」이라고 적었지만, `train`에는 원본 줄기(파일명 `.rf.` 앞)당 최대 3장의 사본이 있고 사본끼리 뒤집기·회전·잡음·밝기가 다릅니다. 증강 사본으로 추정하며, 재분할에서는 같은 줄기를 한 분할에 두고 valid·test에는 줄기당 1장만 씁니다. 전처리는 640×640 늘이기(Stretch)라 원본 종횡비가 바뀌었습니다.

**이용 조건(요약).** CC BY 4.0에 따라 이 데이터셋의 원본이나 변경본을 공유할 때는 저작자(제공자), 출처, 라이선스를 표시합니다(보고서·발표에 데이터셋 출처를 적는 것은 권장 기록 방식입니다). 데이터를 변경해 쓰면(형식 변환·라벨 보정·일부 선택 등) 변경했다는 사실도 함께 적습니다. 정확한 조건은 위 라이선스 원문을 따릅니다.

이 라이선스는 데이터셋에만 적용됩니다. 학습에 쓰는 모델·가중치·라이브러리의 라이선스는 각각 따로 확인하고 기록합니다.
