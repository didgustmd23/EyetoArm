"""저장 입력 단계 L(#189) 목록 기록기: 실행기 `b_list_consumer` 자리에서 목록·봉투 v2를 만들어 목록 소비자에 넘기고 기록한다.

근거: 계획 `_workspace/189_builder2_stage_L_plan.md` r0 §6·§7(reviewer 승인 hop 3), A 상세설계 §5.2 B-09 협의안.

- `wrap(consume)`이 낸 함수는 실행기가 `b_list_gate`를 통과한 응답마다 `(context, display_result)`로 부른다.
  녹화 메타로 촬영 정보·촬영 시각 TF를 만들고(`capture_from_meta`, 단계 T 기록기와 같은 함수) `frame_observations_from_ai` →
  `make_list_envelope`(`eyetoarm.b_input/2`) → `consume(envelope)`을 부른다.
- 목록·봉투 오류(`ValueError`: 메타 없음·대조 실패)는 소비자를 부르지 않고 `b_input_rejected`로 남긴다(실행 끝 판정에서 종료 1).
- 소비자 예외·입력 영상 변경은 그대로 올린다(실행 실패).
- 주석 영상은 만들지 않는다(B 목록 결과 형식 U-3 전, reviewer Q6).
- 출력 폴더는 새로 만든다(있으면 FileExistsError). 파일은 덮어쓰지 않는다.
- B `BStage.process_frame`은 아직 없다. 이 기록기는 시험 전용 가짜 소비자 또는 B가 만든 뒤의 소비자를 받는다.
- 시선 기반 끝점 보정(#207, 선택 `view_bias`): 목록 봉투를 만든 직후 공통 지점 `vision/view_bias.apply_view_bias`로 후보 끝점을
  보정한 봉투를 소비자에 넘긴다. 켰을 때만 `b_list_input.view_bias`(후보별 결과·원 끝점)와 요약을 남긴다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from eyetoarm.vision.ai_convert import frame_observations_from_ai
from eyetoarm.vision.capture_observation_recorder import _image_sha, _Jsonl, _plain, capture_from_meta
from eyetoarm.vision.list_envelope import LIST_ENVELOPE_VERSION, make_list_envelope
from eyetoarm.vision.view_bias import apply_view_bias, summarize


class CaptureListRecorder:
    """목록 소비 기록(`list_frames.jsonl`)·terminal 응답(`frames.jsonl`)·요약(`summary.json`)."""

    def __init__(self, out_dir: str, meta, *, consumer_kind: str, view_bias=None) -> None:
        """출력 폴더를 새로 만든다(있으면 FileExistsError). `consumer_kind`는 요약에 남길 소비자 종류(예: 시험 전용 표시).
        `view_bias`(#207)는 `ViewBiasConfig` 또는 None(꺼짐)."""
        if not isinstance(consumer_kind, str) or not consumer_kind:
            raise ValueError(f"consumer_kind must be a non-empty string: {consumer_kind!r}")

        self._meta = meta
        self._consumer_kind = consumer_kind
        self._view_bias = view_bias
        self._view_bias_records: list = []
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=False, exist_ok=False)
        self._lists = _Jsonl(self.dir / "list_frames.jsonl")
        self._frames = _Jsonl(self.dir / "frames.jsonl")
        self.calls = 0
        self.rejected = 0
        self.candidates_sent = 0
        self.reasons: dict[str, int] = {}

    # ---- display_sink 규약(기록 뒤·B 지점 앞) ----
    def on_outcome(self, outcome, current, **_kw) -> None:
        """terminal 응답 하나를 frames.jsonl에 남긴다(단계 T 관측의 유효·사유는 진단용, 목록 전달은 list_frames.jsonl)."""
        ctx, obs = outcome.context, outcome.observation
        f = ctx.frame
        record = {"kind": "outcome", "image_seq": f.image_seq, "source_index": getattr(f, "source_index", None),
                  "job_seq": ctx.job_seq, "generation": ctx.generation, "status": getattr(outcome.status, "value", None),
                  "applied": outcome.applied, "duplicate": outcome.duplicate, "display_current": bool(current),
                  "capture_stamp_ns": f.ros_stamp_ns, "calibration_id": f.calibration_id, "error": outcome.error}

        if obs is not None:
            record.update(stage_t_valid=obs.valid, stage_t_reason=obs.reason)
            key = "VALID" if obs.valid else obs.reason
            self.reasons[key] = self.reasons.get(key, 0) + 1

        self._frames.write(record)

    def on_input(self, *_a, **_kw) -> None:
        """실시간 입력 알림(저장 입력에서는 오지 않음)."""

    # ---- b_list_consumer ----
    def wrap(self, consume: Callable) -> Callable:
        """`b_list_consumer` 자리 함수. 목록·봉투 오류는 기록만 하고 소비자를 부르지 않는다."""
        def list_consumer(ctx, display):
            """메타 → 목록 → 봉투 v2 → 소비자 → 기록."""
            frame = ctx.frame
            before = _image_sha(frame.image)
            keys = {"image_seq": frame.image_seq, "source_index": getattr(frame, "source_index", None),
                    "job_seq": ctx.job_seq, "generation": ctx.generation}

            try:
                capture, tf = capture_from_meta(self._meta, frame)
                observations = frame_observations_from_ai(ctx, display, capture=capture, camera=self._meta.camera, tf=tf,
                                                          world_frame=self._meta.world_frame)

                # 실행기 b_list_gate가 같은 함수로 비어 있지 않음을 확인했다. None이면 계약 위반이다
                if observations is None:
                    raise RuntimeError("b_list_gate passed but frame_observations_from_ai built no list")

                envelope = make_list_envelope(observations, world_frame=self._meta.world_frame)
            except ValueError as exc:
                self.rejected += 1
                self._lists.write({"kind": "b_input_rejected", "input": keys, "error": str(exc)})
                return None

            # #207 공통 보정 지점(끔이면 같은 봉투)
            envelope, bias = apply_view_bias(envelope, None, self._view_bias)
            out = consume(envelope)

            if _image_sha(frame.image) != before:
                raise RuntimeError("input frame image changed during list consumption")

            self.calls += 1
            self.candidates_sent += len(observations.candidates)
            self._lists.write({"kind": "b_list_input", "b_list_call": self.calls, "input": keys,
                               "b_input": envelope.to_record(), "consumer_kind": self._consumer_kind,
                               "consumer_result": _plain(out), "frame_image_sha256": before,
                               "frame_image_unchanged": True,
                               **({} if self._view_bias is None else {"view_bias": bias})})

            if self._view_bias is not None:
                self._view_bias_records.append(bias)

            return out

        return list_consumer

    def summary(self, record: dict) -> None:
        """끝 요약을 summary.json으로(새 파일)."""
        doc = {"b_input_version": LIST_ENVELOPE_VERSION, "b_list_calls_recorded": self.calls,
               "b_input_rejected": self.rejected, "candidates_sent": self.candidates_sent,
               "consumer_kind": self._consumer_kind, "stage_t_observation_reasons": self.reasons, **record}

        if self._view_bias is not None:
            doc["view_bias"] = {"enabled": True, **self._view_bias.to_json(), **summarize(self._view_bias_records)}

        with open(self.dir / "summary.json", "x", encoding="utf-8") as fh:
            json.dump(_plain(doc), fh, ensure_ascii=False, indent=1, allow_nan=False)

    def close(self) -> None:
        """파일을 닫는다."""
        self._lists.close()
        self._frames.close()


__all__ = ["CaptureListRecorder"]
