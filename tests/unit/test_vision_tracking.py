"""A-6 1차 인스턴스 추적 순수 상태기(#151 S1) 시험 — 발급기·TrackFrame·연결·모호성·경계 입력·유한 자원·M1/M2 메커니즘.

근거: `_workspace/150_builder2_part_track_id_plan.md` r2 §2·§3(ID 규약·모드)·§3.5(발급기)·§3.6(M1/M2 보장 범위)·§4(갱신 규칙)·§6·§7 S1,
reviewer 구현 명세 #151 §1(D1~D11)·§2. 가짜 검출 입력만 쓴다(ROS·GPU·모델 없음).

**이 시험은 메커니즘 시험이다.** 통과 수를 추적 성능·신원 정확도로 해석하지 않는다. M2 휴리스틱의 미탐·오탐 사례는 「보장하지 않음」을
기대값으로 고정한 것이다.
"""
import dataclasses
import gc
import math
import unittest
import weakref

from eyetoarm.vision.tracking import (DEFAULT_TRACKER_CONFIG, FrameKey, PartTracker, PartTrackIdIssuer, TrackAssignment,
                                      TrackerConfig, TrackFrame, TrackingContractError, TrackingInputError, TrackInput,
                                      TrackMode, TrackState, inputs_from_ai)
from test_ai_result import instance, seg_result

RUN, SRC, CFG = "run-t", "gz_link", "c" * 64


def key(image_seq, *, generation=0, job_seq=None, run_id=RUN, source_id=SRC, config=CFG, capture_generation=None):
    """작업 식별자 묶음(job_seq 기본 = image_seq, capture_generation 기본 = generation)."""
    return FrameKey(run_id=run_id, source_id=source_id, image_seq=image_seq,
                    capture_generation=generation if capture_generation is None else capture_generation,
                    generation=generation, job_seq=image_seq if job_seq is None else job_seq, detector_config_id=config)


def det(index, cx, cy, *, cls="bolt", conf=0.9, w=20.0, h=10.0, clipped=False):
    """중심 (cx, cy), 크기 w×h 후보 입력."""
    return TrackInput(index=index, class_name=cls, confidence=conf, box_xyxy=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                      clipped=clipped, candidate=True, untracked_reason=None)


class Base(unittest.TestCase):
    """발급기·추적기 준비와 프레임 공급 도우미."""

    def setUp(self):
        """같은 프로세스 발급기 1개와 M2 추적기."""
        self.issuer = PartTrackIdIssuer(namespace="ns-test")
        self.tracker = self.make()
        self.seq = 0
        self.t = 1000

    def make(self, **kw):
        """발급기를 주입한 추적기."""
        kw.setdefault("mode", TrackMode.HEURISTIC)
        return PartTracker(self.issuer, **kw)

    def step(self, dets, *, dt=100, tracker=None, camera_static=None, **key_over):
        """다음 프레임(SIM_TIME dt ms 뒤)을 넣는다. camera_static은 M1 가짜 정지 신호."""
        self.seq += 1
        self.t += dt
        return (tracker or self.tracker).update(key(self.seq, **key_over), self.t, dets, camera_static=camera_static)

    def ids(self, frame):
        """instance index → part_track_id."""
        return {a.instance_index: a.part_track_id for a in frame.assignments}

    def states(self, frame):
        """instance index → state."""
        return {a.instance_index: a.state for a in frame.assignments}


class IssuerTest(Base):
    """발급기(§3.5, D11)."""

    def test_monotonic_positive_and_namespace(self):
        """번호는 1부터 단조 증가, namespace는 지정값 그대로, 기본 namespace는 발급기마다 다르다."""
        self.assertEqual([self.issuer.next_id() for _ in range(3)], [1, 2, 3])
        self.assertEqual(self.issuer.namespace, "ns-test")
        self.assertNotEqual(PartTrackIdIssuer().namespace, PartTrackIdIssuer().namespace)

    def test_epoch_counter_separate(self):
        """tracker_epoch 순번은 ID 번호와 별개다(ID를 소비하지 않음)."""
        e1 = self.issuer.next_tracker_epoch()
        self.assertEqual(self.issuer.next_id(), 1)
        self.assertGreater(self.issuer.next_tracker_epoch(), e1)
        self.assertEqual(self.issuer.next_id(), 2)

    def test_bad_namespace(self):
        """빈 문자열·비문자열 namespace는 계약 오류."""
        for bad in ("", 3, b"ns"):
            with self.subTest(bad=bad):
                with self.assertRaises(TrackingContractError):
                    PartTrackIdIssuer(namespace=bad)

    def test_tracker_requires_issuer(self):
        """D11: 발급기 없이 추적기를 만들면 계약 오류(몰래 만들거나 꺼짐으로 바꾸지 않음)."""
        for bad in (None, object(), "ns"):
            with self.subTest(bad=bad):
                with self.assertRaises(TrackingContractError):
                    PartTracker(bad, mode=TrackMode.HEURISTIC)

    def test_mode_validation(self):
        """M0(끔)은 추적기를 만들지 않는 것으로 표현한다. 모드 값이 아니면 계약 오류."""
        for bad in (TrackMode.OFF, "fast", None):
            with self.subTest(bad=bad):
                with self.assertRaises(TrackingContractError):
                    PartTracker(self.issuer, mode=bad)


class ConfigTest(unittest.TestCase):
    """개발 임시값 설정 검증."""

    def test_defaults_are_documented_dev_values(self):
        """기본값은 A 개발 임시값(운영 채택값 아님)이며 주입 가능하다."""
        c = DEFAULT_TRACKER_CONFIG
        self.assertEqual((c.min_hits, c.max_gap_ms), (3, 500))
        self.assertEqual(TrackerConfig(min_hits=1).min_hits, 1)

    def test_invalid_values(self):
        """bool·0·음수·NaN·무한은 거부."""
        for name, bad in (("min_hits", 0), ("min_hits", True), ("max_gap_ms", -1), ("gate_box_ratio", math.nan),
                          ("max_tracks", 0), ("ambiguity_margin", math.inf), ("new_track_min_conf", 1.5)):
            with self.subTest(name=name, bad=bad):
                with self.assertRaises(ValueError):
                    TrackerConfig(**{name: bad})


class LinkingTest(Base):
    """정상 이동·여러 개·누락·진입·프레임 건너뜀."""

    def test_confirm_after_min_hits_then_stable(self):
        """tentative 2번(None) → 3번째 hit에서 번호 1 발급 → 같은 번호 유지."""
        frames = [self.step([det(0, 100 + 10 * k, 50)]) for k in range(5)]

        self.assertEqual([self.ids(f)[0] for f in frames], [None, None, 1, 1, 1])
        self.assertEqual([self.states(f)[0] for f in frames][:3],
                         [TrackState.TENTATIVE, TrackState.TENTATIVE, TrackState.CONFIRMED])
        f = frames[-1]
        self.assertEqual((f.run_id, f.source_id, f.image_seq, f.job_seq, f.detector_config_id),
                         (RUN, SRC, 5, 5, CFG))
        self.assertEqual((f.mode, f.track_namespace), (TrackMode.HEURISTIC, "ns-test"))
        self.assertTrue(f.updated)

    def test_two_same_class_separated(self):
        """같은 클래스 둘이 떨어져 평행 이동하면 서로 다른 번호를 유지한다."""
        for k in range(5):
            f = self.step([det(0, 100 + 10 * k, 50), det(1, 100 + 10 * k, 200)])

        self.assertEqual(sorted(self.ids(f).values()), [1, 2])

    def test_short_miss_within_gap_keeps_id(self):
        """confirmed 뒤 누락 허용(500 ms) 안의 재등장은 같은 번호(등속 예측)."""
        for k in range(4):
            f = self.step([det(0, 100 + 10 * k, 50)])

        self.step([])                                     # 무검출 완료(안 보임 1회)
        f = self.step([det(0, 160, 50)])                  # 200 ms 뒤 예측 위치 근처

        self.assertEqual(self.ids(f)[0], 1)

    def test_miss_beyond_gap_new_tentative_no_reuse(self):
        """누락 허용을 넘으면 트랙 은퇴, 같은 자리 재등장도 새 tentative → 새 번호(1 재사용 없음)."""
        for k in range(4):
            self.step([det(0, 100 + 10 * k, 50)])

        frames = [self.step([det(0, 200 + 10 * k, 50)], dt=600 if k == 0 else 100) for k in range(3)]

        self.assertEqual([self.ids(f)[0] for f in frames], [None, None, 2])
        self.assertGreaterEqual(self.tracker.stats()["ids_retired"], 1)

    def test_new_object_enters(self):
        """기존 트랙이 있는 중 새 물체 진입: 기존 번호 유지, 새 물체는 tentative 뒤 새 번호."""
        for k in range(3):
            self.step([det(0, 100 + 10 * k, 50)])

        frames = [self.step([det(0, 130 + 10 * k, 50), det(1, 400, 300 + 5 * k)]) for k in range(3)]

        self.assertEqual([self.ids(f)[0] for f in frames], [1, 1, 1])
        self.assertEqual([self.ids(f)[1] for f in frames], [None, None, 2])

    def test_frame_skip_uses_time(self):
        """프레임 건너뜀(SIM_TIME 간격 300 ms, 허용 안): 속도×시간 예측으로 같은 번호."""
        for k in range(4):
            self.step([det(0, 100 + 10 * k, 50)])         # 100 ms마다 10 px

        f = self.step([det(0, 160, 50)], dt=300)

        self.assertEqual(self.ids(f)[0], 1)

    def test_no_detection_vs_no_information(self):
        """무검출 완료는 miss로 세고, 정보 없음(update 안 부름)은 세지 않는다 — tentative는 무검출 1회에 사라진다."""
        self.step([det(0, 100, 50)])
        self.step([])                                       # tentative 소멸
        f = self.step([det(0, 120, 50)])
        self.assertEqual(self.states(f)[0], TrackState.TENTATIVE)
        self.assertEqual(self.tracker.stats()["tentative_dropped"], 1)


class AmbiguityTest(Base):
    """교차·합쳐짐·분리·클래스 바뀜·저점수·잘림."""

    def confirm_two(self, y0=50, y1=80):
        """같은 클래스 둘을 확정."""
        for k in range(3):
            f = self.step([det(0, 100 + 10 * k, y0), det(1, 100 + 10 * k, y1)])

        self.assertEqual(sorted(self.ids(f).values()), [1, 2])

    def test_merge_one_detection_two_tracks(self):
        """두 트랙 사이에 검출 1개(합쳐짐): ambiguous·None, 두 트랙 종료. 다음에 다시 나뉘어도 1·2는 재사용되지 않는다."""
        self.confirm_two(50, 66)
        f = self.step([det(0, 130, 58)])

        self.assertEqual((self.states(f)[0], self.ids(f)[0]), (TrackState.AMBIGUOUS, None))
        later = [self.step([det(0, 140 + 10 * k, 50), det(1, 140 + 10 * k, 66)]) for k in range(3)]
        self.assertTrue(set(self.ids(later[-1]).values()).isdisjoint({1, 2}))

    def test_split_one_track_two_detections(self):
        """트랙 1개 근처에 비슷한 거리 검출 2개(분리): 둘 다 ambiguous·None, 트랙 종료."""
        for k in range(3):
            self.step([det(0, 100 + 10 * k, 50)])

        f = self.step([det(0, 130, 46), det(1, 130, 54)])

        self.assertEqual(self.states(f), {0: TrackState.AMBIGUOUS, 1: TrackState.AMBIGUOUS})
        self.assertEqual(set(self.ids(f).values()), {None})

    def test_crossing_same_class(self):
        """동종 교차(두 트랙이 같은 자리로 모임) → ambiguous로 끊김, 교차 뒤 번호를 이어 주장하지 않는다."""
        for k in range(2):
            self.step([det(0, 100 + 10 * k, 50 + 10 * k), det(1, 100 + 10 * k, 90 - 10 * k)])

        f = self.step([det(0, 120, 70), det(1, 121, 71)])                # 두 예측이 (120, 70)에서 만남

        self.assertEqual(set(self.states(f).values()), {TrackState.AMBIGUOUS})

    def test_class_change_not_linked(self):
        """같은 자리의 클래스가 바뀐 검출은 연결하지 않고 tentative(None), 원래 트랙 번호를 넘기지 않는다."""
        for k in range(3):
            self.step([det(0, 100 + 10 * k, 50)])

        f = self.step([det(0, 130, 50, cls="nut")])

        self.assertEqual((self.states(f)[0], self.ids(f)[0]), (TrackState.TENTATIVE, None))
        self.assertEqual(f.assignments[0].reason, "class_changed")

    def test_low_confidence_maintains_but_never_starts(self):
        """저점수(<0.25) 검출은 기존 트랙 유지에만 쓰이고 새 트랙을 만들지 않는다."""
        f = self.step([det(0, 100, 50, conf=0.1)])
        self.assertEqual((self.states(f)[0], f.assignments[0].reason), (TrackState.UNTRACKED, "low_confidence"))

        for k in range(3):
            self.step([det(0, 100 + 10 * k, 200)])

        f = self.step([det(0, 130, 200, conf=0.1)])
        self.assertEqual(self.ids(f)[0], 1)

    def test_clipped_does_not_confirm_but_keeps_confirmed(self):
        """잘린 검출만으로는 확정 번호를 발급하지 않는다. 이미 확정된 트랙은 잘림 검출로 번호를 유지한다(reason clipped)."""
        frames = [self.step([det(0, 100 + 10 * k, 50, clipped=True)]) for k in range(4)]
        self.assertEqual({self.ids(f)[0] for f in frames}, {None})

        tracker = self.make()
        for k in range(3):
            self.step([det(0, 100 + 10 * k, 300)], tracker=tracker)

        f = self.step([det(0, 130, 300, clipped=True)], tracker=tracker)
        self.assertEqual((self.ids(f)[0], f.assignments[0].reason), (self.issuer.issued, "clipped"))


class CleanHitTest(Base):
    """S1 r1 F1: 확정 근거는 잘리지 않은 **연속** hit뿐이다(잘림 hit는 유지만 하고 연속을 끊는다)."""

    def m1(self):
        """M1 추적기(정지 신호 True)."""
        return self.make(mode=TrackMode.CONFIRMED_STATIC)

    def feed(self, tracker, pattern):
        """pattern(True=잘림)대로 같은 자리 근처 검출을 넣고 (번호, 상태, hits) 목록을 돌려준다."""
        out = []

        for k, clipped in enumerate(pattern):
            f = self.step([det(0, 100 + k, 50, clipped=clipped)], tracker=tracker, camera_static=True)
            a = f.assignments[0]
            out.append((a.part_track_id, a.state, a.hits))

        return out

    def test_clipped_clipped_normal_not_confirmed(self):
        """잘림·잘림·정상: 첫 정상 hit 하나로는 확정하지 않는다(정상 연속 1)."""
        out = self.feed(self.m1(), [True, True, False])

        self.assertEqual(out[-1], (None, TrackState.TENTATIVE, 1))

    def test_clipped_then_three_normals_confirms_on_third_normal(self):
        """잘림 뒤 정상 3연속이면 세 번째 정상에서 확정."""
        out = self.feed(self.m1(), [True, False, False, False])

        self.assertEqual([o[0] for o in out], [None, None, None, 1])
        self.assertEqual([o[2] for o in out], [0, 1, 2, 3])

    def test_normal_clipped_normal_breaks_streak(self):
        """정상·정상·잘림·정상: 잘림이 연속을 끊어 아직 미확정, 그 뒤 정상 2번 더(연속 3)에서 확정."""
        out = self.feed(self.m1(), [False, False, True, False, False, False])

        self.assertEqual([o[0] for o in out], [None, None, None, None, None, 1])
        self.assertEqual([o[2] for o in out], [1, 2, 0, 1, 2, 3])

    def test_confirmed_keeps_id_through_clipped(self):
        """이미 확정된 트랙은 잘림 hit에서도 번호 유지(reason clipped), 정상 연속 수는 0으로 끊긴다."""
        out = self.feed(self.m1(), [False, False, False, True, False])

        self.assertEqual([o[0] for o in out], [None, None, 1, 1, 1])
        self.assertEqual(out[3], (1, TrackState.CONFIRMED, 0))


class EpochTest(Base):
    """S1 r1 F2: 트랙이 있던 전체 종료마다 새 epoch, 트랙이 없는 반복 종료는 epoch를 올리지 않는다."""

    def confirm(self, tracker, **kw):
        """한 물체를 확정."""
        for k in range(3):
            f = self.step([det(0, 100 + 10 * k, 50)], tracker=tracker, **kw)

        return f

    def test_m1_not_static_new_epoch_once(self):
        """M1: 트랙이 있는 상태에서 정지 신호 False → 새 epoch. 트랙 없이 False 반복은 epoch 그대로."""
        t = self.make(mode=TrackMode.CONFIRMED_STATIC)
        f = self.confirm(t, camera_static=True)
        e0 = f.tracker_epoch

        f1 = self.step([det(0, 140, 50)], tracker=t, camera_static=False)
        f2 = self.step([det(0, 150, 50)], tracker=t, camera_static=False)
        f3 = self.step([det(0, 160, 50)], tracker=t, camera_static=None)

        self.assertGreater(f1.tracker_epoch, e0)
        self.assertEqual((f2.tracker_epoch, f3.tracker_epoch), (f1.tracker_epoch, f1.tracker_epoch))
        self.assertEqual(t.tracker_epoch, f1.tracker_epoch)

    def test_m2_camera_motion_new_epoch_and_new_tracks_in_it(self):
        """M2 카메라 이동 의심 전체 종료 → 새 epoch, 그 프레임에서 새로 만든 트랙은 새 epoch에 속한다."""
        for k in range(3):
            f = self.step([det(0, 100 + 10 * k, 50), det(1, 100 + 10 * k, 200)])

        e0 = f.tracker_epoch
        f = self.step([det(0, 230, 50), det(1, 230, 200)])

        self.assertEqual(f.event, "camera_motion_suspected")
        self.assertGreater(f.tracker_epoch, e0)
        later = self.step([det(0, 240, 50), det(1, 240, 200)])
        self.assertEqual(later.tracker_epoch, f.tracker_epoch)
        self.assertEqual({a.state for a in later.assignments}, {TrackState.TENTATIVE})

    def test_reset_idempotent_without_tracks(self):
        """트랙이 있을 때 reset은 새 epoch, 바로 다시 reset(트랙 없음)은 epoch 그대로(사유 계수는 셈)."""
        self.confirm(self.tracker)
        e0 = self.tracker.tracker_epoch
        self.tracker.reset("stop")
        e1 = self.tracker.tracker_epoch
        self.tracker.reset("stop")

        self.assertGreater(e1, e0)
        self.assertEqual(self.tracker.tracker_epoch, e1)
        self.assertEqual(self.tracker.stats()["resets"]["stop"], 2)

    def test_disable_idempotent(self):
        """disable: 트랙이 있으면 새 epoch, 다시 disable해도 epoch·첫 사유 그대로."""
        self.confirm(self.tracker)
        e0 = self.tracker.tracker_epoch
        self.tracker.disable("first")
        e1 = self.tracker.tracker_epoch
        self.tracker.disable("second")

        self.assertGreater(e1, e0)
        self.assertEqual(self.tracker.tracker_epoch, e1)
        self.assertEqual(self.tracker.stats()["disabled"], "first")

    def test_epoch_does_not_consume_ids(self):
        """epoch 증가는 번호를 소비하지 않는다(전체 종료 뒤 다음 확정 번호는 이어진 값)."""
        self.confirm(self.tracker)
        self.tracker.reset("x")
        f = self.confirm(self.tracker)

        self.assertEqual(f.assignments[0].part_track_id, 2)


class CameraModeTest(Base):
    """M2 휴리스틱(보장 범위 밖 사례 포함)·M1 가짜 신호."""

    def confirm_two(self):
        """두 물체를 같은 속도로 확정."""
        for k in range(3):
            self.step([det(0, 100 + 10 * k, 50), det(1, 100 + 10 * k, 200)])

    def test_m2_coherent_jump_resets_all(self):
        """M2: 모든 트랙이 예측에서 같은 방향으로 크게 튀면 카메라 이동 의심 → 전 트랙 종료(event), 이후 새 번호만."""
        self.confirm_two()
        f = self.step([det(0, 230, 50), det(1, 230, 200)])   # 예측 140 → 230(+90 px, 둘 다)

        self.assertEqual(f.event, "camera_motion_suspected")
        self.assertEqual(set(self.ids(f).values()), {None})
        later = [self.step([det(0, 240 + 10 * k, 50), det(1, 240 + 10 * k, 200)]) for k in range(3)]
        self.assertTrue(set(self.ids(later[-1]).values()).isdisjoint({1, 2}))

    def test_m2_limit_small_camera_shift_not_detected(self):
        """M2 한계(미탐) 고정: 게이트 안의 작은 공통 이동은 카메라 이동인지 구분 못 하고 같은 번호로 이어진다(보장 안 함)."""
        self.confirm_two()
        f = self.step([det(0, 145, 54), det(1, 145, 204)])

        self.assertIsNone(f.event)
        self.assertEqual(sorted(self.ids(f).values()), [1, 2])

    def test_m2_limit_single_track_cannot_judge(self):
        """M2 한계: 트랙이 1개면 공통 이동 판단 불가 — 큰 이동은 연결 실패(새 tentative)일 뿐 event 없음."""
        for k in range(3):
            self.step([det(0, 100 + 10 * k, 50)])

        f = self.step([det(0, 260, 50)])

        self.assertIsNone(f.event)
        self.assertEqual(self.ids(f)[0], None)

    def test_m2_limit_belt_speed_change_false_alarm(self):
        """M2 한계(오탐) 고정: 벨트가 갑자기 빨라져도 모든 부품이 같이 튀므로 카메라 이동으로 의심해 끊는다."""
        self.confirm_two()
        f = self.step([det(0, 220, 50), det(1, 220, 200)])

        self.assertEqual(f.event, "camera_motion_suspected")

    def test_m1_fake_signal(self):
        """M1(가짜 정지 신호): True 구간만 연결, False·None이면 전 트랙 종료·untracked(camera_not_static)."""
        t = self.make(mode=TrackMode.CONFIRMED_STATIC)
        for k in range(3):
            f = self.step([det(0, 100 + 10 * k, 50)], tracker=t, camera_static=True)

        self.assertEqual(self.ids(f)[0], 1)

        for signal in (False, None):
            with self.subTest(signal=signal):
                f = self.step([det(0, 140, 50)], tracker=t, camera_static=signal)
                self.assertEqual((self.states(f)[0], f.assignments[0].reason),
                                 (TrackState.UNTRACKED, "camera_not_static"))

    def test_m2_rejects_camera_signal(self):
        """M2에 정지 신호를 넣으면 계약 오류(M1 신호를 임의로 흉내 내지 않음)."""
        with self.assertRaises(TrackingContractError):
            self.step([det(0, 100, 50)], camera_static=True)


class UpdateRuleTest(Base):
    """두 번 갱신 금지·세대·범위(§4.1)."""

    def test_same_frame_reprocess_not_updated(self):
        """같은 프레임(같은 시각) 재입력은 갱신하지 않고 전부 untracked(not_updated: stale_frame), 상태 무변."""
        for k in range(3):
            f = self.step([det(0, 100 + 10 * k, 50)])

        before = self.tracker.stats()
        again = self.tracker.update(key(self.seq, job_seq=99), self.t, [det(0, 120, 50)])

        self.assertFalse(again.updated)
        self.assertEqual((again.assignments[0].state, again.assignments[0].reason),
                         (TrackState.UNTRACKED, "stale_frame"))
        self.assertEqual(self.tracker.stats()["updates"], before["updates"])

    def test_earlier_frame_reprocess_after_later_not_updated(self):
        """나중 프레임 뒤에 앞 프레임(더 이른 시각)을 재처리해도 갱신하지 않는다(마지막 키만 보는 규칙이 아님)."""
        self.step([det(0, 100, 50)])
        t_first = self.t
        self.step([det(0, 110, 50)])
        self.step([det(0, 120, 50)])

        again = self.tracker.update(key(1, job_seq=50), t_first, [det(0, 100, 50)])
        self.assertFalse(again.updated)

    def test_old_generation_not_updated_new_generation_resets(self):
        """이전 세대 결과는 갱신 안 함(old_generation). 새 세대·새 run·새 source·새 설정이면 전 트랙 종료 뒤 새 epoch."""
        for k in range(3):
            self.step([det(0, 100 + 10 * k, 50)], generation=1)

        epoch = self.tracker.tracker_epoch
        old = self.step([det(0, 130, 50)], generation=0)
        self.assertEqual((old.updated, old.assignments[0].reason), (False, "old_generation"))

        for over in ({"generation": 2}, {"generation": 2, "run_id": "run-2"}, {"generation": 2, "run_id": "run-2", "source_id": "s2"},
                     {"generation": 2, "run_id": "run-2", "source_id": "s2", "config": "d" * 64}):
            with self.subTest(over=over):
                f = self.step([det(0, 140, 50)], **over)
                self.assertTrue(f.updated)
                self.assertEqual(self.ids(f)[0], None)
                self.assertGreater(self.tracker.tracker_epoch, epoch)
                epoch = self.tracker.tracker_epoch

    def test_explicit_reset_and_disable(self):
        """reset(사유): 전 트랙 종료·epoch 증가·사유 계수. disable: 이후 모든 프레임 untracked(tracking_disabled), 이전 번호가 남지 않음."""
        for k in range(3):
            self.step([det(0, 100 + 10 * k, 50)])

        epoch = self.tracker.tracker_epoch
        self.tracker.reset("sim_time_rewind")
        self.assertGreater(self.tracker.tracker_epoch, epoch)
        self.assertEqual(self.tracker.stats()["resets"]["sim_time_rewind"], 1)

        self.tracker.disable("boom")
        f = self.step([det(0, 140, 50)])
        self.assertEqual((f.updated, f.assignments[0].state, f.assignments[0].reason),
                         (False, TrackState.UNTRACKED, "tracking_disabled"))
        self.assertTrue(self.tracker.disabled)


class IssuerSharingTest(Base):
    """발급기 공유·재생성(P3)."""

    def run_confirmed(self, tracker, n=1, y=50):
        """n개 물체를 그 추적기로 확정하고 번호 집합을 돌려준다."""
        for k in range(3):
            f = self.step([det(i, 100 + 10 * k, y + 100 * i) for i in range(n)], tracker=tracker)

        return set(self.ids(f).values())

    def test_new_tracker_object_same_issuer_no_reuse(self):
        """추적기를 새로 만들어도(경계·새 실행) 같은 발급기면 번호 겹침 0."""
        a = self.run_confirmed(self.tracker, 2)
        b = self.run_confirmed(self.make(), 2, y=600)

        self.assertEqual((a, b), ({1, 2}, {3, 4}))

    def test_two_sources_share_issuer(self):
        """source 2개 추적기가 한 발급기를 공유해도 번호가 겹치지 않는다."""
        t1, t2 = self.make(), self.make()
        ids = set()

        for k in range(3):
            f1 = self.step([det(0, 100 + 10 * k, 50)], tracker=t1, source_id="cam-a")
            f2 = self.step([det(0, 100 + 10 * k, 50)], tracker=t2, source_id="cam-b")

        ids = {self.ids(f1)[0], self.ids(f2)[0]}
        self.assertEqual(len(ids), 2)

    def test_two_issuers_distinguished_by_namespace(self):
        """발급기 2개(프로세스 재시작 모사)는 번호가 겹쳐도 namespace가 달라 (namespace, id)로 구분된다."""
        other = PartTrackIdIssuer(namespace="ns-other")
        t2 = PartTracker(other, mode=TrackMode.HEURISTIC)

        for k in range(3):
            f1 = self.step([det(0, 100 + 10 * k, 50)])
            f2 = self.step([det(0, 100 + 10 * k, 50)], tracker=t2)

        self.assertEqual((self.ids(f1)[0], self.ids(f2)[0]), (1, 1))
        self.assertNotEqual((f1.track_namespace, 1), (f2.track_namespace, 1))

    def test_tracker_epoch_independent_of_ids(self):
        """tracker_epoch는 추적기 생성·reset마다 증가, 번호를 소비하지 않는다."""
        e0 = self.tracker.tracker_epoch
        t2 = self.make()
        self.assertGreater(t2.tracker_epoch, e0)
        self.assertEqual(self.issuer.issued, 0)


class InputValidationTest(Base):
    """경계 입력(명시적 거부·상태 무변)."""

    def test_bad_inputs_rejected_without_state_change(self):
        """bool·음수 시각, NaN·뒤집힌 박스, 중복·bool index, NaN 점수, 비문자 클래스는 TrackingInputError·상태 무변."""
        for k in range(3):
            self.step([det(0, 100 + 10 * k, 50)])

        before = self.tracker.stats()
        cases = [
            (key(50), True, [det(0, 130, 50)]),
            (key(50), -1, [det(0, 130, 50)]),
            (key(50), 9999, [dataclasses.replace(det(0, 130, 50), box_xyxy=(math.nan, 0.0, 1.0, 1.0))]),
            (key(50), 9999, [dataclasses.replace(det(0, 130, 50), box_xyxy=(10.0, 10.0, 5.0, 20.0))]),
            (key(50), 9999, [det(0, 130, 50), det(0, 300, 50)]),
            (key(50), 9999, [dataclasses.replace(det(0, 130, 50), index=True)]),
            (key(50), 9999, [dataclasses.replace(det(0, 130, 50), confidence=math.nan)]),
            (key(50), 9999, [dataclasses.replace(det(0, 130, 50), class_name=3)]),
            ("not a key", 9999, [det(0, 130, 50)]),
        ]

        for k_, ts, dets in cases:
            with self.subTest(ts=ts, dets=dets):
                with self.assertRaises(TrackingInputError):
                    self.tracker.update(k_, ts, dets)

        self.assertEqual(self.tracker.stats(), before)


class ResourceTest(Base):
    """유한 자원·참조(§6)."""

    def test_capacity(self):
        """active+lost 상한 초과: 새 후보는 untracked(track_capacity), 상한 유지."""
        tracker = self.make(config=TrackerConfig(max_tracks=3))
        f = self.step([det(i, 50 + 60 * i, 50) for i in range(5)], tracker=tracker)

        self.assertEqual([a.reason for a in f.assignments][3:], ["track_capacity", "track_capacity"])
        self.assertLessEqual(tracker.stats()["tracks"], 3)

    def test_lost_retired_oldest_first_when_full(self):
        """상한이 찼을 때 lost 트랙부터 은퇴해 새 후보 자리를 만든다."""
        tracker = self.make(config=TrackerConfig(max_tracks=2, min_hits=1))
        self.step([det(0, 50, 50), det(1, 300, 50)], tracker=tracker)
        self.step([det(0, 60, 50)], tracker=tracker)           # 1번 lost
        f = self.step([det(0, 70, 50), det(1, 600, 400)], tracker=tracker)

        self.assertIsNotNone(self.ids(f)[1])

    def test_no_image_or_mask_references(self):
        """추적 상태는 AiSegResult·다각형·영상 참조를 붙잡지 않는다(약한 참조 회수)."""
        result = seg_result([instance(0, box_xyxy=(90.0, 45.0, 110.0, 55.0), polygon=[[90, 45], [110, 45], [110, 55], [90, 55]])])
        ref = weakref.ref(result)
        poly_ref = weakref.ref(result.instances[0].polygon)
        self.step(inputs_from_ai(result))
        del result
        gc.collect()

        self.assertIsNone(ref())
        self.assertIsNone(poly_ref())

    def test_frame_is_immutable(self):
        """TrackFrame·TrackAssignment는 frozen, assignments는 tuple."""
        f = self.step([det(0, 100, 50)])

        self.assertIsInstance(f, TrackFrame)
        self.assertIsInstance(f.assignments, tuple)
        self.assertIsInstance(f.assignments[0], TrackAssignment)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            f.assignments[0].part_track_id = 9


class FromAiTest(Base):
    """AiSegResult → 후보 변환(D8): 원래 검출·거부를 바꾸지 않는다."""

    def test_candidate_rules_and_untracked_reasons(self):
        """in_scope·기하 정상만 후보. 비대상·저점수·기하 거부는 untracked 사유, 잘림은 후보(clipped)."""
        result = seg_result([
            instance(0),
            instance(1, rejected="NON_TARGET_CLASS", class_index=1, class_name="nut"),
            instance(2, rejected="LOW_CONFIDENCE", confidence=0.01),
            instance(3, rejected="SMALL_MASK"),
            instance(4, rejected="EMPTY_MASK", polygon=None),
            instance(5, clipped=True),
        ])
        def snapshot():
            """인스턴스 스칼라 필드와 다각형 바이트(입력 불변 비교용)."""
            return [(i.index, i.class_name, i.confidence, i.box_xyxy, i.clipped, i.rejected,
                     None if i.polygon is None else i.polygon.tobytes()) for i in result.instances]

        before = snapshot()
        inputs = inputs_from_ai(result)

        self.assertEqual([(i.index, i.candidate, i.untracked_reason) for i in inputs],
                         [(0, True, None), (1, False, "not_in_scope"), (2, False, "not_in_scope"),
                          (3, False, "geometry_rejected"), (4, False, "geometry_rejected"), (5, True, None)])
        self.assertTrue(inputs[5].clipped)
        f = self.step(inputs)
        self.assertEqual([a.instance_index for a in f.assignments], [0, 1, 2, 3, 4, 5])
        self.assertEqual([a.state for a in f.assignments][1:5], [TrackState.UNTRACKED] * 4)
        self.assertEqual(snapshot(), before)   # 입력 불변

    def test_non_ai_rejected(self):
        """AiSegResult가 아니면 계약 오류."""
        with self.assertRaises(TrackingContractError):
            inputs_from_ai(object())

    def test_numpy_box_values_accepted(self):
        """numpy 실수 박스도 받는다(AiInstance 정규화 결과)."""
        inputs = inputs_from_ai(seg_result([instance(0)]))
        self.assertTrue(all(isinstance(v, float) for v in inputs[0].box_xyxy))


if __name__ == "__main__":
    unittest.main()
