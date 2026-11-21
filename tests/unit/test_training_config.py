"""학습기 공통·후보 설정 검증(`eyetoarm.training.config`) 시험.
저장소 설정 파일(`config/training/`)도 읽어 확인한다."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

from eyetoarm.training.config import (ConfigError, candidates_from_dict, common_from_dict, load_candidates,
                                      load_common, load_training_config, thaw, training_config_from_dicts)


def _write(path, data, mode="w"):
    """파일을 쓰고 닫는다(시험 도우미)."""
    with open(path, mode) as f:
        f.write(data)


def _read(path, mode="r"):
    """파일을 읽고 닫는다(시험 도우미)."""
    with open(path, mode) as f:
        return f.read()


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COMMON = os.path.join(ROOT, "config", "training", "s1_common.json")
CANDS = os.path.join(ROOT, "config", "training", "s1_candidates.json")


def common_doc(**over):
    """저장소 공통 설정 문서 사본(키워드로 최상위 덮어쓰기)."""
    doc = json.loads(_read(COMMON))
    doc.update(over)
    return doc


def cands_doc():
    """저장소 후보 설정 문서 사본."""
    return json.loads(_read(CANDS))


class CommonConfigTest(unittest.TestCase):
    """공통 설정."""

    def test_repo_common_config_loads(self):
        """저장소 s1_common.json은 비교용 640·50·seed 0·deterministic·batch 16·workers 2·상대 경로다."""
        c = load_common(COMMON)

        self.assertEqual((c.kind, c.train["imgsz"], c.train["epochs"], c.train["seed"], c.train["deterministic"],
                          c.train["batch"], c.train["workers"], c.train["cache"], c.train["plots"]),
                         ("comparison", 640, 50, 0, True, 16, 2, False, False))
        self.assertFalse(os.path.isabs(c.data["data_yaml"]))
        self.assertTrue(all(not os.path.isabs(p) for p in c.data["split_lists"]))
        self.assertEqual(c.watch["mem_available_stop_kib"], 2097152)
        self.assertEqual(c.required_env, {"YOLO_AUTOINSTALL": "false", "YOLO_OFFLINE": "1"})

    def test_rejects_bad_common(self):
        """알 수 없는 키·빠진 키·형·범위 위반·절대·상위 경로·잘못된 kind는 ConfigError."""
        base = common_doc()
        bads = []

        def variant(fn):
            """사본을 바꿔 목록에 넣는다."""
            d = copy.deepcopy(base)
            fn(d)
            bads.append(d)

        variant(lambda d: d.update(extra=1))
        variant(lambda d: d.pop("watch"))
        variant(lambda d: d["train"].update(lr0=0.01))
        variant(lambda d: d["train"].pop("seed"))
        variant(lambda d: d["train"].update(epochs=0))
        variant(lambda d: d["train"].update(epochs=True))
        variant(lambda d: d["train"].update(batch=16.0))
        variant(lambda d: d["train"].update(deterministic=1))
        variant(lambda d: d["train"].update(device=0))
        variant(lambda d: d["train"].update(imgsz=650))
        variant(lambda d: d["train"].update(fraction=1.5))
        variant(lambda d: d.update(kind="final"))
        variant(lambda d: d["data"].update(data_yaml="/abs/dataset.yaml"))
        variant(lambda d: d["data"].update(data_yaml="../outside.yaml"))
        variant(lambda d: d["data"].update(class_names=[]))
        variant(lambda d: d["watch"].update(watch_interval_s=0))
        variant(lambda d: d["watch"].update(train_deadline_s=float("inf")))
        variant(lambda d: d.update(required_env={"YOLO_AUTOINSTALL": "true", "YOLO_OFFLINE": "1"}))
        variant(lambda d: d.update(weights_dir="/home/x/weights"))

        for i, bad in enumerate(bads):
            with self.subTest(i=i):
                with self.assertRaises(ConfigError):
                    common_from_dict(bad)

    def test_smoke_kind_allows_fraction(self):
        """smoke 설정은 fraction(0<f≤1)을 허용하고 비교 설정에는 fraction이 없어야 한다."""
        d = common_doc(kind="smoke")
        d["train"] = {**d["train"], "epochs": 1, "fraction": 0.05}

        self.assertEqual(common_from_dict(d).train["fraction"], 0.05)

        d2 = common_doc()
        d2["train"] = {**d2["train"], "fraction": 0.5}
        with self.assertRaises(ConfigError):
            common_from_dict(d2)


class CandidatesConfigTest(unittest.TestCase):
    """후보 설정."""

    def test_repo_candidates_load(self):
        """저장소 후보는 S1-01~03 세 계열이고 공식 ultralytics/assets v8.4.0 출처와 sha256을 가진다."""
        cs = load_candidates(CANDS)

        self.assertEqual([(c.exp_id, c.family, c.weights_file) for c in cs],
                         [("S1-01", "yolov8n-seg", "yolov8n-seg.pt"), ("S1-02", "yolo11n-seg", "yolo11n-seg.pt"),
                          ("S1-03", "yolo26n-seg", "yolo26n-seg.pt")])
        self.assertTrue(all(c.source["repo"] == "ultralytics/assets" and c.source["tag"] == "v8.4.0" for c in cs))
        self.assertEqual(cs[1].sha256, "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152")

    def test_rejects_training_args_and_bad_candidates(self):
        """후보별 학습 인자·미지 키·중복·계열 누락/중복·다른 모델·경로 파일명·끝 개행 해시는 ConfigError."""
        base = cands_doc()
        bads = []

        def variant(fn):
            """사본을 바꿔 목록에 넣는다."""
            d = copy.deepcopy(base)
            fn(d)
            bads.append(d)

        for key, val in (("imgsz", 1280), ("epochs", 10), ("batch", 8), ("optimizer", "SGD"), ("workers", 4)):
            variant(lambda d, k=key, v=val: d["candidates"][0].update({k: v}))
        variant(lambda d: d["candidates"][1].update(exp_id="S1-01"))
        variant(lambda d: d["candidates"].pop())
        variant(lambda d: d["candidates"][2].update(family="yolo11n-seg"))
        variant(lambda d: d["candidates"][0].update(family="yolov8s-seg"))
        variant(lambda d: d["candidates"][0].update(weights_file="../x.pt"))
        variant(lambda d: d["candidates"][0].update(sha256=d["candidates"][0]["sha256"] + "\n"))
        variant(lambda d: d["candidates"][0].update(size_bytes=0))
        variant(lambda d: d["candidates"][0]["source"].update(url="http://example.com/x.pt"))
        variant(lambda d: d["candidates"][0].update(exp_id="S1-1"))

        for i, bad in enumerate(bads):
            with self.subTest(i=i):
                with self.assertRaises(ConfigError) as ctx:
                    candidates_from_dict(bad)
                if i < 5:                                   # 학습 인자 키는 그 이유로 거부됨을 확인
                    self.assertIn("training args", str(ctx.exception))


class LoaderTest(unittest.TestCase):
    """엄격 JSON·config_id·import."""

    def write(self, text):
        """임시 파일 경로."""
        d = tempfile.mkdtemp()
        p = os.path.join(d, "c.json")
        _write(p, text, "w")
        return p

    def test_strict_json(self):
        """중복 키·NaN·1e400·객체 아님은 ConfigError, 파일 없음은 OSError."""
        good = _read(COMMON)
        for text in (good[:-2] + ',"kind":"smoke"}', good.replace('"watch_interval_s": 30', '"watch_interval_s": NaN'),
                     good.replace('"watch_interval_s": 30', '"watch_interval_s": 1e400'), "[]"):
            with self.subTest(text=text[-30:]):
                with self.assertRaises(ConfigError):
                    load_common(self.write(text))

        with self.assertRaises(OSError):
            load_common("/nonexistent/c.json")

    def test_config_id_stable_and_sensitive(self):
        """config_id는 같은 내용이면 같고, 공통 학습 값·후보 해시가 바뀌면 달라진다."""
        a = load_training_config(COMMON, CANDS)
        b = load_training_config(COMMON, CANDS)
        d = common_doc()
        d["train"] = {**d["train"], "seed": 1}

        self.assertEqual(a.config_id, b.config_id)
        self.assertNotEqual(a.config_id, load_training_config(self.write(json.dumps(d)), CANDS).config_id)
        self.assertEqual(len(a.config_id), 64)

    def test_imports_no_torch(self):
        """eyetoarm.training 전체를 불러도 torch·ultralytics가 로드되지 않는다."""
        code = ("import sys, eyetoarm.training.config, eyetoarm.training.weights, eyetoarm.training.fingerprint, "
                "eyetoarm.training.layout, eyetoarm.training.identity, eyetoarm.training.records\n"
                "print(','.join(k for k in sorted(sys.modules) if k.split('.')[0] in ('torch','ultralytics')) or 'none')")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                             env={"PYTHONPATH": os.path.join(ROOT, "src"), "PATH": ""}, cwd=ROOT)

        self.assertEqual(out.stdout.strip(), "none")


class FamilyFileTest(unittest.TestCase):
    """계열과 공식 가중치 파일 대응."""

    def test_family_must_match_weights_file(self):
        """다른 계열의 가중치 정보를 복사하거나 같은 계열의 다른 크기 공식 파일을 쓰면 거부한다."""
        base = cands_doc()
        swapped = copy.deepcopy(base)
        c1, c2 = swapped["candidates"][0], swapped["candidates"][1]
        for k in ("weights_file", "source", "sha256", "size_bytes"):
            c2[k] = copy.deepcopy(c1[k])

        other_size = copy.deepcopy(base)
        c = other_size["candidates"][1]
        c["weights_file"] = "yolo11s-seg.pt"
        c["source"]["url"] = c["source"]["url"].replace("yolo11n-seg.pt", "yolo11s-seg.pt")

        detect = copy.deepcopy(base)
        c = detect["candidates"][2]
        c["weights_file"] = "yolo26n.pt"
        c["source"]["url"] = c["source"]["url"].replace("yolo26n-seg.pt", "yolo26n.pt")

        for name, bad in (("swapped", swapped), ("other size", other_size), ("detect task", detect)):
            with self.subTest(case=name):
                with self.assertRaises(ConfigError) as ctx:
                    candidates_from_dict(bad)
                self.assertIn("family", str(ctx.exception))

        self.assertEqual(len(candidates_from_dict(base)), 3)


class SnapshotTest(unittest.TestCase):
    """검증된 설정 스냅샷 고정."""

    def test_config_fields_are_frozen(self):
        """검증 뒤 train·data·watch·source를 바꿀 수 없고 실행 인자는 config_id와 같은 스냅샷에서 나온다."""
        cfg = load_training_config(COMMON, CANDS)

        for target in (cfg.common.train, cfg.common.data, cfg.common.watch, cfg.candidates[0].source,
                       cfg.common.data["expected_fingerprint"]):
            with self.subTest(target=type(target).__name__):
                with self.assertRaises(TypeError):
                    target["x"] = 1
        with self.assertRaises((TypeError, AttributeError)):
            cfg.common.data["split_lists"].append("x")

        args = cfg.train_args()
        args["epochs"] = 999
        self.assertEqual(cfg.train_args()["epochs"], 50)
        self.assertTrue(cfg.verify())


if __name__ == "__main__":
    unittest.main()


CLEAN_COMMON = os.path.join(ROOT, "config", "training", "s1_clean_train_r1_common.json")
CLEAN_CANDS = os.path.join(ROOT, "config", "training", "s1_clean_train_r1_candidates.json")


def single_doc(n=1):
    """selection_mode single 후보 문서: 저장소 후보에서 앞 n개를 쓴다(n=1이면 S1-01)."""
    d = cands_doc()
    d["selection_mode"] = "single"
    d["candidates"] = d["candidates"][:n]
    return d


class SelectionModeTest(unittest.TestCase):
    """후보 선택 모드: 생략하면 3계열 규칙, single이면 정확히 1개."""

    def test_single_mode_accepts_exactly_one(self):
        """selection_mode single + yolov8n-seg 1개는 통과하고 후보는 그 하나뿐이다."""
        cs = candidates_from_dict(single_doc())

        self.assertEqual([(c.exp_id, c.family) for c in cs], [("S1-01", "yolov8n-seg")])

    def test_single_mode_rejects_wrong_counts_and_bad_candidate(self):
        """single에서 개수·계열↔파일·비공식 url·잘못된 sha256·후보별 학습 인자 오류는 ConfigError."""
        bads = {"empty": single_doc(0), "two": single_doc(2), "three": single_doc(3)}

        d = single_doc()
        d["candidates"][0]["weights_file"] = "yolo11n-seg.pt"
        bads["family-file"] = d

        d = single_doc()
        d["candidates"][0]["source"]["url"] = "http://example.com/yolov8n-seg.pt"
        bads["url"] = d

        d = single_doc()
        d["candidates"][0]["sha256"] = "0" * 63
        bads["sha"] = d

        d = single_doc()
        d["candidates"][0]["epochs"] = 10
        bads["train-arg"] = d

        for name, bad in bads.items():
            with self.subTest(case=name):
                with self.assertRaises(ConfigError):
                    candidates_from_dict(bad)

    def test_unknown_mode_rejected(self):
        """알 수 없는 모드 값(대소문자 다름·다른 문자열·null·숫자·빈 문자열)은 ConfigError."""
        for mode in ("SINGLE", "multi", None, 1, "", "all"):
            d = single_doc()
            d["selection_mode"] = mode
            with self.subTest(mode=mode):
                with self.assertRaises(ConfigError):
                    candidates_from_dict(d)

    def test_omitted_mode_keeps_three_family_rule(self):
        """모드를 생략하면 3계열 문서는 통과하고 1개·2개 문서는 ConfigError."""
        self.assertEqual(len(candidates_from_dict(cands_doc())), 3)

        for n in (1, 2):
            d = cands_doc()
            d["candidates"] = d["candidates"][:n]
            with self.subTest(n=n):
                with self.assertRaises(ConfigError):
                    candidates_from_dict(d)

    def test_mode_is_part_of_config_id(self):
        """single 모드는 config_id 계산 대상 스냅샷에 들어가고 스냅샷 재검증도 통과한다."""
        d = tempfile.mkdtemp()
        path = os.path.join(d, "k.json")
        _write(path, json.dumps(single_doc()))

        cfg = load_training_config(COMMON, path)

        self.assertIn('"selection_mode":"single"', cfg.canonical)
        self.assertTrue(cfg.verify())
        self.assertNotEqual(cfg.config_id, load_training_config(COMMON, CANDS).config_id)

    def test_repo_clean_train_r1_config(self):
        """저장소 clean_train_r1 설정: 후보는 single S1-07 하나, 공통 학습 값·감시는 s1_common과 같다."""
        cfg = load_training_config(CLEAN_COMMON, CLEAN_CANDS)
        base = load_common(COMMON)

        self.assertEqual([(c.exp_id, c.family, c.weights_file, c.source["tag"]) for c in cfg.candidates],
                         [("S1-07", "yolov8n-seg", "yolov8n-seg.pt", "v8.4.0")])
        self.assertEqual(cfg.candidates[0].sha256, load_candidates(CANDS)[0].sha256)
        self.assertEqual(dict(cfg.common.train), dict(base.train))
        self.assertEqual(dict(cfg.common.watch), dict(base.watch))
        self.assertEqual(cfg.common.data["data_yaml"], "data/derived/bolt_nut_clean_train_r1/dataset.yaml")
        self.assertIsNotNone(cfg.common.data["expected_fingerprint"])


HSVV_COMMON = os.path.join(ROOT, "config", "training", "s1_clean_hsvv06_r1_common.json")
HSVV_CANDS = os.path.join(ROOT, "config", "training", "s1_clean_hsvv06_r1_candidates.json")


def hsv_common(value):
    """공통 설정 사본에 train.hsv_v를 넣는다."""
    d = common_doc()
    d["train"] = {**d["train"], "hsv_v": value}
    return d


class HsvVTest(unittest.TestCase):
    """선택 학습 키 hsv_v: 명시할 때만 전달, [0, 1] 유한 실수, 생략 시 config_id 그대로."""

    def test_omitted_keeps_existing_snapshot(self):
        """hsv_v 없는 저장소 설정은 train_args에 hsv_v가 없고 config_id가 기록값과 같다."""
        base = load_training_config(COMMON, CANDS)
        clean = load_training_config(CLEAN_COMMON, CLEAN_CANDS)

        self.assertNotIn("hsv_v", base.train_args())
        self.assertNotIn("hsv_v", dict(base.common.train))
        self.assertEqual(base.config_id, "77368a324a0da01e7bd379b84cc96b3b5603a7db6c25b7973ee02bf1d2d7bb56")
        self.assertEqual(clean.config_id, "1e38cf433ee6c4327145b2853bd121da4e24647115ce42ecdcb47015857c98b1")

    def test_accepts_range_and_passes_to_train_args(self):
        """0·0.6·1(정수 포함)은 통과하고 train_args·검증된 train에 그 값이 들어간다."""
        for v in (0, 0.0, 0.6, 1, 1.0):
            with self.subTest(v=v):
                cfg = training_config_from_dicts(hsv_common(v), cands_doc())
                self.assertEqual(cfg.train_args()["hsv_v"], v)
                self.assertEqual(cfg.common.train["hsv_v"], float(v))

    def test_rejects_bad_values(self):
        """bool·문자열·null·비유한·음수·1 초과·리스트·거대 정수는 ConfigError."""
        for v in (True, False, "0.6", None, float("nan"), float("inf"), -0.1, 1.01, [0.6], 2, -1,
                  10 ** 400, -(10 ** 400), float("-inf")):
            with self.subTest(v=v):
                with self.assertRaises(ConfigError):
                    common_from_dict(hsv_common(v))

    def test_only_hsv_v_is_added(self):
        """다른 증강 키(hsv_s·hsv_h·mosaic·fliplr)는 여전히 공통 train에서 거부된다."""
        for k in ("hsv_s", "hsv_h", "mosaic", "fliplr"):
            d = common_doc()
            d["train"] = {**d["train"], k: 0.5}
            with self.subTest(k=k):
                with self.assertRaises(ConfigError):
                    common_from_dict(d)

    def test_candidate_cannot_carry_hsv_v(self):
        """후보별 hsv_v는 학습 인자로 거부된다."""
        d = cands_doc()
        d["candidates"][0]["hsv_v"] = 0.6

        with self.assertRaises(ConfigError) as ctx:
            candidates_from_dict(d)
        self.assertIn("training args", str(ctx.exception))

    def test_value_changes_config_id(self):
        """hsv_v 0.6과 0.5, 그리고 생략은 서로 다른 config_id."""
        a = training_config_from_dicts(hsv_common(0.6), cands_doc()).config_id
        b = training_config_from_dicts(hsv_common(0.5), cands_doc()).config_id
        c = training_config_from_dicts(common_doc(), cands_doc()).config_id

        self.assertEqual(len({a, b, c}), 3)

    def test_repo_hsvv06_config(self):
        """저장소 hsv_v 설정: single S1-08, train = clean_train_r1 + hsv_v 0.6, 나머지는 clean과 같다."""
        cfg = load_training_config(HSVV_COMMON, HSVV_CANDS)
        clean = load_training_config(CLEAN_COMMON, CLEAN_CANDS)

        self.assertEqual([(c.exp_id, c.family) for c in cfg.candidates], [("S1-08", "yolov8n-seg")])
        self.assertEqual(cfg.candidates[0].sha256, clean.candidates[0].sha256)
        self.assertEqual(cfg.train_args(), {**clean.train_args(), "hsv_v": 0.6})
        self.assertEqual(thaw(cfg.common.raw["data"]), thaw(clean.common.raw["data"]))
        self.assertEqual(dict(cfg.common.watch), dict(clean.common.watch))
