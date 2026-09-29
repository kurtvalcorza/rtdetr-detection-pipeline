# ruff: noqa: E501
"""Regression tests for the closed-set detection workshop review (DET-01..DET-07).

These exec the notebook's own generated cells. Models are either the carried YOLOX architecture with
random weights (no checkpoint) or explicit synthetic adapters; none of this is pretrained-model evidence.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import io
import json
import math
import sys
import time
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from detection_workshop_source import CELLS

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

CLASSES = ("stop-sign", "yield-sign", "speed-limit-sign")
GOOD = {"box": [10.0, 10.0, 30.0, 30.0], "label": "stop-sign", "score": 0.9}


def run(i, ns):
    exec(compile(CELLS[i]["source"], f"cell-{i:02d}", "exec"), ns)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_namespace():
    return {
        "np": np, "pd": pd, "torch": torch, "Image": Image, "ImageDraw": ImageDraw,
        "ImageFont": ImageFont, "math": math, "time": time, "json": json, "Path": Path,
        "gc": gc, "hashlib": hashlib, "io": io, "DEVICE": "cpu", "sha256_file": sha256_file,
        "display": lambda *a, **k: None,
    }


@pytest.fixture
def notebook(tmp_path, monkeypatch):
    """Controls, specs, generated data and evaluator cells, executed in order in tmp_path."""
    monkeypatch.chdir(tmp_path)
    ns = base_namespace()
    for i in (3, 7, 9, 11, 14):
        run(i, ns)
    return ns


@pytest.fixture(scope="module")
def yolox(tmp_path_factory):
    """Carried upstream YOLOX source (cell 18) plus the notebook wrapper (cell 21)."""
    root = tmp_path_factory.mktemp("yolox")
    ns = base_namespace()
    ns.update(WORK_ROOT=root, CLASS_NAMES=CLASSES, EVAL_SCORE_THRESHOLD=0.01, MAX_EVAL_DETECTIONS=100, IOU_THRESHOLDS=(0.5,),
              YOLOX_EVAL_NMS=0.65, YOLOX_VARIANTS={"yolox_s": {"sha256": "0" * 64}})
    module = compile(CELLS[14]["source"].split("empty_predictions=")[0], "cell-14", "exec")
    exec(module, ns)
    run(18, ns)
    run(21, ns)
    return ns


# ---------------------------------------------------------------- DET-01


@pytest.mark.parametrize("depth,width", [(0.33, 0.50), (1.33, 1.25)], ids=["yolox_s", "yolox_x"])
def test_every_yolox_batchnorm_matches_pinned_upstream(yolox, depth, width):
    model = yolox["build_yolox"](depth, width, 3)
    layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert layers and all((m.eps, m.momentum) == (1e-3, 0.03) for m in layers)
    if (depth, width) == (0.33, 0.50):
        assert len(layers) == 74
    del model
    gc.collect()


def test_yolox_forward_equals_upstream_constructor_on_identical_weights(yolox):
    # Upstream Exp.get_model at 419778480ab6ec0590e5d3831b3afb3b46ab2aa3: build, apply init_yolo
    # (BatchNorm eps=1e-3, momentum=0.03), then initialize_biases(1e-2).
    from yolox_ref.yolo_head import YOLOXHead
    from yolox_ref.yolo_pafpn import YOLOPAFPN
    from yolox_ref.yolox import YOLOX

    torch.manual_seed(0)
    upstream = YOLOX(
        YOLOPAFPN(0.33, 0.50, in_channels=[256, 512, 1024], act="silu"),
        YOLOXHead(3, 0.50, in_channels=[256, 512, 1024], act="silu"),
    )

    def init_yolo(module):
        for m in module.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    upstream.apply(init_yolo)
    upstream.head.initialize_biases(1e-2)
    notebook_model = yolox["build_yolox"](0.33, 0.50, 3)
    # PyTorch-default BatchNorm (the pre-fix construction) as a sensitivity control.
    pytorch_default = yolox["build_yolox"](0.33, 0.50, 3)
    for m in pytorch_default.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eps, m.momentum = 1e-5, 0.1
    state = upstream.state_dict()
    generator = torch.Generator().manual_seed(1)
    for key, value in state.items():
        if key.endswith("running_var"):  # varied, non-unit statistics so eps matters
            state[key] = torch.rand(value.shape, generator=generator) * 0.5 + 0.02
    for model in (upstream, notebook_model, pytorch_default):
        model.load_state_dict(state, strict=True)
        model.eval()
    pixels = torch.rand(1, 3, 640, 640, generator=generator)
    with torch.no_grad():
        expected = upstream(pixels)
        assert torch.isfinite(expected).all()
        assert torch.equal(expected, notebook_model(pixels))
        assert not torch.equal(expected, pytorch_default(pixels))


def test_yolox_artifact_from_before_the_batchnorm_fix_is_refused(yolox, tmp_path):
    path = tmp_path / "adapter.pt"
    torch.save({"format": "dimer-yolox-detection-adapter/1", "variant": "yolox_s", "base_sha256": "0" * 64,
                "class_names": list(CLASSES), "state_dict": {}}, path)
    with pytest.raises(ValueError, match="BatchNorm"):
        yolox["YOLOXWorkshop"].load_artifact(path)
    torch.save({"format": "dimer-yolox-detection-adapter/2", "variant": "yolox_s", "base_sha256": "0" * 64,
                "architecture": {"batchnorm": {"eps": 1e-5, "momentum": 0.1}},
                "class_names": list(CLASSES), "state_dict": {}}, path)
    with pytest.raises(ValueError, match="BatchNorm"):
        yolox["YOLOXWorkshop"].load_artifact(path)


# ---------------------------------------------------------------- DET-02: wrappers


class RawOutput(torch.nn.Module):
    def __init__(self, raw):
        super().__init__()
        self.raw = raw

    def forward(self, x):
        return self.raw.clone()


def yolox_raw():
    return torch.tensor([[[160.0, 160.0, 60.0, 60.0, 0.9, 0.9, 0.05, 0.05],
                          [360.0, 360.0, 60.0, 60.0, 0.8, 0.05, 0.9, 0.05]]])


def yolox_detect(yolox, raw):
    pipe = yolox["YOLOXWorkshop"]("yolox_s", RawOutput(raw), CLASSES, "cpu")
    return pipe.detect(Image.new("RGB", (640, 640)), 0.01)


def test_yolox_wrapper_finite_and_valid_empty_outputs(yolox):
    assert len(yolox_detect(yolox, yolox_raw())) == 2
    below = yolox_raw()
    below[:, :, 4] = 0.0  # finite, legitimately below the score cutoff
    assert yolox_detect(yolox, below) == []


@pytest.mark.parametrize("mutate", [
    lambda r: r.__setitem__((0, 0, 4), float("nan")),
    lambda r: r.__setitem__((slice(None), slice(None), 4), float("nan")),
    lambda r: r.__setitem__((0, 1, 0), float("inf")),
], ids=["one_nan_objectness", "all_nan_objectness", "inf_box"])
def test_yolox_wrapper_refuses_nonfinite_raw_output_before_filtering(yolox, mutate):
    raw = yolox_raw()
    mutate(raw)
    with pytest.raises(ValueError, match="non-finite"):
        yolox_detect(yolox, raw)


@pytest.mark.parametrize("raw", [torch.zeros(1, 5, 7), torch.zeros(2, 5, 8), torch.zeros(1, 0, 8), torch.zeros(5, 8)],
                         ids=["missing_class_channel", "batch_of_two", "no_anchors", "two_dimensional"])
def test_yolox_wrapper_refuses_malformed_raw_output(yolox, raw):
    with pytest.raises(ValueError):
        yolox_detect(yolox, raw)


@pytest.fixture(scope="module")
def rtdetr():
    pytest.importorskip("transformers")
    from transformers import RTDetrImageProcessor

    ns = base_namespace()
    ns.update(CLASS_NAMES=CLASSES, EVAL_SCORE_THRESHOLD=0.01, MAX_EVAL_DETECTIONS=100, RTDETR_ROOT=Path("."), IOU_THRESHOLDS=(0.5,),
              RTDETR_SPEC={"model_id": "x", "revision": "y"}, YOLOX_EVAL_NMS=0.65)
    exec(compile(CELLS[14]["source"].split("empty_predictions=")[0], "cell-14", "exec"), ns)
    run(20, ns)
    ns["processor"] = RTDetrImageProcessor()
    return ns


class FixedRTDETR(torch.nn.Module):
    def __init__(self, logits, boxes):
        super().__init__()
        self.logits, self.boxes = logits, boxes

    def forward(self, **inputs):
        return types.SimpleNamespace(logits=self.logits.clone(), pred_boxes=self.boxes.clone())


def rtdetr_detect(ns, logits, boxes):
    pipe = ns["RTDETRWorkshop"](FixedRTDETR(logits, boxes), ns["processor"], CLASSES, "cpu")
    return pipe.detect(Image.new("RGB", (64, 64)), 0.01)


def rtdetr_outputs():
    logits = torch.full((1, 4, 3), -10.0)
    logits[0, 0, 0] = 3.0
    logits[0, 1, 2] = 2.0
    boxes = torch.tensor([[[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2], [0.5, 0.5, 0.1, 0.1], [0.2, 0.8, 0.1, 0.1]]])
    return logits, boxes


def test_rtdetr_wrapper_finite_and_valid_empty_outputs(rtdetr):
    logits, boxes = rtdetr_outputs()
    dets = rtdetr_detect(rtdetr, logits, boxes)
    assert [d["label"] for d in dets] == ["stop-sign", "speed-limit-sign"]
    assert rtdetr_detect(rtdetr, torch.full((1, 4, 3), -20.0), boxes) == []


@pytest.mark.parametrize("case", ["nan_logit", "all_nan_logits", "nan_box", "short_box", "class_count", "missing"])
def test_rtdetr_wrapper_refuses_invalid_raw_output(rtdetr, case):
    logits, boxes = rtdetr_outputs()
    if case == "nan_logit":
        logits[0, 0, 0] = float("nan")
    elif case == "all_nan_logits":
        logits[:] = float("nan")
    elif case == "nan_box":
        boxes[0, 0, 0] = float("nan")
    elif case == "short_box":
        boxes = boxes[:, :, :3]
    elif case == "class_count":
        logits = logits[:, :, :2]
    else:
        with pytest.raises(ValueError, match="missing"):
            rtdetr["validate_rtdetr_outputs"](types.SimpleNamespace(logits=logits), 3)
        return
    with pytest.raises(ValueError):
        rtdetr_detect(rtdetr, logits, boxes)


# ---------------------------------------------------------------- DET-02: evaluators and parity

REF = [{"boxes": [[10, 10, 30, 30]], "labels": ["stop-sign"]}]


@pytest.mark.parametrize("det", [
    {**GOOD, "score": float("nan")}, {**GOOD, "score": float("inf")}, {**GOOD, "score": 2.0},
    {**GOOD, "score": -0.1}, {**GOOD, "box": [10.0, 10.0, 30.0]}, {**GOOD, "box": [float("nan")] * 4},
    {**GOOD, "box": [30.0, 10.0, 10.0, 30.0]}, {**GOOD, "label": "cat"}, {"box": GOOD["box"], "label": "stop-sign"},
], ids=["nan_score", "inf_score", "score_above_one", "negative_score", "short_box", "nan_box",
        "reversed_box", "unknown_label", "missing_score"])
def test_both_evaluators_refuse_invalid_detections(notebook, det):
    with pytest.raises(ValueError):
        notebook["average_precision"]([[det]], REF, CLASSES)
    with pytest.raises(ValueError):
        notebook["operating_metrics"]([[det]], REF, 0.3)


def test_evaluators_accept_valid_and_empty_predictions(notebook):
    assert notebook["average_precision"]([[GOOD]], REF, CLASSES)["ap"] == 1.0
    assert notebook["operating_metrics"]([[GOOD]], REF, 0.3)["recall50"] == 1.0
    assert notebook["average_precision"]([[]], REF, CLASSES)["ap"] == 0.0


PARITY_CASES = {
    "nan_score": [{**GOOD, "score": float("nan")}],
    "nan_box": [{**GOOD, "box": [float("nan")] * 4}],
    "short_box": [{**GOOD, "box": [10.0, 10.0, 30.0]}],
    "unknown_label": [{**GOOD, "label": "cat"}],
    "score_mismatch": [{**GOOD, "score": 0.8}],
    "count_mismatch": [],
}


class Fixed:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, *a):
        return copy.deepcopy(self.dets)


def canonical_parity(ns, reloaded):
    local = dict(ns)
    local.update(MODEL_KEYS=["rtdetr"], live_parity_predictions={"rtdetr": [GOOD]},
                 fresh_adapted={"rtdetr": Fixed(reloaded)}, EFFECTIVE_EVALUATION={"display_threshold": 0.3})
    run(43, local)
    return local["parity"]


def test_canonical_parity_accepts_identical_finite_detections(notebook):
    parity = canonical_parity(notebook, [GOOD])
    assert parity["rtdetr"] == {"n_detections": 1, "max_abs_box_difference": 0.0, "max_abs_score_difference": 0.0}


@pytest.mark.parametrize("case", sorted(PARITY_CASES))
def test_canonical_and_byod_parity_refuse_invalid_or_different_detections(notebook, case):
    with pytest.raises((ValueError, RuntimeError)):
        canonical_parity(notebook, PARITY_CASES[case])
    notebook["USE_BYOD"] = False
    run(47, notebook)
    with pytest.raises((ValueError, RuntimeError)):
        notebook["assert_byod_parity"]([GOOD], PARITY_CASES[case])


# ---------------------------------------------------------------- DET-03: freeze binds the test


class Adapter:
    """Explicit synthetic adapter: returns each reference box at score 0.7. Not a detector."""

    calls = []

    def __init__(self, key="rtdetr", adapted=False):
        self.key, self.adapted = key, adapted

    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls(a[0] if a and isinstance(a[0], str) else "rtdetr")

    @classmethod
    def load_artifact(cls, path, device):
        return cls(json.loads(Path(path).read_text())["key"], True)

    def predict_many(self, records, threshold=0.01):
        Adapter.calls.append(threshold)
        return [[{"box": list(map(float, b)), "label": lab, "score": 0.7}
                 for b, lab in zip(r["boxes"], r["labels"], strict=True) if threshold <= 0.7] for r in records]

    def detect(self, image, threshold=0.3, *unused):
        return [{"box": [1.0, 1.0, 5.0, 5.0], "label": "stop-sign", "score": 0.7}] if threshold <= 0.7 else []

    def save_artifact(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"key": self.key}))
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def frozen(ns):
    """Validation-stage state from synthetic adapters, then the real freeze cell."""
    ns.update(RTDETRWorkshop=Adapter, YOLOXWorkshop=Adapter, YOLOX_CODE_REVISION="rev",
              YOLOX_BATCHNORM={"eps": 1e-3, "momentum": 0.03}, pipes={})
    ns["artifact_descriptors"] = {
        k: Adapter(k, True).save_artifact(ns["OUTPUT_ROOT"] / "artifacts" / k / "adapter.pt") for k in ns["MODEL_KEYS"]}
    ns["pre_val_table"] = pd.DataFrame([{"model": k, "ap": 0.0} for k in ns["MODEL_KEYS"]])
    ns["post_val_table"] = ns["pre_val_table"].copy()
    ns["VALIDATION_SETTINGS"] = {"pre": ns["evaluation_settings"](), "post": ns["evaluation_settings"]()}
    run(30, ns)
    Adapter.calls.clear()
    return ns


def test_unchanged_freeze_scores_test_with_the_frozen_settings(notebook):
    ns = frozen(notebook)
    run(32, ns)
    assert set(Adapter.calls) == {ns["verified_freeze"]["evaluation"]["score_cutoff"]}
    saved = json.loads((ns["OUTPUT_ROOT"] / "test" / "effective_settings.json").read_text())
    assert saved["evaluation"] == ns["evaluation_settings"]()
    assert saved["experiment_id"] == ns["frozen_experiment"]["experiment_id"]
    predictions = json.loads((ns["OUTPUT_ROOT"] / "test" / "predictions.json").read_text())
    assert len(predictions["predictions"]) == 2 * len(ns["MODEL_KEYS"]) * len(ns["test_records"])


@pytest.mark.parametrize("change", [
    lambda ns: ns.update(EVAL_SCORE_THRESHOLD=0.80),
    lambda ns: ns.update(DISPLAY_THRESHOLD=0.50),
    lambda ns: ns.update(YOLOX_EVAL_NMS=0.45),
    lambda ns: ns.update(MAX_EVAL_DETECTIONS=10),
    lambda ns: ns.update(CLASS_NAMES=tuple(reversed(CLASSES))),
    lambda ns: ns.update(test_records=ns["validation_records"]),
    lambda ns: ns.update(MODEL_KEYS=["rtdetr"]),
    lambda ns: (ns["OUTPUT_ROOT"] / "artifacts" / "rtdetr" / "adapter.pt").write_text('{"key": "tampered"}'),
    lambda ns: ns["freeze_path"].write_text(ns["freeze_path"].read_text().replace('"score_cutoff": 0.01', '"score_cutoff": 0.8')),
], ids=["score_cutoff", "display_threshold", "nms", "max_detections", "class_order", "test_cohort",
        "model_selection", "artifact_bytes", "edited_freeze_file"])
def test_changes_after_freeze_are_refused_before_any_test_inference(notebook, change):
    ns = frozen(notebook)
    change(ns)
    with pytest.raises(RuntimeError, match="frozen|Changed"):
        run(32, ns)
    assert Adapter.calls == []
    assert not (ns["OUTPUT_ROOT"] / "test" / "predictions.json").exists()


def test_settings_changed_after_validation_cannot_be_frozen(notebook):
    ns = notebook
    ns.update(VALIDATION_SETTINGS={"pre": ns["evaluation_settings"](), "post": ns["evaluation_settings"]()})
    ns["EVAL_SCORE_THRESHOLD"] = 0.5
    with pytest.raises(RuntimeError, match="after validation"):
        run(30, ns)


def test_only_one_yolox_nms_definition_so_the_control_is_effective():
    assert sum("YOLOX_EVAL_NMS=" in c["source"].replace(" ", "") for c in CELLS if c["kind"] == "code") == 1


# ---------------------------------------------------------------- DET-04: gallery titles


class FakePlt:
    """Records figure titles and saved files; the gallery needs no real matplotlib here."""

    def __init__(self):
        self.titles, self.saved = [], []

    def subplots(self, rows, cols, figsize=None):
        axis = types.SimpleNamespace(imshow=lambda *a: None, axis=lambda *a: None, set_title=lambda *a: None,
                                     add_patch=lambda *a: None, text=lambda *a, **k: None)
        figure = types.SimpleNamespace(suptitle=self.titles.append)
        return figure, [axis] * cols

    def Rectangle(self, *a, **k):
        return None

    def tight_layout(self):
        pass

    def savefig(self, path, **k):
        self.saved.append(Path(path).name)

    def show(self):
        pass


def gallery(ns, rtdetr_wins_on_first):
    records = ns["test_records"]
    perfect = [[{"box": list(map(float, b)), "label": lab, "score": 0.9}
                for b, lab in zip(r["boxes"], r["labels"], strict=True)] for r in records]
    yolox = copy.deepcopy(perfect)
    if rtdetr_wins_on_first:
        yolox[0] = []
    ns["test_predictions"] = {("rtdetr", "adapted"): perfect, ("yolox_s", "adapted"): yolox}
    run(38, ns)
    ns["plt"] = FakePlt()
    run(39, ns)
    return ns["plt"]


def test_tied_results_show_no_advantage_titles(notebook):
    plt = gallery(notebook, rtdetr_wins_on_first=False)
    assert len(plt.titles) == 1 and plt.titles[0].startswith("Three-object scene")
    assert "advantage" not in " ".join(plt.titles) and plt.saved == ["gallery_three-object_scene.png"]


def test_one_sided_results_show_only_the_real_advantage_with_its_difference(notebook):
    plt = gallery(notebook, rtdetr_wins_on_first=True)
    titles = " | ".join(plt.titles)
    assert "RT-DETR advantage" in titles and "YOLOX-S advantage" not in titles
    assert "difference +1.00" in plt.titles[0]


# ---------------------------------------------------------------- DET-05: BYOD headers


def byod_zip(tmp_path, header, rows):
    stream = io.BytesIO()
    Image.new("RGB", (32, 32)).save(stream, format="PNG")
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("annotations.csv", "\n".join([header, *rows]) + "\n")
        z.writestr("0.png", stream.getvalue())
    return path


@pytest.mark.parametrize("header,row", [
    ("image_id,file,label,x0,x0,y0,x1,y1", "0,0.png,stop-sign,1,4,1,10,10"),
    ("image_id,file,label,x0,X0,y0,x1,y1", "0,0.png,stop-sign,1,4,1,10,10"),
    ("image_id,file,label,x0,y0,x1,y1", "0,0.png,stop-sign,1,1,10,10,extra"),
    ("image_id,file,label,x0,y0,x1,y1", "0,0.png,stop-sign,1,1,10"),
], ids=["duplicate_x0", "case_duplicate_x0", "extra_field", "missing_field"])
def test_byod_refuses_ambiguous_headers_and_ragged_rows(notebook, tmp_path, header, row):
    notebook.update(USE_BYOD=False, WORK_ROOT=tmp_path)
    run(47, notebook)
    with pytest.raises(ValueError, match="column|width"):
        notebook["load_byod"](byod_zip(tmp_path, header, [row]))


def test_byod_refuses_to_run_outside_a_complete_canonical_run(notebook, tmp_path):
    notebook.update(USE_BYOD=True, BYOD_ZIP_PATH=str(byod_zip(tmp_path, "a", ["b"])))
    with pytest.raises(RuntimeError, match="complete canonical run"):
        run(47, notebook)


# ---------------------------------------------------------------- DET-06: run directory and replay


def complete(ns):
    frozen(ns)
    run(32, ns)
    ns.update(training_reports={"synthetic_fixture": True}, parity={"synthetic_fixture": True})
    run(52, ns)
    return ns


def test_each_run_gets_its_own_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first, second = base_namespace(), base_namespace()
    run(3, first)
    second["WORKSHOP_TIER"] = "FULL"
    source = CELLS[3]["source"].replace('WORKSHOP_TIER = "STANDARD"', 'WORKSHOP_TIER = "FULL"')
    exec(compile(source, "cell-03", "exec"), second)
    assert first["OUTPUT_ROOT"] != second["OUTPUT_ROOT"]
    assert first["OUTPUT_ROOT"].parent == second["OUTPUT_ROOT"].parent == Path("outputs") / "runs"


def test_export_contains_only_this_run_and_replays_metrics(notebook):
    stale = Path(notebook["OUTPUT_DIR"]) / "runs" / "older-full-run" / "artifacts" / "yolox_x" / "adapter.pt"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale earlier FULL-run marker; not a model")
    ns = complete(notebook)
    with zipfile.ZipFile(ns["bundle"]) as z:
        names = z.namelist()
    assert not any("yolox_x" in n or "older-full-run" in n for n in names)
    assert {"test/predictions.json", "test/effective_settings.json", "frozen/frozen_experiment.json",
            "provenance/experiment_manifest.json"} <= set(names)
    manifest = json.loads((ns["OUTPUT_ROOT"] / "provenance" / "experiment_manifest.json").read_text())
    assert set(manifest["metric_replay_from_saved_predictions"].values()) == {"PASS"}
    assert manifest["run_id"] == ns["RUN_ID"] and manifest["evaluation"] == ns["EFFECTIVE_EVALUATION"]


def test_replay_refuses_an_incomplete_prediction_export(notebook):
    ns = frozen(notebook)
    run(32, ns)
    path = ns["OUTPUT_ROOT"] / "test" / "predictions.json"
    saved = json.loads(path.read_text())
    saved["predictions"] = saved["predictions"][1:]
    path.write_text(json.dumps(saved))
    ns.update(training_reports={}, parity={})
    with pytest.raises(RuntimeError, match="replayed"):
        run(52, ns)


# ---------------------------------------------------------------- DET-07: memory label


def test_training_memory_is_labelled_as_a_stage_peak():
    source = CELLS[25]["source"]
    assert '"stage_peak_vram_MiB"' in source and '"peak_vram_MiB"' not in source
    assert "not an isolated per-model memory requirement" in source
