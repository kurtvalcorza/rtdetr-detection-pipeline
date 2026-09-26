import ast
import contextlib
import gc
import hashlib
import io
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from detection_workshop_source import CELLS

CLASSES = ("stop-sign", "yield-sign", "speed-limit-sign")


def helpers(tmp_path):
    ns = {
        "Path": Path,
        "np": np,
        "pd": pd,
        "Image": Image,
        "hashlib": hashlib,
        "json": json,
        "gc": gc,
        "CLASS_NAMES": CLASSES,
        "WORK_ROOT": tmp_path,
        "SPLIT_SEED": 42,
        "USE_BYOD": False,
        "IOU_THRESHOLDS": (0.5, 0.75),
    }
    module = ast.parse(CELLS[14]["source"])
    module.body = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    exec(compile(module, "evaluation", "exec"), ns)
    exec(CELLS[47]["source"], ns)
    return ns


def fixture(tmp_path, mutate=None):
    rows = []
    images = {}
    for i in range(6):
        name = f"{i}.png"
        stream = io.BytesIO()
        Image.new("RGB", (32, 32), (i * 30, 1, 2)).save(stream, format="PNG")
        images[name] = stream.getvalue()
        for j, label in enumerate(CLASSES):
            rows.append(
                dict(
                    image_id=str(i),
                    file=name,
                    label=label,
                    x0=j,
                    y0=j,
                    x1=j + 10,
                    y1=j + 10,
                    split=("train", "validation", "test")[i // 2],
                )
            )
    if mutate:
        mutate(rows, images)
    import csv

    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path = tmp_path / "input.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("annotations.csv", stream.getvalue())
        for name, data in images.items():
            z.writestr(name, data)
    return path


def test_valid_archive_isolated_repeat_and_preserves_existing(tmp_path):
    ns = helpers(tmp_path)
    path = fixture(tmp_path)
    sentinel = tmp_path / "byod" / "keep.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("keep")
    first = ns["load_byod"](path)
    second = ns["load_byod"](path)
    assert len(first) == len(second) == 6 and sentinel.read_text() == "keep"
    assert len(list(sentinel.parent.glob("dataset-*"))) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r, i: r[0].update(label="cat"),
        lambda r, i: r[0].update(x0="nan"),
        lambda r, i: r[0].update(file="../outside.png"),
        lambda r, i: r[0].update(file="sub\\image.png"),
        lambda r, i: r[0].update(file="1.png"),
        lambda r, i: r[0].update(split="test"),
        lambda r, i: r[0].update(split=""),
        lambda r, i: r.append(dict(r[0])),
        lambda r, i: i.update({"extra.png": i["0.png"]}),
        lambda r, i: i.update({"4.png": i["0.png"]}),
        lambda r, i: r[0].update(image_id="different-id"),
    ],
)
def test_invalid_data_fails_before_model(tmp_path, mutation):
    ns = helpers(tmp_path)
    with pytest.raises(ValueError):
        ns["load_byod"](fixture(tmp_path, mutation))


@pytest.mark.parametrize("name", ["../escape", "C:/escape"])
def test_zip_escape_rejected(tmp_path, name):
    ns = helpers(tmp_path)
    path = tmp_path / "bad.zip"
    info = zipfile.ZipInfo("placeholder")
    info.filename = name
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(info, "bad")
    with pytest.raises(ValueError):
        ns["safe_extract_zip"](path, tmp_path / "extract")


@pytest.mark.parametrize("metric", ["average_precision", "operating_metrics"])
def test_metric_rejects_cardinality_mismatch(tmp_path, metric):
    ns = helpers(tmp_path)
    args = (
        ([], [{"boxes": [], "labels": []}], CLASSES)
        if metric == "average_precision"
        else ([], [{"boxes": [], "labels": []}], 0.3)
    )
    with pytest.raises(ValueError):
        ns[metric](*args)


def test_inline_upstream_original_hashes():
    constants = {}
    for node in ast.parse(CELLS[18]["source"]).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            with contextlib.suppress(ValueError):
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    for path, size, digest in list(constants["UPSTREAM_FILES"].values()) + constants["OPS_FILES"]:
        data = constants["YOLOX_SOURCE_FILES"][path].encode()
        assert len(data) == size and hashlib.sha256(data).hexdigest() == digest
    assert "Apache License" in constants["YOLOX_LICENSE"]
    assert "urlopen" not in CELLS[18]["source"]


def test_real_orchestrator_with_model_double(tmp_path):
    ns = helpers(tmp_path)
    records = ns["load_byod"](fixture(tmp_path))
    events = []

    class Model:
        adapted = False

        @classmethod
        def from_pretrained(cls, *a, **k):
            events.append("load")
            return cls()

        def predict_many(self, rs, threshold):
            events.append(("predict", rs[0]["split"], self.adapted))
            return [self.detect(r["image"], threshold) for r in rs]

        def detect(self, image, threshold):
            return [{"label": CLASSES[0], "score": 0.9, "box": [0, 0, 10, 10]}] if self.adapted else []

        def finetune(self, rs, **kw):
            assert all(r["split"] == "train" for r in rs)
            events.append("train")
            self.adapted = True
            return {"epochs": kw["epochs"]}

        def save_artifact(self, path):
            path.write_bytes(b"adapted")
            events.append("save")
            return {"bytes": 7}

        @classmethod
        def load_artifact(cls, path, device):
            assert (path.parent / "frozen_experiment.json").exists()
            events.append("reload")
            obj = cls()
            obj.adapted = True
            return obj

    ns.update(
        RTDETRWorkshop=Model,
        YOLOXWorkshop=Model,
        DEVICE="cpu",
        EVAL_SCORE_THRESHOLD=0.01,
        DISPLAY_THRESHOLD=0.3,
        RTDETR_EPOCHS=3,
        RTDETR_BATCH_SIZE=4,
        RTDETR_LEARNING_RATE=1e-4,
        YOLOX_EPOCHS=6,
        YOLOX_BATCH_SIZE=2,
        YOLOX_LEARNING_RATE=1e-3,
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
    )
    for key in ["rtdetr", "yolox_s"]:
        rows = ns["run_byod_model"](key, records, tmp_path)
        assert [r["state"] for r in rows] == ["pre", "adapted"]
        result = json.loads((tmp_path / key / "results.json").read_text())
        assert result["parity"] == "live-to-fresh PASS"
        assert (tmp_path / key / "metrics.csv").is_file()
    assert events.index("save") < events.index("reload")


def test_parity_rejects_export_that_loses_adaptation(tmp_path):
    ns = helpers(tmp_path)
    with pytest.raises(RuntimeError):
        ns["assert_byod_parity"]([{"label": "a", "box": [0, 0, 1, 1], "score": 0.5}], [])


def test_stale_guard_accepts_local_build_suffix():
    from packaging.version import Version

    module = ast.parse(CELLS[5]["source"])
    module.body = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    ns = {"Version": Version}
    exec(compile(module, "guard", "exec"), ns)
    assert ns["stale_loaded_packages"]({"torch": "2.14.0+cu130"}, ["torch==2.14.0"]) == []
    assert ns["stale_loaded_packages"]({"torch": "2.13.0+cu130"}, ["torch==2.14.0"]) == ["torch"]


def test_implicit_split_positive_and_missing_coverage_negative(tmp_path):
    ns = helpers(tmp_path)

    def omit(rows, images):
        for row in rows:
            row["split"] = ""

    records = ns["load_byod"](fixture(tmp_path, omit))
    assert {s: sum(r["split"] == s for r in records) for s in ("train", "validation", "test")} == {
        "train": 3,
        "validation": 1,
        "test": 2,
    }

    def missing(rows, images):
        rows[:] = [r for r in rows if not (r["split"] == "test" and r["label"] == CLASSES[2])]

    with pytest.raises(ValueError, match="test must contain"):
        ns["load_byod"](fixture(tmp_path, missing))


def test_failed_model_released_even_with_retained_exception(tmp_path):
    import weakref

    ns = helpers(tmp_path)
    records = ns["load_byod"](fixture(tmp_path))
    references = []

    class FailingModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            obj = cls()
            references.append(weakref.ref(obj))
            return obj

        def predict_many(self, *args):
            raise RuntimeError("simulated inference failure")

    ns.update(
        RTDETRWorkshop=FailingModel,
        DEVICE="cpu",
        EVAL_SCORE_THRESHOLD=0.01,
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
    )
    retained = []
    for attempt in range(2):
        root = tmp_path / str(attempt)
        root.mkdir()
        try:
            ns["run_byod_model"]("rtdetr", records, root)
        except RuntimeError as exc:
            retained.append(exc)
    gc.collect()
    assert len(retained) == 2 and all(reference() is None for reference in references)
