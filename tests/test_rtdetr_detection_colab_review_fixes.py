"""Regression tests for the 2026-10-02 Notebook Review Framework v1 findings on rtdetr_detection_colab (RTD-*).

Only CI's dependencies are used (torch, transformers, numpy, pillow); no weights and no network. Model behaviour is
exercised on tiny stand-in modules; notebook behaviour on the generated notebook's own cell sources.
"""
# ruff: noqa: E501

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from rtdetr_detection_pipeline import (
    RTDetrDetectionPipeline,
    check_split_coverage,
    load_detection_dataset,
    sign_dataset,
    split_dataset,
    validate_dataset,
)
from rtdetr_detection_pipeline import pipeline as pipeline_module

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "tutorials" / "rtdetr_detection_colab.ipynb"


def _nb() -> dict:
    return json.loads(NB_PATH.read_text(encoding="utf-8"))


def _code(nb: dict) -> list[str]:
    return [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]


def _markdown(nb: dict) -> str:
    return "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "markdown")


def _cell(marker: str) -> str:
    found = [s for s in _code(_nb()) if marker in s]
    assert len(found) == 1, marker
    return found[0]


def _record(size=(64, 64), colour=(10, 20, 30), boxes=((4.0, 4.0, 20.0, 20.0),), labels=("stop-sign",)):
    return {"image": Image.new("RGB", size, colour), "boxes": [list(b) for b in boxes], "labels": list(labels)}


# --- RTD-m3: validate_dataset refuses what the prose says it refuses -------------------------------------------------


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([_record(boxes=((10.0, 10.0, 10.0, 10.0),))], "record 0 box 0 .* zero area"),
        ([_record(boxes=((10.0, 10.0, 30.0, 10.0),))], "zero area"),
        ([_record(boxes=(), labels=())], "record 0 has no boxes"),
        ([_record(), _record()], "record 1 repeats the image of record 0"),
        ([_record(labels=("unknown",))], "unknown class"),
        ([_record(boxes=((10.0, 10.0, 99.0, 20.0),))], "outside image bounds"),
    ],
)
def test_rtd_m3_validate_dataset_refuses_with_record_and_rule(records, message):
    with pytest.raises(ValueError, match=message):
        validate_dataset(records, ("stop-sign", "yield-sign"))


def test_rtd_m3_validate_dataset_ceiling_and_sample_still_accepted(monkeypatch):
    assert validate_dataset(sign_dataset(40, seed=0), ("stop-sign", "yield-sign", "speed-limit-sign"))["n_records"] == 40
    monkeypatch.setattr(pipeline_module, "MAX_DATASET_RECORDS", 1)
    with pytest.raises(ValueError, match="MAX_DATASET_RECORDS"):
        validate_dataset([_record(colour=(1, 1, 1)), _record(colour=(2, 2, 2))], ("stop-sign",))


def test_rtd_m3_split_coverage_names_the_split_and_class():
    records = sign_dataset(40, seed=0)
    train, held = split_dataset(records, train_fraction=0.75, seed=0)
    names = ("stop-sign", "yield-sign", "speed-limit-sign")
    assert check_split_coverage({"train": train, "held_out": held}, names)["held_out"] == sorted(names)
    only_stop = [r for r in held if set(r["labels"]) == {"stop-sign"}]
    with pytest.raises(ValueError, match=r"held_out split .* \['yield-sign', 'speed-limit-sign'\]"):
        check_split_coverage({"train": train, "held_out": only_stop}, names)


# --- RTD-M4: a real dataset loader behind a path field -------------------------------------------------------------


def _write_dataset(target: Path, records, *, as_zip: bool, rows_override=None) -> Path:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["file", "label", "x0", "y0", "x1", "y1"])
    images = {}
    for index, record in enumerate(records):
        name = f"images/img_{index:02d}.png"
        png = io.BytesIO()
        record["image"].save(png, format="PNG")
        images[name] = png.getvalue()
        for box, label in zip(record["boxes"], record["labels"], strict=True):
            writer.writerow([name, label, *box])
    text = buffer.getvalue() if rows_override is None else rows_override
    if as_zip:
        path = target.with_suffix(".zip")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("annotations.csv", text)
            for name, data in images.items():
                archive.writestr(name, data)
        return path
    target.mkdir(parents=True)
    (target / "annotations.csv").write_text(text, encoding="utf-8")
    for name, data in images.items():
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        (target / name).write_bytes(data)
    return target


@pytest.mark.parametrize("as_zip", [False, True])
def test_rtd_m4_loader_reads_folder_and_zip_into_valid_records(tmp_path, as_zip):
    source = sign_dataset(12, seed=5)
    path = _write_dataset(tmp_path / "ds", source, as_zip=as_zip)
    records, classes = load_detection_dataset(path)
    assert len(records) == 12 and set(classes) == {lab for r in source for lab in r["labels"]}
    assert classes[0] == source[0]["labels"][0]  # first-appearance order
    for loaded, original in zip(records, source, strict=True):
        assert np.allclose(loaded["boxes"], original["boxes"])
        assert loaded["labels"] == original["labels"]
        assert np.array_equal(np.asarray(loaded["image"]), np.asarray(original["image"]))
    assert validate_dataset(records, classes)["verdict"] == "accepted"


@pytest.mark.parametrize(
    ("csv_text", "message"),
    [
        ("file,label,x0,y0\nimages/img_00.png,stop-sign,1,2\n", "header must start with"),
        ("file,label,x0,y0,x1,y1\nimages/missing.png,stop-sign,1,2,3,4\n", "row 2: image 'images/missing.png' is not in the dataset"),
        ("file,label,x0,y0,x1,y1\nimages/img_00.png,stop-sign,1,nan,3,4\n", "row 2: y0='nan' is not finite"),
        ("file,label,x0,y0,x1,y1\nimages/img_00.png,stop-sign,1,abc,3,4\n", "row 2: y0='abc' is not a number"),
        ("file,label,x0,y0,x1,y1\nnotes.txt,stop-sign,1,2,3,4\n", "row 2: 'notes.txt' is not a"),
    ],
)
def test_rtd_m4_malformed_dataset_is_refused_naming_row_and_rule(tmp_path, csv_text, message):
    path = _write_dataset(tmp_path / "bad", sign_dataset(2, seed=1), as_zip=True, rows_override=csv_text)
    with pytest.raises(ValueError, match=re.escape(message) if "'" in message else message):
        load_detection_dataset(path)


def test_rtd_m4_out_of_bounds_box_from_file_is_refused_by_validation(tmp_path):
    source = sign_dataset(2, seed=1)
    source[1]["boxes"][0] = [10.0, 10.0, 5000.0, 20.0]
    records, classes = load_detection_dataset(_write_dataset(tmp_path / "oob", source, as_zip=True))
    with pytest.raises(ValueError, match="record 1 box 0 .* outside image bounds"):
        validate_dataset(records, classes)


class _FakePipe:
    """Stand-in for RTDetrDetectionPipeline in the Section 13 BYOD cell: records the calls, writes a real file."""

    calls: list = []

    def __init__(self, names):
        self.class_names = tuple(names)

    @classmethod
    def from_pretrained(cls, *, weights_dir, class_names, seed):
        cls.calls.append(("from_pretrained", tuple(class_names)))
        return cls(class_names)

    @classmethod
    def load_artifact(cls, path, *, weights_dir):
        cls.calls.append(("load_artifact", Path(path).name))
        return cls(json.loads(Path(path).read_text())["class_names"])

    def evaluate(self, records):
        return {"ap": 0.5, "ap50": 0.5, "ap75": 0.5, "n_images": len(records)}

    def finetune(self, records, **kw):
        self.calls.append(("finetune", len(records), kw["freeze_backbone"]))
        return {"epochs": kw["epochs"], "freeze_backbone": kw["freeze_backbone"]}

    def save_artifact(self, path, *, notes, training):
        Path(path).write_text(json.dumps({"class_names": list(self.class_names), "training": training}))
        return {"path": str(path), "training": training}

    def raw_outputs(self, images):
        return [{"logits": np.zeros((2, 3)), "pred_boxes": np.zeros((2, 4))} for _ in images]


def _section13_namespace(tmp_path, monkeypatch):
    import rtdetr_detection_pipeline as package

    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    ns = {name: getattr(package, name) for name in package.__all__}
    ns.update(
        os=__import__("os"), json=json, np=np, Path=Path, Image=Image, OUTPUTS=Path("outputs"), threshold=0.3,
        EPOCHS=1, BATCH_SIZE=2, LEARNING_RATE=1e-4, SEED=0, HOLDOUT=0.25, FREEZE_BACKBONE=True, WEIGHTS_DIR=Path("w"),
        PARITY_TOLERANCE=1e-3, pipe=None, RTDetrDetectionPipeline=_FakePipe,
    )
    _FakePipe.calls = []
    return ns


def test_rtd_m4_byod_dataset_branch_runs_to_export_and_reload_without_code_edits(tmp_path, monkeypatch):
    source = _cell("USE_BYOD_DATASET = False  # @param")
    assert "byod_records = []" not in source
    dataset = _write_dataset(tmp_path / "mine", sign_dataset(16, seed=0), as_zip=True)
    ns = _section13_namespace(tmp_path, monkeypatch)
    edited = source.replace("USE_BYOD_DATASET = False  # @param", "USE_BYOD_DATASET = True  # @param").replace(
        "BYOD_DATASET_PATH = ''  # @param", f"BYOD_DATASET_PATH = {str(dataset)!r}  # @param"
    )
    exec(compile(edited, "<section 13 BYOD dataset>", "exec"), ns)
    kinds = [c[0] for c in _FakePipe.calls]
    assert kinds == ["from_pretrained", "finetune", "load_artifact"]
    result = json.loads((tmp_path / "outputs" / "byod_result.json").read_text())
    assert result["split"]["train"] + result["split"]["held_out"] == 16
    assert result["training"]["freeze_backbone"] is True and result["reload_max_abs_difference"] == 0.0
    assert set(result["class_names"]) <= {"stop-sign", "yield-sign", "speed-limit-sign"}


def test_rtd_m4_byod_dataset_branch_refuses_a_label_outside_the_stated_classes(tmp_path, monkeypatch):
    source = _cell("USE_BYOD_DATASET = False  # @param")
    dataset = _write_dataset(tmp_path / "mine", sign_dataset(16, seed=0), as_zip=True)
    ns = _section13_namespace(tmp_path, monkeypatch)
    edited = (
        source.replace("USE_BYOD_DATASET = False  # @param", "USE_BYOD_DATASET = True  # @param")
        .replace("BYOD_DATASET_PATH = ''  # @param", f"BYOD_DATASET_PATH = {str(dataset)!r}  # @param")
        .replace("BYOD_CLASS_NAMES = ''  # @param", "BYOD_CLASS_NAMES = 'stop-sign'  # @param")
    )
    with pytest.raises(ValueError, match="unknown class"):
        exec(compile(edited, "<section 13 BYOD dataset>", "exec"), ns)
    assert _FakePipe.calls == []  # refused before any model was built


# --- RTD-M2: the freeze flag reaches the computation, and every run starts from the pretrained model ---------------


def _tiny_pipeline():
    torch = pytest.importorskip("torch")

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Linear(3, 3)

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()
            self.head = torch.nn.Linear(3, 3)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

        def forward(self, pixel_values, labels=None):
            features = self.model.head(self.model.backbone.conv(pixel_values.mean(dim=(2, 3))))
            return SimpleNamespace(loss=features.pow(2).mean(), logits=features[:, None, :], pred_boxes=torch.sigmoid(features[:, None, :]).repeat(1, 1, 2)[..., :4])

    class Batch(dict):
        def to(self, device):
            return self

    def processor(images, return_tensors):
        arrays = [np.asarray(img.convert("RGB").resize((8, 8)), dtype=np.float32) / 255.0 for img in images] if isinstance(images, list) else [np.asarray(images.convert("RGB").resize((8, 8)), dtype=np.float32) / 255.0]
        return Batch(pixel_values=torch.tensor(np.stack(arrays)).permute(0, 3, 1, 2))

    torch.manual_seed(0)
    return RTDetrDetectionPipeline(model=Model(), processor=processor, device="cpu", class_names=("stop-sign", "yield-sign", "speed-limit-sign")), torch


def test_rtd_m2_freeze_flag_is_applied_in_both_directions():
    pipe, _torch = _tiny_pipeline()
    records = sign_dataset(4, seed=0)
    frozen = pipe.finetune(records, epochs=1, batch_size=2, seed=0, freeze_backbone=True)
    assert not any(p.requires_grad for p in pipe.model.model.backbone.parameters())
    assert frozen["trainable_parameters"] == 12 and frozen["started_from_adapted"] is False
    unfrozen = pipe.finetune(records, epochs=1, batch_size=2, seed=0, freeze_backbone=False)
    assert all(p.requires_grad for p in pipe.model.parameters())  # failed before: the backbone stayed frozen
    assert unfrozen["trainable_parameters"] == unfrozen["total_parameters"] == 24
    assert unfrozen["started_from_adapted"] is True and unfrozen["freeze_backbone"] is False
    assert isinstance(unfrozen["train_seconds"], float)


def test_rtd_m2_section8_rebuilds_the_adapter_before_training_and_checks_the_model_state():
    source = _cell("FREEZE_BACKBONE = True  # @param")
    rebuild = source.index("adapter = RTDetrDetectionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=SIGN_CLASSES, seed=SEED)")
    assert rebuild < source.index("run = adapter.finetune(")
    assert "assert backbone_frozen == FREEZE_BACKBONE" in source and "assert not run['started_from_adapted']" in source
    assert "EPOCHS = 3  # @param" in source  # the schedule is set where it is used
    markdown = _markdown(_nb())
    assert "rerun **Section 8 and then Section 9**" in markdown
    assert "Predict → Change → Run → Observe → Explain" in markdown


# --- RTD-m4 / RTD-m2: raw-output reload parity and exported provenance --------------------------------------------


def test_rtd_m4_raw_outputs_and_training_record_round_trip(tmp_path):
    pipe, torch = _tiny_pipeline()
    run = pipe.finetune(sign_dataset(4, seed=0), epochs=1, batch_size=2, seed=0)
    rows = pipe.raw_outputs([Image.new("RGB", (32, 32), (200, 0, 0))])
    assert set(rows[0]) == {"logits", "pred_boxes"} and rows[0]["logits"].shape == (1, 3)
    descriptor = pipe.save_artifact(tmp_path / "a.pt", notes="t", training=run)
    payload = torch.load(tmp_path / "a.pt", map_location="cpu", weights_only=True)
    for key in ("learning_rate", "batch_size", "seed", "freeze_backbone", "epoch_losses", "train_seconds"):
        assert key in payload["training"] and key in descriptor["training"]


def test_rtd_m4_section11_compares_raw_outputs_on_all_images_and_cannot_pass_empty():
    source = _cell("PARITY_TOLERANCE = 1e-3")
    assert "compare_images = [r['image'] for r in held_out] + [r['image'] for r in new_records]" in source
    assert "assert detections_compared > 0" in source and "max_abs_logit_difference" in source
    assert "exactly" not in source.lower() and "identically" not in _markdown(_nb())


def test_rtd_m2_result_json_and_new_data_outputs_are_exported():
    source = _cell("result_export = {")
    for key in ("'training':", "'zero_training_reference': reference_table", "'new_data': new_data_rows", "'reload_parity': reload_parity", "'degenerate_probes': degenerate"):
        assert key in source, key
    assert "_new_data.csv" in source and "_new_data_annotated.png" in source and "best_same_label_reference_iou" in source


# --- RTD-M3: a zero-training reference next to the baseline, and honest prose ---------------------------------------


def test_rtd_m3_section9_prints_a_zero_training_reference_on_the_same_split():
    source = _cell("adapted = adapter.evaluate(held_out)")
    assert "pipe.detect(r['image'], threshold=EVAL_DETECTION_THRESHOLD)" in source and "for r in held_out" in source
    assert "coco_reference = average_precision(" in source and "'unadapted_coco_detector'" in source
    markdown = _markdown(_nb())
    assert "**The baseline is zero by construction.**" in markdown
    assert "colour-and-shape rule" in markdown and "AP 1.000" in markdown
    assert "that is the expected starting point" not in markdown


# --- RTD-M5 / RTD-m1 / RTD-m6 / RTD-m5: guided layer, labelled numbers, interpretation, docs -----------------------


def test_rtd_m5_degenerate_probe_and_new_data_are_interpreted():
    markdown = _markdown(_nb())
    assert markdown.count("**What to notice.**") >= 3
    assert "Any detection at 0.3 on a blank or noise image is a false positive" in markdown
    assert "**Who owns it?**" in markdown and "duplicate" in markdown and "miss" in markdown


def test_rtd_m1_numbers_are_labelled_and_not_stale():
    markdown = _markdown(_nb())
    for stale in ("~30–50 s", "~1–2 s", "0.91–0.97", "under a minute"):
        assert stale not in markdown, stale
    assert "estimate" in markdown and "train_seconds" in markdown


def test_rtd_m5_docs_describe_the_e2e_notebook_and_spec_22():
    nb = _nb()
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.2"
    for doc in ("README.md", "docs/release-verification.md", "STATUS.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        assert "TASK-INFERENCE" not in text, doc
    record = (ROOT / "docs/release-verification.md").read_text(encoding="utf-8")
    assert "1 automatic restart" not in record and "check_split_coverage" in record
    assert "Run-all: verified" not in (ROOT / "tutorials/README.md").read_text(encoding="utf-8")


# --- RTD-M1 lock coverage: every backend the model path calls is in the hash lock ----------------------------------


def test_rtd_m1_lock_carries_scipy_for_the_hungarian_matcher():
    lock = (ROOT / "tutorials" / "requirements-colab.lock.txt").read_text(encoding="utf-8")
    pinned = set(re.findall(r"^([a-z0-9][a-z0-9._-]*)==", lock, re.M))
    # transformers' RTDetrHungarianMatcher calls requires_backends(self, ["scipy"]) inside the training loss.
    for name in ("scipy", "torch", "torchvision", "transformers", "numpy", "pillow", "safetensors", "huggingface-hub"):
        assert name in pinned, name
    assert "scipy==1.18.1" in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
