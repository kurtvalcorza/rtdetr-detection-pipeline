import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from rtdetr_detection_pipeline import (
    ARTIFACT_FORMAT,
    DEFAULT_WEIGHTS_DIR,
    DETECTION_THRESHOLD,
    LABELS,
    MAX_DETECTIONS,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    SIGN_CLASSES,
    RTDetrDetectionPipeline,
    average_precision,
    box_iou,
    sign_dataset,
    split_dataset,
    stage_missing_files,
    validate_dataset,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]


def test_identity_constants():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "PekingU/rtdetr_r50vd"
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    assert 0 < DETECTION_THRESHOLD < 1 and len(LABELS) == 80 and MAX_DETECTIONS == 300
    assert LABELS[0] == "person" and LABELS[11] == "stop sign" and LABELS[-1] == "toothbrush"
    assert ARTIFACT_FORMAT == "rtdetr-adapter-v1"
    manifest = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["modelId"] == MODEL_ID
        assert data["revision"] == MODEL_REVISION


def _write_snapshot(root: Path, content: bytes, sha: str | None = None, size: int | None = None) -> None:
    (root / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "config.json",
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_snapshot_accepts_matching_manifest(tmp_path):
    _write_snapshot(tmp_path, b'{"model_type": "rt_detr"}')
    info = verify_snapshot(tmp_path)
    assert info["revision"] == MODEL_REVISION and info["files"] == 1


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"model_type": "rt_detr"}'
    good = hashlib.sha256(content).hexdigest()
    flipped = ("0" if good[0] != "0" else "1") + good[1:]
    _write_snapshot(tmp_path, content, sha=flipped)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_size_missing_file_and_revision(tmp_path):
    _write_snapshot(tmp_path, b"abc", size=99)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["revision"] = "0" * 40
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    listed = verify_snapshot(tmp_path)["files"]
    assert (listed if isinstance(listed, int) else len(listed)) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


class MockPipeline(RTDetrDetectionPipeline):
    """Mock pipeline that overrides _run for fast offline unit testing."""

    def __init__(self, runner_fn=None, class_names=LABELS, device="cpu"):
        self.runner_fn = runner_fn
        super().__init__(
            model=None,
            processor=None,
            device=device,
            class_names=tuple(class_names),
            source="mock",
        )

    def _run(self, image: Image.Image, threshold: float) -> list[dict[str, Any]]:
        if self.runner_fn:
            return self.runner_fn(image, threshold)
        return [
            {"box": [1.0, 2.0, 10.0, 20.0], "label": "clock", "score": 0.91},
            {"box": [0.0, 0.0, 5.0, 5.0], "label": "stop sign", "score": 0.99},
        ]


def test_detect_output_fields_and_defaults():
    calls: list = []

    def runner(img, thresh):
        calls.append((img.mode, thresh))
        return [
            {"box": [1.0, 2.0, 10.0, 20.0], "label": "clock", "score": 0.91},
            {"box": [0.0, 0.0, 5.0, 5.0], "label": "stop sign", "score": 0.99},
        ]

    pipe = MockPipeline(runner)
    result = pipe.detect(Image.new("L", (40, 30)))
    assert [d["label"] for d in result["detections"]] == ["stop sign", "clock"]
    assert result["threshold"] == DETECTION_THRESHOLD
    assert (result["width"], result["height"]) == (40, 30)
    assert result["model_id"] == MODEL_ID and result["model_revision"] == MODEL_REVISION
    assert calls == [("RGB", DETECTION_THRESHOLD)]
    assert pipe.detect(Image.new("RGB", (40, 30)), threshold=0.5)["threshold"] == 0.5


def test_detect_rejects_bad_inputs():
    pipe = MockPipeline()
    with pytest.raises(TypeError):
        pipe.detect(np.zeros((30, 40, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        pipe.detect(Image.new("RGB", (MIN_IMAGE_SIDE - 1, 64)))
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        pipe.detect(Image.new("RGB", (MAX_IMAGE_SIDE + 1, 64)))
    with pytest.raises(ValueError, match="threshold"):
        pipe.detect(Image.new("RGB", (64, 64)), threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        pipe.detect(Image.new("RGB", (64, 64)), threshold=True)


def test_detect_rejects_malformed_backend_output():
    pipe = MockPipeline(lambda *_: [{"box": [0, 0, 1], "label": "stop sign", "score": 0.1}])
    with pytest.raises(RuntimeError):
        pipe.detect(Image.new("RGB", (64, 64)))

    pipe2 = MockPipeline(lambda *_: [{"box": [0, 0, 1, 1], "label": "unicorn", "score": 0.1}])
    with pytest.raises(RuntimeError, match="malformed"):
        pipe2.detect(Image.new("RGB", (64, 64)))

    pipe3 = MockPipeline(
        lambda *_: [{"box": [0, 0, 1, 1], "label": "stop sign", "score": 0.5}] * (MAX_DETECTIONS + 1)
    )
    with pytest.raises(RuntimeError, match="num_queries"):
        pipe3.detect(Image.new("RGB", (64, 64)))


def test_box_iou():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    with pytest.raises(ValueError):
        box_iou([10, 0, 0, 10], [0, 0, 1, 1])


def test_average_precision_matching():
    preds = [
        [
            {"box": [10.0, 10.0, 50.0, 50.0], "label": "stop-sign", "score": 0.95},
            {"box": [100.0, 100.0, 150.0, 150.0], "label": "yield-sign", "score": 0.85},
        ]
    ]
    refs = [
        {
            "boxes": [[10.0, 10.0, 50.0, 50.0], [100.0, 100.0, 150.0, 150.0]],
            "labels": ["stop-sign", "yield-sign"],
        }
    ]
    metrics = average_precision(preds, refs, ("stop-sign", "yield-sign"))
    assert metrics["ap50"] == pytest.approx(1.0)
    assert metrics["ap75"] == pytest.approx(1.0)
    assert metrics["ap"] == pytest.approx(1.0)
    assert metrics["per_class_ap50"]["stop-sign"] == pytest.approx(1.0)
    assert metrics["per_class_ap50"]["yield-sign"] == pytest.approx(1.0)

    # Completely wrong boxes yield 0.0 AP
    wrong_preds = [[{"box": [500.0, 500.0, 600.0, 600.0], "label": "stop-sign", "score": 0.99}]]
    metrics_zero = average_precision(wrong_preds, refs, ("stop-sign", "yield-sign"))
    assert metrics_zero["ap50"] == pytest.approx(0.0)


def test_sign_dataset_and_split():
    dataset = sign_dataset(n_images=10, seed=42)
    assert len(dataset) == 10
    for rec in dataset:
        assert isinstance(rec["image"], Image.Image)
        assert len(rec["boxes"]) == len(rec["labels"])
        for label in rec["labels"]:
            assert label in SIGN_CLASSES

    train, val = split_dataset(dataset, train_fraction=0.7, seed=42)
    assert len(train) == 7
    assert len(val) == 3


def test_validate_dataset():
    records = [
        {
            "image": Image.new("RGB", (100, 100)),
            "boxes": [[10.0, 10.0, 50.0, 50.0]],
            "labels": ["stop-sign"],
        }
    ]
    manifest = validate_dataset(records, SIGN_CLASSES, epochs=2)
    assert manifest["verdict"] == "accepted"
    assert manifest["n_records"] == 1
    assert manifest["n_boxes"] == 1

    # Box out of bounds
    with pytest.raises(ValueError, match="outside image bounds"):
        validate_dataset(
            [
                {
                    "image": Image.new("RGB", (100, 100)),
                    "boxes": [[10.0, 10.0, 150.0, 50.0]],
                    "labels": ["stop-sign"],
                }
            ],
            SIGN_CLASSES,
        )

    # Unknown label
    with pytest.raises(ValueError, match="unknown class"):
        validate_dataset(
            [
                {
                    "image": Image.new("RGB", (100, 100)),
                    "boxes": [[10.0, 10.0, 50.0, 50.0]],
                    "labels": ["unknown-label"],
                }
            ],
            SIGN_CLASSES,
        )


def test_labels_match_snapshot_config_order():
    config = REPO / "weights" / MODEL_KEY / "config.json"
    if not config.is_file():
        pytest.skip("snapshot config.json not present")
    id2label = json.loads(config.read_text(encoding="utf-8"))["id2label"]
    assert tuple(id2label[str(i)] for i in range(len(id2label))) == LABELS
