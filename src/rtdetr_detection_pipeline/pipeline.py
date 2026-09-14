"""Real-time object detection with the pinned ``PekingU/rtdetr_r50vd`` checkpoint (RT-DETR, COCO classes).

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the RT-DETR architecture comes from the pinned ``transformers`` release,
the weights are SafeTensors, and no model-repository code is executed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "PekingU/rtdetr_r50vd"
MODEL_REVISION = "df939e661d8c52e80608d1ec566561aabd25a4e7"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "rtdetr-r50vd"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# The 80 COCO 2017 classes the checkpoint was trained on, in config.json id2label order (the
# upstream spelling: "motorbike", "aeroplane", "sofa", "pottedplant", "tvmonitor", ...).
LABELS = (
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)
# Detection threshold: the value the pinned README's transformers example passes to
# post_process_object_detection (threshold=0.3). RT-DETR is trained with a focal (sigmoid) loss, so
# each score is an independent per-class sigmoid, not a softmax over classes; the value was not
# calibrated for any deployment and the deployment owns tuning it on labelled images.
DETECTION_THRESHOLD = 0.3
# The decoder emits exactly num_queries proposals (config.json) and the processor keeps at most that
# many (query, class) pairs, so no image can yield more than this many boxes.
MAX_DETECTIONS = 300
# Input ceilings. The processor resizes every image to 640x640 (preprocessor_config.json `size`,
# aspect ratio not preserved) and rescales to [0, 1] without mean/std normalisation, so image cost
# is bounded whatever the caller sends; the side ceiling only guards memory during decoding.
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two xyxy pixel boxes; the building block for any caller-side mAP."""
    if len(a) != 4 or len(b) != 4:
        raise ValueError("boxes must be [x0, y0, x1, y1]")
    if a[2] < a[0] or a[3] < a[1] or b[2] < b[0] or b[3] < b[1]:
        raise ValueError("boxes must satisfy x0 <= x1 and y0 <= y1")
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


def _check_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise ValueError(f"threshold must be a number in [0, 1], got {value!r}")
    return float(value)


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one image as PIL.Image.Image (any mode, converted to RGB): a photograph or a rendered scene",
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "threshold": [0.0, 1.0],
    "labels": list(LABELS),
    "max_detections": MAX_DETECTIONS,
    "preprocessing": (
        "image converted to RGB; the processor resizes to 640x640 without preserving the aspect ratio "
        "and rescales to [0, 1] (no mean/std normalisation); returned boxes are mapped back to input pixels"
    ),
}


def _check_inputs(image: Any, threshold: Any) -> tuple[Image.Image, float]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request.

    ``detect`` and ``validate_inputs`` both route through this function so their acceptance
    criteria cannot diverge.
    """
    return validate_image(image), _check_threshold(threshold)


def validate_inputs(
    image: Image.Image,
    *,
    threshold: float = DETECTION_THRESHOLD,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Rejection is reported by raising exactly as ``detect`` would; a caller that wants the finding
    recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    _rgb, checked = _check_inputs(image, threshold)
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (detect takes one image)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": names[0] if names else "image-0", "mode": image.mode, "size": list(image.size)}],
        "threshold": checked,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any],
    ground_truth_boxes: Mapping[str, Sequence[Sequence[float]]] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``ground_truth_boxes`` (a mapping label -> xyxy reference boxes of the objects in the image)
    the report carries one ``box_iou`` entry per reference — the best-overlapping detection **of the
    same label** — as sample-sanity geometry evidence; without them the verdict is ``not-measurable``
    and the report says what labelled data would make the task measurable.
    """
    detections = list(result["detections"])
    base = {
        "task": "object detection over the 80 COCO classes on one image",
        "decision_rule": (
            "a (query, class) pair survives when its sigmoid class score reaches the threshold; the score "
            "is an independent per-class sigmoid under the model's own focal-loss head, not a calibrated "
            "probability for the deployment's images, and one query can surface under several classes"
        ),
        "threshold": result.get("threshold", DETECTION_THRESHOLD),
        "sample_kind": sample_kind,
        "n_detections": len(detections),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if not ground_truth_boxes:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no ground-truth object boxes were supplied for the evaluated image",
            "needs": (
                "labelled boxes per class on your own images, scored per object with box_iou and aggregated "
                "into COCO-style mean average precision (AP@[.50:.95], AP50) at stated IoU thresholds; no "
                "such labelled set ships with this repository"
            ),
        }
    metrics = []
    for label, boxes in ground_truth_boxes.items():
        if label not in LABELS:
            raise ValueError(f"unknown reference label {label!r}; expected one of the 80 COCO LABELS")
        same_label = [det for det in detections if det["label"] == label]
        for index, box in enumerate(boxes):
            ious = [box_iou(det["box"], box) for det in same_label]
            best = max(range(len(ious)), key=ious.__getitem__) if ious else None
            metrics.append(
                {
                    "id": "box_iou",
                    "reference": f"{label}-{index}",
                    "value": ious[best] if best is not None else 0.0,
                    "matched_score": same_label[best]["score"] if best is not None else None,
                    "n_detected_same_label": len(same_label),
                    "estimation": "one reference box per object on a single image, no dispersion estimate",
                }
            )
    return {
        **base,
        "metrics": metrics,
        "verdict": "sample-sanity",
        "reason": (
            f"{len(metrics)} reference box(es) on one tutorial image; geometry sanity evidence, "
            "not a detection benchmark"
        ),
        "needs": (
            "a labelled image set from the deployment domain (cameras, scenes, object classes) for any "
            "COCO-style mean-average-precision or precision/recall claim"
        ),
    }


@dataclass
class RTDetrDetectionPipeline:
    """COCO-class object detection over the pinned RT-DETR (ResNet-50-vd) checkpoint."""

    _runner: Callable[[Image.Image, float], list[dict[str, Any]]]
    device: str

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> RTDetrDetectionPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            source, kwargs = str(root), {"local_files_only": True}
        elif allow_download:
            source, kwargs = MODEL_ID, {}
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage {MODEL_ID}@{MODEL_REVISION} under weights/{MODEL_KEY}"
            )
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import AutoImageProcessor, RTDetrForObjectDetection

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        processor = AutoImageProcessor.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        # config.json already says use_pretrained_backbone=false and use_timm_backbone=false (the
        # ResNet-50-vd backbone is the in-library RTDetrResNet); it is passed explicitly anyway so a
        # future config change cannot re-introduce a construction-time weight download.
        model = RTDetrForObjectDetection.from_pretrained(
            source,
            revision=MODEL_REVISION,
            trust_remote_code=False,
            use_pretrained_backbone=False,
            **kwargs,
        )
        model = model.to(resolved_device).eval()
        id2label = {int(k): v for k, v in model.config.id2label.items()}

        def runner(image: Image.Image, threshold: float) -> list[dict]:
            inputs = processor(images=image, return_tensors="pt").to(resolved_device)
            with torch.inference_mode():
                outputs = model(**inputs)
            # use_focal_loss=True: per-class sigmoid scores, top num_queries (query, class) pairs,
            # then the threshold — the pinned processor's default for this checkpoint.
            result = processor.post_process_object_detection(
                outputs, threshold=threshold, target_sizes=[image.size[::-1]]
            )[0]
            return [
                {
                    "box": [float(v) for v in box.tolist()],
                    "label": id2label[int(label)],
                    "score": float(score),
                }
                for box, label, score in zip(result["boxes"], result["labels"], result["scores"], strict=True)
            ]

        return cls(runner, resolved_device)

    def detect(self, image: Image.Image, *, threshold: float = DETECTION_THRESHOLD) -> dict[str, Any]:
        """Detect COCO-class objects on one image; boxes are xyxy pixel coordinates in the input image."""
        rgb, checked = _check_inputs(image, threshold)
        detections = self._runner(rgb, checked)
        if len(detections) > MAX_DETECTIONS:
            raise RuntimeError(
                f"backend returned {len(detections)} detections > num_queries {MAX_DETECTIONS}"
            )
        for det in detections:
            if set(det) != {"box", "label", "score"} or len(det["box"]) != 4 or det["label"] not in LABELS:
                raise RuntimeError(f"backend returned a malformed detection: {det!r}")
        return {
            "detections": sorted(detections, key=lambda d: -d["score"]),
            "threshold": checked,
            "width": rgb.width,
            "height": rgb.height,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
