"""Real-time object detection and bounded detection fine-tuning with the pinned RT-DETR R50-VD checkpoint.

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the RT-DETR architecture comes from the pinned ``transformers`` release,
the weights are SafeTensors, and no model-repository code is executed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MODEL_ID = "PekingU/rtdetr_r50vd"
MODEL_REVISION = "df939e661d8c52e80608d1ec566561aabd25a4e7"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "rtdetr-r50vd"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
ARTIFACT_FORMAT = "rtdetr-adapter-v1"

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
EVAL_DETECTION_THRESHOLD = 0.05
MAX_DETECTIONS = 300
MAX_EVAL_DETECTIONS = 100
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
MAX_CLASSES = 1000

# Training defaults for the bounded tutorial adaptation.
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_SEED = 20260915
HEAD_PRIOR_PROB = 0.01

COCO_IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))


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
    """Intersection-over-union of two xyxy pixel boxes; the building block for caller-side mAP."""
    if len(a) != 4 or len(b) != 4:
        raise ValueError("boxes must be [x0, y0, x1, y1]")
    if a[2] < a[0] or a[3] < a[1] or b[2] < b[0] or b[3] < b[1]:
        raise ValueError("boxes must satisfy x0 <= x1 and y0 <= y1")
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def average_precision(
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    references: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
    *,
    iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
) -> dict[str, Any]:
    """Small, faithful COCO-style average precision over a scored dataset.

    For each class and IoU threshold, detections are matched greedily to references by descending
    score. Each detection matches at most one ground-truth box; recall-precision curves are sampled
    with 101-point interpolation.
    """
    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} prediction lists but {len(references)} references")
    names = list(class_names)
    recall_points = np.linspace(0.0, 1.0, 101)
    per_threshold: dict[float, dict[str, float]] = {}

    for threshold in iou_thresholds:
        per_class: dict[str, float] = {}
        for name in names:
            scored: list[tuple[float, bool]] = []
            n_references = 0
            for dets, reference in zip(predictions, references, strict=True):
                ref_boxes = [
                    box
                    for box, label in zip(reference["boxes"], reference["labels"], strict=True)
                    if label == name
                ]
                n_references += len(ref_boxes)
                claimed = [False] * len(ref_boxes)
                candidates = sorted((d for d in dets if d["label"] == name), key=lambda d: -float(d["score"]))
                for det in candidates:
                    best, best_iou = -1, 0.0
                    for j, ref_box in enumerate(ref_boxes):
                        if claimed[j]:
                            continue
                        value = box_iou(det["box"], ref_box)
                        if value > best_iou:
                            best, best_iou = j, value
                    hit = best >= 0 and best_iou >= threshold
                    if hit:
                        claimed[best] = True
                    scored.append((float(det["score"]), hit))
            if n_references == 0:
                continue
            scored.sort(key=lambda pair: -pair[0])
            true_positives = np.cumsum([1 if hit else 0 for _score, hit in scored])
            false_positives = np.cumsum([0 if hit else 1 for _score, hit in scored])
            if not len(scored):
                per_class[name] = 0.0
                continue
            recall = true_positives / n_references
            precision = true_positives / np.maximum(true_positives + false_positives, 1)
            precision = np.maximum.accumulate(precision[::-1])[::-1]
            sampled = np.zeros_like(recall_points)
            indices = np.searchsorted(recall, recall_points, side="left")
            valid = indices < len(precision)
            sampled[valid] = precision[indices[valid]]
            per_class[name] = float(sampled.mean())
        per_threshold[threshold] = per_class

    scored_classes = sorted({name for values in per_threshold.values() for name in values})
    means = {
        threshold: (float(np.mean(list(values.values()))) if values else 0.0)
        for threshold, values in per_threshold.items()
    }
    return {
        "ap": float(np.mean(list(means.values()))) if means else 0.0,
        "ap50": means.get(0.5, 0.0),
        "ap75": means.get(0.75, 0.0),
        "per_class_ap50": per_threshold.get(0.5, {}),
        "iou_thresholds": [float(t) for t in iou_thresholds],
        "scored_classes": scored_classes,
        "n_images": len(references),
        "n_references": sum(len(r["boxes"]) for r in references),
    }


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


def _check_threshold(value: Any, name: str = "threshold") -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
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
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request."""
    return validate_image(image), _check_threshold(threshold)


def validate_inputs(
    image: Image.Image,
    *,
    threshold: float = DETECTION_THRESHOLD,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict)."""
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


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
    *,
    epochs: int = DEFAULT_EPOCHS,
) -> dict[str, Any]:
    """Validate adaptation dataset records and return the dataset manifest."""
    if not records:
        raise ValueError("dataset must hold at least one record")
    if not 1 <= epochs <= 100:
        raise ValueError(f"epochs must be in 1..100, got {epochs}")
    valid_classes = set(class_names)
    total_boxes = 0
    observed_classes: set[str] = set()

    for idx, record in enumerate(records):
        if "image" not in record or "boxes" not in record or "labels" not in record:
            raise ValueError(f"record {idx} must contain 'image', 'boxes', and 'labels'")
        img = validate_image(record["image"])
        boxes = record["boxes"]
        labels = record["labels"]
        if len(boxes) != len(labels):
            raise ValueError(f"record {idx}: {len(boxes)} boxes but {len(labels)} labels")
        width, height = img.size
        for b_idx, box in enumerate(boxes):
            if len(box) != 4:
                raise ValueError(f"record {idx} box {b_idx} must have 4 elements, got {len(box)}")
            x0, y0, x1, y1 = box
            if not (0.0 <= x0 <= x1 <= width and 0.0 <= y0 <= y1 <= height):
                raise ValueError(
                    f"record {idx} box {b_idx} [{x0}, {y0}, {x1}, {y1}] "
                    f"outside image bounds {(width, height)}"
                )
        for label in labels:
            if label not in valid_classes:
                raise ValueError(f"record {idx} has unknown class {label!r}; expected one of {class_names}")
            observed_classes.add(label)
        total_boxes += len(boxes)

    return {
        "n_records": len(records),
        "n_boxes": total_boxes,
        "class_names": list(class_names),
        "observed_classes": sorted(observed_classes),
        "epochs": epochs,
        "verdict": "accepted",
    }


def evaluation_report(
    result: Mapping[str, Any],
    ground_truth_boxes: Mapping[str, Sequence[Sequence[float]]] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Single-image evaluation stage: machine-readable report with per-object box_iou."""
    detections = list(result["detections"])
    labels = tuple(result.get("class_names") or LABELS)
    base = {
        "task": f"object detection over {len(labels)} classes on one image",
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
        if label not in labels:
            raise ValueError(f"unknown reference label {label!r}; expected one of {len(labels)} classes")
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
    """COCO-class object detection and transfer fine-tuning over RT-DETR (ResNet-50-vd)."""

    model: Any
    processor: Any
    device: str
    class_names: tuple[str, ...] = LABELS
    source: str = "snapshot"
    base_state_digest: str | None = None
    adapted: bool = False
    reinitialised: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        class_names: Sequence[str] | None = None,
        seed: int = DEFAULT_SEED,
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
        import torch
        from transformers import AutoImageProcessor, RTDetrForObjectDetection

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        processor = AutoImageProcessor.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        names = tuple(class_names) if class_names is not None else LABELS
        if not 1 <= len(names) <= MAX_CLASSES:
            raise ValueError(f"class_names must hold 1..{MAX_CLASSES} names, got {len(names)}")

        # Calculate base checkpoint digest if local file exists
        base_digest = None
        weights_file = root / "model.safetensors"
        if weights_file.is_file():
            base_digest = _sha256(weights_file)

        reinitialised: tuple[str, ...] = ()
        if class_names is not None:
            # Deterministic head re-initialization
            torch.manual_seed(seed)
            model = RTDetrForObjectDetection.from_pretrained(
                source,
                revision=MODEL_REVISION,
                num_labels=len(names),
                ignore_mismatched_sizes=True,
                trust_remote_code=False,
                use_pretrained_backbone=False,
                **kwargs,
            )
            # Prior probability bias initialization for focal loss (standard in RetinaNet / DETR).
            # Sets initial background bias to -log((1 - p) / p), so random background queries
            # start with low scores (~0.01) instead of flooding predictions.
            prior_bias = -math.log((1.0 - HEAD_PRIOR_PROB) / HEAD_PRIOR_PROB)
            torch.nn.init.constant_(model.model.enc_score_head.bias, prior_bias)
            for embed in model.model.decoder.class_embed:
                torch.nn.init.constant_(embed.bias, prior_bias)

            reinitialised = (
                "model.enc_score_head.bias",
                "model.enc_score_head.weight",
                "model.denoising_class_embed.weight",
                *(f"model.decoder.class_embed.{i}.bias" for i in range(6)),
                *(f"model.decoder.class_embed.{i}.weight" for i in range(6)),
            )
        else:
            model = RTDetrForObjectDetection.from_pretrained(
                source,
                revision=MODEL_REVISION,
                trust_remote_code=False,
                use_pretrained_backbone=False,
                **kwargs,
            )

        model = model.to(resolved_device).eval()
        return cls(
            model=model,
            processor=processor,
            device=resolved_device,
            class_names=names,
            source=source,
            base_state_digest=base_digest,
            adapted=False,
            reinitialised=reinitialised,
        )

    def _run(self, image: Image.Image, threshold: float) -> list[dict[str, Any]]:
        import torch

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            outputs = self.model(**inputs)
        if was_training:
            self.model.train()

        # post_process_object_detection returns list of dicts with boxes, scores, labels
        result = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=[(image.height, image.width)]
        )[0]
        detections = []
        for box, label_idx, score in zip(result["boxes"], result["labels"], result["scores"], strict=True):
            idx = int(label_idx)
            label_name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
            detections.append(
                {
                    "box": [float(v) for v in box.tolist()],
                    "label": label_name,
                    "score": float(score),
                }
            )
        return detections

    def detect(self, image: Image.Image, *, threshold: float = DETECTION_THRESHOLD) -> dict[str, Any]:
        """Detect objects on one image; boxes are xyxy pixel coordinates in the input image."""
        rgb, checked = _check_inputs(image, threshold)
        detections = self._run(rgb, checked)

        if len(detections) > MAX_DETECTIONS:
            raise RuntimeError(
                f"backend returned {len(detections)} detections > num_queries {MAX_DETECTIONS}"
            )
        for det in detections:
            if (
                set(det) != {"box", "label", "score"}
                or len(det["box"]) != 4
                or det["label"] not in self.class_names
            ):
                raise RuntimeError(f"backend returned a malformed detection: {det!r}")

        return {
            "detections": sorted(detections, key=lambda d: -d["score"]),
            "threshold": checked,
            "width": rgb.width,
            "height": rgb.height,
            "class_names": list(self.class_names),
            "adapted": self.adapted,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def detect_many(
        self,
        images: Sequence[Image.Image],
        *,
        threshold: float = EVAL_DETECTION_THRESHOLD,
    ) -> list[list[dict[str, Any]]]:
        """Detections for several images, defaulting to the evaluation threshold."""
        return [self.detect(image, threshold=threshold)["detections"] for image in images]

    def finetune(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        seed: int = DEFAULT_SEED,
        freeze_backbone: bool = True,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning on ``records`` using RT-DETR native loss (Varifocal + L1 + GIoU).

        Mutates this pipeline in place (``adapted`` becomes True) and leaves model in eval mode.
        """
        import torch

        validate_dataset(records, self.class_names, epochs=epochs)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
        if not isinstance(learning_rate, int | float) or isinstance(learning_rate, bool):
            raise ValueError(f"learning_rate must be a number, got {learning_rate!r}")
        if not 0.0 < float(learning_rate) <= 1.0:
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate!r}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)

        if freeze_backbone:
            for p in self.model.model.backbone.parameters():
                p.requires_grad = False

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)

        epoch_losses: list[float] = []
        n_records = len(records)
        class_to_id = {name: i for i, name in enumerate(self.class_names)}

        for epoch in range(epochs):
            self.model.train()
            order = rng.permutation(n_records)
            running_loss = 0.0
            n_batches = 0

            for start in range(0, n_records, batch_size):
                batch_indices = order[start : start + batch_size]
                batch_records = [records[i] for i in batch_indices]
                images = [validate_image(r["image"]) for r in batch_records]

                # Format ground truth targets for RT-DETR (normalized cx, cy, w, h in [0, 1])
                labels = []
                for r in batch_records:
                    w, h = r["image"].size
                    boxes_norm = []
                    for box in r["boxes"]:
                        cx = (box[0] + box[2]) / 2.0 / w
                        cy = (box[1] + box[3]) / 2.0 / h
                        bw = (box[2] - box[0]) / w
                        bh = (box[3] - box[1]) / h
                        boxes_norm.append([cx, cy, bw, bh])
                    class_ids = [class_to_id[name] for name in r["labels"]]
                    labels.append(
                        {
                            "class_labels": torch.tensor(class_ids, dtype=torch.long, device=self.device),
                            "boxes": torch.tensor(boxes_norm, dtype=torch.float32, device=self.device),
                        }
                    )

                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(pixel_values=inputs["pixel_values"], labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                running_loss += float(loss.detach().cpu())
                n_batches += 1

            mean_epoch_loss = running_loss / max(1, n_batches)
            epoch_losses.append(mean_epoch_loss)
            if progress is not None:
                progress({"epoch": epoch + 1, "epochs": epochs, "loss": mean_epoch_loss})

        self.model.eval()
        self.adapted = True

        return {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": float(learning_rate),
            "seed": seed,
            "freeze_backbone": freeze_backbone,
            "trainable_parameters": sum(p.numel() for p in trainable),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "epoch_losses": epoch_losses,
            "final_loss": epoch_losses[-1] if epoch_losses else None,
            "loss": "upstream RTDetrForObjectDetection loss (Varifocal + L1 + GIoU)",
            "device": self.device,
            "class_names": list(self.class_names),
            "reinitialised_tensors": list(self.reinitialised),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        threshold: float = EVAL_DETECTION_THRESHOLD,
        iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
        max_detections: int = MAX_EVAL_DETECTIONS,
    ) -> dict[str, Any]:
        """Score a held-out dataset: average precision plus evaluation metadata."""
        images = [r["image"] for r in records]
        predictions = self.detect_many(images, threshold=threshold)
        raw_counts = [len(dets) for dets in predictions]
        capped_predictions = [dets[:max_detections] for dets in predictions]
        metrics = average_precision(
            capped_predictions, records, self.class_names, iou_thresholds=iou_thresholds
        )
        return {
            **metrics,
            "threshold": _check_threshold(threshold, "threshold"),
            "max_detections": max_detections,
            "detections_before_cap": raw_counts,
            "adapted": self.adapted,
            "class_names": list(self.class_names),
            "implementation": (
                "package-local average_precision: COCO matching and 101-point interpolation, without "
                "pycocotools area ranges or crowd handling"
            ),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def save_artifact(self, path: str | Path, *, notes: str | None = None) -> dict[str, Any]:
        """Write adapted weights and provenance as one artifact; return its descriptor."""
        import torch

        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": ARTIFACT_FORMAT,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_key": MODEL_KEY,
            "class_names": list(self.class_names),
            "adapted": self.adapted,
            "base_state_digest": self.base_state_digest,
            "notes": notes or "",
            "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
        }
        torch.save(payload, artifact_path)
        return {
            "path": str(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "sha256": _sha256(artifact_path),
            "format": ARTIFACT_FORMAT,
            "class_names": list(self.class_names),
            "tensors": len(payload["state_dict"]),
            "base_state_digest": self.base_state_digest,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    @classmethod
    def load_artifact(
        cls,
        path: str | Path,
        *,
        weights_dir: str | Path | None = None,
        device: str | None = None,
    ) -> RTDetrDetectionPipeline:
        """Rebuild an adapted pipeline from an exported artifact."""
        import torch

        artifact_path = Path(path)
        payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
        if payload.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {payload.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if payload.get("model_id") != MODEL_ID or payload.get("model_revision") != MODEL_REVISION:
            raise ValueError(
                f"artifact was built on {payload.get('model_id')}@{payload.get('model_revision')}, "
                f"package pins {MODEL_ID}@{MODEL_REVISION}"
            )
        if payload.get("model_key") != MODEL_KEY:
            raise ValueError(
                f"artifact was built on {payload.get('model_key')!r}, package pins {MODEL_KEY!r}"
            )

        names = tuple(payload["class_names"])
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, class_names=names)
        pipe.model.load_state_dict(payload["state_dict"], strict=True)
        pipe.adapted = True
        pipe.source = f"artifact:{artifact_path.name}"
        return pipe
