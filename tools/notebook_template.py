"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "rtdetr_detection_pipeline",
    "repo_name": "rtdetr-detection-pipeline",
    "stem": "rtdetr_detection",
    "notebook_name": "rtdetr_detection_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "pipeline_class": "RTDetrDetectionPipeline",
    "weights_key": "rtdetr-r50vd",
    "runtime_imports": ["torch", "transformers"],
    "title": "RT-DETR R50-VD (COCO) — DIMER real-time object detection tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/rtdetr-detection-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/rtdetr-detection-pipeline/blob/main/tutorials/rtdetr_detection_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PekingU%2Frtdetr__r50vd-ffcc4d?style=flat",
            "https://huggingface.co/PekingU/rtdetr_r50vd",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-lyuwenyu%2FRT--DETR-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/lyuwenyu/RT-DETR",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2304.08069-b31b1b.svg", "https://arxiv.org/abs/2304.08069"),
    ],
    "capability": "object detection over the 80 COCO classes on one image (score-ordered xyxy boxes with a per-class sigmoid score under a caller-owned threshold) using the pinned `PekingU/rtdetr_r50vd` weights",
    "intro": (
        "At inference the RT-DETR model reads one image resized to 640×640 (aspect ratio not preserved), runs a ResNet-50-vd "
        "backbone, a one-layer hybrid encoder over three feature scales and a 6-layer transformer decoder with 300 learned "
        "object queries, and emits, per query, a box and an **independent sigmoid score per COCO class** (the model is trained "
        "with a focal loss, so scores are not a softmax over classes); the processor keeps the top (query, class) pairs above "
        "the threshold and maps their boxes back to input pixels. **No adaptation occurs:** no training, fine-tuning, in-context "
        "conditioning, or preprocessing fitting happens in this notebook — the upstream checkpoint supplies the weights and "
        "image-processor configuration, and the carried module adds snapshot verification, the input contract, a fixed output "
        "contract and the `box_iou`, `validate_inputs` and `evaluation_report` helpers. The default sample is a street-like "
        "scene drawn in code (a stop sign, a traffic light, an analogue clock and a sports ball) whose drawn boxes serve as "
        "references; the resulting per-object `box_iou` values are demonstration (plumbing) evidence for one image, not a "
        "detection benchmark — and one of the four drawn objects is not detected at all, which the notebook records rather "
        "than hides."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, draw a synthetic scene with reference boxes per COCO class (or upload your own "
        "image) and validate it into an input manifest, run the supported task, read the 80 labels, the sigmoid scores and the "
        "caller-owned threshold correctly, exercise an optional BYOD path, produce an evaluation report that is `sample-sanity` "
        "with per-object `box_iou` only when reference boxes exist and `not-measurable` otherwise, and export machine-readable "
        "detections plus an annotated image and provenance."
    ),
    "exclusions": (
        "classes outside the 80 COCO categories (a closed vocabulary: use an open-vocabulary detector such as OWLv2 or "
        "Grounding DINO for text prompts), instance segmentation, tracking, batched or video inference, COCO mean-average-"
        "precision evaluation (which needs a labelled image set; only per-object `box_iou` against drawn references is "
        "computed here), the upstream latency claims (108 FPS on a T4 with TensorRT; this notebook measures CPU wall time "
        "only), or any training. The model was trained on COCO 2017 photographs; drawn icons, documents, medical or aerial "
        "imagery and non-COCO objects are outside what this notebook measures, and a scene with no objects still yields boxes."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. CPU is adequate: the repository's model card records 4.8 s to load and 0.25–0.34 s per `detect` on the 640×480 synthetic scene in the Windows venv (Intel Core Ultra 9 275HX). The pinned `torch==2.14.0` install and the 172 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what a bounding box in xyxy pixel coordinates is; what intersection-over-union measures; the difference between a sigmoid per-class score and a softmax over classes.",
        "- **Data:** the default sample is a deterministic 640×480 scene drawn in code (sky, ground, a road edge, a red octagonal stop sign with the word STOP, a three-lamp traffic light, a white analogue clock with numerals and hands, and an orange sports ball with seams), so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one image decodable by Pillow (PNG/JPEG/WebP and similar), ideally a photograph of everyday scenes, any colour mode, sides between 16 and 4096 px. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Draw the synthetic scene or optional BYOD\n\n"
                "The default sample is **synthetic** and carries its own reference boxes: a 640×480 scene drawn with Pillow — "
                "sky and ground, a red octagonal **stop sign** with the word STOP on a post, a black three-lamp **traffic light**, "
                "a white analogue **clock** with numerals and hands, and an orange **sports ball** with seams — the same scene the "
                "repository's smoke run used. The drawn boxes, keyed by their COCO label, are the references for the per-object "
                "`box_iou` sanity check later; they are not a labelled dataset, so nothing here is a mean-average-precision "
                "measurement, and drawn icons are not the photographs the model was trained on. The image digest is printed for "
                "the record. BYOD is optional and disabled by default; when enabled, upload one image — no reference boxes exist "
                "for it, so the evaluation report will be `not-measurable`.\n\n"
                "The detection threshold is a **caller-owned request parameter**, not a pipeline constant: a (query, class) pair "
                "survives when its sigmoid class score reaches it. The package default (`DETECTION_THRESHOLD = 0.3`) is the value "
                "the pinned README's transformers example passes, not a calibration; it is exposed here as a form parameter and "
                "passed explicitly on every call. Nothing is validated in this cell — the next section hands the image and the "
                "threshold to the pipeline's own validation stage, which is the only checker. Look for a dictionary naming the "
                "sample kind, the image size and digest, the threshold, and the reference boxes per label."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import math\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "threshold = 0.3  # @param {{type:\"number\"}}\n\n\n"
                "def synthetic_scene(width=640, height=480):\n"
                "    \"\"\"Sky/ground scene with a stop sign, a traffic light, an analogue clock and a sports ball; returns image + label->boxes.\"\"\"\n"
                "    img = Image.new('RGB', (width, height), (135, 190, 235))\n"
                "    d = ImageDraw.Draw(img)\n"
                "    d.rectangle([0, 330, width, height], fill=(96, 128, 72))\n"
                "    d.rectangle([0, 300, width, 330], fill=(110, 110, 110))\n"
                "    refs = {{}}\n"
                "    cx, cy, r = 110, 150, 62\n"
                "    pts = [(cx + r * math.cos(math.pi / 8 + k * math.pi / 4), cy + r * math.sin(math.pi / 8 + k * math.pi / 4)) for k in range(8)]\n"
                "    d.rectangle([cx - 5, cy, cx + 5, 330], fill=(90, 90, 90))\n"
                "    d.polygon(pts, fill=(200, 20, 30), outline=(255, 255, 255))\n"
                "    f = ImageFont.load_default(size=30)\n"
                "    d.text((cx - d.textlength('STOP', font=f) / 2, cy - 17), 'STOP', fill='white', font=f)\n"
                "    refs['stop sign'] = [[cx - r, cy - r, cx + r, cy + r]]\n"
                "    x0, y0 = 270, 60\n"
                "    d.rectangle([x0 + 22, y0 + 150, x0 + 30, 330], fill=(70, 70, 70))\n"
                "    d.rectangle([x0, y0, x0 + 52, y0 + 150], fill=(25, 25, 25), outline=(60, 60, 60))\n"
                "    for k, col in enumerate([(230, 30, 30), (240, 200, 30), (40, 200, 60)]):\n"
                "        d.ellipse([x0 + 8, y0 + 8 + k * 47, x0 + 44, y0 + 44 + k * 47], fill=col)\n"
                "    refs['traffic light'] = [[x0, y0, x0 + 52, y0 + 150]]\n"
                "    cx, cy, r = 480, 140, 70\n"
                "    d.rectangle([cx - 6, cy, cx + 6, 330], fill=(120, 80, 40))\n"
                "    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(250, 250, 245), outline=(20, 20, 20), width=5)\n"
                "    f2 = ImageFont.load_default(size=16)\n"
                "    for h in range(1, 13):\n"
                "        a = math.radians(h * 30 - 90)\n"
                "        d.text((cx + (r - 18) * math.cos(a) - 5, cy + (r - 18) * math.sin(a) - 8), str(h), fill='black', font=f2)\n"
                "    d.line([(cx, cy), (cx + 0.5 * r * math.cos(math.radians(-60)), cy + 0.5 * r * math.sin(math.radians(-60)))], fill='black', width=5)\n"
                "    d.line([(cx, cy), (cx + 0.8 * r * math.cos(math.radians(30)), cy + 0.8 * r * math.sin(math.radians(30)))], fill='black', width=3)\n"
                "    refs['clock'] = [[cx - r, cy - r, cx + r, cy + r]]\n"
                "    cx, cy, r = 330, 400, 45\n"
                "    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(235, 120, 30), outline=(40, 20, 10), width=3)\n"
                "    d.line([(cx - r, cy), (cx + r, cy)], fill=(40, 20, 10), width=3)\n"
                "    d.line([(cx, cy - r), (cx, cy + r)], fill=(40, 20, 10), width=3)\n"
                "    d.arc([cx - r * 1.6, cy - r, cx - r * 0.2, cy + r], 300, 60, fill=(40, 20, 10), width=3)\n"
                "    d.arc([cx + r * 0.2, cy - r, cx + r * 1.6, cy + r], 120, 240, fill=(40, 20, 10), width=3)\n"
                "    refs['sports ball'] = [[cx - r, cy - r, cx + r, cy + r]]\n"
                "    return img, {{label: [[float(v) for v in box] for box in boxes] for label, boxes in refs.items()}}\n\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    image_name = next(iter(uploaded))\n"
                "    image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    image.load()\n"
                "    drawn_boxes = None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic synthetic scene: no randomness, so no seed is needed and the digest is stable per Pillow build.\n"
                "    image, drawn_boxes = synthetic_scene()\n"
                "    image_name = 'synthetic_scene_640x480.png'\n"
                "    sample_kind = 'synthetic'\n\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': image_sha256, 'threshold': threshold, 'reference_boxes': drawn_boxes}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `detect` applies — "
                "image type and sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px and a threshold in `[0, 1]` — and returns an "
                "**input manifest** naming the schema (including the 80 labels and the 300-query ceiling on detections), the "
                "input's observed mode and size, the threshold, and the verdict. The manifest is written to "
                "`outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates a threshold "
                "outside `[0, 1]` and records the pipeline's own error message as a finding. Inside the pipeline the image is "
                "converted to RGB and resized to 640×640 by the processor (the aspect ratio is not preserved — a tall or wide "
                "image is squashed); boxes are mapped back to input pixels, and nothing else is dropped or altered."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_DETECTIONS': MAX_DETECTIONS, 'n_labels': len(LABELS), 'DETECTION_THRESHOLD': DETECTION_THRESHOLD}}}})\n"
                "print({{'LABELS': list(LABELS)}})\n"
                "input_manifest = validate_inputs(image, threshold=threshold, names=[image_name])\n"
                "# Demonstrate rejection on a request that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(image, threshold=1.5)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'out-of-range-threshold-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in input_manifest.items() if k != 'schema'}}, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Detect and read the scores correctly\n\n"
                "`detect` returns a dict with `detections` — a list of `{{box, label, score}}` **ordered by descending score**, "
                "`box` in xyxy pixel coordinates of the input, `label` one of the 80 COCO classes — plus the threshold used, "
                "`width`, `height` and the model identity. At most 300 boxes can ever be returned (the decoder has 300 queries "
                "and the processor keeps at most that many (query, class) pairs, so one query can surface twice under two "
                "labels). Each `score` is the **per-class sigmoid under the model's own focal-loss head, not a calibrated "
                "estimate for your images**: it was never fitted to the frequency with which a box is a real object on your "
                "data, and the scores of different classes for one query do not sum to one. The threshold you passed is the only "
                "decision rule; the pipeline ships 0.3 as a default (the README example's value), not as a calibration, and the "
                "caller owns it per deployment. Inference is deterministic on a fixed device and dtype (no sampling, "
                "`torch.inference_mode`); CUDA kernel selection can move scores in the third or fourth decimal place. As recorded "
                "in the model card, the repository's CPU smoke on this same scene returned exactly three boxes — `stop sign` 0.977, "
                "`clock` 0.965, `traffic light` 0.931 — and **no `sports ball`** even at threshold 0.1; the same three boxes "
                "appeared at 0.1, 0.5 and 0.9. That is one observation on drawn icons, not a calibration point."
            ),
            "code": (
                "import time\n\n"
                "t0 = time.time()\n"
                "result = pipe.detect(image, threshold=threshold)\n"
                "elapsed = time.time() - t0\n"
                "print({{'n_detections': len(result['detections']), 'threshold': result['threshold'], 'device': pipe.device, 'seconds': round(elapsed, 2)}})\n"
                "for rank, det in enumerate(result['detections'], start=1):\n"
                "    print(f\"{{rank:>3}}. score {{det['score']:.4f}}  label {{det['label']!r:16}}  box {{[round(v, 1) for v in det['box']]}}\")"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No detection metric "
                "is reported by default: COCO mean average precision needs a labelled image set, and this repository ships none. "
                "The repository's only metric helper is `box_iou(a, b)` (intersection-over-union of two xyxy boxes), the building "
                "block a caller would use to compute mAP on their own labelled images; when reference boxes are supplied, keyed by "
                "COCO label, the report carries one `box_iou` entry per reference — matched only against detections **of the same "
                "label**, with the matched detection's score and how many same-label detections existed — with the verdict "
                "`sample-sanity`. A reference with no same-label detection scores 0.0 and `n_detected_same_label` 0: that is what "
                "a miss looks like, and on this scene the sports ball is one. On the synthetic path those references are icons "
                "**you drew yourself**, so a high IoU proves only that the input contract, forward pass and coordinate mapping "
                "round-trip. On BYOD no reference exists, the verdict is `not-measurable`, and the report states what would make "
                "the task measurable. The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, drawn_boxes, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in report.items() if k != 'metrics'}}, indent=2))\n"
                "for metric in report['metrics']:\n"
                "    matched = 'no same-label detection' if metric['matched_score'] is None else f\"matched score {{metric['matched_score']:.3f}}\"\n"
                "    print(f\"{{metric['reference']:18}} iou {{metric['value']:.3f}}  ({{matched}}, same-label detections {{metric['n_detected_same_label']}})\")\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No reference boxes exist for this input, so box_iou is not computed; inspect the annotated PNG instead.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the full result (score-ordered detections with boxes and labels, the threshold), "
                "the evaluation report, the input manifest, the sample identity, digest and reference boxes, the notebook's source "
                "(repository, revision, embedded module digest, generator), the model identifier, the immutable model revision, "
                "the model licence, and the runtime identity (Python, `torch`, `transformers`, device). The detections are also "
                "written as CSV with explicit `image`, `rank`, `label`, `score`, `x0`, `y0`, `x1`, `y1` columns so score ordering "
                "survives downstream use, and an annotated PNG draws every returned box in green with its label and score, and "
                "every reference box in red, for visual inspection (a supplement to, not a replacement for, the machine-readable "
                "files). No credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "annotated = image.convert('RGB').copy()\n"
                "draw = ImageDraw.Draw(annotated)\n"
                "for label, boxes in (drawn_boxes or {{}}).items():\n"
                "    for box in boxes:\n"
                "        draw.rectangle(box, outline=(220, 30, 30), width=2)\n"
                "for det in result['detections']:\n"
                "    draw.rectangle(det['box'], outline=(0, 160, 0), width=3)\n"
                "    draw.text((det['box'][0] + 4, det['box'][1] + 4), f\"{{det['label']}} {{det['score']:.3f}}\", fill=(0, 160, 0))\n"
                "annotated.save('outputs/{stem}_annotated.png')\n"
                "payload = {{\n"
                "    'prediction': result,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256, 'reference_boxes': drawn_boxes}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_detections.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'rank', 'label', 'score', 'x0', 'y0', 'x1', 'y1'])\n"
                "    for rank, det in enumerate(result['detections'], start=1):\n"
                "        writer.writerow([image_name, rank, det['label'], f\"{{det['score']:.6f}}\", *[f\"{{v:.2f}}\" for v in det['box']]])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The boxes locate regions the model classifies as one of the 80 COCO classes; the vocabulary is closed, the sigmoid "
        "score is not calibrated for your images, and the threshold is a request parameter you own (the default is the README "
        "example's value, not a tuned operating point). On the synthetic scene the per-object `box_iou` values in the evaluation "
        "report compare detections to icons you drew yourself and the verdict is `sample-sanity`, which proves only that the "
        "input contract, forward pass and coordinate mapping work — and the drawn sports ball, which the model does not find at "
        "any threshold, shows that a drawn icon is not a photograph; they say nothing about real scenes, small or occluded "
        "objects, crowded images, unusual viewpoints or non-COCO objects, and a BYOD result is a single-image observation with "
        "the verdict `not-measurable`. **The model emits boxes for any image**: the repository's smoke run fed it a blank "
        "4096×4096 image and got one `train` at 0.33, and a uniform-noise image one `cat` at 0.32, so an empty scene at the "
        "default threshold produces a confident nonsense box rather than an empty result. Everything is resized to 640×640, so "
        "tiny objects and extreme aspect ratios suffer. The pipeline provides no open-vocabulary prompting, no segmentation, no "
        "tracking, no mAP evaluation and no training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model, validate the demonstrated request, execute the public pipeline path, and "
        "emit the shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on "
        "an unseen domain.\n\n"
        "**Next experiments:** lower `threshold` to 0.05 and look for the sports ball (the smoke run found nothing at 0.1); "
        "redraw the ball with a soccer-ball pattern or a photograph-like shading and see whether it appears; raise `threshold` "
        "to 0.9 and check the three boxes survive (they did); enable `USE_BYOD` with a street photograph, hand-label a few "
        "objects by COCO class and pass them to `evaluation_report` to see the verdict switch to `sample-sanity` — the first step "
        "towards a real precision/recall number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/rtdetr-detection-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/rtdetr-detection-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/rtdetr-detection-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/lyuwenyu/RT-DETR\n"
        "- DETRs Beat YOLOs on Real-time Object Detection (Zhao et al., 2023): https://arxiv.org/abs/2304.08069\n"
        "- Microsoft COCO: Common Objects in Context (Lin et al., 2014): https://arxiv.org/abs/1405.0312"
    ),
}
