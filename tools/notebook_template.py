"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package, and the
model pin/stage/verify cells are produced by the generator from repository sources so they cannot
drift from the package.

This is an `E2E` template, so it must state `run_all` itself, and its default path really adapts:
NOTEBOOK_SPEC 2.0 RUN7/FT2 make a bounded fine-tune mandatory rather than optional for this profile.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "rtdetr_detection_pipeline",
    "repo_name": "rtdetr-detection-pipeline",
    "stem": "rtdetr_detection",
    "notebook_name": "rtdetr_detection_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (generator /2.2): managed CPython, a
    # size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "pipeline_class": "RTDetrDetectionPipeline",
    "weights_key": "rtdetr-r50vd",
    "modules": [
        "samples.py",
        "pipeline.py",
    ],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "transformers", "numpy", "PIL", "scipy"],
    "title": "RT-DETR R50-VD (COCO) — DIMER real-time object detection and bounded detection fine-tuning (standalone)",
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
        ("License", "https://img.shields.io/badge/License-Apache--2.0-green.svg", "https://github.com/kurtvalcorza/rtdetr-detection-pipeline/blob/main/LICENSE"),
    ],
    "capability": "real-time object detection over the 80 COCO classes, and a bounded detection fine-tune that re-heads RT-DETR onto your own class vocabulary, evaluates it against a held-out split with COCO-style average precision, and exports a reloadable artifact",
    "intro": (
        "RT-DETR with a ResNet-50-vd backbone (`PekingU/rtdetr_r50vd`) is the first real-time end-to-end object detector: "
        "a ResNet-50-vd backbone and an efficient hybrid encoder (CCFM) feed a 6-layer transformer decoder with 300 learned "
        "object queries that directly emit bounding boxes and class scores without non-maximum suppression (NMS). At inference, "
        "the model reads an image resized to 640×640, predicts xyxy boxes with an independent sigmoid score under a caller-owned "
        "threshold, and maps coordinates back to input pixels.\n\n"
        "**The default path really adapts the model:** it re-heads RT-DETR onto a three-class traffic sign vocabulary that does "
        "not exist in COCO (`stop-sign`, `yield-sign`, `speed-limit-sign`), initializes class classification biases to suppress "
        "background query flooding, measures a pre-adaptation baseline, runs a bounded fine-tune with the backbone frozen, scores "
        "the result against a held-out split with COCO-style average precision (AP@[.50:.95] and AP50), runs the adapted model on "
        "three unseen images, exports the weights as a standalone `.pt` artifact, and reloads that artifact from disk to check that "
        "its raw outputs match the in-memory model within a stated tolerance. Every number you see is measured locally in this notebook "
        "runtime.\n\n"
        "**Read the adaptation result honestly.** The baseline is a freshly re-headed model, so its average precision is zero by "
        "construction, and the drawn sign task is easy: each class has a fixed colour and shape on a plain background. A colour-and-shape "
        "rule with no learning at all can score near-perfect AP on it. The lesson here is the workflow (re-head, baseline, train, score, "
        "export, reload), not the size of the gain. Section 9 prints a zero-training reference next to the baseline so you can see this."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable upstream model "
        "revision; run COCO detection on a drawn scene and score per-object `box_iou`, noting where the pretrained model succeeds "
        "and where it misses; build and validate a labelled detection dataset over a new three-class sign vocabulary; partition the "
        "dataset and measure a pre-adaptation baseline; run a bounded fine-tune using RT-DETR native loss (Varifocal + L1 + GIoU); "
        "score the adapted model on the held-out split with COCO-style AP; run inference on unseen test data; and export, reload and "
        "verify the adapted artifact."
    ),
    "exclusions": (
        "real-world traffic sign deployment claims (the adaptation dataset is drawn in code, so the model learns these synthetic "
        "renderings and nothing about road photographs); published COCO test-dev benchmarks (the average-precision helper here is "
        "a compact implementation without pycocotools crowd or area-range filtering); full unfreezing without large training sets "
        "(the default freezes the ResNet backbone to prevent destroying pretrained features); video tracking; and instance segmentation."
    ),
    "guided": {"opening": [(
        "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Jupyter and has met bounding boxes, and wants to see two things in one run: what a pretrained real-time detector does on a scene it was not drawn for, and what a bounded fine-tune changes when the detector is re-headed onto three classes COCO does not have. The audience is students and practitioners deciding whether a detector can be adapted to their own labelled images; no prior experience with RT-DETR, transformers or average precision is assumed — each term is explained where it first matters and again in the **Glossary**. CPU is enough (the fine-tune took 80–140 s in local CPU checks); a GPU makes it seconds (about 11 s on a Tesla T4 in a recorded hosted run).\n\n**Input → Model → Output.**\n\n| | |\n|---|---|\n| Input | images resized to 640 × 640, with a caller-owned score threshold; the default is a drawn 640 × 480 street scene with four reference boxes, then a 40-image drawn traffic-sign dataset over `stop-sign`, `yield-sign`, `speed-limit-sign` |\n| Model | RT-DETR R50-VD: a ResNet-50-vd backbone, a hybrid encoder and a 6-layer transformer decoder with 300 object queries emitting boxes and per-class sigmoid scores without NMS; for adaptation, the classification heads are re-initialised for the three sign classes and the backbone is frozen (19.3 M of 42.7 M parameters train) |\n| Output | COCO detections and `box_iou` against the drawn references, two degenerate-input probes, a pre-adaptation baseline and a post-adaptation AP / AP50 / AP75 on a held-out split, detections on three unseen images, a reloadable `.pt` adapter whose detections are checked against the in-memory model, and six machine-readable outputs |\n\n**How to use this notebook.** Choose any runtime, then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed (the recorded hosted run of the previous version needed one; this version removes it). Sections 1–3 are **infrastructure** — the isolated environment, the carried modules and the verified snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it comes a **What to notice** note and a collapsible **Check your reasoning** with a worked answer. The worked answers quote a local CPU run of this revision (9 October 2026, the pinned checkpoint, `torch` 2.13 CPU); a GPU runtime gives the same picture with numbers that differ from the third decimal on — read your own output first. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Budget about ten minutes on a GPU runtime, longer on CPU.\n\n**Roadmap:** 1–3 infrastructure → 4 the pretrained detector on a drawn scene, with one honest miss *(core concept: sigmoid scores under a caller-owned threshold, no NMS)* → 5 blank and noise probes *(evaluation practice: a detector should find nothing where there is nothing)* → 6 a labelled dataset over a new vocabulary and its validation *(core concept: the data contract)* → 7 split, re-head and the pre-adaptation baseline *(core concept: prior-probability bias initialisation)* → 8 the bounded fine-tune with the backbone frozen, always from the pretrained model *(core concept: what 45 % of the parameters can learn in three epochs)* → 9 held-out AP, AP50 and AP75 next to a zero-training reference *(evaluation practice: a baseline that is zero by construction says nothing about how hard the task is)* → 10 unseen images → 11 export, reload and parity *(engineering)* → 12 outputs and provenance → 13 optional BYOD → conclude."
    )]},
    "prerequisites": [
        "- **Learner:** basic Python and Colab or Jupyter familiarity; no prior experience with RT-DETR or detection fine-tuning. Object queries, sigmoid scores, IoU, AP, re-heading and reload parity are explained where they are first used and again in the Glossary.",
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or a Linux Jupyter server). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the Python version of the kernel itself does not matter and nothing is installed into it; a Windows or macOS kernel is not supported. CPU is the documented default and CUDA GPU is used automatically when available. Estimated fine-tune time (Section 8 prints the measured `train_seconds`): 80–140 s on a laptop CPU (local checks, October 2026) and about 11 s on a Tesla T4 (recorded Kaggle run, 15 September 2026). Pinned `torch==2.14.0` and the 172 MB checkpoint are the primary downloads.",
        "- **Knowledge:** basic Python and PIL; bounding box representation in xyxy pixel coordinates; intersection-over-union (IoU); and the interpretation of average precision (AP50 and AP@[.50:.95]).",
        "- **Data:** everything is generated deterministically in code by `samples.py`, requiring zero external dataset download: one 640×480 COCO demonstration scene and a 40-image labelled sign adaptation dataset. Optional BYOD is gated off by default. Expected BYOD input: one PNG/JPEG/WebP image, or a labelled dataset as a folder or `.zip` holding `annotations.csv` (header `file,label,x0,y0,x1,y1`, one row per box, pixel coordinates) and the images it names — the full rules are in Section 13. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so; uploaded inputs stay in this runtime and are not sent to any inference API.",
    ],
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated, hash-locked environment with the pinned dependencies "
        "(nothing is installed into the notebook kernel, so no restart is needed), stages and digest-verifies the pinned checkpoint, "
        "runs COCO detection on a drawn scene, validates the 40-image sign adaptation dataset, splits it into train and validation parts, "
        "measures the pre-adaptation baseline, **runs the bounded fine-tune**, re-evaluates on the held-out split next to a zero-training "
        "reference, detects on three unseen test images, exports the adapted artifact, reloads it from disk to verify its outputs match "
        "within a stated tolerance, and writes machine-readable outputs with provenance. Nothing is skipped behind a default-off flag, and no clone or DIMER worker is required (NOTEBOOK_SPEC 2.0 §5, RUN7, FT2)."
    ),
    "byod": (
        "Two optional BYOD branches are included and both are disabled by default (`USE_BYOD_IMAGE = False`, `USE_BYOD_DATASET = False`). "
        "`USE_BYOD_IMAGE` runs your own image (from `BYOD_IMAGE_PATH`, or the Colab upload dialog when the path is empty) through the detection and validation contract. `USE_BYOD_DATASET` takes your own labelled "
        "detection dataset (`BYOD_DATASET_PATH`: a folder or `.zip` with `annotations.csv`, or the Colab upload dialog when the path is "
        "empty) through the full adaptation workflow (validate, split, coverage check, baseline, fine-tune, evaluate, export, reload) "
        "under NOTEBOOK_SPEC 2.2 DAT14."
    ),
    "cells": [
        # ---------------------------------------------------------------- 4. COCO scene
        {
            "md": (
                "## 4. What the pretrained detector does, and where it fails\n\n"
                "Before adapting anything, inspect the pretrained model you start from. The carried `samples` module draws a deterministic "
                "street scene containing four objects with reference boxes: a **stop sign**, a **traffic light**, an analogue **clock**, and "
                "an orange **sports ball**.\n\n"
                "The detection threshold is a **caller-owned request parameter**, not a pipeline constant: each score is an independent "
                "**per-class sigmoid under the model's own focal-loss head, not a calibrated** probability for this domain. The default threshold "
                "`0.3` is passed explicitly.\n\n"
                "**Expect one honest failure.** The drawn sports ball is not detected by this checkpoint; the evaluation report records it "
                "with `box_iou = 0.0` rather than hiding it. COCO mean average precision needs a labelled image set; on a single unlabelled scene, the verdict is `sample-sanity`.\n\n"
                "**Predict:** four drawn objects — a stop sign, a traffic light, a clock and a sports ball — at threshold 0.3. How many will the COCO detector find, which is most at risk, and will the boxes it does find be tight (IoU above 0.9) or loose?"
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "OUTPUTS = Path('outputs')\n\n"
                'threshold = 0.3  # @param {{type:"number"}}\n\n'
                "scene, references = tutorial_scene()\n"
                "buffer = io.BytesIO()\n"
                "scene.save(buffer, format='PNG')\n"
                "print({{'sample_kind': 'synthetic', 'size': list(scene.size), 'sha256': hashlib.sha256(buffer.getvalue()).hexdigest()[:16],\n"
                "       'references': {{label: len(boxes) for label, boxes in references.items()}}}})\n\n"
                "input_manifest = validate_inputs(scene, threshold=threshold, names=['tutorial-scene'])\n"
                "print({{'verdict': input_manifest['verdict'], 'findings': input_manifest['findings'], 'inputs': input_manifest['inputs']}})\n\n"
                "coco_result = pipe.detect(scene, threshold=threshold)\n"
                "for det in coco_result['detections']:\n"
                "    print(f\"{{det['label']:>14s}} {{det['score']:.3f}}  [{{', '.join(f'{{v:.0f}}' for v in det['box'])}}]\")\n\n"
                "coco_report = evaluation_report(coco_result, references, sample_kind='synthetic')\n"
                "print({{'verdict': coco_report['verdict'], 'n_detections': coco_report['n_detections']}})\n"
                "for metric in coco_report['metrics']:\n"
                "    print(f\"  {{metric['reference']:>18s}}  box_iou {{metric['value']:.3f}}  same-label detections {{metric['n_detected_same_label']}}\")\n"
                "hits = sum(1 for m in coco_report['metrics'] if m['value'] >= 0.5)\n"
                "print(f'{{hits}}/{{len(coco_report[\"metrics\"])}} drawn objects matched at IoU >= 0.5')\n"
                "scene"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Three of four. In the worked run the stop sign, the traffic light and the clock were found with tight boxes (IoU 0.877, 0.965 and 0.983 against the drawn references) and the orange sports ball was missed — `box_iou = 0.0` — which the report records rather than hides. A drawn disc on a drawn street is far from COCO photographs, and a miss at 0.3 is a finding about the threshold and the domain, not a bug: lowering the threshold is the first experiment to try.</details>'
            ),
        },
        # ---------------------------------------------------------------- 5. Degenerate inputs
        {
            "md": (
                "## 5. Degenerate input probes: blank canvas and noise\n\n"
                "A detector should be evaluated on structure-free inputs. A model that invents confident detections on a blank canvas or uniform "
                "noise will invent them on real unlabelled scenes. We probe the model with both a blank image and a random RGB noise image at "
                "both the standard detection threshold (`0.3`) and the evaluation threshold (`0.05`).\n\n"
                "**Predict:** on a blank canvas and on uniform noise, how many detections at 0.3, and how many at 0.05? If the lenient count is not zero, is that a defect? And if the 0.3 count is not zero?"
            ),
            "code": (
                "degenerate = {{}}\n"
                "for name, image in (('blank', blank_scene()), ('noise', noise_scene(0))):\n"
                "    standard = pipe.detect(image, threshold=threshold)['detections']\n"
                "    lenient = pipe.detect(image, threshold=EVAL_DETECTION_THRESHOLD)['detections']\n"
                "    degenerate[name] = {{\n"
                "        'at_standard_threshold': len(standard),\n"
                "        'at_evaluation_threshold': len(lenient),\n"
                "        'top': [(d['label'], round(d['score'], 3)) for d in lenient[:3]],\n"
                "    }}\n"
                "print(json.dumps(degenerate, indent=2))"
            ),
        },
        {
            "md": (
                "**What to notice.** Compare the two thresholds. Hundreds of low-score queries at 0.05 are normal for a detector with 300 queries and no NMS. Any detection at 0.3 on a blank or noise image is a false positive at the threshold you use for real scenes: note its label and score, because it tells you how close to the threshold the model's own noise sits."
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Not zero. In the worked run the blank canvas gave one detection at 0.3 (`train`, 0.333) and 87 at 0.05; the noise image gave one at 0.3 (`cat`, 0.391) and 254 at 0.05. At 0.05 that is expected: each score is an independent uncalibrated sigmoid, and 300 queries with no NMS always leave some above a very lenient cut-off. At 0.3 it is a real finding: the pretrained model invents one object on an image with nothing in it, with a score just above the threshold. So 0.3 is not a safe threshold on its own — the deployment owns the threshold and should choose it on labelled images from its own cameras, and a single low-score detection should not be trusted without that check.</details></details>'
            ),
        },
        # ---------------------------------------------------------------- 6. Dataset & Validation
        {
            "md": (
                "## 6. Labelled adaptation dataset and validation\n\n"
                "The pretrained model knows 80 COCO classes. Suppose your task requires traffic signs not present in COCO. `sign_dataset` "
                "synthesizes a deterministic 40-image dataset over `SIGN_CLASSES`: `stop-sign`, `yield-sign`, and `speed-limit-sign`.\n\n"
                "**Keep the two vocabularies apart.** COCO contains `stop sign` (space-separated). Our adaptation vocabulary uses `stop-sign` (hyphenated), "
                "`yield-sign`, and `speed-limit-sign`. They represent distinct class identities.\n\n"
                "`validate_dataset` checks the record schema and refuses, naming the record and the rule: an image with no boxes, a box outside the image, a "
                "box with zero width or height (`x0 < x1` and `y0 < y1` are required), a label outside the vocabulary, the same image twice (identical "
                "pixels), more than 100 boxes in one image or more than 500 records. It returns the dataset manifest. Section 7 then checks that every "
                "split covers every class."
            ),
            "code": (
                'N_IMAGES = 40  # @param {{type:"integer"}}\n'
                'DATASET_SEED = 0  # @param {{type:"integer"}}\n\n'
                "records = sign_dataset(N_IMAGES, seed=DATASET_SEED)\n"
                "dataset_manifest = validate_dataset(records, SIGN_CLASSES)\n"
                "print(json.dumps(dataset_manifest, indent=2))\n"
                "print('Adaptation vocabulary:', list(SIGN_CLASSES))\n\n"
                "preview = Image.new('RGB', (480, 320))\n"
                "for index, record in enumerate(records[:6]):\n"
                "    preview.paste(record['image'].resize((160, 160)), (160 * (index % 3), 160 * (index // 3)))\n"
                "print('Previewing first six synthetic sign images:')\n"
                "preview"
            ),
        },
        # ---------------------------------------------------------------- 7. Split & Baseline
        {
            "md": (
                "## 7. Split, re-head, and measure the pre-adaptation baseline\n\n"
                "We partition the dataset into training (75%, 30 images) and held-out validation (25%, 10 images) splits. The held-out split is "
                "never shown to the fine-tuning optimizer.\n\n"
                "`from_pretrained(class_names=SIGN_CLASSES)` instantiates RT-DETR with classification heads tailored to the 3 target classes. "
                "Crucially, the classification biases are initialized to $-4.595$ ($p=0.01$ prior probability), suppressing random background query "
                "firings while retaining the ResNet-50-vd backbone and bbox regression weights.\n\n"
                "Evaluating the held-out split before adaptation establishes the **pre-adaptation baseline**.\n\n"
                "**The baseline is zero by construction.** With the prior biases, the untrained class heads emit no detections on the held-out "
                "set, so AP is 0.000. That confirms the re-heading worked; it does **not** tell you how hard the task is or how good a later AP "
                "is. Section 9 adds a reference that does.\n\n"
                "**Predict:** the re-headed model has never seen a sign class. Will the pre-adaptation AP be exactly 0, near 0, or somewhere above — and what would a non-zero baseline tell you about the bias initialisation?"
            ),
            "code": (
                'HOLDOUT = 0.25  # @param {{type:"number"}}\n'
                'SEED = 0  # @param {{type:"integer"}}\n\n'
                "train_records, held_out = split_dataset(records, train_fraction=1.0 - HOLDOUT, seed=SEED)\n"
                "print('Classes per split:', check_split_coverage({{'train': train_records, 'held_out': held_out}}, SIGN_CLASSES))\n"
                "print({{'train': len(train_records), 'held_out': len(held_out),\n"
                "       'train_boxes': sum(len(r['boxes']) for r in train_records),\n"
                "       'held_out_boxes': sum(len(r['boxes']) for r in held_out)}})\n\n"
                "adapter = RTDetrDetectionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=SIGN_CLASSES, seed=SEED)\n"
                "print({{'class_names': list(adapter.class_names), 'device': adapter.device, 'adapted': adapter.adapted}})\n"
                "print('Re-initialized parameter heads:', len(adapter.reinitialised))\n\n"
                "baseline = adapter.evaluate(held_out)\n"
                "print(json.dumps({{'ap': round(baseline['ap'], 4), 'ap50': round(baseline['ap50'], 4),\n"
                "                  'per_class_ap50': {{k: round(v, 4) for k, v in baseline['per_class_ap50'].items()}},\n"
                "                  'n_references': baseline['n_references'], 'max_detections': baseline['max_detections']}}, indent=2))"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Exactly zero: the recorded baseline AP was 0.000 (AP50 and the per-class AP50 as well). With the new classification biases at −4.595 (p = 0.01), the untrained heads score every query below the evaluation threshold, so there are no detections and nothing to be right or wrong. A clearly non-zero baseline would mean the heads were not re-initialised, or the bias prior was not applied.</details>'
            ),
        },
        # ---------------------------------------------------------------- 8. Bounded Fine-Tuning
        {
            "md": (
                "## 8. Bounded detection fine-tuning\n\n"
                "This cell executes the real adaptation step in the notebook runtime. The optimizer trains the hybrid encoder and decoder heads "
                "using RT-DETR's native composite loss: Varifocal Loss for classification, L1 loss, and GIoU loss for bounding box regression, "
                "accumulated across all 6 decoder layers.\n\n"
                "- **The backbone is frozen by default.** With `FREEZE_BACKBONE = True` the ResNet-50-vd backbone does not train; the cell prints how "
                "many parameters do (19.3 M of 42.7 M on this checkpoint) and checks that the freeze state it reports is the one the model really has.\n"
                "- **Schedule:** `EPOCHS` epochs (default 3) with AdamW at learning rate `1e-4` and batch size 4.\n\n"
                "**Every run starts from the pretrained model.** The cell first rebuilds `adapter` from the verified snapshot with the same seeded "
                "re-headed classifier that Section 7 scored, then trains it. So after changing a form value here, rerun **Section 8 and then "
                "Section 9**: the comparison with the Section 7 baseline stays valid, and nothing carries over from an earlier run.\n\n"
                "**Predict:** three epochs over 30 drawn images. Will the loss fall at every epoch, and by how much from the first to the last epoch — a few percent, or more than half?"
            ),
            "code": (
                'EPOCHS = 3  # @param {{type:"integer"}}\n'
                'LEARNING_RATE = 1e-4  # @param {{type:"number"}}\n'
                'BATCH_SIZE = 4  # @param {{type:"integer"}}\n'
                'FREEZE_BACKBONE = True  # @param {{type:"boolean"}}\n\n'
                "# Start from the pretrained model every time: the same seeded re-head that Section 7's baseline scored.\n"
                "adapter = RTDetrDetectionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=SIGN_CLASSES, seed=SEED)\n"
                "run = adapter.finetune(\n"
                "    train_records,\n"
                "    epochs=EPOCHS,\n"
                "    batch_size=BATCH_SIZE,\n"
                "    learning_rate=LEARNING_RATE,\n"
                "    seed=SEED,\n"
                "    freeze_backbone=FREEZE_BACKBONE,\n"
                "    progress=lambda row: print(\n"
                "        f\"epoch {{row['epoch']}}/{{EPOCHS}}  loss {{row['loss']:.4f}}\"\n"
                "    ),\n"
                ")\n"
                "backbone_frozen = not any(p.requires_grad for p in adapter.model.model.backbone.parameters())\n"
                "assert backbone_frozen == FREEZE_BACKBONE, 'the reported freeze state must be the model state'\n"
                "assert not run['started_from_adapted'], 'the fine-tune must start from the pretrained model'\n"
                "print(json.dumps({{'freeze_backbone': run['freeze_backbone'], 'backbone_frozen_in_model': backbone_frozen,\n"
                "                  'trainable_parameters': run['trainable_parameters'],\n"
                "                  'total_parameters': run['total_parameters'],\n"
                "                  'epochs': run['epochs'], 'batch_size': run['batch_size'],\n"
                "                  'learning_rate': run['learning_rate'], 'train_seconds': run['train_seconds'], 'device': run['device'],\n"
                "                  'epoch_losses': [round(x, 4) for x in run['epoch_losses']]}}, indent=2))\n"
                "print(f\"Loss progression: {{run['epoch_losses'][0]:.4f}} -> {{run['epoch_losses'][-1]:.4f}}\")"
            ),
        },
        {
            "md": (
                "**What to notice.** `backbone_frozen_in_model` is read from the model itself, not from the form, and `trainable_parameters` changes when you change `FREEZE_BACKBONE`. `train_seconds` is this runtime's measured time; compare it with the estimate in the prerequisites."
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>It falls at every epoch, by about half: in the worked run the epoch losses were 39.62, 25.58 and 19.30 (19,259,121 of 42,733,137 parameters trained, 137 s on a laptop CPU). The dataset is small and regular and the heads start from a bias prior, so the loss drops fast; the loss is the optimiser's signal, and AP in Section 9 is the evaluation. Rerunning this cell unchanged gives the same first-epoch loss, because it rebuilds the model before training.</details>"
            ),
        },
        # ---------------------------------------------------------------- 9. Evaluate Held-Out
        {
            "md": (
                "## 9. Evaluate on the held-out split\n\n"
                "We re-run `evaluate` on the exact same held-out validation set using the same thresholds. "
                "`ap50` is average precision at IoU 0.50; `ap` is COCO-standard AP@[.50:.95] across 10 IoU thresholds.\n\n"
                "**A zero-training reference.** The Section 7 baseline is zero by construction, so it cannot tell you whether the adapted number is "
                "good. The cell therefore also scores the **unadapted COCO detector** (`pipe`, never trained here) on the same held-out images for "
                "the one class COCO already has: its `stop sign` detections are renamed `stop-sign` and scored against the drawn stop signs. That "
                "is what you get for that class with no adaptation at all. Keep a second fact in mind: the drawn signs differ by a fixed colour and "
                "shape on a plain background, and in the 2 October 2026 review a short colour-and-shape rule written for this generator, with no "
                "learning, scored AP 1.000 on these same ten held-out images. The task is easy; the adapted number shows that the workflow runs, "
                "not that the model learned something hard.\n\n"
                "**Predict:** from a baseline of zero, where will AP50 land after three epochs on drawn signs — below 0.5, around 0.7, or above 0.9? Will AP (averaged up to IoU 0.95) be much lower than AP50? And will the unadapted COCO detector find the drawn stop signs?"
            ),
            "code": (
                "adapted = adapter.evaluate(held_out)\n"
                "print(json.dumps({{'ap': round(adapted['ap'], 4), 'ap50': round(adapted['ap50'], 4), 'ap75': round(adapted['ap75'], 4),\n"
                "                  'per_class_ap50': {{k: round(v, 4) for k, v in adapted['per_class_ap50'].items()}},\n"
                "                  'n_images': adapted['n_images'], 'n_references': adapted['n_references']}}, indent=2))\n"
                "print()\n"
                "print(f\"Adapted column: {{run['epochs']}} epochs, freeze_backbone={{run['freeze_backbone']}}, lr={{run['learning_rate']}}; baseline column: 0 epochs (Section 7).\")\n"
                "print(f\"{{'metric':<10s}} {{'baseline':>10s}} {{'adapted':>10s}} {{'change':>10s}}\")\n"
                "for key in ('ap', 'ap50', 'ap75'):\n"
                "    before_val, after_val = baseline[key], adapted[key]\n"
                "    print(f\"{{key:<10s}} {{before_val:>10.4f}} {{after_val:>10.4f}} {{after_val - before_val:>+10.4f}}\")\n\n"
                "# Zero-training reference: the unadapted COCO detector's 'stop sign' class, scored on the same held-out stop signs.\n"
                "coco_stop_predictions = [\n"
                "    [{{**d, 'label': 'stop-sign'}} for d in pipe.detect(r['image'], threshold=EVAL_DETECTION_THRESHOLD)['detections'] if d['label'] == 'stop sign']\n"
                "    for r in held_out\n"
                "]\n"
                "stop_references = [\n"
                "    {{'boxes': [b for b, lab in zip(r['boxes'], r['labels'], strict=True) if lab == 'stop-sign'], 'labels': [lab for lab in r['labels'] if lab == 'stop-sign']}}\n"
                "    for r in held_out\n"
                "]\n"
                "coco_reference = average_precision(coco_stop_predictions, stop_references, ['stop-sign'])\n"
                "adapted_stop = average_precision(adapter.detect_many([r['image'] for r in held_out]), stop_references, ['stop-sign'])\n"
                "reference_table = {{\n"
                "    'class': 'stop-sign', 'held_out_stop_signs': coco_reference['n_references'],\n"
                "    'unadapted_coco_detector': {{'ap': round(coco_reference['ap'], 4), 'ap50': round(coco_reference['ap50'], 4)}},\n"
                "    're_headed_baseline': {{'ap50': round(baseline['per_class_ap50'].get('stop-sign', 0.0), 4)}},\n"
                "    'adapted': {{'ap': round(adapted_stop['ap'], 4), 'ap50': round(adapted_stop['ap50'], 4)}},\n"
                "}}\n"
                "print()\n"
                "print('Zero-training reference on the held-out stop signs:')\n"
                "print(json.dumps(reference_table, indent=2))"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Around 0.85, not higher. In the worked run three epochs took the held-out split from AP 0.000 to AP 0.8494 and AP50 0.8535 (the earlier hosted T4 record reached AP 0.9161; GPU and CPU runs differ, so treat the second decimal as run-to-run variation). AP is only a little below AP50, so most matched boxes are tight. The zero-training reference is the more useful comparison. The unadapted COCO detector, never trained here, scored AP 1.000 on the five held-out stop signs — the same as the adapted model (stop-sign AP 1.000). So for the one class COCO already knew, adaptation added nothing measurable; what it added is the two classes COCO lacks (yield-sign AP50 0.914, speed-limit-sign 0.646 in the worked run). And remember the colour-and-shape rule that scored AP 1.000 on all three classes: the adapted detector does not beat a trivial rule on this task, which is why this notebook claims a working workflow, not a strong model.</details>'
            ),
        },
        # ---------------------------------------------------------------- 10. New-data inference
        {
            "md": (
                "## 10. Inference on unseen test data\n\n"
                "We synthesize 3 new images from an unseen seed (`seed=99`), run the adapted pipeline on them at the threshold from Section 4, and "
                "compute the intersection-over-union of the best same-label detection against each ground-truth box.\n\n"
                "**Predict:** will every sign be found exactly once?"
            ),
            "code": (
                'NEW_DATA_SEED = 99  # @param {{type:"integer"}}\n\n'
                "new_records = sign_dataset(3, seed=NEW_DATA_SEED)\n"
                "new_data_rows = []\n"
                "for index, record in enumerate(new_records):\n"
                "    out = adapter.detect(record['image'], threshold=threshold)\n"
                "    ious = []\n"
                "    for box, label in zip(record['boxes'], record['labels'], strict=True):\n"
                "        same_label = [d for d in out['detections'] if d['label'] == label]\n"
                "        ious.append(round(max((box_iou(d['box'], box) for d in same_label), default=0.0), 3))\n"
                "    row = {{\n"
                "        'image': index, 'truth': record['labels'],\n"
                "        'detections': [(d['label'], round(d['score'], 3)) for d in out['detections']],\n"
                "        'same_label_iou': ious,\n"
                "    }}\n"
                "    new_data_rows.append(row)\n"
                "    print(json.dumps(row))\n\n"
                "contact = Image.new('RGB', (480, 160))\n"
                "for index, record in enumerate(new_records):\n"
                "    contact.paste(record['image'].resize((160, 160)), (160 * index, 0))\n"
                "contact"
            ),
        },
        {
            "md": (
                "**What to notice.** Read each row against its `truth`. A `same_label_iou` of 0.0 is a **miss**; more detections of one label than "
                "that label has objects is a **duplicate** (no NMS, so two queries can fire on one sign); a score just above 0.3 is a detection the "
                "threshold barely kept. In the worked run image 0 had three `yield-sign` detections for two signs (one sign found twice), image 1's speed-limit "
                "sign was missed and its yield sign was found twice, and two of the kept scores were 0.302. The threshold 0.3 was carried over from the COCO model in Section 4; nobody chose it for this three-class model. **Who owns it?** "
                "You do: a deployment would pick it on its own validation images (for example, the threshold that maximises F1 on the training "
                "split) and report misses and duplicates at that threshold."
            ),
        },
        # ---------------------------------------------------------------- 11. Export, reload & verify
        {
            "md": (
                "## 11. Artifact export, fresh reload, and boundary verification\n\n"
                "We export the fine-tuned adapter weights, with the training settings that produced them, to `outputs/rtdetr_adapter.pt`. To meet "
                "NOTEBOOK_SPEC §18 (VER1–VER5), we reload the artifact into a fresh pipeline and compare its **raw outputs before any threshold** — "
                "the class logits and the boxes of all 300 queries — with the in-memory model on the 10 held-out and 3 unseen images. The check "
                "passes when the largest absolute difference is within `PARITY_TOLERANCE` and the compared images hold at least one detection at "
                "the threshold, so an empty comparison cannot pass.\n\n"
                "**Predict:** will the largest difference be exactly 0, below 1e-3, or larger?"
            ),
            "code": (
                "artifact_path = OUTPUTS / 'rtdetr_adapter.pt'\n"
                "descriptor = adapter.save_artifact(artifact_path, notes='RT-DETR R50-VD sign adaptation tutorial artifact', training=run)\n"
                "print('Exported artifact descriptor:', json.dumps(descriptor, indent=2))\n\n"
                "reloaded = RTDetrDetectionPipeline.load_artifact(artifact_path, weights_dir=WEIGHTS_DIR)\n"
                "print({{'reloaded_source': reloaded.source, 'adapted': reloaded.adapted, 'class_names': list(reloaded.class_names)}})\n\n"
                "assert reloaded.training == descriptor['training'], 'the artifact must carry its training settings'\n\n"
                "PARITY_TOLERANCE = 1e-3\n"
                "compare_images = [r['image'] for r in held_out] + [r['image'] for r in new_records]\n"
                "raw_live = adapter.raw_outputs(compare_images)\n"
                "raw_fresh = reloaded.raw_outputs(compare_images)\n"
                "max_logit_diff = max(float(np.abs(a['logits'] - b['logits']).max()) for a, b in zip(raw_live, raw_fresh, strict=True))\n"
                "max_box_diff = max(float(np.abs(a['pred_boxes'] - b['pred_boxes']).max()) for a, b in zip(raw_live, raw_fresh, strict=True))\n"
                "detections_compared = sum(len(adapter.detect(img, threshold=threshold)['detections']) for img in compare_images)\n"
                "reload_parity = {{'images': len(compare_images), 'queries_per_image': int(raw_live[0]['logits'].shape[0]),\n"
                "                 'detections_at_threshold': detections_compared, 'max_abs_logit_difference': max_logit_diff,\n"
                "                 'max_abs_box_difference': max_box_diff, 'tolerance': PARITY_TOLERANCE}}\n"
                "print(json.dumps(reload_parity, indent=2))\n"
                "assert detections_compared > 0, 'the parity check compared no detections; it would pass vacuously'\n"
                "assert max_logit_diff <= PARITY_TOLERANCE and max_box_diff <= PARITY_TOLERANCE, reload_parity\n"
                "print(f'Fresh reload verification passed: raw outputs on {{len(compare_images)}} images match within {{PARITY_TOLERANCE}}.')"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Exactly 0 in the worked run (CPU), on 13 images × 300 queries with 24 detections at the threshold. Same weights on the same device usually give identical outputs; a GPU may show differences around 1e-6, which is why the check uses a tolerance and does not claim "identical". A difference beyond the tolerance means the artifact does not carry the adapted weights, and the cell stops rather than accepting a broken adapter.</details>'
            ),
        },
        # ---------------------------------------------------------------- 12. Outputs & provenance
        {
            "md": (
                "## 12. Write machine-readable outputs and provenance\n\n"
                "We export the required machine-readable artifacts:\n"
                "- `outputs/rtdetr_detection_input_manifest.json`\n"
                "- `outputs/rtdetr_detection_evaluation_report.json`\n"
                "- `outputs/rtdetr_detection_result.json`\n"
                "- `outputs/rtdetr_detection_detections.csv`\n"
                "- `outputs/rtdetr_detection_annotated.png`\n"
                "- `outputs/rtdetr_detection_new_data.csv` — one row per adapted detection on the unseen images, with its image, label, score, box and the IoU of the best same-label reference\n"
                "- `outputs/rtdetr_detection_new_data_annotated.png` — the adapted model's detections on the unseen images\n"
                "- `outputs/rtdetr_adapter.pt`\n\n"
                "`result.json` also records the training settings (learning rate, batch size, seed, freeze state, epoch losses, time), the degenerate-probe counts, the zero-training reference and the reload-parity numbers."
            ),
            "code": (
                "import csv\n"
                "from PIL import ImageDraw\n\n"
                "with open(OUTPUTS / '{stem}_input_manifest.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(input_manifest, f, indent=2)\n\n"
                "with open(OUTPUTS / '{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(coco_report, f, indent=2)\n\n"
                "annotated = scene.copy()\n"
                "draw = ImageDraw.Draw(annotated)\n"
                "for det in coco_result['detections']:\n"
                "    x0, y0, x1, y1 = det['box']\n"
                "    draw.rectangle([x0, y0, x1, y1], outline='red', width=3)\n"
                "    draw.text((x0 + 4, y0 + 4), f\"{{det['label']}} {{det['score']:.2f}}\", fill='red')\n"
                "annotated.save(OUTPUTS / '{stem}_annotated.png')\n\n"
                "new_data_csv = OUTPUTS / '{stem}_new_data.csv'\n"
                "new_sheet = Image.new('RGB', (480, 160))\n"
                "with open(new_data_csv, 'w', newline='', encoding='utf-8') as f:\n"
                "    writer = csv.writer(f)\n"
                "    writer.writerow(['image', 'rank', 'label', 'score', 'x0', 'y0', 'x1', 'y1', 'best_same_label_reference_iou'])\n"
                "    for index, record in enumerate(new_records):\n"
                "        dets = adapter.detect(record['image'], threshold=threshold)['detections']\n"
                "        drawn = record['image'].copy()\n"
                "        pen = ImageDraw.Draw(drawn)\n"
                "        for r, d in enumerate(dets):\n"
                "            refs = [b for b, lab in zip(record['boxes'], record['labels'], strict=True) if lab == d['label']]\n"
                "            best = max((box_iou(d['box'], b) for b in refs), default=0.0)\n"
                "            writer.writerow([f'new-{{NEW_DATA_SEED}}-{{index}}', r, d['label'], f\"{{d['score']:.4f}}\", *(f\"{{v:.1f}}\" for v in d['box']), f'{{best:.3f}}'])\n"
                "            pen.rectangle(d['box'], outline='red', width=3)\n"
                "            pen.text((d['box'][0] + 4, d['box'][1] + 4), f\"{{d['label']}} {{d['score']:.2f}}\", fill='red')\n"
                "        new_sheet.paste(drawn.resize((160, 160)), (160 * index, 0))\n"
                "new_sheet.save(OUTPUTS / '{stem}_new_data_annotated.png')\n\n"
                "csv_path = OUTPUTS / '{stem}_detections.csv'\n"
                "with open(csv_path, 'w', newline='', encoding='utf-8') as f:\n"
                "    writer = csv.writer(f)\n"
                "    writer.writerow(['image', 'rank', 'label', 'score', 'x0', 'y0', 'x1', 'y1'])\n"
                "    for r, d in enumerate(coco_result['detections']):\n"
                "        writer.writerow(['scene', r, d['label'], f\"{{d['score']:.4f}}\", *(f\"{{v:.1f}}\" for v in d['box'])])\n\n"
                "result_export = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'device': pipe.device,\n"
                "    'coco_detections': coco_result['detections'],\n"
                "    'degenerate_probes': degenerate,\n"
                "    'adaptation': {{\n"
                "        'dataset': dataset_manifest,\n"
                "        'split': {{'train': len(train_records), 'held_out': len(held_out)}},\n"
                "        'training': {{k: v for k, v in run.items() if not callable(v)}},\n"
                "        'baseline': baseline,\n"
                "        'adapted': adapted,\n"
                "        'zero_training_reference': reference_table,\n"
                "        'new_data': new_data_rows,\n"
                "        'reload_parity': reload_parity,\n"
                "        'artifact': descriptor,\n"
                "    }},\n"
                "}}\n"
                "with open(OUTPUTS / '{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_export, f, indent=2, default=str)\n\n"
                "print('Written release outputs:')\n"
                "for p in sorted(OUTPUTS.iterdir()):\n"
                "    print(f'  {{p.name:<40s}} {{p.stat().st_size:>12,d}} bytes')\n"
                "new_sheet"
            ),
        },
        # ---------------------------------------------------------------- 13. BYOD
        {
            "md": (
                "## 13. Optional: Bring Your Own Data (BYOD)\n\n"
                "Two BYOD branches are provided. Both are disabled by default so the default `Run all` path completes non-interactively:\n"
                "- `USE_BYOD_IMAGE`: run a single image through inference — set `BYOD_IMAGE_PATH` to a file in the runtime (Kaggle, Jupyter), or leave it empty on Colab to open the upload dialog.\n"
                "- `USE_BYOD_DATASET`: adapt the detector to your own labelled images through the same stages as Sections 6–11 (validate, split, "
                "coverage check, baseline, fine-tune, evaluate, export, fresh reload and parity), with the Section 8 settings.\n\n"
                "**Dataset format (read before you switch it on).** Set `BYOD_DATASET_PATH` to a folder or a `.zip` file in the runtime (on Kaggle, an "
                "attached dataset path; on Colab, leave it empty to open the upload dialog for one `.zip`). It must hold `annotations.csv` with the "
                "header `file,label,x0,y0,x1,y1` — one row per box, pixel coordinates with `x0 < x1` and `y0 < y1`, `file` relative to the folder or "
                "zip root — and the PNG/JPEG/WebP images it names. The class vocabulary is read from the `label` column in first-appearance order, "
                "or set `BYOD_CLASS_NAMES` to a comma-separated list to fix it (then every label must be in that list). Limits: at most 500 images, "
                "100 boxes per image and 1000 classes; every image needs at least one box; no image twice; and after the split (`HOLDOUT` from "
                "Section 7) both the train and held-out parts must contain every class — in practice at least four images per class. Each refusal "
                "names the row, file or record and the rule. Results go to `outputs/byod_result.json` and `outputs/byod_adapter.pt`."
            ),
            "code": (
                'USE_BYOD_IMAGE = False  # @param {{type:"boolean"}}\n'
                "BYOD_IMAGE_PATH = ''  # @param {{type:\"string\"}}\n"
                'USE_BYOD_DATASET = False  # @param {{type:"boolean"}}\n'
                "BYOD_DATASET_PATH = ''  # @param {{type:\"string\"}}\n"
                "BYOD_CLASS_NAMES = ''  # @param {{type:\"string\"}}\n\n"
                "# Verify rejection of malformed input:\n"
                "for desc, fn in (\n"
                "    ('non-image object', lambda: validate_inputs('/not/an/image.png')),\n"
                "    ('out-of-bounds box', lambda: validate_dataset([{{'image': blank_scene(), 'boxes': [[0, 0, 9999, 10]], 'labels': [SIGN_CLASSES[0]]}}], SIGN_CLASSES)),\n"
                "):\n"
                "    try:\n"
                "        fn()\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print(f'Refusal check passed: {{desc}} -> {{type(exc).__name__}}: {{exc}}')\n\n"
                'def byod_file(path, kind, suffixes=()):\n'
                '    """BYOD path first (works on Colab, Kaggle and Jupyter); on Colab an empty path opens the upload dialog."""\n'
                '    if str(path).strip():\n'
                '        source = Path(str(path).strip()).expanduser()\n'
                '        if not source.is_file():\n'
                "            raise FileNotFoundError(f'BYOD path {{str(source)!r}} does not exist or is not a file (relative paths start at {{os.getcwd()}}); give the path of one {{kind}}.')\n"
                '    else:\n'
                '        try:\n'
                '            from google.colab import files\n'
                '        except ImportError:\n'
                "            raise RuntimeError(f'BYOD is on but its path field is empty, and the upload dialog exists only in Google Colab: copy the {{kind}} into this runtime (or attach it as a Kaggle dataset) and set the path field.') from None\n"
                '        uploaded = files.upload()\n'
                '        if len(uploaded) != 1:\n'
                "            raise ValueError(f'Upload exactly one {{kind}} (received {{len(uploaded)}} files; a cancelled dialog sends none). Run this cell again.')\n"
                '        name, payload = next(iter(uploaded.items()))\n'
                "        source = Path('work') / Path(name).name\n"
                '        source.parent.mkdir(parents=True, exist_ok=True)\n'
                '        source.write_bytes(payload)\n'
                '    if suffixes and not source.name.lower().endswith(tuple(suffixes)):\n'
                '        raise ValueError(f\'{{source.name}}: expected a {{kind}} ending in {{" or ".join(suffixes)}}.\')\n'
                '    return source\n'
                '\n'
                'if USE_BYOD_IMAGE:\n'
                "    byod_source = byod_file(BYOD_IMAGE_PATH, 'image file (PNG, JPEG or WebP)')\n"
                '    byod_image = Image.open(byod_source)\n'
                '    byod_image.load()\n'
                "    print(validate_inputs(byod_image, threshold=threshold, names=[byod_source.name])['verdict'])\n"
                "    byod_res = pipe.detect(byod_image, threshold=threshold)\n"
                "    for det in byod_res['detections'][:20]:\n"
                "        print(f\"{{det['label']:>16s}} {{det['score']:.3f}}\")\n"
                "    print(evaluation_report(byod_res, None, sample_kind='byod')['verdict'])\n"
                "else:\n"
                "    print('BYOD image branch is off; set USE_BYOD_IMAGE = True to run inference on custom images.')\n\n"
                "if USE_BYOD_DATASET:\n"
                "    byod_dataset_source = BYOD_DATASET_PATH if str(BYOD_DATASET_PATH).strip() else byod_file('', 'dataset .zip', ('.zip',))\n"
                "    byod_records, byod_found_classes = load_detection_dataset(byod_dataset_source)\n"
                "    byod_classes = tuple(c.strip() for c in str(BYOD_CLASS_NAMES).split(',') if c.strip()) or byod_found_classes\n"
                "    byod_manifest = validate_dataset(byod_records, byod_classes, epochs=EPOCHS)\n"
                "    print(json.dumps(byod_manifest, indent=2))\n"
                "    byod_train, byod_held = split_dataset(byod_records, train_fraction=1.0 - HOLDOUT, seed=SEED)\n"
                "    print('Classes per split:', check_split_coverage({{'train': byod_train, 'held_out': byod_held}}, byod_classes))\n"
                "    byod_pipe = RTDetrDetectionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=byod_classes, seed=SEED)\n"
                "    byod_baseline = byod_pipe.evaluate(byod_held)\n"
                "    byod_run = byod_pipe.finetune(byod_train, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,\n"
                "                                  seed=SEED, freeze_backbone=FREEZE_BACKBONE)\n"
                "    byod_adapted = byod_pipe.evaluate(byod_held)\n"
                "    for key in ('ap', 'ap50', 'ap75'):\n"
                "        print(f\"BYOD {{key:<5s}} baseline {{byod_baseline[key]:.4f}}  adapted {{byod_adapted[key]:.4f}}\")\n"
                "    byod_descriptor = byod_pipe.save_artifact(OUTPUTS / 'byod_adapter.pt', notes='BYOD adaptation artifact', training=byod_run)\n"
                "    byod_reloaded = RTDetrDetectionPipeline.load_artifact(OUTPUTS / 'byod_adapter.pt', weights_dir=WEIGHTS_DIR)\n"
                "    byod_images = [r['image'] for r in byod_held]\n"
                "    byod_live, byod_fresh = byod_pipe.raw_outputs(byod_images), byod_reloaded.raw_outputs(byod_images)\n"
                "    byod_max_diff = max(float(np.abs(a[k] - b[k]).max()) for a, b in zip(byod_live, byod_fresh, strict=True) for k in ('logits', 'pred_boxes'))\n"
                "    assert byod_max_diff <= PARITY_TOLERANCE, f'BYOD reload parity failed: {{byod_max_diff}}'\n"
                "    print(f'BYOD reload parity: max |difference| {{byod_max_diff:.2e}} on {{len(byod_images)}} held-out images')\n"
                "    with open(OUTPUTS / 'byod_result.json', 'w', encoding='utf-8') as f:\n"
                "        json.dump({{'source': str(byod_dataset_source), 'dataset': byod_manifest, 'class_names': list(byod_classes),\n"
                "                   'split': {{'train': len(byod_train), 'held_out': len(byod_held)}}, 'training': byod_run,\n"
                "                   'baseline': byod_baseline, 'adapted': byod_adapted, 'reload_max_abs_difference': byod_max_diff,\n"
                "                   'artifact': byod_descriptor}}, f, indent=2, default=str)\n"
                "    print('Written outputs/byod_result.json and outputs/byod_adapter.pt')\n"
                "else:\n"
                "    print('BYOD dataset branch is off; set USE_BYOD_DATASET = True to adapt on custom labelled datasets.')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "**What this notebook established, in this runtime.** The pinned `PekingU/rtdetr_r50vd` snapshot was verified against a committed "
        "SHA-256 manifest. The pretrained RT-DETR detector found three of the four drawn objects with the IoU values Section 4 printed "
        "(0.877–0.983 in the worked run) and missed the sports ball. On blank and noise images it still produced one detection each at 0.3 "
        "(Section 5), so the threshold belongs to the deployment. A non-COCO three-class sign vocabulary was adapted by re-heading the detector, "
        "setting prior probability biases, fine-tuning the hybrid encoder and decoder with the ResNet backbone frozen, and scoring held-out "
        "validation with COCO-style average precision before and after (Section 9 printed both). Detections on unseen data were written out "
        "with their misses and duplicates, and the adapter artifact was saved with its training settings, reloaded, and its raw outputs "
        "matched the in-memory model within the stated tolerance.\n\n"
        "**How to read the adaptation number.** The baseline is zero by construction, so \"0 → about 0.85\" shows that a re-headed model "
        "learns, not how much. Compare the adapted stop-sign AP with the unadapted COCO detector's line in Section 9 (both 1.000 in the worked run: "
        "the pretrained model already found these drawn stop signs), and remember that a colour-and-shape rule with no learning scored AP 1.000 on the same held-out images in the review of this notebook: the drawn task is "
        "easy, and the adapted detector does not beat that rule. What the run demonstrates is a correct, reproducible adaptation workflow.\n\n"
        "**Run-to-run variation.** CPU and GPU runs give different adapted metrics: the worked CPU run reached AP 0.8494, the earlier hosted T4 "
        "record 0.9161. With ten held-out images, one detection more or less moves AP by several hundredths, so read differences in the "
        "second decimal as variation, not as an effect.\n\n"
        "**What a green run proves.** Successful execution proves that the recorded repository revision, the pinned dependency set and the "
        "pinned checkpoint together reproduce these stages in a fresh runtime, without the repository being cloned or installed and without "
        "any DIMER worker or service. It does **not** establish benchmark superiority, fitness for any deployment, or that the adapted model "
        "generalises beyond the synthetic data it was fitted to.\n\n"
        "**Reproducibility.** Seeds are exposed as form parameters (`DATASET_SEED = 0`, `SEED = 0`). Computation runs in float32 without "
        "stochastic data augmentation, and Section 8 rebuilds the model before each fine-tune. Running unchanged in an identical runtime "
        "reproduces these results; a different device or library build changes them in the later decimals.\n\n"
        '## Troubleshooting\n'
        '\n'
        '- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n'
        '- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused and an incomplete one is finished. If it repeats, `files.pythonhosted.org` or `pypi.org` is blocked or altered.\n'
        '- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable. After a session restart, run from the top.\n'
        '- **"The isolated environment\'s Python process exited"** — usually out of memory; restart the session and choose **Run all**.\n'
        '- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete the folder Section 3 prints as `weights_dir` and run Section 3 again (172 MB).\n'
        "- **Section 4 finds fewer than three of the four drawn objects, or Section 9 stays near zero** — on CPU and GPU the numbers can differ in the third decimal; a very different result means a different runtime (check `torch` and `transformers` versions in Section 1's record) or a changed form field.\n"
        '- **You changed a Section 8 form value** — rerun Section 8 and then Section 9. Section 8 rebuilds the model from the snapshot first, so nothing carries over from an earlier run; if you also changed `HOLDOUT` or `SEED`, rerun from Section 7.\n'
        '- **The fine-tune is slow** — on CPU three epochs over 30 images took 80–140 s in local checks (estimate; Section 8 prints the measured `train_seconds`); a GPU runtime makes it seconds. `FREEZE_BACKBONE = False` trains all 42.7 M parameters and is slower.\n'
        "- **Section 11's reload check fails** — the export or reload is broken, or the compared images held no detection at the threshold; the message prints the numbers. Run Sections 8–11 again. Do not use the artifact.\n"
        '- **BYOD: "BYOD path … does not exist" / "the upload dialog exists only in Google Colab" / "Upload exactly one"** — set `BYOD_IMAGE_PATH` to an image in the runtime (it works on Kaggle and Jupyter); on Colab an empty path opens the dialog, and a cancelled dialog stops with that message.\n'
        '- **A `ValueError` or `TypeError` from `validate_inputs`, `validate_dataset`, `load_detection_dataset` or `check_split_coverage`** — it names the rule and the record, row or file: not an image, a missing `annotations.csv` or wrong header, an image the CSV names but the folder lacks, a box outside the image or with zero area, an image with no boxes, the same image twice, a label outside `BYOD_CLASS_NAMES`, or a split without every class (add images of that class or change `SEED`).\n'
        '\n'
        '## Change one thing (next experiments)\n'
        '\n'
        '**Activity — unfreeze the backbone (Predict → Change → Run → Observe → Explain).** *Predict:* with `FREEZE_BACKBONE = False`, how many parameters will train, will the first-epoch loss match the frozen run, and will held-out AP go up or down? *Change:* set `FREEZE_BACKBONE = False` in Section 8. *Run:* Section 8, then Section 9. *Observe:* `trainable_parameters` (42,733,137 on this checkpoint), `backbone_frozen_in_model: false`, `train_seconds`, the epoch losses and the AP table. *Explain:* why the first-epoch loss is close to the frozen run (both start from the same pretrained model), and whether more trainable parameters helped on 30 drawn images. Set it back to `True` and rerun Sections 8–9 to return to the default.\n'
        '\n'
        'Other single changes: lower `threshold` in Section 4 to 0.1 and watch whether the sports ball appears (and what else does); raise `EPOCHS` to 6 or lower `LEARNING_RATE` to 3e-5 and rerun Sections 8–9; change `DATASET_SEED` for a different set of 40 drawn signs (rerun from Section 6); or bring your own labelled dataset through `USE_BYOD_DATASET`.\n'
        '\n'
        '## Glossary\n'
        '\n'
        '- **Object query** — one of the 300 learned decoder slots that each emit a box and a class score; no anchors and no non-maximum suppression (NMS), so overlapping duplicates are possible at a low threshold.\n'
        '- **Sigmoid score / threshold** — each class score is an independent sigmoid, not a calibrated probability; the caller chooses the cut-off (`0.3` by default, `0.05` for evaluation).\n'
        '- **IoU** — intersection over union between a detected box and a reference box (1 = identical); `box_iou` in Section 4.\n'
        "- **AP, AP50, AP75** — COCO-style average precision: the area under the precision–recall curve at IoU 0.50 (AP50), at 0.75 (AP75), and averaged over ten thresholds from 0.50 to 0.95 (AP); computed by the carried helper without pycocotools' crowd or area-range filtering.\n"
        '- **Re-heading** — replacing the 80-class classification heads with heads for the new vocabulary while keeping the backbone, encoder and box regression.\n'
        '- **Prior-probability bias** — initialising the new classification biases to −4.595 (p = 0.01) so the untrained heads fire on almost nothing, which is why the baseline is zero.\n'
        '- **Frozen backbone** — the ResNet-50-vd weights are not updated; 19.3 M of 42.7 M parameters (45.1 %) train.\n'
        "- **Varifocal + L1 + GIoU loss** — RT-DETR's native training objective: a focal-style classification loss weighted by IoU, plus box-coordinate and generalised-IoU terms, summed over the six decoder layers.\n"
        '- **Held-out split** — the 25 % of images (10 of 40) never shown to the optimiser; AP is measured there.\n'
        '- **Reload parity** — the `.pt` adapter written to disk, loaded into a fresh pipeline, reproduces the in-memory model\'s raw outputs (logits and boxes of every query, before any threshold) within a stated tolerance (1e-3).\n'
        '- **Zero-training reference** — a score obtained with no adaptation at all (here, the unadapted COCO detector on the stop signs); it shows what the adapted number must beat, which a zero-by-construction baseline cannot.\n'
        '- **Isolated environment** — the separate Python 3.12.12 environment Section 1 builds from the hash lock; every later cell runs there.\n'
        '- **BYOD** — bring your own data: an image via `BYOD_IMAGE_PATH` or the Colab dialog, or a labelled dataset (folder or `.zip` with `annotations.csv`) via `BYOD_DATASET_PATH`.\n'
        '\n'
        '## Conclusion (your notes)\n'
        '\n'
        'Before you leave, write three lines in this cell: (1) which drawn object the pretrained detector missed and why a miss on a drawn scene is a finding, not an error; (2) the baseline, adapted and zero-training-reference numbers from Section 9, and why the adapted AP alone does not show the task was hard; (3) what you would need — labelled photographs, a held-out split, a threshold chosen on validation — before trusting such numbers on real images.\n'
        '\n'
        "## References\n\n"
        "- Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y. and Chen, J. (2023). *DETRs Beat YOLOs on Real-time Object Detection.* [arXiv:2304.08069](https://arxiv.org/abs/2304.08069).\n"
        "- Upstream repository: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) — Apache-2.0.\n"
        "- Hugging Face checkpoint: [PekingU/rtdetr_r50vd](https://huggingface.co/PekingU/rtdetr_r50vd) — Apache-2.0.\n"
        "- Lin, T.-Y. et al. (2014). *Microsoft COCO: Common Objects in Context.* [arXiv:1405.0312](https://arxiv.org/abs/1405.0312).\n"
        "- Repository model card: https://github.com/kurtvalcorza/rtdetr-detection-pipeline/blob/main/MODEL_CARD.md\n"
        "- [`kurtvalcorza/rtdetr-detection-pipeline`](https://github.com/kurtvalcorza/rtdetr-detection-pipeline) — source repository for this pipeline."
    ),
}
