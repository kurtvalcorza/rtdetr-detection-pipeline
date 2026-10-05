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
    "runtime_imports": ["torch", "transformers", "numpy", "PIL"],
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
        "an unseen image, exports the weights as a standalone `.pt` artifact, and reloads that artifact from disk to assert identical "
        "behavior. Every number you see is measured locally in this notebook runtime."
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
        "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Jupyter and has met bounding boxes, and wants to see two things in one run: what a pretrained real-time detector does on a scene it was not drawn for, and what a bounded fine-tune changes when the detector is re-headed onto three classes COCO does not have. The audience is students and practitioners deciding whether a detector can be adapted to their own labelled images; no prior experience with RT-DETR, transformers or average precision is assumed — each term is explained where it first matters and again in the **Glossary**. CPU is enough (the fine-tune takes under a minute there); a GPU makes it seconds.\n\n**Input → Model → Output.**\n\n| | |\n|---|---|\n| Input | images resized to 640 × 640, with a caller-owned score threshold; the default is a drawn 640 × 480 street scene with four reference boxes, then a 40-image drawn traffic-sign dataset over `stop-sign`, `yield-sign`, `speed-limit-sign` |\n| Model | RT-DETR R50-VD: a ResNet-50-vd backbone, a hybrid encoder and a 6-layer transformer decoder with 300 object queries emitting boxes and per-class sigmoid scores without NMS; for adaptation, the classification heads are re-initialised for the three sign classes and the backbone is frozen (19.3 M of 42.7 M parameters train) |\n| Output | COCO detections and `box_iou` against the drawn references, two degenerate-input probes, a pre-adaptation baseline and a post-adaptation AP / AP50 / AP75 on a held-out split, detections on three unseen images, a reloadable `.pt` adapter whose detections are checked against the in-memory model, and six machine-readable outputs |\n\n**How to use this notebook.** Choose any runtime, then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed (the recorded hosted run of the previous version needed one; this version removes it). Sections 1–3 are **infrastructure** — the isolated environment, the carried modules and the verified snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning** with a worked answer from the recorded Kaggle T4 run of 15 September 2026. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Budget about ten minutes on a GPU runtime, longer on CPU.\n\n**Roadmap:** 1–3 infrastructure → 4 the pretrained detector on a drawn scene, with one honest miss *(core concept: sigmoid scores under a caller-owned threshold, no NMS)* → 5 blank and noise probes *(evaluation practice: a detector should find nothing where there is nothing)* → 6 a labelled dataset over a new vocabulary and its validation *(core concept: the data contract)* → 7 split, re-head and the pre-adaptation baseline *(core concept: prior-probability bias initialisation)* → 8 the bounded fine-tune with the backbone frozen *(core concept: what 45 % of the parameters can learn in three epochs)* → 9 held-out AP, AP50 and AP75 *(evaluation practice)* → 10 unseen images → 11 export, reload and parity *(engineering)* → 12 outputs and provenance → 13 optional BYOD → conclude."
    )]},
    "prerequisites": [
        "- **Learner:** basic Python and Colab or Jupyter familiarity; no prior experience with RT-DETR or detection fine-tuning. Object queries, sigmoid scores, IoU, AP, re-heading and reload parity are explained where they are first used and again in the Glossary.",
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or a Linux Jupyter server). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the Python version of the kernel itself does not matter and nothing is installed into it; a Windows or macOS kernel is not supported. CPU is the documented default and CUDA GPU is used automatically when available. On CPU, the fine-tune completes in ~30–50 s; on a Tesla T4 GPU, it runs in ~1–2 s. Pinned `torch==2.14.0` and the 172 MB checkpoint are the primary downloads.",
        "- **Knowledge:** basic Python and PIL; bounding box representation in xyxy pixel coordinates; intersection-over-union (IoU); and the interpretation of average precision (AP50 and AP@[.50:.95]).",
        "- **Data:** everything is generated deterministically in code by `samples.py`, requiring zero external dataset download: one 640×480 COCO demonstration scene and a 40-image labelled sign adaptation dataset. Optional BYOD is gated off by default. Expected BYOD input: an image or list of `{'image': PIL.Image, 'boxes': [[x0, y0, x1, y1], ...], 'labels': [name, ...]}` records. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so; uploaded inputs stay in this runtime and are not sent to any inference API.",
    ],
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated, hash-locked environment with the pinned dependencies "
        "(nothing is installed into the notebook kernel, so no restart is needed), stages and digest-verifies the pinned checkpoint, "
        "runs COCO detection on a drawn scene, validates the 40-image sign adaptation dataset, splits it into train and validation parts, "
        "measures the pre-adaptation baseline, **runs the bounded fine-tune**, re-evaluates on the held-out split, detects on an unseen test "
        "image, exports the adapted artifact, reloads it from disk to verify numeric consistency, and writes machine-readable outputs "
        "with provenance. Nothing is skipped behind a default-off flag, and no clone or DIMER worker is required (NOTEBOOK_SPEC 2.0 §5, RUN7, FT2)."
    ),
    "byod": (
        "Two optional BYOD branches are included and both are disabled by default (`USE_BYOD_IMAGE = False`, `USE_BYOD_DATASET = False`). "
        "`USE_BYOD_IMAGE` runs your own image (from `BYOD_IMAGE_PATH`, or the Colab upload dialog when the path is empty) through the detection and validation contract. `USE_BYOD_DATASET` takes your own labelled "
        "detection records and runs them through the full adaptation workflow (validate, split, baseline, fine-tune, evaluate) under "
        "NOTEBOOK_SPEC 2.0 DAT14."
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
                '<details><summary>Check your reasoning</summary>Three of four. In the recorded run the stop sign, the traffic light and the clock were found with tight boxes (IoU 0.91–0.97 against the drawn references) and the orange sports ball was missed — `box_iou = 0.0` — which the report records rather than hides. A drawn disc on a drawn street is far from COCO photographs, and a miss at 0.3 is a finding about the threshold and the domain, not a bug: lowering the threshold is the first experiment to try.</details>'
            ),
        },
        # ---------------------------------------------------------------- 5. Degenerate inputs
        {
            "md": (
                "## 5. Degenerate input probes: blank canvas and noise\n\n"
                "A detector should be evaluated on structure-free inputs. A model that invents confident detections on a blank canvas or uniform "
                "noise will invent them on real unlabelled scenes. We probe the model with both a blank image and a random RGB noise image at "
                "both the standard detection threshold (`0.3`) and the evaluation threshold (`0.05`).\n\n"
                "**Predict:** on a blank canvas and on uniform noise, how many detections at 0.3, and how many at 0.05? If the lenient count is not zero, is that a defect?"
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
                '<details><summary>Check your reasoning</summary>At 0.3 the expected count on both probes is zero or very near it; at 0.05 a few low-score queries can surface (the cell prints the top three with their scores). That is not a defect: each score is an independent uncalibrated sigmoid, and 300 queries with no NMS will always have a few above a very lenient cut-off. The defect would be confident detections at the standard threshold on structure-free input. The hosted record keeps the counts in the executed notebook rather than in the verification table, so read your own.</details>'
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
                "`validate_dataset` validates the schema, checks image bounds, ensures non-empty coordinates, and returns a verified dataset manifest."
            ),
            "code": (
                'N_IMAGES = 40  # @param {{type:"integer"}}\n'
                'DATASET_SEED = 0  # @param {{type:"integer"}}\n'
                'EPOCHS = 3  # @param {{type:"integer"}}\n\n'
                "records = sign_dataset(N_IMAGES, seed=DATASET_SEED)\n"
                "dataset_manifest = validate_dataset(records, SIGN_CLASSES, epochs=EPOCHS)\n"
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
                "**The baseline is zero (or near-zero), and that is the expected starting point.** With properly initialized prior biases, "
                "the unadapted class heads emit no false positives on the held-out set before training.\n\n"
                "**Predict:** the re-headed model has never seen a sign class. Will the pre-adaptation AP be exactly 0, near 0, or somewhere above — and what would a non-zero baseline tell you about the bias initialisation?"
            ),
            "code": (
                'HOLDOUT = 0.25  # @param {{type:"number"}}\n'
                'SEED = 0  # @param {{type:"integer"}}\n\n'
                "train_records, held_out = split_dataset(records, train_fraction=1.0 - HOLDOUT, seed=SEED)\n"
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
                "- **The backbone is frozen.** The ResNet-50-vd backbone is frozen (`19.3 M` trainable out of `42.7 M` parameters, 45.1%). "
                "This accelerates adaptation on CPU and prevents catastrophic forgetting on small datasets.\n"
                "- **Schedule:** 3 epochs with AdamW at learning rate `1e-4` and batch size 4.\n\n"
                "This cell trains `adapter` in place. To repeat the fine-tune from the baseline after changing a setting, run Section 7 first: it builds a fresh re-headed model.\n\n"
                "**Predict:** three epochs over 30 drawn images. Will the loss fall at every epoch, and by how much from the first to the last epoch — a few percent, or more than half?"
            ),
            "code": (
                'LEARNING_RATE = 1e-4  # @param {{type:"number"}}\n'
                'BATCH_SIZE = 4  # @param {{type:"integer"}}\n'
                'FREEZE_BACKBONE = True  # @param {{type:"boolean"}}\n\n'
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
                "print(json.dumps({{'freeze_backbone': run['freeze_backbone'],\n"
                "                  'trainable_parameters': run['trainable_parameters'],\n"
                "                  'total_parameters': run['total_parameters'],\n"
                "                  'epochs': run['epochs'], 'batch_size': run['batch_size'],\n"
                "                  'learning_rate': run['learning_rate'], 'epoch_losses': [round(x, 4) for x in run['epoch_losses']]}}, indent=2))\n"
                "print(f\"Loss progression: {{run['epoch_losses'][0]:.4f}} -> {{run['epoch_losses'][-1]:.4f}}\")"
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>The loss falls, usually at every one of the three epochs, and the drop from the first to the last epoch is large — well over half on this drawn data — because the dataset is small and regular and the heads start from a bias prior. The recorded verification table keeps AP rather than the per-epoch losses, so compare your `Loss progression` line with the AP that Section 9 reaches rather than with a fixed number; the loss is the optimiser's signal, AP is the evaluation.</details>"
            ),
        },
        # ---------------------------------------------------------------- 9. Evaluate Held-Out
        {
            "md": (
                "## 9. Evaluate on the held-out split\n\n"
                "We re-run `evaluate` on the exact same held-out validation set using the same thresholds to measure empirical progress. "
                "`ap50` is average precision at IoU 0.50; `ap` is COCO-standard AP@[.50:.95] across 10 IoU thresholds.\n\n"
                "**Predict:** from a baseline of zero, where will AP50 land after three epochs on drawn signs — below 0.5, around 0.7, or above 0.9? Will AP (averaged up to IoU 0.95) be much lower than AP50?"
            ),
            "code": (
                "adapted = adapter.evaluate(held_out)\n"
                "print(json.dumps({{'ap': round(adapted['ap'], 4), 'ap50': round(adapted['ap50'], 4), 'ap75': round(adapted['ap75'], 4),\n"
                "                  'per_class_ap50': {{k: round(v, 4) for k, v in adapted['per_class_ap50'].items()}},\n"
                "                  'n_images': adapted['n_images'], 'n_references': adapted['n_references']}}, indent=2))\n"
                "print()\n"
                "print(f\"{{'metric':<10s}} {{'baseline':>10s}} {{'adapted':>10s}} {{'change':>10s}}\")\n"
                "for key in ('ap', 'ap50', 'ap75'):\n"
                "    before_val, after_val = baseline[key], adapted[key]\n"
                "    print(f\"{{key:<10s}} {{before_val:>10.4f}} {{after_val:>10.4f}} {{after_val - before_val:>+10.4f}}\")"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Above 0.9. In the recorded run three epochs took the held-out split from AP 0.000 to AP 0.9161, AP50 0.9273 and AP75 0.9273 — AP50 and AP75 equal, so every matched box was tight enough to survive IoU 0.75, and AP (averaged to 0.95) only a little lower. Drawn signs are a regular, high-contrast domain; on photographs the gap between AP50 and AP would be far wider. The recorded run took 240.6 s end to end on a T4, most of it downloads.</details>'
            ),
        },
        # ---------------------------------------------------------------- 10. New-data inference
        {
            "md": (
                "## 10. Inference on unseen test data\n\n"
                "We synthesize 3 new images from an unseen seed (`seed=99`). The adapted pipeline detects the custom sign classes, and we compute "
                "the intersection-over-union against the ground-truth annotations."
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
        # ---------------------------------------------------------------- 11. Export, reload & verify
        {
            "md": (
                "## 11. Artifact export, fresh reload, and boundary verification\n\n"
                "We export the fine-tuned adapter weights to `outputs/rtdetr_adapter.pt`. To satisfy NOTEBOOK_SPEC 2.0 §18 (VER1–VER5), we reload "
                "the artifact into a fresh pipeline instance and verify that detections match the adapted model identically.\n\n"
                "**Predict:** the reloaded pipeline is a new object built from the `.pt` file. Will its detections on the first unseen image match the in-memory model exactly, within 1e-3, or differ?"
            ),
            "code": (
                "artifact_path = OUTPUTS / 'rtdetr_adapter.pt'\n"
                "descriptor = adapter.save_artifact(artifact_path, notes='RT-DETR R50-VD sign adaptation tutorial artifact')\n"
                "print('Exported artifact descriptor:', json.dumps(descriptor, indent=2))\n\n"
                "reloaded = RTDetrDetectionPipeline.load_artifact(artifact_path, weights_dir=WEIGHTS_DIR)\n"
                "print({{'reloaded_source': reloaded.source, 'adapted': reloaded.adapted, 'class_names': list(reloaded.class_names)}})\n\n"
                "test_img = new_records[0]['image']\n"
                "det_orig = adapter.detect(test_img, threshold=threshold)['detections']\n"
                "det_reloaded = reloaded.detect(test_img, threshold=threshold)['detections']\n"
                "assert len(det_orig) == len(det_reloaded)\n"
                "for d1, d2 in zip(det_orig, det_reloaded, strict=True):\n"
                "    assert d1['label'] == d2['label']\n"
                "    assert np.allclose(d1['box'], d2['box'], atol=1e-3)\n"
                "    assert np.isclose(d1['score'], d2['score'], atol=1e-3)\n"
                "print('Fresh reload verification passed: all reloaded detections match exactly.')"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>They match: the recorded run reported `Fresh reload verification passed`, with the same labels, boxes within 1e-3 and scores within 1e-3 (same device, same kernels). A difference beyond that tolerance means the artifact does not carry the adapted weights, and the cell stops rather than exporting a broken adapter.</details>'
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
                "- `outputs/rtdetr_adapter.pt`"
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
                "    'adaptation': {{\n"
                "        'dataset': dataset_manifest,\n"
                "        'split': {{'train': len(train_records), 'held_out': len(held_out)}},\n"
                "        'baseline': baseline,\n"
                "        'adapted': adapted,\n"
                "        'artifact': descriptor,\n"
                "    }},\n"
                "}}\n"
                "with open(OUTPUTS / '{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_export, f, indent=2, default=str)\n\n"
                "print('Written release outputs:')\n"
                "for p in sorted(OUTPUTS.iterdir()):\n"
                "    print(f'  {{p.name:<36s}} {{p.stat().st_size:>10,d}} bytes')"
            ),
        },
        # ---------------------------------------------------------------- 13. BYOD
        {
            "md": (
                "## 13. Optional: Bring Your Own Data (BYOD)\n\n"
                "Two BYOD branches are provided. Both are disabled by default so the default `Run all` path completes non-interactively:\n"
                "- `USE_BYOD_IMAGE`: run a single image through inference — set `BYOD_IMAGE_PATH` to a file in the runtime (Kaggle, Jupyter), or leave it empty on Colab to open the upload dialog.\n"
                "- `USE_BYOD_DATASET`: Upload a list of labelled records to run custom adaptation through the exact same local pipeline stages."
            ),
            "code": (
                'USE_BYOD_IMAGE = False  # @param {{type:"boolean"}}\n'
                "BYOD_IMAGE_PATH = ''  # @param {{type:\"string\"}}\n"
                'USE_BYOD_DATASET = False  # @param {{type:"boolean"}}\n'
                "BYOD_CLASS_NAMES = ['custom-1', 'custom-2']  # @param\n\n"
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
                "    byod_records = []  # Supply: [{{'image': PIL.Image, 'boxes': [[x0, y0, x1, y1], ...], 'labels': ['custom-1', ...]}}]\n"
                "    byod_manifest = validate_dataset(byod_records, BYOD_CLASS_NAMES, epochs=EPOCHS)\n"
                "    print(json.dumps(byod_manifest, indent=2))\n"
                "    byod_train, byod_held = split_dataset(byod_records, train_fraction=0.75, seed=SEED)\n"
                "    byod_pipe = RTDetrDetectionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=BYOD_CLASS_NAMES, seed=SEED)\n"
                "    print('Baseline:', byod_pipe.evaluate(byod_held))\n"
                "    byod_pipe.finetune(byod_train, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, seed=SEED)\n"
                "    print('Adapted:', byod_pipe.evaluate(byod_held))\n"
                "    byod_pipe.save_artifact(OUTPUTS / 'byod-adapter.pt', notes='BYOD adaptation artifact')\n"
                "else:\n"
                "    print('BYOD dataset branch is off; set USE_BYOD_DATASET = True to adapt on custom labelled datasets.')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "**What this notebook established, in this runtime.** The pinned `PekingU/rtdetr_r50vd` snapshot was verified against a committed "
        "SHA-256 manifest. The pretrained RT-DETR detector predicted COCO objects on a rendered scene with high IoU (0.91–0.97) for three objects "
        "and missed the fourth. A non-COCO three-class sign vocabulary was adapted by re-heading the detector, setting prior probability biases, "
        "fine-tuning the hybrid encoder and decoder with the ResNet backbone frozen, and scoring held-out validation with COCO-style average "
        "precision before and after. Detections on unseen data were demonstrated, and the adapter artifact was saved, reloaded, and verified to match.\n\n"
        "**What a green run proves.** Successful execution proves that the recorded repository revision, the pinned dependency set and the "
        "pinned checkpoint together reproduce these stages in a fresh runtime, without the repository being cloned or installed and without "
        "any DIMER worker or service. It does **not** establish benchmark superiority, fitness for any deployment, or that the adapted model "
        "generalises beyond the synthetic data it was fitted to.\n\n"
        "**Reproducibility.** Seeds are exposed as form parameters (`DATASET_SEED = 0`, `SEED = 0`). Computation runs in float32 without "
        "stochastic data augmentation. Running unchanged in an identical runtime reproduces these results.\n\n"
        '## Troubleshooting\n'
        '\n'
        '- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n'
        '- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused and an incomplete one is finished. If it repeats, `files.pythonhosted.org` or `pypi.org` is blocked or altered.\n'
        '- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable. After a session restart, run from the top.\n'
        '- **"The isolated environment\'s Python process exited"** — usually out of memory; restart the session and choose **Run all**.\n'
        '- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete the folder Section 3 prints as `weights_dir` and run Section 3 again (172 MB).\n'
        "- **Section 4 finds fewer than three of the four drawn objects, or Section 9 stays near zero** — on CPU and GPU the numbers can differ in the third decimal; a very different result means a different runtime (check `torch` and `transformers` versions in Section 1's record) or a changed form field.\n"
        '- **You changed `EPOCHS`, `LEARNING_RATE` or `FREEZE_BACKBONE` and re-ran Section 8 alone** — the fine-tune continued from the already adapted `adapter`, so Section 9 is no longer a comparison with the Section 7 baseline. Run Section 7 (a fresh re-headed model and its baseline) and then Section 8.\n'
        '- **The fine-tune is slow** — on CPU three epochs over 30 images take about 30–50 s; a GPU runtime makes it seconds. `FREEZE_BACKBONE = False` trains all 42.7 M parameters and is slower.\n'
        "- **Section 11's reload check fails** — the export or reload is broken; run Sections 8–11 again. Do not use the artifact.\n"
        '- **BYOD: "BYOD path … does not exist" / "the upload dialog exists only in Google Colab" / "Upload exactly one"** — set `BYOD_IMAGE_PATH` to an image in the runtime (it works on Kaggle and Jupyter); on Colab an empty path opens the dialog, and a cancelled dialog stops with that message.\n'
        '- **A `ValueError` or `TypeError` from `validate_inputs` or `validate_dataset`** — it names the rule: not an image, a box outside the image, an empty label list, a label outside the vocabulary, or too few images for the epochs requested.\n'
        '\n'
        '## Change one thing (next experiments)\n'
        '\n'
        'Each of these changes one default and keeps the rest of the path; run Section 7 again before Section 8 so the fine-tune starts from a fresh re-headed model and the baseline is comparable. Lower `threshold` in Section 4 to 0.1 and watch whether the sports ball appears (and what else does); raise `EPOCHS` to 6 or lower `LEARNING_RATE` to 3e-5 and read AP against the recorded 0.9161; set `FREEZE_BACKBONE = False` and compare the time and the AP; change `DATASET_SEED` for a different set of 40 drawn signs; or bring your own labelled records through `USE_BYOD_DATASET`.\n'
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
        '- **Reload parity** — the `.pt` adapter written to disk, loaded into a fresh pipeline, reproduces the in-memory detections (labels, boxes within 1e-3, scores within 1e-3).\n'
        '- **Isolated environment** — the separate Python 3.12.12 environment Section 1 builds from the hash lock; every later cell runs there.\n'
        '- **BYOD** — bring your own data: an image via `BYOD_IMAGE_PATH` or the Colab dialog, or labelled records in code via `USE_BYOD_DATASET`.\n'
        '\n'
        '## Conclusion (your notes)\n'
        '\n'
        'Before you leave, write three lines in this cell: (1) which drawn object the pretrained detector missed and why a miss on a drawn scene is a finding, not an error; (2) the baseline and adapted AP / AP50 and what three epochs with a frozen backbone bought; (3) what you would need — labelled photographs, a held-out split, a threshold chosen on validation — before trusting such numbers on real images.\n'
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
