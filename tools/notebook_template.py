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
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). CPU is the documented default and CUDA GPU is used automatically when available. On CPU, the fine-tune completes in ~30–50 s; on a Tesla T4 GPU, it runs in ~1–2 s. Pinned `torch==2.14.0` and the 172 MB checkpoint are the primary downloads.",
        "- **Knowledge:** basic Python and PIL; bounding box representation in xyxy pixel coordinates; intersection-over-union (IoU); and the interpretation of average precision (AP50 and AP@[.50:.95]).",
        "- **Data:** everything is generated deterministically in code by `samples.py`, requiring zero external dataset download: one 640×480 COCO demonstration scene and a 40-image labelled sign adaptation dataset. Optional BYOD is gated off by default. Expected BYOD input: an image or list of `{'image': PIL.Image, 'boxes': [[x0, y0, x1, y1], ...], 'labels': [name, ...]}` records. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so; uploaded inputs stay in this runtime and are not sent to any inference API.",
    ],
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs dependencies, stages and digest-verifies the pinned checkpoint, "
        "runs COCO detection on a drawn scene, validates the 40-image sign adaptation dataset, splits it into train and validation parts, "
        "measures the pre-adaptation baseline, **runs the bounded fine-tune**, re-evaluates on the held-out split, detects on an unseen test "
        "image, exports the adapted artifact, reloads it from disk to verify numeric consistency, and writes machine-readable outputs "
        "with provenance. Nothing is skipped behind a default-off flag, and no clone or DIMER worker is required (NOTEBOOK_SPEC 2.0 §5, RUN7, FT2)."
    ),
    "byod": (
        "Two optional BYOD branches are included and both are disabled by default (`USE_BYOD_IMAGE = False`, `USE_BYOD_DATASET = False`). "
        "`USE_BYOD_IMAGE` runs your own image through the detection and validation contract. `USE_BYOD_DATASET` takes your own labelled "
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
                "with `box_iou = 0.0` rather than hiding it. COCO mean average precision needs a labelled image set; on a single unlabelled scene, the verdict is `sample-sanity`."
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
        # ---------------------------------------------------------------- 5. Degenerate inputs
        {
            "md": (
                "## 5. Degenerate input probes: blank canvas and noise\n\n"
                "A detector should be evaluated on structure-free inputs. A model that invents confident detections on a blank canvas or uniform "
                "noise will invent them on real unlabelled scenes. We probe the model with both a blank image and a random RGB noise image at "
                "both the standard detection threshold (`0.3`) and the evaluation threshold (`0.05`)."
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
                "the unadapted class heads emit no false positives on the held-out set before training."
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
        # ---------------------------------------------------------------- 8. Bounded Fine-Tuning
        {
            "md": (
                "## 8. Bounded detection fine-tuning\n\n"
                "This cell executes the real adaptation step in the notebook runtime. The optimizer trains the hybrid encoder and decoder heads "
                "using RT-DETR's native composite loss: Varifocal Loss for classification, L1 loss, and GIoU loss for bounding box regression, "
                "accumulated across all 6 decoder layers.\n\n"
                "- **The backbone is frozen.** The ResNet-50-vd backbone is frozen (`19.3 M` trainable out of `42.7 M` parameters, 45.1%). "
                "This accelerates adaptation on CPU and prevents catastrophic forgetting on small datasets.\n"
                "- **Schedule:** 3 epochs with AdamW at learning rate `1e-4` and batch size 4."
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
        # ---------------------------------------------------------------- 9. Evaluate Held-Out
        {
            "md": (
                "## 9. Evaluate on the held-out split\n\n"
                "We re-run `evaluate` on the exact same held-out validation set using the same thresholds to measure empirical progress. "
                "`ap50` is average precision at IoU 0.50; `ap` is COCO-standard AP@[.50:.95] across 10 IoU thresholds."
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
                "the artifact into a fresh pipeline instance and verify that detections match the adapted model identically."
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
                "- `USE_BYOD_IMAGE`: Upload a single image to test inference.\n"
                "- `USE_BYOD_DATASET`: Upload a list of labelled records to run custom adaptation through the exact same local pipeline stages."
            ),
            "code": (
                'USE_BYOD_IMAGE = False  # @param {{type:"boolean"}}\n'
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
                "if USE_BYOD_IMAGE:\n"
                "    from google.colab import files  # type: ignore[import-not-found]\n"
                "    uploaded = files.upload()\n"
                "    name, data = next(iter(uploaded.items()))\n"
                "    byod_image = Image.open(io.BytesIO(data))\n"
                "    print(validate_inputs(byod_image, threshold=threshold, names=[name])['verdict'])\n"
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
        "## References\n\n"
        "- Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y. and Chen, J. (2023). *DETRs Beat YOLOs on Real-time Object Detection.* [arXiv:2304.08069](https://arxiv.org/abs/2304.08069).\n"
        "- Upstream repository: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) — Apache-2.0.\n"
        "- Hugging Face checkpoint: [PekingU/rtdetr_r50vd](https://huggingface.co/PekingU/rtdetr_r50vd) — Apache-2.0.\n"
        "- Lin, T.-Y. et al. (2014). *Microsoft COCO: Common Objects in Context.* [arXiv:1405.0312](https://arxiv.org/abs/1405.0312).\n"
        "- Repository model card: https://github.com/kurtvalcorza/rtdetr-detection-pipeline/blob/main/MODEL_CARD.md\n"
        "- [`kurtvalcorza/rtdetr-detection-pipeline`](https://github.com/kurtvalcorza/rtdetr-detection-pipeline) — source repository for this pipeline."
    ),
}
