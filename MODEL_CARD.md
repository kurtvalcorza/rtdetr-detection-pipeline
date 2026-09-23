---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: object-detection
task: "Object Detection"
base_model: PekingU/rtdetr_r50vd
date_published: "2024-05-29"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt` 2024-05-29T01:36:24Z, https://huggingface.co/api/models/PekingU/rtdetr_r50vd — the Transformers-format conversion); the RT-DETR paper is arXiv:2304.08069 (2023-04) and the pinned revision is the Hub's `main` as of 2026-09-14"
---

# RT-DETR R50-VD, COCO — Real-Time Object Detection (Inference and Bounded Fine-Tuning)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PekingU%2Frtdetr__r50vd-ffcc4d?style=flat)](https://huggingface.co/PekingU/rtdetr_r50vd)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-lyuwenyu%2FRT--DETR-181717?style=flat&logo=github&logoColor=white)](https://github.com/lyuwenyu/RT-DETR)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2304.08069-b31b1b.svg)](https://arxiv.org/abs/2304.08069)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — stage and verify the pinned upstream revision in a fresh runtime, validate an input, run the task, adapt the model to custom class vocabularies, and inspect and export the outputs:

- **End-to-End Task & Adaptation Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/rtdetr-detection-pipeline/blob/main/tutorials/rtdetr_detection_colab.ipynb) [`rtdetr_detection_colab.ipynb`](https://github.com/kurtvalcorza/rtdetr-detection-pipeline/blob/main/tutorials/rtdetr_detection_colab.ipynb)  
  *COCO-class detection on a synthetic scene with pinned `PekingU/rtdetr_r50vd` weights, followed by a bounded fine-tune re-heading RT-DETR onto a 3-class traffic sign vocabulary with focal-loss prior bias initialization, pre- vs post-adaptation COCO-style average precision evaluation (AP@[.50:.95], AP50), new-data inference, and `.pt` artifact export/reload.*

---

#### Description

`PekingU/rtdetr_r50vd` is the Transformers-format release of RT-DETR with a ResNet-50-vd backbone from "DETRs Beat YOLOs on Real-time Object Detection" (Zhao et al., arXiv:2304.08069; Baidu / Peking University), converted by the Hugging Face team and pinned here to revision `df939e661d8c52e80608d1ec566561aabd25a4e7` (the Hub's `main` on 2026-09-14). The snapshot `config.json` declares `RTDetrForObjectDetection` with an in-library `RTDetrResNet` backbone (`backbone_config` out features `stage2`–`stage4` at strides 8/16/32, channels 512/1024/2048; `use_timm_backbone` and `use_pretrained_backbone` both false), a one-layer hybrid encoder at hidden size 256 over three feature levels, and a 6-layer deformable-attention decoder (8 heads, feed-forward width 1024) with exactly 300 learned object queries (`num_queries`) and 80 output classes (`id2label`: the COCO 2017 categories in the upstream spelling — `motorbike`, `aeroplane`, `sofa`, `pottedplant`, `diningtable`, `tvmonitor`); the upstream README reports 42M parameters, and the float32 `model.safetensors` is 172 MB. At inference the processor (`RTDetrImageProcessor`, `preprocessor_config.json`) resizes every image to 640×640 without preserving the aspect ratio and rescales it to [0, 1] with no mean/std normalisation; the model emits, per query, a box and a **per-class sigmoid score** (RT-DETR is trained with a focal loss, so the 80 scores of one query are independent, not a softmax), and `post_process_object_detection` keeps the top 300 (query, class) pairs, applies the threshold and maps boxes back to input pixels.

In addition to pretrained COCO inference, this package implements the `E2E` adaptation profile: transfer learning onto custom class vocabularies via `RTDetrDetectionPipeline.from_pretrained(class_names=...)`. The classification heads (`enc_score_head` and decoder `class_embed`) are re-initialized with focal loss prior probability biases (`prior_bias = -math.log((1 - p) / p)` for $p=0.01$, approximately $-4.595$), suppressing random background query flooding before training starts. Fine-tuning (`finetune`) trains the hybrid encoder and decoder while freezing the ResNet-50-vd backbone (19.3M trainable out of 42.7M parameters) using RT-DETR's native composite loss (Varifocal + L1 + GIoU across all 6 decoder layers). The pipeline provides COCO-style average precision evaluation (`evaluate`, `average_precision`), dataset validation (`validate_dataset`), batched inference (`detect_many`), and portable artifact export/reload (`save_artifact`, `load_artifact`) using the `rtdetr-adapter-v1` format.

What this repository adds is packaging: `verify_snapshot` and `stage_missing_files` (manifest digest checking and fresh-clone staging), `RTDetrDetectionPipeline.from_pretrained` (verified local loading with `trust_remote_code=False` and `use_pretrained_backbone=False`), `detect` (input validation, threshold checks, sorted pixel-space output), the `validate_inputs` and `evaluation_report` stage helpers (references keyed by COCO label, matched to same-label detections), `box_iou`, and the complete adaptation suite.

#### Intended Use and Limitations

The uses below are the ones the package was built to support; everything else is either out of scope (§Out-of-scope use cases) or prohibited (§Use cases).

###### Primary Intended Uses

The task is closed-vocabulary object detection: input one image (`PIL.Image.Image`, any mode, converted to RGB) and a threshold; output a list of at most 300 detections, each an xyxy pixel box, a `label` among the 80 COCO classes, and the model's sigmoid `score`, sorted by score. Envisioned applications are everyday-scene detection where the COCO vocabulary fits — people and vehicles in street or traffic imagery, animals, furniture and household objects in indoor imagery, sports and food scenes — as the first stage of counting, cropping, tracking-by-detection or downstream classification, and as the person detector feeding a pose estimator (the sibling ViTPose pipeline). The pipeline is an inference component and a zero-configuration baseline for general object detection, not a certified detector for any specific camera, scene or class.

###### Primary Intended Users

Intended users are machine-learning engineers, computer-vision developers, and data analysts integrating general object detection into research prototypes or in-house tooling. A user is expected to understand that the vocabulary is the 80 COCO classes and nothing else (a forklift, a drone or a logo is either mislabelled or missed), that each score is an independent per-class sigmoid — a ranking signal within one image, not a calibrated probability on their data, and not exclusive across classes for the same query — that the threshold trades recall against spurious boxes and must be tuned per deployment, that the model was trained on COCO photographs so drawn graphics, documents, aerial, medical and infrared imagery are distribution shifts (the tutorial's drawn sports ball is missed), that every image is squashed to 640×640 so tiny objects and extreme aspect ratios suffer, and that precision, recall or mAP can only be measured on a labelled image set they supply. Users who need open-vocabulary prompts, segmentation masks, tracking IDs or the upstream's TensorRT throughput are expected to know none of that is provided here.

###### Out-of-scope use cases

1. **Capability boundary:** no classes outside COCO's 80 (no text prompts; use an open-vocabulary detector), no instance or semantic segmentation, no tracking, no batching or video, no keypoints, and no calibrated confidence. The upstream latency figures (108 FPS on a T4 with TensorRT) are not reproduced; this repository measures CPU wall time only.
2. **Input boundary:** `detect` rejects non-PIL images (`TypeError`), sides below `MIN_IMAGE_SIDE = 16` px or above `MAX_IMAGE_SIDE = 4096` px, and thresholds outside `[0, 1]` (`ValueError`). One image per call; the backend can return at most `MAX_DETECTIONS = 300` (query, class) pairs, and one query can surface under two labels. Every image is resized to 640×640 with the aspect ratio discarded, so objects that are only a few pixels at that scale, and panoramas or tall crops, are handled worse than the COCO evaluation setting suggests.
3. **Input boundary:** the training data is COCO 2017 train — consumer photographs of everyday scenes. Drawn or rendered graphics (the tutorial's icons), documents and screenshots, aerial, satellite, medical, thermal or underwater imagery, night or heavily degraded frames, and objects outside the 80 classes fall outside what the upstream authors evaluated and what this repository measured; results on them are undefined, not merely degraded. An image with no objects at all still produces boxes (see §Risks and harms).
4. **Decision boundary:** not for autonomous decisions that act on detections — vehicle control, security or surveillance alerts, safety interlocks, medical or industrial inspection — without a human reviewing the detections and a locally measured precision/recall on the deployment's own labelled images at the chosen threshold.

#### Factors

###### Groups

This pipeline is human-centric in one respect: `person` is COCO class 0, so the model localises people in any image and its recall and score for people vary with the population — COCO's known skews toward North American and European consumer photography, and the object-detection literature's documented disparities in person detection by skin tone, body size, age, clothing (including religious and cultural dress), mobility aids and lighting apply to any COCO-trained detector and were not audited by this repository or, to our knowledge, for this checkpoint. Neither the upstream authors nor this repository evaluated per-group performance; the fairness of a person-detection deployment is unknown, not known to be equal. Objects also vary by region: vehicle types, furniture, foods and signage outside COCO's collection geography are the classes whose recall is unknown. The operator who detects people is responsible for a per-group audit on their own imagery, stratified by the factors above, before relying on the output.

###### Instrumentation

The upstream training instrument is the consumer camera behind COCO — daytime, colour, mostly well-exposed photographs at web resolution, with human-drawn boxes following COCO annotation conventions (tight around visible extent, one box per instance, crowd regions marked). Inference images arrive from whatever produced them — a phone, a CCTV or dashboard camera, a drone, a rendering engine — and resolution, motion blur, compression, exposure, lens distortion and viewpoint all change the visual evidence; the 640×640 resize discards resolution and distorts aspect on every image regardless of source. The pipeline validates type and size only; it cannot detect a night frame, a fisheye lens, a rendered scene or an image with no objects. The synthetic tutorial scene (Pillow shapes, flat colours, a bundled font) is itself a rendering instrument far from any camera: three of its four icons are detected with tight boxes, the fourth (the sports ball) is not detected at any threshold, and neither outcome says anything about photographs.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, float32 on CPU; CUDA is used automatically when visible but was not exercised for this card. No timm is needed: the backbone is the config's in-library `RTDetrResNet`. The snapshot declares the slow `RTDetrImageProcessor` (transformers prints a `use_fast` notice); it is used as declared. Measured on the reference machine with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`) and the Hub offline (`HF_HUB_OFFLINE=1`): `verify_snapshot` on the 4-file, 172 MB snapshot 0.09 s, load 4.77 s, a 640×480 drawn scene 0.34 s (first call) / 0.25 s (second), a 4096×4096 blank image 0.52 s — cost is dominated by the fixed 640×640 working resolution and the 300-query decoder, not by the caller's pixel count. Data environment: the model assumes a photograph of an everyday scene containing COCO objects; the synthetic scene only partly satisfies that (icons, not photographs) and is where the measured behaviour holds. Real cameras, non-COCO objects, degraded frames and empty scenes violate it to degrees this repository did not measure, and the pipeline reports no signal when they do.

#### Metrics

###### Performance Measures

The pipeline reports no accuracy measure. Each detection carries `score`, the sigmoid output for its class under the model's own focal-loss head — a ranking signal within one image, not a probability that the box is a real object of that class on the deployment's images, not exclusive across classes for the same query, and not a measure of correctness. The repository ships `box_iou(a, b)`, the intersection-over-union of two xyxy boxes, because it is the primitive every detection metric is built from; COCO mean average precision (AP@[.50:.95], AP50, AP75, per-scale AP) and per-class precision/recall are not implemented, since they need a labelled image set with matching conventions that the caller must choose. To evaluate, the caller supplies ground-truth boxes per class and computes precision/recall at their chosen IoU with `box_iou`, or runs the COCO evaluator. The public `evaluation_report(result, ground_truth_boxes=None)` stage returns that report in machine-readable form: one `box_iou` entry per supplied reference box (its best-overlapping detection **of the same label**, that detection's score, and how many same-label detections existed — a reference with none scores 0.0) with the verdict `sample-sanity`, or the verdict `not-measurable` naming the labelled set that would be required when no reference is supplied. The upstream README's COCO val2017 numbers (RT-DETR-R50: AP 53.1, AP50 71.3, AP75 57.7, 108 FPS on a T4 with TensorRT FP16) are upstream-reported and this pipeline does not reproduce or claim them.

###### Decision thresholds

One threshold is applied and exposed as a module constant: `DETECTION_THRESHOLD = 0.3` keeps a (query, class) pair only if its sigmoid score is at least 0.3; below it the pair is discarded, and there is no non-maximum suppression beyond what DETR's set prediction already provides. This is the value the pinned README's transformers example passes to `post_process_object_detection`; it was not tuned by this repository and is not calibrated for any deployment. It can be overridden per call (`detect(..., threshold=)`), and the smoke run shows the synthetic scene is insensitive to it (the same three boxes at 0.1, 0.5 and 0.9, all scores above 0.93; the sports ball absent even at 0.1) while empty inputs are not (a blank image and a noise image each yield one box in the 0.32–0.33 band just above the default). A deployment owns tuning it on its own labelled images: lower the threshold when a missed object costs more than a spurious box, raise it — likely above 0.35 on the evidence of the empty-image probes — when a false detection triggers downstream action, and re-tune whenever the camera, scene or class mix changes.

###### Approaches to uncertainty and variability

This repository reports no central metric value and therefore no dispersion: the smoke run records timings, boxes, scores and per-object IoU on one drawn scene, not accuracy. Run-to-run variability comes only from floating-point kernel selection across CPU builds and accelerators; there is no sampling and no seed to set, so a fixed input on fixed hardware is repeatable but not guaranteed bitwise-identical across machines, and the synthetic scene's own bytes depend on the Pillow build's bundled font. The `score` is a sigmoid, not a calibrated confidence: on the drawn scene the three found icons scored 0.93–0.98 with `box_iou` 0.88–0.98 against the drawn boxes, the drawn sports ball scored below 0.1 for its class, and a blank image produced a `train` at 0.33 — three observations that illustrate uncalibration, not a calibration curve. A caller who needs calibrated confidences must fit a calibration map on their own labelled images; a caller who needs an uncertainty estimate for a metric must supply labelled images and compute it over many images or bootstrap resamples themselves.

#### Ethical considerations and biases

No external ethics board, red-team, or population-specific clearance reviewed this repository or, to our knowledge, the upstream checkpoint; nothing below should be read as implying one.

###### Data

The snapshot README states that the model was trained on COCO 2017 object detection (118k training images, 5k validation images, per the README); the COCO paper describes images collected from Flickr under Creative Commons licences and annotated by crowd workers. COCO contains photographs of identifiable people, licence plates, house numbers and personal environments, so personal data in the training corpus is present by construction and was not audited here; the `person` class exists precisely to localise people. This repository distributes code, tests, and documentation; it does not distribute the 172,175,856-byte `model.safetensors`, which is staged locally under `weights/rtdetr-r50vd/` and git-ignored, and it ships no sample photographs — the tutorial scene is drawn in code. The operator must audit the images they submit for personal, proprietary, or otherwise restricted content; the pipeline performs no such check and will localise a person in a private photograph as readily as a chair.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by this repository, the upstream authors, or any regulator. Foreseeable but unintended sensitive uses — pedestrian detection for vehicle control, person detection for surveillance, crowd counting for policing, weapon-adjacent object detection for security screening, PPE or hazard detection for safety interlocks — would be admissible only with human review of the detections, a locally measured precision/recall on the deployment's own labelled imagery stratified by the groups named above, a documented threshold and re-validation policy, and whatever regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest whose `modelId`/`revision` differ from the package constants and fetches only manifest-listed files at that revision when `allow_download=True`; `verify_snapshot` then checks all 4 listed files' byte sizes and SHA-256 before any load; `from_pretrained` loads only from the verified directory with `local_files_only=True`, always passes `trust_remote_code=False` and `use_pretrained_backbone=False`, and the smoke run loaded with `HF_HUB_OFFLINE=1`. No pickle checkpoint exists at the pinned revision. A test flips one hex digit of a manifest digest and asserts the loader refuses; another asserts a foreign manifest is refused; the import-boundary tests assert that a missing or tampered snapshot is refused before `torch` or `transformers` is imported; another asserts the module's `LABELS` equal the snapshot's `id2label` order.
- **Input integrity:** the public `validate_inputs(image, *, threshold)` stage applies exactly the checks `detect` applies (both route through one shared private checker) and returns an input manifest recording the schema, the ceilings, the observed input and the verdict; `validate_image` rejects non-PIL inputs and sides outside 16–4096 px; thresholds outside `[0, 1]` (and booleans) are rejected; `detect` raises on a malformed backend object, a label outside the 80, or more than 300 detections; `evaluation_report` rejects reference labels outside the 80 classes.
- **Reproducibility:** exact `==` pins in `pyproject.toml`; the processor is used as the snapshot declares it; every result carries `model_id`, `model_revision` and the threshold used.
- **Refusals:** no batching, no download without the explicit flag, no threshold default hidden inside the runner, no pickle deserialisation, no attempt to guess whether the image contains objects.
- No statistical mitigation (class balancing, subsampling) applies: no training happens in this repository.

###### Risks and harms

- **Boxes on empty input:** the model emits boxes for any image — a blank 4096×4096 image returned one `train` at 0.33 and a uniform-noise image one `cat` at 0.32 at the default threshold in the smoke run; an operator who feeds an empty frame gets a confident nonsense detection, and every downstream count or alert built on it is fabricated. The pipeline cannot detect this.
- **Missed objects:** a real object with an unusual appearance, small size or unfamiliar rendering is simply absent (the drawn sports ball); the downstream consumer bears the harm of the miss, and no field of the output flags it.
- **Mislabelling within the closed vocabulary:** anything outside the 80 classes that resembles one is labelled as it (a scooter as `motorbike`, a drone as `bird` or `kite`); a system that acts on labels inherits the error.
- **Person detection and surveillance:** `person` is a first-class output; the pipeline makes people in any image locatable and countable with no consent or purpose check, and its per-group recall is unaudited.
- **Automation bias:** tight boxes with 0.9+ scores invite trust that an uncalibrated sigmoid has not earned.
- **Aspect distortion:** the 640×640 squash silently degrades tall or wide images and small objects.
- **Bias amplification:** any scene, object or population COCO under-represents is reproduced as uneven recall, undetected because no per-group evaluation exists.
- **Resource use:** small model (172 MB, ~0.3 s per image on the reference CPU); a video stream can still saturate a shared host.

###### Use cases

Prohibited even where the model would work: detecting people in order to surveil, track, profile, or score them, or to enable unlawful discrimination in employment, housing, credit, insurance, education, healthcare access or law enforcement; processing imagery the operator has no right to process, or in breach of consent, privacy or data-protection obligations; deceptive uses that present detections as verified facts or as evidence; autonomous physical control or safety interlocks based on unreviewed detections; and any use that violates the upstream Apache-2.0 licence terms or the terms of the deployment that runs the pipeline. Autonomous high-consequence actions triggered by unreviewed detections are prohibited by the intended-use contract above.

## Immutable provenance

- Model: `PekingU/rtdetr_r50vd`
- Revision: `df939e661d8c52e80608d1ec566561aabd25a4e7`
- Snapshot manifest: `weights/rtdetr-r50vd/dimer-base-manifest.json`, 4 files, `totalBytes` 172190863
- `model.safetensors` SHA-256: `5263d5521eff3e356f6cd8a371fd5dfb891725beda5f713674f79669115cdc64` (172,175,856 bytes, float32)
- `config.json` SHA-256: `2ed2a305c51eef46715eb755a02b2a266ecfb752936cc9574bb5714601c2742d` (5,113 bytes; `RTDetrForObjectDetection`, 300 queries, 80 labels, in-library `RTDetrResNet` backbone)
- `preprocessor_config.json` SHA-256: `ffb4b9461a1dad746be8f0f9c8330ed7743a1ba5fba4f75c232cd281b3d4c64a` (841 bytes; `RTDetrImageProcessor`, 640×640, rescale only)
- Weight format: SafeTensors; loader `RTDetrForObjectDetection.from_pretrained(<dir>, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False, use_pretrained_backbone=False)` with `AutoImageProcessor` from the same directory. No pickle checkpoint exists at this revision.

## Input/output contract

- `RTDetrDetectionPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False, class_names=None, seed=0)` — stages missing manifest files (only with `allow_download=True`), verifies digests, loads; optionally re-heads onto custom `class_names` with focal-loss prior bias initialization (`HEAD_PRIOR_PROB = 0.01`, bias `-4.595`).
- `detect(image, *, threshold=0.3) -> dict` with keys `detections` (list of `{"box": [x0, y0, x1, y1], "label": <class_name>, "score": float}` in input-pixel coordinates, sorted by descending score, at most 300 entries), `threshold`, `width`, `height`, `model_id`, `model_revision`.
- `detect_many(images, *, threshold=0.3, batch_size=4) -> list[dict]` — batched inference over multiple images.
- `finetune(records, *, epochs=3, batch_size=4, learning_rate=1e-4, seed=0, freeze_backbone=True, progress=None) -> dict` — bounded detection fine-tuning with RT-DETR composite loss (Varifocal + L1 + GIoU across 6 decoder layers).
- `evaluate(records, *, threshold=0.05, max_detections=100, iou_thresholds=(0.50, ... 0.95)) -> dict` — COCO-style average precision evaluation returning `ap`, `ap50`, `ap75`, `per_class_ap50`.
- `save_artifact(path, *, notes="") -> dict` — exports fine-tuned adapter weights to standalone `.pt` file.
- `load_artifact(path, *, weights_dir=None, device=None) -> RTDetrDetectionPipeline` — restores fine-tuned pipeline with full weights-only deserialization safety.
- Ceilings and constants: `MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_DETECTIONS = 300`, `LABELS` (80 COCO classes in `id2label` order), `DETECTION_THRESHOLD = 0.3`, `EVAL_DETECTION_THRESHOLD = 0.05`, `ARTIFACT_FORMAT = "rtdetr-adapter-v1"`, `HEAD_PRIOR_PROB = 0.01`.
- `box_iou(a, b) -> float` on xyxy boxes; `validate_inputs(image, *, threshold, names) -> dict`; `validate_dataset(records, class_names, epochs=1) -> dict`; `average_precision(predictions, references, class_names, ...) -> dict`; `evaluation_report(result, ground_truth_boxes=None, *, sample_kind) -> dict`; `verify_snapshot(path=None) -> dict`; `stage_missing_files(path=None, *, allow_download=False, downloader=None) -> list[str]`.

## Runtime

- Pins: `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, `huggingface-hub==0.36.2`; Python 3.12.
- Precision: float32; preprocessing resize to 640×640 (aspect ratio not preserved), rescale to [0, 1], no mean/std normalisation (`RTDetrImageProcessor`, snapshot defaults, slow processor as declared).
- Measured 2026-09-14 in the Windows venv (`torch 2.14.0+cu130`) with `CUDA_VISIBLE_DEVICES=-1` and `HF_HUB_OFFLINE=1`, device `cpu`: `verify_snapshot` 0.09 s (4 files, 172 MB); load 4.77 s; `detect` on a synthetic 640×480 scene (sky and ground, a red octagonal stop sign with the word STOP, a three-lamp traffic light, a white analogue clock with numerals and hands, an orange sports ball with seams, drawn with Pillow) at the default threshold 0.3 → 3 detections in 0.34 s (0.25 s on the second call): `stop sign` [52.0, 92.3, 168.1, 208.5] 0.977, `clock` [410.4, 70.6, 550.7, 210.7] 0.965, `traffic light` [269.8, 60.6, 322.9, 211.5] 0.931; `box_iou` against the drawn boxes: stop sign 0.877, traffic light 0.965, clock 0.983, sports ball 0.000 (no `sports ball` detection); identical three boxes at thresholds 0.1, 0.5 and 0.9; 4096×4096 blank image → 1 detection (`train` 0.333) in 0.52 s; 640×480 uniform noise (NumPy default generator, seed 0) → 1 detection (`cat` 0.319).
- Fine-tuning performance: 19,268,260 trainable parameters out of 42,674,089 total (45.1%) with ResNet-50-vd backbone frozen. 3 epochs on 30 training images (batch size 4) executes in ~30–50 s on Intel CPU and ~1–2 s on Tesla T4 GPU.
- Tutorial execution: `tutorials/rtdetr_detection_colab.ipynb` runs top-to-bottom in a fresh local or hosted kernel with full adaptation and artifact reload; recorded in `docs/release-verification.md`.
- Tests: `pytest -q -o addopts= tests` — 31 tests offline in ~0.4s, no weights required; `ruff check src tests tools` clean.
- Not executed: half precision, the fast image processor, published COCO test-dev benchmarks (the smoke and tutorial use synthetic drawn data), crowded scenes.

## References

- Zhao et al. DETRs Beat YOLOs on Real-time Object Detection. CVPR 2024. https://arxiv.org/abs/2304.08069
- Lin et al. Microsoft COCO: Common Objects in Context. ECCV 2014. https://arxiv.org/abs/1405.0312
- Carion et al. End-to-End Object Detection with Transformers (DETR). ECCV 2020. https://arxiv.org/abs/2005.12872
- Upstream code: https://github.com/lyuwenyu/RT-DETR
- Upstream card: https://huggingface.co/PekingU/rtdetr_r50vd
- Transformers `RT-DETR` documentation: https://huggingface.co/docs/transformers/model_doc/rt_detr
