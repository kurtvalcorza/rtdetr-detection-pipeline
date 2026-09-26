# DIMER Workshop Specification: Comparing Closed-Set Object Detectors

**Status:** Proposed  
**Notebook specification:** DIMER `NOTEBOOK_SPEC` **2.1**  
**Notebook profile:** `E2E`  
**Pedagogical mode:** `WORKSHOP`  
**Proposed filename:** `DIMER_MultiModel_Closed_Set_Object_Detection_Workshop.ipynb`  
**Canonical runtime:** NVIDIA Tesla T4 or equivalent  
**Canonical execution:** standalone, credential-free, top-to-bottom `Run all`

---

# 1. Purpose

This workshop compares modern **closed-vocabulary object detectors** on exactly the same downstream detection problem.

Canonical models:

| Model | Detection family | Canonical role |
|---|---|---|
| **RT-DETR R50-VD** | transformer / set prediction | core |
| **YOLOX-S** | anchor-free dense detector | core |
| **YOLOX-X** | large anchor-free dense detector | optional `FULL` tier |

The workshop asks:

> Given the same labelled images, same three classes, same boxes, same held-out split, and same evaluator, how do a DETR-style set detector and an anchor-free YOLO detector differ in adaptation quality, localization accuracy, computational footprint, and inference behavior?

It MUST NOT present the results as a universal architecture ranking.

---

# 2. Notebook profile

The notebook SHALL declare:

**Profile:** `E2E`  
**Mode:** `WORKSHOP`

The canonical workflow includes actual supervised detection adaptation:

```text id="06o25o"
pretrained detector
→ custom three-class re-head
→ pre-adaptation evaluation
→ bounded fine-tuning
→ validation inspection
→ freeze experiment
→ independent test
→ artifact export
→ fresh reload
→ new-image inference
```

No core model is inference-only.

---

# 3. Core models

## 3.1 RT-DETR R50-VD

**Model:** `PekingU/rtdetr_r50vd`  
**Revision:**  
`df939e661d8c52e80608d1ec566561aabd25a4e7`

License:

**Apache-2.0**

Checkpoint:

```text id="bceo6m"
model.safetensors
172,175,856 bytes
SHA-256:
5263d5521eff3e356f6cd8a371fd5dfb891725beda5f713674f79669115cdc64
```

Preprocessor:

```text id="366op9"
preprocessor_config.json
841 bytes
SHA-256:
ffb4b9461a1dad746be8f0f9c8330ed7743a1ba5fba4f75c232cd281b3d4c64a
```

Approximate parameters:

**42.7M**

Architecture:

```text id="33m52z"
ResNet-50-vd backbone
→ hybrid encoder
→ 6-layer deformable transformer decoder
→ 300 learned object queries
```

Native pretrained vocabulary:

**80 COCO classes**

---

# 4. YOLOX-S

**Model family:** `Megvii-BaseDetection/YOLOX`  
**Release tag:** `0.1.1rc0`  
**Upstream code revision:**  
`419778480ab6ec0590e5d3831b3afb3b46ab2aa3`

Checkpoint:

```text id="xfxwto"
yolox_s.pth
72,089,125 bytes
SHA-256:
f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30
```

License:

**Apache-2.0**

Parameters:

**8,968,255**

Architecture:

```text id="0yn0lr"
CSPDarknet
→ PAFPN
→ decoupled anchor-free head
→ strides 8 / 16 / 32
```

At 640×640:

```text id="jqgnm7"
80² + 40² + 20² = 8,400 candidate locations
```

---

# 5. Optional YOLOX-X

`FULL` mode SHALL optionally add:

**YOLOX-X**

Checkpoint:

```text id="5ur6tx"
yolox_x.pth
793,463,373 bytes
SHA-256:
5652330b6ae860043f091b8f550a60c10e1129f416edfdb65c259be6caf355cf
```

Parameters:

**99,071,455**

Architecture scale:

```text id="3wyfp4"
depth = 1.33
width = 1.25
```

This enables a second question:

> Does dramatically increasing YOLOX model size improve this small transfer-detection task enough to justify its additional compute and artifact footprint?

YOLOX-X MUST NOT be required for the canonical default path.

---

# 6. Execution tiers

Recommended control:

```python id="fdlf3z"
WORKSHOP_TIER = "STANDARD"  # @param ["STANDARD", "FULL"]
```

## STANDARD

Runs:

- RT-DETR R50-VD
- YOLOX-S

This is the canonical release `Run all`.

## FULL

Runs:

- RT-DETR R50-VD
- YOLOX-S
- YOLOX-X

Both require separate clean-runtime qualification.

---

# 7. Why Swin-T Mask R-CNN is excluded from the core

The live:

**Swin-T + Mask R-CNN**

pipeline SHOULD NOT be included in the canonical comparison.

Reasons:

1. its current DIMER release is `TASK-INFERENCE`, not a completed adaptation carrier;
2. its gradient-adaptation path remains scaffolded;
3. it requires a separate OpenMMLab / CPython 3.10 / MMCV runtime;
4. the underlying architecture includes instance masks even though the DIMER output exposes boxes;
5. it cannot currently be adapted to the same three-class fixture under the same release-grade contract.

It MAY appear in an optional conceptual section:

> Other closed-set detector families in the DIMER fleet

but MUST NOT enter the adapted-model result table.

---

# 8. Common downstream task

All core models are adapted to the same custom vocabulary:

```text id="cr9bjv"
stop-sign
yield-sign
speed-limit-sign
```

These exact hyphenated names intentionally form a new target vocabulary rather than directly reusing COCO label strings.

Even though COCO contains `"stop sign"`, the custom task's `"stop-sign"` is treated as a separate class head.

This ensures the workshop genuinely demonstrates adaptation.

---

# 9. Canonical dataset

Reuse the existing deterministic sign-data formulation already implemented in both the RT-DETR and YOLOX pipelines.

Canonical dataset size:

**60 images**

Image size:

```text id="03qafr"
640 × 640 RGB
```

Generator seed:

```text id="uqhmne"
0
```

Maximum objects per scene:

```text id="90k8fj"
3
```

Every scene contains:

```text id="o9wqbt"
1–3 traffic signs
```

drawn on a varied synthetic road/background scene.

---

# 10. Why increase from the existing 40-image fixtures

The individual model notebooks currently use:

```text id="244mqd"
40 images
→ 30 train
→ 10 held out
```

That is sufficient for an individual E2E smoke/tutorial.

A comparative workshop needs:

- training;
- validation; and
- a genuinely independent test set.

Therefore the common workshop SHOULD generate:

```text id="u3qe5c"
60 deterministic images
```

and divide them into:

| Split | Images |
|---|---:|
| Train | **36** |
| Validation | **12** |
| Test | **12** |

The test images MUST remain untouched until the experiment is frozen.

---

# 11. Deterministic split

Recommended split seed:

```text id="ysr984"
42
```

Procedure:

```text id="hgb3a9"
generate 60 scenes with generator seed 0
→ deterministic permutation seed 42
→ 36 train
→ 12 validation
→ 12 test
```

The notebook MUST verify that every split contains at least one instance of every class.

If class coverage fails, the canonical fixture definition itself MUST be changed and repinned rather than performing an ad hoc runtime resplit.

---

# 12. Dataset record

Each record:

```text id="xa8z7u"
{
  "id": string,
  "image": PIL.Image,
  "boxes": [
    [x0, y0, x1, y1],
    ...
  ],
  "labels": [
    "stop-sign",
    ...
  ],
  "split": "train" | "validation" | "test"
}
```

Bounding boxes are exact by construction.

---

# 13. Why 640×640 is particularly useful

The common fixture is already:

**640×640**

This controls a major preprocessing confound.

RT-DETR normally:

```text id="onuv8v"
squash-resizes input to 640×640
```

YOLOX normally:

```text id="ci89iq"
letterboxes input to 640×640
```

For the canonical fixture, both transformations are effectively spatial no-ops.

Therefore:

> Both models see the same source pixels at the same working geometry.

This makes the built-in comparison substantially cleaner than a comparison on arbitrary aspect ratios.

---

# 14. Dataset generation contract

The canonical generator SHOULD retain the existing DIMER semantics:

```text id="9q9l9i"
background colour randomized within bounded range
road/background region
six possible non-overlapping object slots
1–3 selected slots
bounded x/y jitter
sign radius 46–70 px
```

Sign types:

### Stop sign

- red octagon;
- white border/text;
- exact circumscribed box.

### Yield sign

- triangular sign;
- exact reference box.

### Speed-limit sign

- rectangular sign;
- speed values selected from:

```text id="x0ps7y"
30
50
60
80
```

Every box MUST remain fully inside the image.

---

# 15. Data manifest

Before model acquisition, export:

```text id="739lpx"
outputs/data/dataset_manifest.json
```

containing:

```text id="z8u5oa"
generator_version
generator_seed
split_seed
image_size
class_names
n_images
n_boxes
boxes_per_class
split_image_ids
split_counts
split_box_counts
pixel_sha256_per_image
annotation_sha256
```

This makes the synthetic dataset itself a pinned experiment artifact.

---

# 16. Validation

Before loading any detector, check:

- image is RGB-convertible;
- width/height positive;
- canonical sample exactly 640×640;
- IDs unique;
- boxes finite;
- boxes use xyxy;
- `x1 > x0`;
- `y1 > y0`;
- boxes within image bounds;
- labels belong to declared class vocabulary;
- box/label counts agree;
- every class occurs in train;
- every class occurs in validation;
- every class occurs in test;
- no identical image digest crosses splits.

Invalid records MUST fail explicitly.

---

# 17. Object-detection primer

Before execution explain the output:

```text id="sd2tly"
bounding box
class label
score
```

and distinguish:

### Classification

```text id="b55661"
one image → one label
```

from:

### Detection

```text id="u4q7cd"
one image → zero or more objects
```

Each predicted object has:

- location;
- class;
- model score.

---

# 18. Architecture primer — RT-DETR

Explain the set-prediction formulation:

```text id="jrzuv5"
image
→ CNN backbone
→ multi-scale features
→ transformer decoder
→ learned object queries
→ boxes/classes
```

RT-DETR predicts a bounded set of objects through learned queries rather than scanning anchor locations followed by NMS.

Its class scores are independent per-class sigmoid outputs.

---

# 19. Architecture primer — YOLOX

Explain:

```text id="6pab6g"
image
→ CSPDarknet
→ PAFPN
→ dense multi-scale candidate locations
→ objectness + class + box
→ NMS
```

YOLOX is:

- anchor-free;
- dense;
- objectness-based;
- NMS-based.

This contrast is central to the workshop.

---

# 20. Score semantics

Raw scores MUST NOT be directly interpreted as calibrated probabilities.

RT-DETR score:

> independent class sigmoid from the focal-loss detection head.

YOLOX score:

\[
score =
sigmoid(objectness)
\times
sigmoid(class)
\]

Therefore:

> A score of 0.8 from RT-DETR is not numerically equivalent to a score of 0.8 from YOLOX.

The workshop MUST compare geometric/detection metrics rather than raw score averages.

---

# 21. Common baseline — empty detector

The primary task baseline:

```text id="1z37nb"
predict no boxes
```

Expected:

```text id="pp6dz9"
AP = 0
AP50 = 0
AP75 = 0
```

This verifies that the common evaluator behaves correctly.

---

# 22. Pre-adaptation baseline

Each detector is re-headed onto:

```text id="9rwygc"
stop-sign
yield-sign
speed-limit-sign
```

before training.

Evaluate this random/new classification head on **validation**.

This produces a more informative transfer baseline:

> How much detection structure transfers from the COCO pretrained model before the new class head has learned the target vocabulary?

For YOLOX, box/objectness layers retain pretrained information.

For RT-DETR, pretrained backbone/decoder geometry similarly provides reusable localization structure.

---

# 23. Common evaluation thresholds

To make the comparison fair, the common workshop evaluator SHOULD retain detections with:

```text id="1476id"
score >= 0.01
```

for AP calculation.

Maximum predictions:

```text id="5okzua"
100 per image
```

This deliberately favors recall during AP calculation.

For visualization only:

```text id="tmhp1a"
display threshold = 0.30
```

The visualization threshold MUST NOT affect AP.

---

# 24. YOLOX NMS

YOLOX retains its model-native NMS:

```text id="f3hm07"
NMS IoU threshold = 0.65
```

for evaluation.

RT-DETR uses no conventional NMS because set prediction is part of its architecture.

This difference MUST remain visible rather than forcing artificial NMS onto RT-DETR.

---

# 25. Common evaluator

The workshop MUST use one model-neutral evaluator rather than taking each repository's reported AP table at face value.

Inputs:

```text id="k5iw0v"
ground truth:
  image_id
  boxes
  labels

predictions:
  image_id
  boxes
  labels
  scores
```

Per class:

1. sort detections by descending score;
2. match each prediction to the highest-IoU unclaimed same-class reference;
3. count TP/FP for each IoU threshold;
4. calculate interpolated precision-recall.

---

# 26. COCO-style metrics

IoU thresholds:

```text id="wd0u9i"
0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
0.90
0.95
```

Report:

### AP

Mean AP across IoU thresholds 0.50–0.95.

### AP50

AP at IoU 0.50.

### AP75

AP at IoU 0.75.

### Per-class AP50

For:

```text id="a0v055"
stop-sign
yield-sign
speed-limit-sign
```

---

# 27. Additional workshop metrics

Also calculate:

- mean detections/image;
- precision@IoU50 at display threshold 0.30;
- recall@IoU50 at display threshold 0.30;
- mean matched IoU;
- false positives/image;
- missed objects/image.

These make the AP result easier to interpret.

---

# 28. Common adaptation principle

Unlike the image-classification workshop, the two detector architectures do **not** share one sensible optimization algorithm.

The workshop MUST therefore preserve **model-native bounded adaptation**.

Fairness is established through common:

- images;
- labels;
- boxes;
- split;
- input geometry;
- seed where applicable;
- test evaluator.

It is **not** established by forcing both networks to use the same loss or optimizer.

---

# 29. RT-DETR adaptation contract

Canonical recipe:

```text id="aexxbd"
epochs = 3
batch size = 4
learning rate = 1e-4
optimizer = AdamW
weight decay = 1e-4
freeze ResNet backbone = true
```

Trainable approximately:

**19.3M of 42.7M parameters**

Training objective:

RT-DETR native composite objective:

```text id="fknez6"
Varifocal classification
+ L1 box loss
+ GIoU
```

across its decoder layers.

The custom class heads use focal-loss prior initialization:

```text id="xcy15a"
prior probability = 0.01
bias ≈ -4.595
```

---

# 30. YOLOX-S adaptation contract

Canonical existing DIMER recipe:

```text id="2puvtg"
epochs = 6
batch size = 2
learning rate = 1e-3
optimizer = SGD
momentum = 0.9
weight decay = 5e-4
freeze backbone = true
use_l1 = false
```

Trainable approximately:

**1.89M parameters**

YOLOX backbone + PAFPN remain frozen.

Training objective:

```text id="04sx1f"
SimOTA assignment
+ IoU box loss
+ objectness loss
+ classification loss
```

---

# 31. YOLOX-X adaptation contract

In `FULL` mode use the same YOLOX adaptation policy:

```text id="0ogw50"
epochs = 6
batch size = 2
learning rate = 1e-3
freeze backbone = true
```

Trainable approximately:

**11.79M parameters**

The workshop MUST NOT assume that a larger model should outperform YOLOX-S on this small synthetic transfer problem.

---

# 32. Why hyperparameters differ

The notebook MUST explain:

> RT-DETR and YOLOX solve detection with different parameterizations, label-assignment procedures and training losses. Using the same learning rate and optimizer would not make the comparison scientifically fair; it would simply disadvantage one or both models.

Therefore the workshop compares:

> **qualified DIMER detector systems under their established bounded adaptation recipes**

rather than claiming to isolate architecture alone.

---

# 33. Fixed recipe before test

The model-specific recipes are declared before any validation/test result appears.

Validation MAY be used to diagnose:

- under-training;
- class failure;
- divergence;
- obvious implementation defects.

The workshop MUST NOT change the default recipe based on test results.

---

# 34. Validation stage

After adaptation, evaluate all selected detectors on the same **12-image validation split**.

Show:

| Model | Pre AP | Post AP | Post AP50 | Post AP75 | Precision@50 | Recall@50 |
|---|---:|---:|---:|---:|---:|---:|

This is a diagnostic table.

Do not call it the final comparison.

---

# 35. Per-class validation

Report:

| Model | Stop AP50 | Yield AP50 | Speed-limit AP50 |
|---|---:|---:|---:|

A high aggregate AP MUST NOT hide a class that never gets detected.

---

# 36. Freeze-before-test

After validation, create:

```text id="8vjl7m"
outputs/frozen/frozen_experiment.json
```

containing:

```text id="hm852r"
dataset digest
split IDs
class order

model identities
model revisions/tags
weight digests

RT-DETR adaptation recipe
YOLOX adaptation recipe

evaluation score threshold
YOLOX NMS threshold
max detections
IoU thresholds
metric implementation revision

artifact digests
validation results
```

No detector setting may change after this point.

---

# 37. Independent held-out test

Final test:

**12 previously unopened images**

Run:

1. empty detector;
2. pre-adaptation RT-DETR;
3. adapted RT-DETR;
4. pre-adaptation YOLOX-S;
5. adapted YOLOX-S;
6. optional pre/adapted YOLOX-X.

The pre-adaptation predictions SHOULD be cached before training or generated from freshly reconstructed re-headed models with the same seeded initialization.

---

# 38. Principal test table

Core table:

| Model | Params | Trainable | AP | AP50 | AP75 | Precision@50 | Recall@50 | ms/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Empty | — | — | 0 | 0 | 0 | — | 0 | — |
| RT-DETR pre | 42.7M | — | | | | | | |
| RT-DETR adapted | 42.7M | 19.3M | | | | | | |
| YOLOX-S pre | 9.0M | — | | | | | | |
| YOLOX-S adapted | 9.0M | 1.89M | | | | | | |
| YOLOX-X adapted | 99.1M | 11.79M | | | | | | |

YOLOX-X rows appear only in `FULL`.

---

# 39. Do not produce an overall winner

The notebook MUST NOT calculate a composite score.

Students should compare dimensions such as:

- AP;
- AP75/localization;
- recall;
- runtime;
- checkpoint size;
- trainable parameter count;
- artifact size.

The correct engineering choice depends on constraints.

---

# 40. Architecture-level questions

The workshop SHOULD highlight:

### RT-DETR

- fixed query set;
- no conventional NMS;
- transformer decoder;
- relatively large trainable adaptation surface.

### YOLOX

- 8,400 dense locations;
- explicit objectness;
- per-class NMS;
- lightweight head-only adaptation.

---

# 41. Localization analysis

AP50 can saturate on an easy synthetic task.

Therefore AP@[.50:.95] and AP75 SHOULD remain primary.

Ask:

> Do two detectors that both reach AP50 ≈ 1.0 localize boxes equally well?

Higher-IoU thresholds reveal localization differences.

---

# 42. Object-size diagnostic

The synthetic generator already varies sign radius.

Divide ground-truth boxes into relative-size groups:

```text id="7ghfh1"
small
medium
large
```

based on box area as a fraction of image area.

Report AP50 or recall by size as a workshop diagnostic.

Do not call these COCO small/medium/large categories unless using COCO's exact area thresholds.

---

# 43. Object-density diagnostic

Group scenes by:

```text id="s5zqsn"
1 object
2 objects
3 objects
```

Compare:

- recall;
- false positives;
- AP50 if sample counts permit.

This illustrates whether crowding/object count affects detector behavior.

---

# 44. Confusion analysis

Build a class-level confusion summary using the best same-box/same-threshold assignment.

Inspect:

- missed stop signs;
- yield → speed-limit confusion;
- speed-limit → stop confusion;
- false boxes with no matched ground truth.

Detection confusion must include a:

```text id="zqo8as"
background / unmatched
```

category.

---

# 45. Deterministic error gallery

Choose images using fixed rules:

1. highest AP image for both;
2. lowest AP image for both;
3. largest RT-DETR advantage;
4. largest YOLOX-S advantage;
5. largest disagreement in number of detections;
6. one scene with three objects.

Do not hand-pick examples after looking at them.

---

# 46. Detection visualization

For each selected image show:

```text id="b0uvne"
ground-truth boxes
RT-DETR boxes
YOLOX-S boxes
```

Use consistent:

- class names;
- box labels;
- score display;
- coordinate system.

For `FULL`, YOLOX-X MAY use a separate panel to avoid overcrowding.

---

# 47. Confidence-score exercise

Pick matched detections from both models.

Ask:

> If RT-DETR reports 0.90 and YOLOX reports 0.90, are these equivalent?

Answer:

**No.**

Their score-generating mechanisms differ and neither is calibrated on this downstream task.

---

# 48. Threshold sensitivity

Using **validation only**, plot precision/recall across display thresholds such as:

```text id="5c0pkw"
0.05
0.10
0.20
0.30
0.50
0.70
```

Do this separately for each architecture.

The AP evaluator still uses the fixed low evaluation cutoff.

The workshop MUST NOT choose an "optimal" threshold from test.

---

# 49. Pretrained COCO demo

A short optional/introduction stage MAY show each original detector before re-heading on a common synthetic COCO-style scene.

Purpose:

> Observe what the original COCO detector knows before transferring it to a new vocabulary.

These results MUST NOT enter the custom three-class leaderboard.

---

# 50. Supply-chain handling

## RT-DETR

Uses SafeTensors directly.

Verify:

```text id="jo4dp4"
model.safetensors
172,175,856 bytes
5263d552…
```

before loading.

No remote code.

---

# 51. YOLOX source weights

YOLOX upstream publishes `.pth`.

The standalone workshop SHOULD preserve the DIMER trust boundary:

```text id="cizvjt"
download exact release asset
→ byte-size check
→ SHA-256 check
→ torch.load(weights_only=True)
→ strict state-dict load
```

It MUST NOT use:

```text id="rhjrxa"
torch.hub.load_state_dict_from_url
```

without digest verification.

---

# 52. YOLOX standalone implementation

The workshop cannot rely on repository-local DIMER source.

Therefore its cells MUST carry the minimal qualified YOLOX architecture implementation required for:

- YOLOX-S;
- YOLOX-X in FULL;
- inference;
- SimOTA adaptation;
- artifact reconstruction.

The carried code SHOULD be source-bound to upstream:

```text id="37uhxb"
419778480ab6ec0590e5d3831b3afb3b46ab2aa3
```

with any DIMER-specific edits explicitly documented.

No `pip install yolox` dependency should be introduced merely to shorten the notebook.

---

# 53. Shared runtime

A common union environment is practical:

```text id="7fh8r3"
Python 3.12
torch==2.14.0
torchvision==0.29.0
torchaudio==2.11.0
transformers==4.57.6
safetensors==0.8.0
numpy==2.5.3
pillow==11.3.0
huggingface-hub==0.36.2
```

RT-DETR uses Transformers.

YOLOX uses the carried PyTorch architecture and requires no Transformers API.

One environment SHOULD be preferred if clean qualification confirms no conflict.

---

# 54. Model-memory strategy

Train/evaluate one detector at a time:

```text id="5mg11k"
load model
→ pre-adaptation predictions
→ fine-tune
→ validation predictions
→ export artifact
→ unload
→ clear CUDA
→ next model
```

Test predictions happen only after freeze.

YOLOX-X SHOULD be loaded last.

---

# 55. Artifact export

Preserve model-specific formats.

### RT-DETR

```text id="d0va54"
rtdetr-adapter-v1
```

### YOLOX

```text id="mh45ds"
dimer-yolox-detection-adapter/1
```

Artifacts MUST record:

- base model identity;
- base model revision/tag;
- architecture variant;
- custom class order;
- input geometry;
- adaptation request;
- tensor/state identity;
- file digest.

---

# 56. Fresh reload

For each selected detector:

```text id="bvvcdv"
destroy in-memory adapted detector
→ verify artifact
→ reconstruct fresh detector
→ load artifact through safe loader
→ rerun fixed validation/test image
→ compare boxes, labels, scores
```

Require:

- identical class ordering;
- same detection count;
- same labels;
- box parity within tolerance;
- score parity within tolerance.

Also recompute test AP after fresh reload.

The metric SHOULD match the pre-reload test result.

---

# 57. New-data inference

Generate a small set using a **new generator seed not present in train/val/test**.

Recommended:

```text id="vr0tso"
seed = 2026
3 images
```

No part of these scenes enters training or model selection.

Show adapted detections side-by-side.

Because their ground truth is known from the generator, the notebook MAY report same-label IoU as a final sanity check, but these images MUST NOT be folded into the main test metrics.

---

# 58. Computational tradeoffs

Measure:

- checkpoint bytes;
- parameters;
- trainable parameters;
- re-head time;
- training wall time;
- inference ms/image;
- peak VRAM;
- artifact bytes.

Suggested table:

| Model | Weights | Params | Trainable | Train time | ms/image | Artifact |
|---|---:|---:|---:|---:|---:|---:|
| RT-DETR R50 | 172 MB | 42.7M | ~19.3M | | | |
| YOLOX-S | 72 MB | 9.0M | ~1.89M | | | |
| YOLOX-X | 793 MB | 99.1M | ~11.79M | | | |

No derived "efficiency winner" is required.

---

# 59. Why YOLOX-X belongs in FULL

YOLOX-X is:

- ~11× larger than YOLOX-S by parameters;
- ~11× larger by checkpoint bytes;
- materially slower;
- trained with the same detection formulation.

Therefore it provides a clean **scale comparison** inside one architecture family.

The workshop can ask:

> On a small simple transfer task, does bigger actually help?

Existing individual pipeline evidence suggests that this is not guaranteed.

The workshop MUST recompute the answer on its new independent split.

---

# 60. BYOD format

Preferred:

```text id="ut1q8t"
dataset.zip
├── annotations.csv
└── images/
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

`annotations.csv`:

```text id="yytvuk"
image_id,file,label,x0,y0,x1,y1,split
img001,images/image001.jpg,class_a,10,20,100,150,train
img001,images/image001.jpg,class_b,200,50,300,180,train
...
```

Multiple rows per image are expected.

---

# 61. BYOD split

If the `split` column is supplied, accept:

```text id="8k8c5z"
train
validation
test
```

and preserve it.

This is preferred.

If no split is supplied, perform a deterministic image-level:

```text id="fvn7bx"
60% train
20% validation
20% test
seed 42
```

Then verify that every declared class occurs in every split.

If class coverage fails:

> refuse and ask the user to supply explicit splits or more data.

Do not silently move individual boxes between splits.

---

# 62. BYOD validation

Validate:

- ZIP traversal/symlink safety;
- file exists;
- image decodes;
- image ID unique;
- 2–100 classes;
- finite boxes;
- boxes inside image bounds;
- nonzero box area;
- at least one box per labelled image;
- labels non-empty;
- same image not in multiple splits;
- every class represented in training;
- evaluation classes represented in validation/test.

---

# 63. BYOD preprocessing caveat

Unlike the canonical 640×640 synthetic fixture, arbitrary BYOD images expose different model-native geometry:

### RT-DETR

squash to:

```text id="61dhq7"
640×640
```

### YOLOX

preserve aspect ratio and letterbox to:

```text id="ojmd1r"
640×640
```

Therefore BYOD comparisons are comparisons of the **complete deployed detector systems**, not architecture-only controlled comparisons.

The notebook MUST state this explicitly.

---

# 64. BYOD E2E path

Because the notebook profile is `E2E`, BYOD MUST execute:

```text id="o9trkt"
load
→ validate
→ split
→ custom re-head
→ pre-adaptation evaluation
→ model-native fine-tuning
→ validation
→ freeze
→ independent test
→ artifact export
→ fresh reload
→ new-image inference
```

It must not degrade into inference-only BYOD.

---

# 65. BYOD privacy guidance

The notebook MUST state:

> User-supplied images and annotations are processed inside the selected notebook runtime and are not submitted to a DIMER worker or API. A hosted notebook remains an external compute environment. Do not upload confidential, biometric, surveillance, personal, security-sensitive, restricted, or proprietary imagery unless authorized.

---

# 66. Interpretation boundaries

### Synthetic task

The built-in signs are drawn graphics, not real traffic-sign photographs.

High AP means:

> the adaptation workflow learned this controlled rendering domain.

It does NOT establish real traffic-sign performance.

### Tiny test set

Twelve test images produce high statistical uncertainty.

No general detector superiority may be inferred.

### Different adaptation algorithms

RT-DETR and YOLOX preserve their model-native training procedures.

The experiment therefore compares **qualified detector systems**, not pure architectures under identical optimization.

### Pretraining difference

Both start from COCO-pretrained models, but their learned representations, heads, losses, and localization parameterizations differ.

### Scores uncalibrated

Detection scores are ranking signals.

### Closed vocabulary

After custom adaptation the detector knows only:

```text id="9h541m"
stop-sign
yield-sign
speed-limit-sign
```

for this exercise.

It cannot recognize arbitrary new categories by text prompt.

---

# 67. Closed-set versus open-vocabulary transition

The concluding section SHOULD explicitly connect this workshop to the next detection workshop.

Closed-set:

```text id="m6dovw"
fixed learned class head
```

Open-vocabulary:

```text id="kybj1f"
text/query-driven object concepts
```

This naturally sets up the later:

**Grounding DINO / OWLv2 / CountGD**

workshop.

---

# 68. Output structure

```text id="smok4a"
outputs/
├── data/
│   ├── dataset_manifest.json
│   ├── train.json
│   ├── validation.json
│   └── test.json
├── pre_adaptation/
│   ├── rtdetr.csv
│   ├── yolox_s.csv
│   └── yolox_x.csv
├── adaptation/
│   ├── rtdetr_training.json
│   ├── yolox_s_training.json
│   └── yolox_x_training.json
├── validation/
│   ├── predictions.csv
│   ├── metrics.csv
│   └── threshold_sweep.csv
├── frozen/
│   └── frozen_experiment.json
├── test/
│   ├── predictions.csv
│   ├── aggregate_metrics.csv
│   ├── per_class_metrics.csv
│   ├── size_metrics.csv
│   ├── density_metrics.csv
│   └── disagreements.csv
├── artifacts/
│   ├── rtdetr/
│   ├── yolox_s/
│   └── yolox_x/
├── figures/
├── new_data/
├── provenance/
│   ├── model_manifest.json
│   └── experiment_manifest.json
└── workshop_summary.json
```

YOLOX-X paths MAY be absent in `STANDARD`.

---

# 69. Provenance

`experiment_manifest.json` SHOULD include:

```text id="kf8g4n"
notebook_spec
notebook_profile
notebook_mode
workshop_revision
execution_tier

dataset:
  generator_version
  generator_seed
  split_seed
  dataset_digest
  class_order
  train_ids
  validation_ids
  test_ids

models:
  id
  revision_or_tag
  code_revision
  weight_sha256
  weight_bytes
  parameters
  preprocessing

adaptation:
  per-model optimizer
  loss
  epochs
  learning_rate
  batch_size
  frozen_modules
  trainable_parameters
  seed

evaluation:
  score_cutoff
  nms_policy
  max_detections
  iou_thresholds
  metric_implementation

artifacts:
  path
  format
  bytes
  sha256
  reload_parity
```

---

# 70. Notebook metadata

```json id="j9m3tn"
{
  "dimer": {
    "notebook_spec": "2.1",
    "notebook_profile": "E2E",
    "notebook_mode": "WORKSHOP",
    "standalone": true,
    "capability": "multi-model-closed-set-object-detection",
    "carrier": "comparative bounded-adaptation object-detection workshop",
    "dataset": "60 deterministic 640x640 three-class synthetic traffic-sign scenes",
    "default_tier": "STANDARD",
    "canonical_runtime": "NVIDIA Tesla T4",
    "worker_required": false,
    "credentials_required": false,
    "clean_runtime_evidence": "pending"
  }
}
```

---

# 71. Release acceptance

| Requirement | STANDARD | FULL |
|---|---:|---:|
| Notebook Spec 2.1 | PASS | PASS |
| `E2E` / `WORKSHOP` | PASS | PASS |
| Fresh T4 `Run all` | PASS | PASS |
| No Git clone | PASS | PASS |
| No runtime DIMER source fetch | PASS | PASS |
| No DIMER services | PASS | PASS |
| No credentials | PASS | PASS |
| Common 60-image fixture | PASS | PASS |
| Independent 36/12/12 split | PASS | PASS |
| Class coverage checks | PASS | PASS |
| Empty baseline | PASS | PASS |
| RT-DETR pre-adaptation | PASS | PASS |
| RT-DETR adaptation | PASS | PASS |
| YOLOX-S pre-adaptation | PASS | PASS |
| YOLOX-S adaptation | PASS | PASS |
| YOLOX-X adaptation | N/A | PASS |
| Common AP evaluator | PASS | PASS |
| AP / AP50 / AP75 | PASS | PASS |
| Per-class AP50 | PASS | PASS |
| Freeze-before-test | PASS | PASS |
| Independent test | PASS | PASS |
| Error gallery | PASS | PASS |
| Threshold sensitivity on validation | PASS | PASS |
| Runtime/VRAM comparison | PASS | PASS |
| Artifact export | PASS | PASS |
| Fresh reload parity | PASS | PASS |
| BYOD E2E positive case | PASS | PASS |
| BYOD invalid annotation refusals | PASS | PASS |
| Provenance export | PASS | PASS |

---

# 72. Suggested registry entry

```markdown id="0uyphv"
| Notebook | Profile | Mode | Capability | Runtime | Sample | BYOD | Run-all | Status |
|---|---|---|---|---|---|---|---|---|
| `DIMER_MultiModel_Closed_Set_Object_Detection_Workshop.ipynb` | `E2E` | `WORKSHOP` | RT-DETR vs YOLOX closed-set detector adaptation | T4 | 60 deterministic 640×640 three-class sign scenes, 36/12/12 split | yes | pending | candidate |
```

---

# 73. Implementation principle

The experiment should hold constant what can meaningfully be held constant:

> **same pixels → same boxes → same class vocabulary → same splits → same common evaluator**

while preserving what should remain model-specific:

> **detection architecture → preprocessing semantics → assignment algorithm → loss → optimizer → adaptation scope**

That makes the comparison scientifically more defensible than forcing RT-DETR and YOLOX through one artificial training recipe.

The core workshop question becomes:

> **How do set-prediction and dense anchor-free detectors transfer differently to the same small closed-vocabulary detection problem, and what accuracy/compute tradeoffs emerge once the data and evaluation are controlled?**

## 2026-09-26 implementation clarification

The supplemental guided notebook carries the pinned YOLOX source and Apache-2.0 license inline (nine upstream files, eight runtime modules). It no longer fetches upstream Python at runtime. Model pins and the canonical 60-image recipe remain unchanged.

BYOD is bounded adaptation for the same three declared sign labels, not arbitrary-vocabulary detection. Supply 5–60 images, at most six boxes each, side lengths 16–4096 and at most 16 megapixels. `annotations.csv` must annotate every image, use consistent ID/file/split mappings, and supply all explicit train/validation/test splits or none. Each split must cover every class; duplicate pixels across splits are rejected. Validation completes before model acquisition. Artifacts, frozen validation settings, live-to-fresh parity, baseline/adapted held-out metrics and prediction JSON are separate from canonical output. No real-model hosted BYOD result is claimed.
