# Release verification

`tutorials/rtdetr_detection_colab.ipynb` (`E2E`, **standalone** carrier) is a
**release candidate** until the exact notebook revision has executed top-to-bottom in a clean
supported runtime. Unit tests, JSON validation, code-cell compilation, the generator parity checks
and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under
DIMER Notebook Specification 2.0. This file is the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that
  profile, spec `2.0`, a pedagogical mode, `standalone: true` and `generated_from` (repository, revision, module
  SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on
  the primary path; exactly one cell tagged `embedded_module` equal to
  `src/rtdetr_detection_pipeline/pipeline.py` after the generator's documented rewrites; the
  inline `MANIFEST` equal to the committed snapshot manifest and the inline `PINS` equal to the
  `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to `tools/build_notebook.py`
  output for its recorded revision; the pinned-install cell with its restart-on-stale-import guard;
  `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline
  manifest, which the notebook asserts against the module before fetching), the revision is a 40-hex
  immutable commit, and the same identity string appears in `README.md`, `MODEL_CARD.md`, and
  `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `RTDetrDetectionPipeline.from_pretrained(weights_dir=...)`, `validate_inputs`, `detect`,
  `evaluation_report`), the ceiling print (`MIN_IMAGE_SIDE`, `MAX_IMAGE_SIDE`, `MAX_DETECTIONS`,
  the 80 `LABELS`, `DETECTION_THRESHOLD`), the exports, the learner-facing statements (caller-owned
  threshold, uncalibrated per-class sigmoid score, score ordering, no mAP, IoU as sanity check, the
  model emits boxes for any image, closed vocabulary, capability exclusions) and the gated-off BYOD
  default listed in the validator; forbidden patterns (credential-in-URL, any `git clone` /
  `github.com` / repository import on the primary path, a mutable `revision='main'`, direct
  `from transformers import` / `RTDetrForObjectDetection` / `AutoImageProcessor` /
  `post_process_object_detection(` / `from huggingface_hub import` use **outside the carried module
  cell**, `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `safetensors`, `numpy` and
`pillow`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit
suite (`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_notebook_parity.py`;
injected runner, no weights). These are source/provenance and unit checks. They are **not** execution
evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (**no repository checkout is needed — the notebook is standalone**) |
| Local Windows-venv harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, `CUDA_VISIBLE_DEVICES=-1` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle
   executor above) with **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`, `threshold = 0.3`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded
   in `metadata.dimer.generated_from` and that the installed core package versions equal the inline
   `PINS` (= `pyproject.toml`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executes (defines `RTDetrDetectionPipeline`, `validate_inputs`,
     `evaluation_report`, `box_iou`, `verify_snapshot`, `stage_missing_files`, the 80 `LABELS`) with no
     import of the repository package;
   - synthetic 640×480 scene drawn in code with its RGB SHA-256 printed and the ceilings
     (`MIN_IMAGE_SIDE` 16, `MAX_IMAGE_SIDE` 4096, `MAX_DETECTIONS` 300, 80 labels,
     `DETECTION_THRESHOLD` 0.3) surfaced;
   - pinned `PekingU/rtdetr_r50vd` acquisition at the immutable revision through the carried module:
     the inline `MANIFEST` is asserted against the module identity and written to
     `weights/rtdetr-r50vd/`, `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports all
     four manifest entries on a clean runtime, `verify_snapshot` returns its summary dict, and
     `from_pretrained(weights_dir=WEIGHTS_DIR)` loads from the verified directory;
   - `validate_inputs` writes `outputs/rtdetr_detection_input_manifest.json` (verdict `accepted`, one
     recorded rejection finding from the out-of-range-threshold probe);
   - `detect` returning score-ordered boxes; record the labels and scores (the card-pass CPU smoke
     returned exactly three boxes — `stop sign` 0.977, `clock` 0.965, `traffic light` 0.931 — and no
     `sports ball` at any threshold; a materially different result is a finding to record, not a
     failure by itself, because no metric is asserted);
   - `evaluation_report` writes `outputs/rtdetr_detection_evaluation_report.json` with verdict
     `sample-sanity` and one same-label `box_iou` entry per drawn reference box on the synthetic sample
     (three matched, the sports ball at 0.0 with no same-label detection; `not-measurable` on BYOD),
     stated as such;
   - `outputs/rtdetr_detection_result.json`, `outputs/rtdetr_detection_detections.csv` and
     `outputs/rtdetr_detection_annotated.png` written with `NOTEBOOK_SOURCE`, model revision, model
     licence, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/rtdetr_detection_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/rtdetr_detection_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Local pre-flight evidence (not a supported runtime)

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | notebook blob `901896a1d0e5` (commit `caba03e`, generated at `c15c834`; `NOTEBOOK_SOURCE.repository_revision` = `c15c834…`) | Local Windows-venv harness (`run_nb_local.py`: nbclient 0.11.0, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6 | Default synthetic path, all 8 code cells: pinned install skipped (pre-installed), `stage_missing_files` fetched all 4 manifest entries (172 MB) from the Hub cache at the pinned revision into the scratch `weights/`, `verify_snapshot` PASS (4 files), `detect` → 3 boxes (`stop sign` 0.977, `clock` 0.965, `traffic light` 0.931), `evaluation_report` `sample-sanity` (IoU stop sign 0.877, traffic light 0.965, clock 0.983, sports ball 0.000 — no same-label detection), 5 outputs written | 25.5 s | PASS — pre-flight only; not promotion evidence |
| 2026-09-15 | `feat/rtdetr-e2e-finetuning` | Local Windows-venv harness (nbclient 0.11.0, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6 | Default E2E path, all 13 sections: COCO demonstration (3 boxes matched), degenerate input probes, 40-image sign dataset generation & validation, baseline evaluation (AP 0.000, AP50 0.000), 3-epoch bounded fine-tuning (backbone frozen, 19.3M trainable parameters), post-adaptation evaluation (AP 0.849, AP50 0.854; stop-sign 1.000, yield-sign 0.914, speed-limit-sign 0.646), unseen image inference, artifact export to `rtdetr_adapter.pt` (171.7 MB) and fresh reload identity assertion, 6 outputs written | 142.2 s | PASS — pre-flight only; ready for hosted clean-room GPU verification |

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `1fe27a4` / `76385610a91b` | Kaggle CPU (`kurtvalcorza/dimer-nb2-rtdetr-detection` v1) | Default sample path | 217.4 s | **PASSED** — 8/8 ok code cells executed cleanly, 10 files, 172 MB staged |
| 2026-09-15 | `26fd892` / `315909a9b1a7` | Kaggle GPU (Tesla T4, `kurtvalcorza/dimer-nb2-rtdetr-detection` v2) | Default E2E adaptation path (all 14 code cells: COCO demo, input probes, 40-image sign dataset validation, baseline AP 0.000, 3-epoch bounded FT with frozen ResNet-50-vd backbone, post-adaptation AP 0.9161 / AP50 0.9273 / AP75 0.9273, unseen inference, fresh reload verification, all 6 outputs written) | 240.6 s | **PASSED** — 14/14 ok code cells executed cleanly (1 restart after install cell), 10 files, 172 MB staged, adapter exported |

## Current status

Clean-room execution in a **supported runtime** (Kaggle GPU, Tesla T4) has been recorded above. All 14/14 code cells executed cleanly (1 automatic restart after the pinned install cell), staging the 4-file 172 MB snapshot, validating the 40-image sign dataset, establishing baseline AP 0.000, running 3-epoch bounded fine-tuning in 11.4 s on GPU, achieving post-adaptation AP 0.9161 and AP50 0.9273, verifying fresh reload equivalence, and writing all 6 release outputs (`rtdetr_adapter.pt`, `result.json`, `evaluation_report.json`, `input_manifest.json`, `detections.csv`, `annotated.png`). Static validation (`tools/validate_release_assets.py`), generator parity checks (`--check` OK), the offline unit suite (31/31 passed), local pre-flight execution, and hosted clean-room GPU execution all confirm the E2E adaptation profile.

## Supplemental closed-set guided notebook — 2026-09-26 remediation

This entry applies only to `DIMER_MultiModel_Closed_Set_Object_Detection_Workshop.ipynb`, not the earlier primary-notebook executions above. Status remains **Candidate**. No new hosted runtime execution was performed or inferred from the primary notebook's evidence.

Baseline: 10 workshop contract tests and 7 primary parity tests passed (independently recorded by the integrator). Confirmed gaps were a BYOD loader without downstream adaptation, destructive reuse of extraction destination, ambiguous image identities/splits, evaluator sequence truncation, reload-to-reload rather than live-to-reload parity, and runtime source fetching. The fixes carry nine upstream files with original SHA-256 checks and the Apache license, enforce bounded BYOD validation, run sequential model adaptation/reconstruction/evaluation with isolated exports, and compare live adapted validation predictions to a fresh artifact. Guided prompts, a validation-only threshold activity, infrastructure collapse and a stale imported-package guard were added. Public versions accept local build suffixes. A required session restart is explicitly reported and does not satisfy an uninterrupted Run all gate.

Local lightweight checks execute the actual notebook validation, evaluation and orchestration functions with small ZIP fixtures and model doubles. These verify control/data flow, not real model quality, GPU memory or hosted package compatibility. Measured counts and exits are finalized below.

Pending qualification recipe:

1. Start a fresh T4 Colab runtime; use default STANDARD and Run all. Preserve commit/blob, outputs, versions, wall time, peak VRAM, any restart and the final summary. Repeat FULL in a separate fresh runtime. Do not claim uninterrupted execution if the package guard requests a restart.
2. Build a valid ZIP with 5–60 distinct images of the declared three classes and explicit train/validation/test assignments, every split covering all classes, at most six boxes per image. Set `USE_BYOD=True` and `BYOD_ZIP_PATH`. Run through model adaptation, frozen validation, baseline reconstruction, fresh adapter reconstruction, held-out inference and per-model exports under `outputs/byod/run-*`. Check `input_manifest.json` hashes/model identities, frozen artifact digest, `results.json`, `metrics.csv`, and live-to-fresh parity. Keep these results separate from canonical synthetic results.
3. In a separate run replace a class label with `unknown`, introduce an out-of-bounds/nonfinite box, omit an annotated file, or duplicate pixels across splits. Confirm a clear failure before BYOD model loading and no BYOD result report. Preserve the rejection output.

REL12 real-model valid/invalid BYOD and STANDARD/FULL hosted evidence remain open. Local tests do not promote this notebook.

Measured local checks: workshop contracts **10/10** and BYOD/evaluator/runtime guard checks **22/22** passed (`python -m pytest tests/test_detection_workshop.py tests/test_detection_workshop_byod.py -q --noconftest`, exit 0). Release validator and supplemental generator parity each returned exit 0. No model weights were executed by these tests.


### Colab NumPy setup failure — 2026-09-26

The maintainer-supplied run stopped in setup before model execution: NumPy 2.1.3 was already loaded, while the notebook installed 2.5.3. The [failure record](execution-evidence/2026-09-26/colab-setup-failure.json) records the independently inspected error. The supplemental notebook retains an already loaded NumPy 2.x, integrating the concurrent host-preservation fix, and uses 2.1.3 as the fallback pin when NumPy is not yet loaded. The observed Colab 2.1.3 is preserved instead of replaced. Other model/runtime pins are unchanged; stale-module detection remains enabled. Declared upstream requirements permit 2.1.3 (Transformers and datasets require >=1.17; the closed-set SciPy pin requires >=2.0,<2.8).

A regression executes the real setup prefix against a simulated Colab preloaded NumPy and package installer: it reproduces the original restart error before the fix and completes without a restart after it. This is setup regression evidence, not a full model/Colab rerun. A new hosted Run all is still required to discover any downstream issues. Use a fresh runtime for that rerun; the prior failed session already replaced installed packages.


### Maintainer-supplied successful Colab run — 2026-09-26

The maintainer supplied the [executed notebook](execution-evidence/2026-09-26/DIMER_MultiModel_Closed_Set_Object_Detection_Workshop.ipynb) and explicitly authorized merging PR #8. This later record supersedes the earlier default-path setup failure. The file is archived byte-for-byte, SHA-256 `19301d8b57626b07f681233e0fa4492c4141983ec49dd367b3dd7cd21974b519`. All 27 code cells have execution counts, 38 saved outputs and zero saved errors. Executable Python ASTs match commit `047f416ef663f922f5a73ce47887af39171b1ec0`, tutorial blob `f696be062bad0c0faaef31786d1f3152c4f1d621`. This evidence commit does not change tutorial code.

Scope: STANDARD: RT-DETR and YOLOX-S; 60 synthetic images split 36 train / 12 validation / 12 test. FULL and BYOD were not exercised.

Saved runtime: Python 3.13.15, torch 2.14.0+cu130, Transformers 4.57.6, NumPy 2.1.3, CUDA Tesla T4. Execution reaches the final summary/export checks. The separate exported files were not supplied, so their bytes/digests were not independently inspected. Saved counts run sequentially from 1 to 27; runtime freshness and absence of manual restarts/reruns are not independently established by the artifact.

Merge approval and this successful canonical run do not close the remaining optional-path/REL12 qualification gates or imply a blanket gold-standard promotion. Retain the earlier limitations except where this default-path execution directly supersedes them.
