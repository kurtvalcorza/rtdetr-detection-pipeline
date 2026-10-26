# Release verification

`tutorials/rtdetr_detection_colab.ipynb` (`E2E`, **standalone** carrier) is a
**release candidate** until the exact notebook revision has executed top-to-bottom in a clean
supported runtime. The current blob `8e4a0fb7ff09` (commit `9206a7d`, historical wording) passed its confirming hosted Colab T4 run on 2026-10-25 (PR #17 head `4444386`, 15/15 code cells, one pass, 0 errors; held-out AP 0.9626; drift recorded in `docs/release-verification.md`); the previous blob `e921b62e6b68` (prose kept true across runs, notebook commit `34d7254`) executed top to bottom in one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-25 at PR #17 head `95de437` (Colab CLI sequential execution, not a browser Run all; 15/15 code cells, 110.0 s wall; held-out AP 0.9774 / AP50 0.9825 from baseline 0.000; `rtdetr_adapter.safetensors` reloaded with raw-output parity 0.0 on 13 images). The blob before it, `becde2c0d5e3` (tier C review fixes, `2497882`), did likewise on 2026-10-25 (127.6 s; held-out AP 0.979). The earlier blob `0dd61e64a3cc` (commit `7456a3c`) executed top to bottom in one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-25 (Colab CLI sequential execution, not a browser Run all; 15/15 code cells in the exec log, 111.1 s wall; held-out AP 0.9344 / AP50 0.942 from baseline 0.000; `rtdetr_adapter.safetensors` plus its JSON sidecar reloaded with raw-output parity 0.0 on 13 images); status stays **Candidate**: the remaining gates are REL12 (hosted BYOD) and REL2 (a browser run of the interactive journeys). Runs of earlier blobs are history (REL14). Unit tests, JSON validation, code-cell compilation, the generator parity checks
and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under
DIMER Notebook Specification 2.5. This file is the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- the primary tutorial notebook (plus the supplemental workshop), named in `tutorials/README.md` with its `E2E`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that
  profile, spec `2.2`, a pedagogical mode, `standalone: true` and `generated_from` (repository, revision, module
  SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on
  the primary path; exactly one cell tagged `embedded_module` equal to
  `src/rtdetr_detection_pipeline/pipeline.py` after the generator's documented rewrites; the
  inline `MANIFEST` equal to the committed snapshot manifest and the inline `PINS` equal to the
  `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to `tools/build_notebook.py`
  output for its recorded revision; the isolated-environment bootstrap cell (generator /2.2: hash-locked `uv`
  environment, nothing installed into the kernel, no restart);
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
   defaults for the sample path: `USE_BYOD_IMAGE = False`, `USE_BYOD_DATASET = False`, `threshold = 0.3`);
   Run all must complete in one pass with no restart (Section 1 builds the isolated environment);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded
   in `metadata.dimer.generated_from` and that the installed core package versions equal the inline
   `PINS` (= `pyproject.toml`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` (hash-locked, into the isolated environment of Section 1) with no GitHub access;
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
   - the adaptation stages: the 40-image sign dataset validated (`validate_dataset`) and split 30/10 with
     every class in both splits (`check_split_coverage`); the re-headed baseline AP 0.000 on the held-out
     split; the bounded fine-tune (Section 8 rebuilds the model from the snapshot first, prints
     `train_seconds`, and asserts the reported freeze state equals the model's `requires_grad` state);
     held-out AP / AP50 / AP75 next to the zero-training reference (the unadapted COCO detector on the
     held-out stop signs); inference on the three seed-99 images; export of `outputs/rtdetr_adapter.safetensors`
     with its JSON sidecar `outputs/rtdetr_adapter.json` (training settings, weights SHA-256); fresh reload with raw-output parity (logits and boxes of every query on
     13 images within `PARITY_TOLERANCE`, at least one detection compared); and
     `outputs/rtdetr_detection_new_data.csv` / `outputs/rtdetr_detection_new_data_annotated.png`;
   - REL12 (separate runs): `USE_BYOD_DATASET = True` with `BYOD_DATASET_PATH` on a valid folder/zip
     reaches evaluate, export and fresh reload; a malformed dataset is refused naming the row and rule;
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
| 2026-10-09 | review branch `review/rtdetr_detection_colab-2026-10-02` (RTD fixes, generator /2.2; worked-answer source) | Sequential in-process cell executor (`DIMER_NOTEBOOK_CI_PREINSTALLED=1`, `CUDA_VISIBLE_DEVICES=-1`, pre-staged verified snapshot), Windows laptop CPU, torch 2.13.0+cpu (not the 2.14.0 pin), transformers 4.57.6, scipy 1.18.1 | Default E2E path, all code cells (repeated on the final source: identical numbers; BYOD probes: a 16-image synthetic zip through `BYOD_DATASET_PATH` reached evaluate, export and fresh reload with parity 0.0; a zip with an out-of-bounds box was refused naming record 0 box 0; `BYOD_IMAGE_PATH` ran outside Colab): COCO scene 3/4 matched (IoU 0.877 / 0.965 / 0.983, sports ball 0.0); degenerate probes blank 1 at 0.3 (`train` 0.333) / 87 at 0.05, noise 1 at 0.3 (`cat` 0.391) / 254 at 0.05; baseline AP 0.000; fine-tune 19,259,121 of 42,733,137 parameters, epoch losses 39.62 / 25.58 / 19.30, `train_seconds` 137.4; held-out AP 0.8494 / AP50 0.8535 / AP75 0.8535; zero-training reference (unadapted COCO detector, 5 held-out stop signs) AP 1.000, adapted stop-sign AP 1.000; unseen images: 2 duplicate yield-sign detections (images 0 and 1), 1 missed speed-limit sign, two scores 0.302; reload raw-output parity on 13 images (24 detections at 0.3) max difference 0.0. Rerun probe: Section 8 rerun unchanged reproduced first-epoch loss 39.6166; with `FREEZE_BACKBONE = False` 42,733,137 trainable, backbone unfrozen in the model | not summed | **Local pre-flight PASS** — not a supported runtime; worked answers in the notebook quote this run |

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-09 | `0feefe5` / `e2ab45177aec` (`NOTEBOOK_SOURCE.repository_revision` `2ce1925`, the source revision the notebook was generated from; `2ce1925..0feefe5` is the review-fix commit `23bb7a7` and a test-only commit; generator `build_notebook.py/2.2`, `notebook_spec` 2.2) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-rtdetr-0feefe5-6128`), committed blob fetched at the 40-char SHA and Git-blob verified, no repository checkout, clean model cache, BYOD gates off, sample-path defaults; kernel Python 3.13.15; isolated uv environment Python 3.12.12, 48 locked packages, built in 67 s; torch 2.14.0+cu130, transformers 4.57.6, NumPy 2.5.3, Pillow 11.3.0, SciPy 1.18.1; device `cuda:0` | Default E2E path, all 15 code cells (cells 3–4 are the carried modules and print nothing): 4 snapshot files fetched and verified; COCO scene 3/4 matched (IoU 0.877 / 0.965 / 0.983, sports ball 0.0); degenerate probes blank 1 at 0.3 (`train` 0.333) / 87 at 0.05, noise 1 at 0.3 (`cat` 0.391) / 254 at 0.05; 40 records / 82 boxes, split 30 / 10 with every class in both; baseline AP 0.000; fine-tune 19,259,121 of 42,733,137 parameters, backbone frozen in the model, epoch losses 39.65 / 25.20 / 19.44, `train_seconds` 11.5; held-out AP 0.9452 / AP50 0.9578 / AP75 0.9578 (per-class AP50 stop 0.934, yield 0.939, speed-limit 1.0); zero-training reference on 5 held-out stop signs: unadapted COCO detector AP 1.000, adapted 0.934; unseen images all 5 signs found, image 1 with two extra `yield-sign` boxes, lowest kept score 0.307; artifact 174,923,973 B (sha256 `2b90fa3c…`) carries its training settings; fresh reload raw-output parity on 13 images × 300 queries, 26 detections at 0.3, max difference 0.0; 8 outputs written | 130.4 s | **PASSED in one pass, no restart** — 0 errors. Evidence (byte-exact): `docs/execution-evidence/2026-10-09-0feefe5/` — executed notebook sha256 `56a1d13467ff…`, `exec.log` `0dbdad489ce3…`, `run_summary.json` `50839cfd6ade…`. The worked answers quote the 2026-10-09 local CPU run; this run differs: AP 0.9452 not ~0.85, and the adapted stop-sign AP (0.934) is below the COCO reference (1.000) rather than equal. One seeded split, one runtime, no dispersion estimate; BYOD (REL12) not exercised on a hosted runtime |
| 2026-10-25 | `7456a3c` / `0dd61e64a3cc` (main after PR #14 and #15; safetensors adapter export) | Colab CLI sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4 VM (session `suite-rtdetr-7456a3c-6128`), committed blob fetched at the 40-char SHA and Git-blob verified, no repository checkout; kernel Python 3.13.16; isolated uv environment Python 3.12.12, 48 locked packages, built in 50 s; device `cuda:0` | Default E2E path, 15/15 code cells (see the 2026-10-25 record below) | 111.1 s | **PASSED in one pass, no restart** — 0 errors; current blob |
| 2026-10-25 | `7456a3c` / `6de85e6314e9` (workshop revision `0.4.0-candidate`) | Colab CLI sequential execution, fresh Colab Tesla T4 (session `suite-rtdetr-7456a3c-e141`); isolated uv CPython 3.12.12, 57 locked packages, torch 2.14.0+cu130 | Workshop `DIMER_MultiModel_Closed_Set_Object_Detection_Workshop.ipynb`, default STANDARD path only (`USE_BYOD=False`) | 151.5 s | **PASSED**: 32/32 code cells in order (exec log), no errors; current workshop blob |
| 2026-09-14 | `1fe27a4` / `76385610a91b` | Kaggle CPU (`kurtvalcorza/dimer-nb2-rtdetr-detection` v1) | Default sample path | 217.4 s | **PASSED** — 8/8 ok code cells executed cleanly, 10 files, 172 MB staged |
| 2026-09-15 | `26fd892` / `315909a9b1a7` | Kaggle GPU (Tesla T4, `kurtvalcorza/dimer-nb2-rtdetr-detection` v2) | Default E2E adaptation path (all 14 code cells: COCO demo, input probes, 40-image sign dataset validation, baseline AP 0.000, 3-epoch bounded FT with frozen ResNet-50-vd backbone, post-adaptation AP 0.9161 / AP50 0.9273 / AP75 0.9273, unseen inference, fresh reload verification, all 6 outputs written) | 240.6 s | **PASSED** — 14/14 ok code cells executed cleanly (1 restart after install cell), 10 files, 172 MB staged, adapter exported |
| 2026-10-04 | `46a300c` / `929be36ca616` (PR #10) | Google Colab (browser, maintainer-run), Tesla T4; uv-isolated CPython 3.12.12, torch 2.14.0+cu130 | Workshop `DIMER_MultiModel_Closed_Set_Object_Detection_Workshop.ipynb`, default STANDARD path only (`USE_BYOD=False`) | not recorded (environment setup 70 s) | **RAN CLEAN, METRICS DIFFER**: 29/29 code cells, counts 1..29, no errors; adapted test AP differs from the 2026-09-30 T4 run (see the 2026-10-04 record below); not merge evidence |
| 2026-10-15 | `7240fd1` / `5937db0e0bce` (PR #13) | Colab CLI 0.7.4, fresh Colab Tesla T4 (session `suite-rtdetr-7240fd1-e141`; `colab exec -f` sequential execution, not a browser Run all); uv-isolated CPython 3.12.12, torch 2.14.0+cu130 | Workshop `DIMER_MultiModel_Closed_Set_Object_Detection_Workshop.ipynb`, default STANDARD path only (`USE_BYOD=False`, multi-seed repeat skipped as designed) | 191.4 s | **PASSED**: 31/31 code cells in order (exec log), no errors; pre-adaptation rows identical to 2026-09-30 and 2026-10-04; adapted test AP inside the earlier T4 range (see the 2026-10-15 record below) |

## Final prose pass (2026-10-26)

**Code commit `2ce7561`, notebook commit `df75665`.** The current blob of `tutorials/rtdetr_detection_colab.ipynb` is `7579885cd623a53fd7b0d27cb01c6b32c29cf883`, **Candidate**; that exact blob (PR #17 head `da680b8`) passed its confirming hosted run on 2026-10-26: Colab CLI sequential execution on a fresh Colab Tesla T4, one pass, no restart, 0 errors (15/15 code cells, 109.0 s wall; held-out AP 0.972 / AP50 0.979 from baseline 0.000; reload parity 0.0 on 13 images). The paths this blob's runs cover: the default path only. Remaining gates: REL12 (a hosted BYOD run, one input accepted and one refused) and REL2 (a browser `Run all`). Runs of earlier blobs are history (REL14). See "Colab CLI execution of `da680b8` (blob `7579885c`)" below. Kurt's 2026-10-26 ruling: fix only the prose rule breaks the final evidence flagged; recorded drift stays recorded, not fixed. Markdown only: the opening cells no longer give a time budget or a third-decimal spread claim, and the unnamed T4 timing points at the named runs under Runtime. The runs of blob `8e4a0fb7` and earlier blobs are history (REL14); the records below are kept; follow-up the same day: md0 relabels the worked answers' local CPU run to the earlier revision (it said this revision). It supersedes blob `8d1e0775` (code commit `625c1cf`, notebook commit `19eca6f`), which was never run. Second follow-up the same day (code commit `eac55be` → `2ce7561`, notebook commit `ea1fadc` → `df75665`): Troubleshooting no longer says a very different adapted AP means a different runtime; it supersedes blob `80434242`, which was never run.

## Current status

**DAT20 refusal display (2026-10-26): Candidate.** The current blob `b19ea7db` (code commit `705cc5b`) shows a refused BYOD input as one `Refused: <file>: <reason>` line with no traceback, removes that branch's earlier results from the session and from `outputs/` and stops Run all at that cell, while genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20); its hosted Colab T4 runs (the default path and a scripted refusal, `USE_BYOD_DATASET=True;BYOD_DATASET_PATH='/etc/hostname'`) are pending, and the hosted runs of earlier blobs, including the 2026-10-26 run of `7579885c`, are history for it (REL14). The status below is history.

**Historical wording (2026-10-26, code commits `7cdba8f` and `6100128`, notebook commit `9206a7d`).** The previous blob of `tutorials/rtdetr_detection_colab.ipynb` was `8e4a0fb7ff0922dfd505aa72ea96eefdd7c44584`, **Candidate**, regenerated under Kurt's 2026-10-26 ruling: every run-derived number in the prose names the runs that printed it (local CPU worked run 2026-10-09; Kaggle T4 of `315909a9` 2026-09-15; Colab T4 of `e2ab4517` 2026-10-09, `0dd61e64`, `becde2c0` and `e921b62e` 2026-10-25), with no expectation ranges, timing promise or not-re-run status phrases; it adds the `e921b62e` run (AP 0.9774, `train_seconds` 11.3), which printed no prose mismatch. Markdown only. This exact blob (PR #17 head `4444386`) **passed** its confirming hosted run: Colab CLI sequential execution on a fresh Colab Tesla T4, 2026-10-25, 15/15 code cells in one pass, no restart, 0 errors, 123.6 s wall; held-out AP 0.9626, AP50 0.9736; see "Colab CLI execution of `4444386`" below; evidence in `docs/execution-evidence/2026-10-25-4444386/`. This PR's runs cover the default path only. Remaining gates: REL12 (a hosted BYOD run with one accepted and one refused input) and REL2 (a browser Run all). Runs of earlier blobs are history (REL14). The records below stay as history.

**Prose kept true across runs (2026-10-25, code commit `a24b044`, notebook commit `34d7254`).** The previous blob of `tutorials/rtdetr_detection_colab.ipynb` is `e921b62e6b68225db557c4b0c224b6355df11fa9`, **Candidate**. At PR #17 head `95de437` it executed top to bottom in one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-25 (Colab CLI sequential execution, not a browser Run all; 15/15 code cells, 110.0 s wall; held-out AP 0.9774 / AP50 0.9825 from baseline 0.000; `rtdetr_adapter.safetensors` reloaded with raw-output parity 0.0 on 13 images; record "Colab CLI execution of `95de437`" below). The run covers the default path only; remaining gates REL12 (hosted BYOD) and REL2 (a browser run of the interactive journeys). Markdown only: the troubleshooting range for the adapted AP is now about 0.85 (local CPU) to 0.98 (hosted T4 0.916–0.979), and the worked answers and run-to-run paragraph list every recorded hosted run (Kaggle T4 of `315909a9`, 2026-09-15; Colab T4 of `e2ab4517`, 2026-10-09, and of `0dd61e64` and `becde2c0`, 2026-10-25) beside the labelled 2026-10-09 local CPU worked run, with the Colab epoch-loss ranges and per-class AP50 ranges. The paragraph below describes the previous blob `becde2c0` and stays as history.

**Candidate.** On PR #17 head `2497882` (2026-10-25):

- `tutorials/rtdetr_detection_colab.ipynb`, blob `becde2c0d5e349b9ff5ff43bb72399a4f2f93f68` (tier C review fixes T77-1..T77-3, generated at `0563816`): the default E2E path ran on a hosted Colab T4 on 2026-10-25 in one pass, no restart, 0 errors (record below: held-out AP 0.979 / AP50 0.979 from baseline 0.000; safetensors reload parity 0.0 on 13 images). This PR's run covers the default path only.
- `tutorials/DIMER_MultiModel_Closed_Set_Object_Detection_Workshop.ipynb`, blob `6de85e6314e979856a71c0857e6af8a2b24b5e62` (revision `0.4.0-candidate`), unchanged by this PR: the default STANDARD path ran on a hosted Colab T4 (record below, 2026-10-25). FULL (including the multi-seed repeat) has not run on this blob.
- Remaining gates: **REL12** (a hosted BYOD run with one accepted and one refused input) and **REL2** (a browser run of the interactive journeys: forms, BYOD upload). The scripted CLI runs cover the default paths only.
- Every earlier record in this file describes an earlier blob and is history (REL14), including the 2026-10-25 run of `0dd61e64` (`7456a3c`), the 2026-09-15 Kaggle T4 run, the 2026-10-09 Colab CLI run of `e2ab45177aec` and the 2026-10-15 workshop run of `5937db0e`.

Promotion is an integrator decision; this record does not change the status.

## Supplemental closed-set guided notebook — 2026-09-26 remediation

This entry applies only to `DIMER_MultiModel_Closed_Set_Object_Detection_Workshop.ipynb`, not the earlier primary-notebook executions above. Status remains **Candidate**. No new hosted runtime execution was performed or inferred from the primary notebook's evidence.

Baseline: 10 workshop contract tests and 7 primary parity tests passed (independently recorded by the integrator). Confirmed gaps were a BYOD loader without downstream adaptation, destructive reuse of extraction destination, ambiguous image identities/splits, evaluator sequence truncation, reload-to-reload rather than live-to-reload parity, and runtime source fetching. The fixes carry nine upstream files with original SHA-256 checks and the Apache license, enforce bounded BYOD validation, run sequential model adaptation/reconstruction/evaluation with isolated exports, and compare live adapted validation predictions to a fresh artifact. Guided prompts, a validation-only threshold activity, infrastructure collapse and a stale imported-package guard were added. Public versions accept local build suffixes. A required session restart is explicitly reported and does not satisfy an uninterrupted Run all gate.

Local lightweight checks execute the actual notebook validation, evaluation and orchestration functions with small ZIP fixtures and model doubles. These verify control/data flow, not real model quality, GPU memory or hosted package compatibility. Measured counts and exits are finalized below.

Pending qualification recipe:

1. Start a fresh T4 Colab runtime; use default STANDARD and Run all. Preserve commit/blob, outputs, versions, wall time, peak VRAM, any restart and the final summary. Repeat FULL in a separate fresh runtime. Do not claim uninterrupted execution if the package guard requests a restart.
2. Build a valid ZIP with 5–60 distinct images of the declared three classes and explicit train/validation/test assignments, every split covering all classes, at most six boxes per image. Set `USE_BYOD=True` and `BYOD_ZIP_PATH`. Run through model adaptation, frozen validation, baseline reconstruction, fresh adapter reconstruction, held-out inference and per-model exports under `outputs/runs/<RUN_ID>/byod/run-*` (before the 2026-09-30 revision: `outputs/byod/run-*`). Check `input_manifest.json` hashes/model identities, frozen artifact digest, `results.json`, `metrics.csv`, and live-to-fresh parity. Keep these results separate from canonical synthetic results.
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


### Notebook review remediation (DET-01 to DET-07) — 2026-09-30

This entry applies only to the supplemental closed-set notebook and responds to the review of commit `16115d3` (notebook blob `b1a8b9ef2d41`). Status remains **Candidate**, and readiness is **Verification pending** until a hosted run of this revision exists. The successful 2026-09-26 Colab record above remains attached to its original revision (`047f416`). It is not evidence for this revision: the YOLOX construction changed.

| Finding | Correction |
|---|---|
| DET-01 (major) | `build_yolox` applies the pinned upstream `Exp.get_model` BatchNorm settings (`eps=1e-3`, `momentum=0.03`) before weights load, in both the fresh and reload paths, and checks every layer. Artifacts record the setting (`dimer-yolox-detection-adapter/2`). `/1` adapters are refused because they were trained with PyTorch-default BatchNorm. |
| DET-02 (major) | Raw-output contracts in both wrappers (shape and finiteness, checked before thresholding or NMS). A shared detection-validity gate (four finite, ordered coordinates; a declared label; a finite score in [0, 1]) is applied in both evaluators and in both reload-parity paths. A valid empty list stays valid. The parity prose now says it compares thresholded, postprocessed detections. |
| DET-03 (major) | The freeze gets an `experiment_id` digest and refuses settings that changed after validation. The test cell re-reads the freeze from disk, verifies the digest, the evaluation settings (cutoff, display threshold, NMS, detection cap, IoU thresholds, class order), the dataset and test cohort, the selected models and the artifact digests, and scores with the frozen values. Drift is refused before any test inference. The duplicate `YOLOX_EVAL_NMS=0.65` redefinition in the wrapper cell was removed, so the section-1 control is the only source. |
| DET-04 | Gallery "advantage" examples are shown only when that sign exists. Titles carry both per-image AP50 values and their difference. |
| DET-05 | BYOD refuses duplicate (including case-duplicate) column names and rows whose width differs from the header. |
| DET-06 | Every Run all writes into `OUTPUT_DIR/runs/<RUN_ID>/`, and the report bundle contains only that run. All normalized test and validation detections are exported (`test/predictions.json`, `validation/predictions.json`) with the effective settings, and the export replays the test metrics from the saved detections. |
| DET-07 | The training summary column is `stage_peak_vram_MiB`, and the report states that it is a stage peak with every selected detector resident. |

The earlier hand edit that added the AI Use Disclosure (`16115d3`) is now carried by the generator. Before this, `main` failed CI's generator-parity check.

**User-visible changes.** Output paths move under `outputs/runs/<RUN_ID>/`. The bundle is named `outputs/DIMER_Closed_Set_Detection_Workshop_Report_<RUN_ID>.zip`. YOLOX adapter format `/2` is required. BYOD must be enabled in section 1 before Run all. Newly refused inputs: non-finite or malformed model outputs, duplicate annotation headers, ragged annotation rows, and changes made after the freeze.

**Offline verification (not clean-runtime evidence).**

- **pytest:** 130 passed. The baseline at `16115d3` was 70 passed and 1 failed (generator parity). 59 of the 130 are new regression tests that execute the generated cells. The suite also passes without matplotlib or SciPy, as in CI.
- **Static checks:** ruff, `validate_release_assets.py`, and both generator `--check` runs pass.
- **Real pinned checkpoint (CPU, no training):**
  - `yolox_s.pth` was staged through the notebook's own digest check, and the notebook model's raw outputs were **bit-identical** to the pinned upstream constructor on 3/3 generated scenes.
  - The pre-fix BatchNorm construction changed raw box-regression outputs by up to 380 px and objectness by up to 0.16 on the same weights.
  - YOLOX-X was built only with random weights; the 793 MB checkpoint was not downloaded.
- **Journey harness** (every code cell except pip install and the RT-DETR Hub download, with explicit synthetic adapters replacing both detectors):
  - STANDARD Run all plus an export re-run: same bundle members, metric replay PASS.
  - FULL Run all into the same `OUTPUT_DIR`: a separate run directory, and no cross-run files in either bundle.
  - BYOD Run all: the BYOD run sits inside the run directory and bundle.
  - Refusals: cutoff, NMS and class-order drift are refused with 0 test inference calls; the reviewer's duplicate-header, unsafe-path and missing-image ZIPs are refused before models load.
- **Scope:** no RT-DETR model execution, no gradient training, and no GPU.

Still required before promotion: a fresh T4 STANDARD Run all of this revision; FULL in a separate fresh runtime; real-model valid and invalid BYOD (REL12); and the learner observation named in the review. Model scores after the BatchNorm correction are new measurements and are not expected to match the earlier run.


### Maintainer-supplied Colab execution of revision `ade681b` — 2026-09-30

- **File:** [executed notebook](execution-evidence/2026-09-30/DIMER_MultiModel_Closed_Set_Object_Detection_Workshop_ade681b.ipynb), archived byte-for-byte, SHA-256 `7fbfd0b07fbf36438d8eb0a0afc0d00a3d08158f1757da2d419733051319f31e`.
- **Source match:** the source of all 56 cells matches the PR head `ade681b` (notebook blob `395cb935`) with no diffs at all, not even `# @param` lines. Cell ids and order are unchanged.
- **Runtime:** Colab Tesla T4. Python 3.13.15, torch 2.14.0+cu130, torchvision 0.29.0+cu130, Transformers 4.57.6, NumPy 2.1.3 (preloaded by the host kernel).
- **Execution:** 27/27 code cells, counts 1..27 in order, no saved errors. Tier STANDARD (RT-DETR, YOLOX-S); `USE_BYOD=False`. `run_id` `20260929T215642Z-cad619`, `experiment_id` `7999e116…6ba3a`.
- **Dataset:** `dataset_sha256` `823df4cf…302169`, split 36/12/12, identical to the 2026-09-26 run.
- **Training-stage peak VRAM:** RT-DETR 2553 MiB; YOLOX-S 621 MiB. Both are stage peaks with both detectors resident.

| Model | State | Test AP | AP50 | AP75 | Recall@0.30 | 2026-09-26 AP (`047f416`) |
|---|---|---:|---:|---:|---:|---:|
| RT-DETR | pre | 0.0495 | 0.0495 | 0.0495 | 0.000 | 0.0495 |
| RT-DETR | adapted | 0.9704 | 0.9883 | 0.9883 | 0.682 | 0.9737 |
| YOLOX-S | pre | 0.0344 | 0.0376 | 0.0376 | 0.000 | 0.1163 |
| YOLOX-S | adapted | 0.8273 | 1.0000 | 1.0000 | 1.000 | 0.8520 |

RT-DETR's code path is unchanged: its pre-adaptation row is identical, and the adapted AP differs by 0.003, which is consistent with GPU training nondeterminism. The YOLOX-S rows are new measurements after the DET-01 BatchNorm correction. Neither set of scores is an acceptance gate.

| Journey / check | Verdict |
|---|---|
| STANDARD Run all (default) | **PASS**: the final summary was reached. |
| DET-01 BatchNorm | **PASS**: YOLOX-S built, loaded and adapted without the BatchNorm check refusing it; the adapter uses format `/2`. |
| DET-02 numerical gates | **PASS** on valid outputs: no refusals. Reload parity: RT-DETR and YOLOX-S each 1 detection, differences 0.0. |
| DET-03 freeze binding | **PASS**: the test used the verified frozen settings (cutoff 0.01, display 0.30, NMS 0.65). |
| DET-04 gallery | **PASS**: every test image tied on per-image AP50, so both "no advantage" lines were printed and only the three-object scene was drawn, titled with the 0.00 difference. |
| DET-06 export | **PASS**: 27 files in the run directory, bundle SHA-256 `5ea11ea3…d185be1`, metric replay from `test/predictions.json` PASS 4/4. |
| DET-07 label | **PASS**: `stage_peak_vram_MiB` and the note are shown. |
| Export re-run | not assessed in this run (export cell executed once) |
| FULL tier | not assessed in this run |
| BYOD (valid and invalid, real models) | not assessed in this run |

**Evidence boundary:** saved outputs were inspected; execution was not independently repeated. The exported bundle's bytes were not supplied. Saved execution counts do not establish runtime freshness or the absence of restarts.

**Still open:** FULL in a fresh runtime, real-model valid and invalid BYOD (REL12), and the learner observation. Status remains **Candidate**.

### 2026-10-03 uv isolated environment

Notebook blob `395cb935` (`b7bb099`, the `ade681b` source) → `929be36c` (workshop revision `0.2.0-candidate`). **Hosted re-run pending; status remains Candidate.** The recorded runs above are unchanged and describe the earlier blob.

- **What changed:** the section 2 in-kernel `pip install`, the preloaded-NumPy handling and the "Restart session" guards are gone. Two infrastructure cells inserted before section 1 verify a pinned `uv` 0.12.15 wheel (size and SHA-256), create a uv-managed CPython 3.12.12 environment and install the carried hash lock (`tools/detection-workshop-requirements.lock`, 57 packages) with `--require-hashes --only-binary :all:`; a router then runs every later code cell in one persistent worker process on that environment's Python. Nothing is installed into the kernel and Run all needs no restart. The DET-01..DET-07 cells keep their ids, order and code; section 2 now checks the environment against the pins instead of installing them.
- **User-visible:** **Linux x86_64 only** (Google Colab, Kaggle, Linux Jupyter). Model code runs on Python 3.12.12 instead of the host kernel's Python (3.13.15 on Colab on 2026-09-30). The first run downloads the locked wheels. The two former ranges are fixed inside their ranges: matplotlib 3.10.9, pandas 3.0.6; every former exact pin is unchanged (NumPy 2.1.3 is the version the 2026-09-30 run kept).
- **Line limit:** the 49,902-character YOLOX carrier line and the carried lock are written as parenthesised runs of string pieces of at most 1,000 characters (`ast.literal_eval` equality asserted by the generator and tests); the longest notebook line is now under 2,000 characters.
- **Local CPU stand-in (not clean-runtime evidence):** one Run all of blob `929be36c` with nbclient in a Linux x86_64 WSL kernel, CUDA hidden, real weights, tier STANDARD: 29/29 code cells, no errors, 3,914 s. Dataset `823df4cf…302169` and split 36/12/12 match the 2026-09-30 run; pre-adaptation rows match it exactly (RT-DETR AP 0.0495; YOLOX-S AP 0.0344, AP50 0.0376). Adapted rows on CPU: RT-DETR AP 0.8874 / AP50 0.9153; YOLOX-S AP 0.6994 / AP50 1.0000 (T4 2026-09-30: 0.9704 / 0.9883 and 0.8273 / 1.0000). The adapted gap is attributed to CPU-versus-CUDA training numerics (inferred, not isolated); a hosted T4 run of this blob is the comparison that counts.
- **Still required:** a hosted T4 STANDARD Run all of this blob; then the items already listed as open above.

### Maintainer-supplied Colab execution of `46a300c` (blob `929be36c`) — 2026-10-04

- **File:** [executed notebook](execution-evidence/2026-10-04/DIMER_MultiModel_Closed_Set_Object_Detection_Workshop_46a300c_colab-browser-t4.ipynb), archived byte-for-byte, SHA-256 `3fa98f83f3185a8113a8341b01d1c1a9e4f3ab2333824b06179762a83b07655b`.
- **Source match:** all 60 cells match blob `929be36c` at `46a300c` with no source diffs at all, not even `# @param` lines. Cell ids and order are unchanged.
- **Executor:** Google Colab (browser, maintainer-run), Tesla T4. Kernel Python 3.13.15. Isolated uv environment `/content/dimer_isolated_env`: CPython 3.12.12, 57 locked packages, built in 70 s, no restart. torch 2.14.0+cu130, torchvision 0.29.0+cu130, Transformers 4.57.6, NumPy 2.1.3, `pin_mismatches` empty, device `cuda:0`.
- **Execution:** 29/29 code cells, counts 1..29 in order, no saved errors. Tier STANDARD (RT-DETR, YOLOX-S); `USE_BYOD=False`. `run_id` `20261003T234112Z-c89195`, `experiment_id` `4870e50b…ef439529`. The saved notebook does not record wall time.
- **Identical to 2026-09-30:**
  - `dataset_sha256` `823df4cf…302169` and the 36/12/12 split;
  - the validation pre-adaptation row (RT-DETR AP 0.0858; YOLOX-S AP 0.0756) and the test pre-adaptation rows;
  - training-stage peak VRAM (2553 / 621 MiB);
  - reload parity (differences 0.0) and metric replay (PASS 4/4).

| Model | State | Test AP | AP50 | AP75 | 2026-09-30 T4 AP (`ade681b`) | 2026-09-26 T4 AP |
|---|---|---:|---:|---:|---:|---:|
| RT-DETR | pre | 0.0495 | 0.0495 | 0.0495 | 0.0495 | 0.0495 |
| RT-DETR | adapted | 0.9299 | 0.9590 | 0.9590 | 0.9704 | 0.9737 |
| YOLOX-S | pre | 0.0344 | 0.0376 | 0.0376 | 0.0344 | 0.1163 |
| YOLOX-S | adapted | 0.8932 | 1.0000 | 1.0000 | 0.8273 | 0.8520 |

- **What differs:** training losses diverge from epoch 1 (RT-DETR 37.1504 vs 37.1656), and so does everything downstream of training.
  - Adapted validation AP: RT-DETR 0.9278 vs 0.8376; YOLOX-S 0.8794 vs 0.8245.
  - Adapted test AP: RT-DETR −0.0405; YOLOX-S +0.0659.
  - The gallery now draws a YOLOX-S advantage image (`sign-036`). This is consistent with the export holding 28 files instead of 27 (inferred).
  - Bundle SHA-256 `9708a96f…60c508b3118`.
- **Why this is unexplained:** T4 training is not bit-reproducible; epoch-1 losses also differed between the two earlier T4 runs. But the earlier same-code spread in RT-DETR adapted AP was 0.003, and here it is 0.04. The environment also changed at the same time (Python 3.12 in the uv venv, different wheel builds). Neither cause is isolated.
- **CPU stand-in gap still open:** the T4 values sit between the CPU stand-in (RT-DETR 0.8874; YOLOX-S 0.6994) and the 2026-09-30 run, so this run neither confirms nor refutes the CPU-versus-CUDA attribution.
- **Verdict:** the default path completed cleanly, but the adapted metrics don't match the 2026-09-30 T4 run, so this run is **not merge evidence** for this head.
- **Maintainer decision (2026-10-04):** Kurt accepted this run for merging. Every pre-adaptation metric, the dataset hash, VRAM, reload parity and the metric replay match 2026-09-30, and the recorded acceptance criteria do not gate on adapted scores. The adapted-score difference (RT-DETR AP 0.930 vs 0.970, YOLOX-S 0.893 vs 0.827) is recorded as run-to-run variance of bounded adaptation in the new environment, not yet isolated. Status stays **Candidate**.
- **Boundary:** default STANDARD path only; FULL, export re-run and BYOD were not exercised. Saved outputs were inspected; execution was not independently repeated. Status remains **Candidate**.

### DET2 review fixes — 2026-10-14

Notebook blob `929be36c` (`00130c3`) → new blob on branch `fix/closed-det-workshop-review-2026-10-12` (workshop revision `0.3.0-candidate`). **Hosted re-run pending; status remains Candidate.** The recorded runs above describe the earlier blobs.

- **What changed (generator `tools/build_detection_workshop.py` and `tools/detection_workshop_source.py`):**
  - DET2-M1: section 12 picks a per-model operating point on validation (highest validation F1 at IoU 0.5 over the swept thresholds; ties go higher); section 13 freezes it and section 14 refuses drift; section 15 keeps the shared 0.30 table, labelled as a raw-score threshold whose meaning differs between models, and adds recall at each model's operating point plus threshold-free AP50 per density group.
  - DET2-M2 (owner ruling B): section 16 is retitled "Disagreement and error gallery (this run)" and discloses the recorded spread by revision (gap +0.143 at `ade681b`, +0.037 at `46a300c`, cause not isolated); a FULL-only multi-seed repeat (seeds 1, 2, 3; RT-DETR and YOLOX-S) reports mean, minimum and maximum. STANDARD skips it.
  - DET2-M3 (owner ruling A): the canonical question and goal 6 are whole-system; section 11 discloses untuned recipes, 27 vs 108 optimizer steps and 19.3M vs 1.9M trainable parameters.
  - DET2-M4: section 10 names the re-initialised heads per family and states that the class-aware pre-adaptation AP is not a localisation-transfer measure; a display-only class-agnostic pre-adaptation AP is added.
  - DET2-M5: class-confusion tables (reference × predicted, `missed`, `no object`) at each model's operating point, with a prompt.
  - DET2-m1..m6: BYOD refusals print one message naming the CSV line, image_id or file and the rule (no traceback); NOTEBOOK_SPEC 2.5 and `notebook_kind`; the stale "fetch YOLOX at runtime" gate removed from the freeze file; downloads stated in the opening; exercises for goals 4 and 11; first-run wording; the reload warning silenced in `load_artifact` and explained in section 10.
- **Default-path numbers:** every previously computed metric, table and file is unchanged in code. Additive only: new printed tables and the files `validation/operating_points.csv`, `test/size_density_operating_point_metrics.csv` and `test/class_confusion.csv`; the frozen record gains `validation_operating_points`, `operating_point_rule` and `multi_seed_repeat`, and its `notebook_spec` and `candidate_release_gate` change, so `experiment_id`, the export file count and the bundle digest change.
- **Verification (not clean-runtime evidence):** generator `--check`, `validate_release_assets.py`, ruff and the offline pytest suite on Windows CPU; new regression tests exec the notebook's own cells with synthetic adapters. No model was trained and nothing ran on a GPU.
- **Still required:** a hosted T4 STANDARD Run all of the new blob; FULL in a fresh runtime (now including the multi-seed repeat); real-model valid and invalid BYOD (REL12).
- **Peer-review follow-up (2026-10-15, same branch):** Exercise F no longer ties false-detection counts to calibration and defines calibration with an explicit correctness event; section 15 prints each slice's class mix, `hits / objects` and contributing images, states the class/size confounding (from the generator: only stop signs drawn at radius 70 reach the 0.040 `large` cutoff; yield ≤ 0.038 and speed-limit ≤ 0.024 never do), renames the density AP column to `ap50_over_retained_predictions` and marks groups missing a class as not directly comparable; density is "objects per scene"; the confusion tables are labelled label-blind spatial-association counts with an `unmatched detection` row; the FULL multi-seed repeat runs from the verified freeze, refuses drift and saves `test/multi_seed_protocol.json`; Exercise G separates retraining a fixed vocabulary from open-vocabulary queries; a refused BYOD records `byod_status` `refused` and the reason in the provenance manifest, `workshop_summary.json` and the completion summary. Also added: a worked TP/FP/FN and AP example in section 6, BYOD group-split guidance, the timing label in section 17, and the per-image AP / gallery threshold note in section 16. No previously computed number changes; additive outputs: `test/size_density_group_composition.csv`, `byod` / `byod_status` fields, and (FULL) `test/multi_seed_protocol.json`.

### Colab CLI execution of `7240fd1` (blob `5937db0e`) — 2026-10-15

- **Files (byte for byte):** [executed notebook](execution-evidence/2026-10-15/DIMER_MultiModel_Closed_Set_Object_Detection_Workshop_7240fd1_colab-cli-t4.ipynb) SHA-256 `f865e23808694d40e22a408b8e078c980a71c63e5cbe7e5fa0c9cefec1a7d02b`; [exec log](execution-evidence/2026-10-15/DIMER_MultiModel_Closed_Set_Object_Detection_Workshop_7240fd1_colab-cli-t4_exec.log) `967d818cbf6fb3500acff088a682bf571f8b4d25e022c0824d608a637d16bed8`; [run summary](execution-evidence/2026-10-15/DIMER_MultiModel_Closed_Set_Object_Detection_Workshop_7240fd1_colab-cli-t4_run_summary.json) `4096cf01d947024bd3a13a9221fa33ca9791aa2551fb38514d1f361d94b5110f`.
- **Source match:** the committed blob was fetched at the 40-character SHA and Git-blob verified (`5937db0e0bce8171909af1377ea083ede946b1a3`); the executed notebook's 64 cells match it with zero source differences, ids and order unchanged.
- **Executor:** Colab CLI 0.7.4, `colab exec -f` on a fresh Colab Tesla T4 VM, session released afterwards. Sequential CLI execution, not a browser Run all: cell order comes from the exec log and the saved notebook has no execution counts. Kernel Python 3.13.15; isolated uv environment CPython 3.12.12, 57 locked packages, built in 51 s, no restart; torch 2.14.0+cu130, torchvision 0.29.0+cu130, Transformers 4.57.6, NumPy 2.1.3, `pin_mismatches` empty, device `cuda:0`. Forms and uploads were not exercised.
- **Execution:** 31/31 code cells (cells 5, 12 and 13 define code and print nothing), no saved errors, 191.4 s wall. Tier STANDARD (RT-DETR, YOLOX-S), `USE_BYOD=False`; `run_id` `20261015T003922Z-be5c02`, `experiment_id` `0bda30c5…b066480c7`.
- **Training ran for both detectors:** optimizer steps RT-DETR 27, YOLOX-S 108 (printed by section 11). RT-DETR epoch losses 37.20 / 23.76 / 19.13 (13.4 s); YOLOX-S total loss 4.49 / 2.88 / 2.27 / 1.93 / 1.48 / 1.52 (5.4 s). The training cell has no cache or skip path; RT-DETR's heads were re-initialised from the COCO checkpoint (80 → 3 classes). The short wall time is the 36-image training split.
- **Identical to 2026-09-30 and 2026-10-04:** `dataset_sha256` `823df4cf…302169` and the 36/12/12 split; validation pre-adaptation AP (RT-DETR 0.0858, YOLOX-S 0.0756); test pre-adaptation rows (below); training-stage peak VRAM 2553 / 621 MiB; reload parity (differences 0.0); metric replay PASS 4/4.

| Model | State | Test AP | AP50 | AP75 | 2026-10-04 AP / AP50 (`46a300c`) | 2026-09-30 AP / AP50 (`ade681b`) |
|---|---|---:|---:|---:|---:|---:|
| RT-DETR | pre | 0.0495 | 0.0495 | 0.0495 | 0.0495 / 0.0495 | 0.0495 / 0.0495 |
| RT-DETR | adapted | 0.9529 | 0.9717 | 0.9717 | 0.9299 / 0.9590 | 0.9704 / 0.9883 |
| YOLOX-S | pre | 0.0344 | 0.0376 | 0.0376 | 0.0344 / 0.0376 | 0.0344 / 0.0376 |
| YOLOX-S | adapted | 0.8760 | 1.0000 | 1.0000 | 0.8932 / 1.0000 | 0.8273 / 1.0000 |

- **Run-to-run variation (DET2-M2):** training losses diverge from epoch 1 (RT-DETR 37.2046 vs 37.1504 on 2026-10-04), so everything downstream of training differs. Every adapted value lies inside the range of the earlier T4 runs (RT-DETR AP 0.9299–0.9737 over 2026-09-26/09-30/10-04; YOLOX-S AP 0.8273–0.8932 after DET-01; adapted validation AP RT-DETR 0.8614 vs 0.8376–0.9278, YOLOX-S 0.8482 vs 0.8245–0.8794). No difference exceeds the earlier observed spread. The adapted test-AP gap is +0.077 (earlier +0.143 and +0.037). RT-DETR small-object recall at the shared 0.30 threshold is 0.50 (2026-10-04: 0.25); YOLOX-S 1.00 in both.
- **New DET2 outputs, as designed:** validation-chosen operating points RT-DETR 0.30, YOLOX-S 0.70 (frozen and re-checked at test); group-composition table; label-blind confusion tables (0 cross-class confusions for either model; RT-DETR misses 5 of 8 speed-limit signs at 0.30); display-only class-agnostic pre-adaptation AP (RT-DETR 0.0866, YOLOX-S 0.6122); multi-seed repeat printed as skipped (STANDARD); `byod_status` `disabled`. The export bundle holds 31 files (SHA-256 `d915d3c4…9addd6`).
- **Verdict:** **PASSED** on the default STANDARD path. This run satisfies "a hosted T4 STANDARD Run all of the new blob" above with the boundary below.
- **Boundary:** CLI sequential execution, not a browser Run all; forms and uploads not exercised; FULL (including the multi-seed repeat) and real-model BYOD (REL12) still not exercised. Status remains **Candidate**.

### Colab CLI execution of `95de437` (blob `e921b62e`) — 2026-10-25

Queued and executed with the Colab CLI serial suite; the files below are its durable copies. `run_summary.json` records `verdict` `PASS`, `exec_exit_code` 0, `input_unchanged` true and no overrides. The notebook bytes were fetched at the full SHA `95de437e8d9059872268f71d691be2c8a945dcef` and their git blob `e921b62e6b68225db557c4b0c224b6355df11fa9` equals PR #17's blob at `95de437`; the SHA-256 of the executed notebook equals the summary's `executed_sha256` (`c12aa18b…`). Local paths are redacted byte-wise to `<local-run-dir>` (the Colab CLI install path would become `<colab-cli-tool-dir>`; none occurred); the notebook and summary carried none.

| File (under `docs/execution-evidence/2026-10-25-95de437/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `rtdetr_detection_colab_95de437_colab-cli-t4_default.ipynb` | `c12aa18b8866b185e1d0c780199ee40ae2bc2ca207eddfd95034429cfce6aa71` | `c12aa18b8866b185e1d0c780199ee40ae2bc2ca207eddfd95034429cfce6aa71` (unchanged, no local path) |
| `rtdetr_detection_colab_95de437_colab-cli-t4_default_exec.log` | `0c2c4754274632e2adeb4e5f6398dbb081ddce801252753ac9a4840afd376151` | `004d40165d366d488cef5ce951d6b519bbaf5ced001946eab15720c0469335a7` (2 local-path replacements) |
| `rtdetr_detection_colab_95de437_colab-cli-t4_default_run_summary.json` | `9cefaeddc666b49bc20a2f0b17f4bf7541aecf43157c3e6570d2244d875407c3` | `9cefaeddc666b49bc20a2f0b17f4bf7541aecf43157c3e6570d2244d875407c3` (unchanged, no local path) |

- **Executor:** Colab CLI `colab exec -f` on a fresh Colab Tesla T4 VM (started 2026-10-25T14:26:19Z, released afterwards). Sequential CLI execution, not a browser Run all; order from the exec log, no execution counts; device `cuda:0`. 15/15 code cells logged (the carried modules print nothing), 0 errors, 110.0 s wall.
- **Printed results:** COCO scene 3/4 matched (stop sign 0.977, clock 0.965, traffic light 0.931; IoU 0.877 / 0.965 / 0.983, sports ball 0.000); degenerate probes blank 1 at 0.3 (`train` 0.333) / 87 at 0.05, noise 1 at 0.3 (`cat` 0.391) / 254 at 0.05; 40 records / 82 boxes; baseline AP 0.0; fine-tune 19,259,121 parameters trained, backbone frozen, epoch losses 39.6241 / 25.2125 / 19.6108, `train_seconds` 11.3; held-out AP 0.9774 / AP50 0.9825 / AP75 0.9825 (per-class AP50 stop-sign 1.0, yield-sign 0.9474, speed-limit-sign 1.0); zero-training reference on 5 held-out stop signs: unadapted COCO detector AP 1.0, re-headed baseline AP50 0.0, adapted AP 1.0; new-seed images: image 0 both yield signs found (IoU 0.971 / 0.983), image 1 the speed-limit sign missed (same-label IoU 0.0) and the yield sign found (0.983), image 2 the stop sign found (0.986); reload on 13 images × 300 queries, 26 detections at the threshold, max abs logit and box difference 0.0.
- **Prose check:** every run-derived number the notebook quotes holds for this run — the Section 4–5 values are equal; the epoch losses lie inside the stated Colab T4 ranges (39.57–39.65, 25.17–25.27, 19.16–20.12); AP 0.9774 lies inside "about 0.92 to 0.98"; yield-sign AP50 0.9474 inside 0.937–0.991 and speed-limit-sign 1.0 inside 0.835–1.000; the adapted stop-sign AP50 equals the unadapted detector's 1.000; `train_seconds` 11.3 is about 12 s; the reload difference is 0.0 with 26 detections.

### Colab CLI execution of `2497882` (blob `becde2c0`) — 2026-10-25

Queued and executed by a separate maintainer session with the Colab CLI serial suite; the files below are its durable copies. `run_summary.json` records `verdict` `PASS`, `exec_exit_code` 0, `input_unchanged` true, `stop_verified` true and no overrides. The SHA-256 of the executed notebook equals the summary's `executed_sha256` (`d9eb72fb…`), and the blob equals PR #17's blob at `2497882`. Local paths are redacted byte-wise to `<local-run-dir>`; the notebook and summary carried none.

| File (under `docs/execution-evidence/2026-10-25/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `rtdetr_detection_colab_2497882_colab-cli-t4_default.ipynb` | `d9eb72fb839d717593110894918e5e25c52c6905cdbe8e4da614f659babdb60b` | `d9eb72fb839d717593110894918e5e25c52c6905cdbe8e4da614f659babdb60b` (unchanged, no local path) |
| `rtdetr_detection_colab_2497882_colab-cli-t4_default_exec.log` | `01e10f5da2aaa51459b965333e4773af1332eedf7e6d4ad6fd8cd93b5f9df4a8` | `d1b6c44334e262b04943bc20ad1bd9142eb9f3b77611c5de383cbe7110871ad1` (2 local-path replacements) |
| `rtdetr_detection_colab_2497882_colab-cli-t4_default_run_summary.json` | `420300ddf333d5ba775f7cc71ba892b40e90bed0c3942dcd34a67ec689c26f1e` | `420300ddf333d5ba775f7cc71ba892b40e90bed0c3942dcd34a67ec689c26f1e` (unchanged, no local path) |

- **Executor:** Colab CLI `colab exec -f` on a fresh Colab Tesla T4 VM (started 2026-10-25T06:41:22Z, released afterwards). Sequential CLI execution, not a browser Run all; order from the exec log, no execution counts. Kernel Python 3.13.16; isolated uv environment Python 3.12.12, 48 locked packages, built in 62 s; `torch 2.14.0+cu130`, `transformers 4.57.6`; device `cuda:0`. 15/15 code cells logged (cells 3–4 are the carried modules and print nothing), 0 errors, 127.6 s wall.
- **Printed results:** 4 snapshot files fetched and verified at revision `df939e66…`; COCO scene 3/4 matched (stop sign 0.977, clock 0.965, traffic light 0.931; IoU 0.877 / 0.965 / 0.983, sports ball 0.000); degenerate probes blank 1 at 0.3 (`train` 0.333) / 87 at 0.05, noise 1 at 0.3 (`cat` 0.391) / 254 at 0.05; 40 records / 82 boxes, split 30 / 10 (65 / 17 boxes) with every class in both; baseline AP 0.0; fine-tune 19,259,121 of 42,733,137 parameters, backbone frozen, epoch losses 39.5658 / 25.2743 / 19.1633, `train_seconds` 11.6; held-out AP 0.979 / AP50 0.979 / AP75 0.979 (per-class AP50 stop-sign 1.0, yield-sign 0.937, speed-limit-sign 1.0); zero-training reference on 5 held-out stop signs: unadapted COCO detector AP 1.0, re-headed baseline AP50 0.0, adapted AP 1.0; new-seed images: image 0 both yield signs found (IoU 0.98 / 0.977), image 1 the speed-limit sign missed (same-label IoU 0.0) and the yield sign found (0.99), image 2 the stop sign found (0.983), every adapted score between 0.352 and 0.499.
- **Safetensors export and reload:** `"path": "outputs/rtdetr_adapter.safetensors", "bytes": 174748756`, `"sha256": "bcd21180204b166bcc10448ffc27e6eec0185ac4ae38ec307f8d75db12e21461"`, `"sidecar": "outputs/rtdetr_adapter.json"`, `"format": "rtdetr-adapter-v2"`, `"tensors": 812`; parity `"images": 13, "queries_per_image": 300, "detections_at_threshold": 24, "max_abs_logit_difference": 0.0, "max_abs_box_difference": 0.0`, then "Fresh reload verification passed: raw outputs on 13 images match within 0.001." 9 release outputs written; both BYOD refusal checks passed, BYOD branches off.
- **Differences from the 2026-10-25 run of the previous blob `0dd61e64`:** epoch losses 39.58 / 25.17 / 20.12 then, 39.57 / 25.27 / 19.16 now; held-out AP 0.9344 / AP50 0.942 then, 0.979 / 0.979 now; per-class AP50 then 1.0 / 0.991 / 0.835, now 1.0 / 0.937 / 1.0; reload parity counted 36 detections at 0.3 then, 24 now, max difference 0.0 both times; COCO scene, probes, dataset and split identical. Training is not bitwise reproducible across T4 sessions; one seeded split, no dispersion estimate.
- **Prose numbers checked against this run (REL13, notebook not edited):** The troubleshooting notes in the closing summary say the adapted AP in Section 9 "has ranged from about 0.85 to 0.95 between runs"; this run printed AP 0.979, above that range. Every other number the prose quotes is attributed to the worked CPU run or an earlier blob's hosted run and was not re-measured as such; the shared lines (COCO scene IoU 0.877 / 0.965 / 0.983, probes, dataset, split, 19,259,121 of 42,733,137 parameters, `train_seconds` 11.6, stop-sign reference 1.000) equal this run's output.
- **Boundary:** CLI sequential execution, not a browser Run all; forms and uploads not exercised; BYOD (REL12) not exercised.

### Colab CLI executions of `7456a3c` (blobs `0dd61e64`, `6de85e63`) — 2026-10-25

Both runs were queued and executed by a separate maintainer session with the same Colab CLI serial suite (its ledger row is in that session's local suite ledger, not committed here); the files below are its durable copies. Each run's `run_summary.json` records `verdict` `PASS`, `exec_exit_code` 0, `input_unchanged` true, `stop_verified` true and no overrides. The SHA-256 of each executed notebook equals the summary's `executed_sha256`, and each blob equals main's blob at `7456a3c`. Local paths are redacted byte-wise to `<local-run-dir>`; the notebooks and summaries carried none.

#### `rtdetr_detection_colab.ipynb` (blob `0dd61e64`), default E2E path

| File (under `docs/execution-evidence/2026-10-25/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `rtdetr_detection_colab_7456a3c_colab-cli-t4_default.ipynb` | `bd078ced80d1cba2868e53dd1513360ff2983666cfaac64934dfbd77a5be9f33` | `bd078ced80d1cba2868e53dd1513360ff2983666cfaac64934dfbd77a5be9f33` (unchanged, no local path) |
| `rtdetr_detection_colab_7456a3c_colab-cli-t4_default_exec.log` | `c35a151a939806d9052fb110b0330652bb83ab0f7a924046108c2da55e183b4b` | `f2c0196c1a124452aec39eb3c6bcd3aade55782b3e54e7e72b0705f4174c1c93` (2 local-path replacements) |
| `rtdetr_detection_colab_7456a3c_colab-cli-t4_default_run_summary.json` | `d4f0963138699f88ddd1645efed7ba9c18f3b20016833ee45ca1af832037c262` | `d4f0963138699f88ddd1645efed7ba9c18f3b20016833ee45ca1af832037c262` (unchanged, no local path) |

- **Executor:** Colab CLI `colab exec -f` on a fresh Colab Tesla T4 VM (session `suite-rtdetr-7456a3c-6128`, started 2026-10-25T00:24:15Z, released afterwards). Sequential CLI execution, not a browser Run all; order from the exec log, no execution counts. Kernel Python 3.13.16; isolated uv environment Python 3.12.12, 48 locked packages, built in 50 s; device `cuda:0`. 15/15 code cells logged (cells 3–4 are the carried modules and print nothing), 0 errors, 111.1 s wall.
- **Printed results:** 4 snapshot files fetched and verified at revision `df939e66…`; COCO scene 3/4 matched (IoU 0.877 / 0.965 / 0.983, sports ball 0.000); degenerate probes blank 1 at 0.3 (`train` 0.333) / 87 at 0.05, noise 1 at 0.3 (`cat` 0.391) / 254 at 0.05; 40 records / 82 boxes, split 30 / 10 with every class in both; baseline AP 0.0; 15 re-initialized parameter heads; fine-tune 19,259,121 of 42,733,137 parameters, backbone frozen in the model, epoch losses 39.5821 / 25.1714 / 20.1248, `train_seconds` 11.6; held-out AP 0.9344 / AP50 0.942 / AP75 0.942 (per-class AP50 stop-sign 1.0, yield-sign 0.991, speed-limit-sign 0.835); zero-training reference on 5 held-out stop signs: unadapted COCO detector AP 1.0, adapted AP 1.0; new-seed images: image 0 two yield signs with three `yield-sign` detections, image 1 the speed-limit sign missed (same-label IoU 0.0), image 2 the stop sign found (0.404).
- **Safetensors export and reload (first hosted exercise of this path):** `"path": "outputs/rtdetr_adapter.safetensors", "bytes": 174748756`, `"sha256": "de0f8e2ababac9ddbb24d9410a2bc82de9e8c2e53bf815b6894a3ae037c684e3"`, `"sidecar": "outputs/rtdetr_adapter.json"`, `"format": "rtdetr-adapter-v2"`, `"tensors": 812`; reload `'reloaded_source': 'artifact:rtdetr_adapter.safetensors', 'adapted': True`; parity `"images": 13, "queries_per_image": 300, "detections_at_threshold": 36, "max_abs_logit_difference": 0.0, "max_abs_box_difference": 0.0`, then "Fresh reload verification passed: raw outputs on 13 images match within 0.001." 9 release outputs written; both BYOD refusal checks passed, BYOD branches off.
- **Differences from the previous hosted run (2026-10-09, blob `e2ab45177aec`, `.pt` export):** epoch losses 39.65 / 25.20 / 19.44 then, 39.58 / 25.17 / 20.12 now; held-out AP 0.9452 / AP50 0.9578 then, 0.9344 / 0.942 now; per-class AP50 then stop 0.934, yield 0.939, speed-limit 1.0, now 1.0 / 0.991 / 0.835; adapted stop-sign reference AP 0.934 then, 1.0 now; reload parity counted 26 detections at 0.3 then, 36 now, max difference 0.0 both times; the artifact was a 174,923,973-byte `.pt` file then and is a 174,748,756-byte `.safetensors` file plus a 1,953-byte sidecar now (9 outputs, 8 then); new-seed images then had all 5 signs found, now the image-1 speed-limit sign is missed. The COCO scene, degenerate probes, dataset and split lines are identical. Training is not bitwise reproducible across T4 sessions, so these are run-to-run differences on one seeded split; no dispersion estimate exists.
- **Boundary:** CLI sequential execution, not a browser Run all; forms and uploads not exercised; BYOD (REL12) not exercised.

#### Workshop `DIMER_MultiModel_Closed_Set_Object_Detection_Workshop.ipynb` (blob `6de85e63`, revision `0.4.0-candidate`), default STANDARD path

| File (under `docs/execution-evidence/2026-10-25/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `DIMER_MultiModel_Closed_Set_Object_Detection_Workshop_7456a3c_colab-cli-t4.ipynb` | `6bd42d25b21d4eb3828a0a51bbb2b7635ae313e56fba4dbd4ffd126ee5e87ee2` | `6bd42d25b21d4eb3828a0a51bbb2b7635ae313e56fba4dbd4ffd126ee5e87ee2` (unchanged, no local path) |
| `DIMER_MultiModel_Closed_Set_Object_Detection_Workshop_7456a3c_colab-cli-t4_exec.log` | `6c0c461c2017d833ccb6e10ddd69cafdb812f374a71bf625b4ea1dbda2e04e4b` | `2bc6695195f858610adf524aa411456dd41d73dfc499ffc1f0e76684ff2c6fa4` (2 local-path replacements) |
| `DIMER_MultiModel_Closed_Set_Object_Detection_Workshop_7456a3c_colab-cli-t4_run_summary.json` | `8c05c64f86b582024e6d6262cef8e06a724814cb656289cc6f1bf91484efe7d5` | `8c05c64f86b582024e6d6262cef8e06a724814cb656289cc6f1bf91484efe7d5` (unchanged, no local path) |

- **Executor:** Colab CLI `colab exec -f` on a fresh Colab Tesla T4 VM (session `suite-rtdetr-7456a3c-e141`, started 2026-10-25T00:26:08Z, released afterwards), not a browser Run all. Kernel Python 3.13.16; isolated uv environment CPython 3.12.12, 57 locked packages, built in 53 s; torch 2.14.0+cu130, torchvision 0.29.0+cu130, Transformers 4.57.6, NumPy 2.1.3, `pin_mismatches` empty, device `cuda:0`. 32/32 code cells logged, 0 errors, 151.5 s wall. Tier STANDARD (RT-DETR, YOLOX-S), `USE_BYOD=False`; `run_id` `20261025T002717Z-580588`, `experiment_id` `36728823…a0a91d49`.
- **Training ran for both detectors:** optimizer steps RT-DETR 27, YOLOX-S 108. RT-DETR epoch losses 37.2211 / 24.2639 / 19.4408; YOLOX-S total loss 4.4908 / 2.8647 / 2.2232 / 1.9573 / 1.8820 / 1.7021; training seconds 13.63 / 5.29. Artifacts 166.65 MiB (RT-DETR) and 34.23 MiB (YOLOX-S), written by the cell titled "Adapter artifact files: safetensors weights plus a plain-JSON sidecar".
- **Identical to 2026-10-15:** `dataset_sha256` `823df4cf…ca302169` and the 36/12/12 split; validation pre-adaptation AP (RT-DETR 0.0858, YOLOX-S 0.0756); class-agnostic pre-adaptation AP (RT-DETR 0.0866, YOLOX-S 0.6122); the test pre-adaptation rows (below); training-stage peak VRAM 2553 / 621 MiB; validation operating points RT-DETR 0.30, YOLOX-S 0.70; reload parity (box and score differences 0.0 for both); metric replay PASS 4/4; 0 cross-class confusions for either model; multi-seed repeat printed as skipped (STANDARD); `byod_status` `disabled`.

| Model | State | Test AP | AP50 | AP75 | 2026-10-15 AP / AP50 (`7240fd1`) |
|---|---|---:|---:|---:|---:|
| RT-DETR | pre | 0.0495 | 0.0495 | 0.0495 | 0.0495 / 0.0495 |
| RT-DETR | adapted | 0.9157 | 0.9292 | 0.9292 | 0.9529 / 0.9717 |
| YOLOX-S | pre | 0.0344 | 0.0376 | 0.0376 | 0.0344 / 0.0376 |
| YOLOX-S | adapted | 0.8498 | 1.0000 | 1.0000 | 0.8760 / 1.0000 |

- **Differences from the 2026-10-15 run (DET2-M2 run-to-run variation):** training losses diverge from epoch 1 (RT-DETR 37.2211 vs 37.2046), so everything downstream of training differs. Adapted validation AP RT-DETR 0.8864 (then 0.8614), YOLOX-S 0.8791 (then 0.8482). The adapted RT-DETR test AP 0.9157 is below the earlier T4 range recorded above (0.9299–0.9737); the YOLOX-S value 0.8498 is inside its earlier range (0.8273–0.8932). The adapted test-AP gap is +0.066 (earlier +0.143, +0.037, +0.077). RT-DETR small-object recall at the shared 0.30 threshold is 0.875 (2026-10-15: 0.50); YOLOX-S 1.00 both times. Per-class adapted AP50 RT-DETR stop 0.9670, yield 1.0, speed-limit 0.8207; YOLOX-S 1.0 for all three. The export bundle holds 34 files (SHA-256 `afc10742…58126b51`; 31 on 2026-10-15). One seed, one runtime; these numbers do not rank the detectors.
- **Verdict:** **PASSED** on the default STANDARD path of blob `6de85e63`.
- **Boundary:** CLI sequential execution, not a browser Run all; forms and uploads not exercised; FULL (including the multi-seed repeat) and real-model BYOD (REL12) not exercised. Status remains **Candidate**.

### Colab CLI execution of `4444386` (blob `8e4a0fb7`) — 2026-10-25

Queued and executed with the Colab CLI serial suite; the files below are its durable copies. `run_summary.json` records `verdict` `PASS`, `exec_exit_code` 0, `input_unchanged` true and no overrides. The notebook bytes were fetched at the full SHA `44443867079120157004ff05669cae9748b70e60` and their git blob `8e4a0fb7ff0922dfd505aa72ea96eefdd7c44584` equals PR #17's blob at `4444386`; the SHA-256 of the executed notebook equals the summary's `executed_sha256` (`11db1a35…`). This is the confirming run for the historical-wording revision.

Evidence files for the 2026-10-25 default run of blob `8e4a0fb7`, copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-25-4444386/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `rtdetr_detection_colab_4444386_colab-cli-t4_default.ipynb` | `11db1a3542db9a5e09cd6eb0ecfdc3f4d93d5d201aaa0cc0eacada95117d483f` | `11db1a3542db9a5e09cd6eb0ecfdc3f4d93d5d201aaa0cc0eacada95117d483f` (unchanged, no local path) |
| `rtdetr_detection_colab_4444386_colab-cli-t4_default_exec.log` | `fea2483664a49d1b61317ec5af2b1e5344cbf74887bf0248ade384ad89a8bfc2` | `5aad749a3c02fc442c72510bb71a067c3332db27769c1955aa4a221f48932536` (2 local-path replacements) |
| `rtdetr_detection_colab_4444386_colab-cli-t4_default_run_summary.json` | `b6b4be2aa80bbdd2e79e755a8ef0425c363270b52c85a90a4f8253e8f774d41b` | `b6b4be2aa80bbdd2e79e755a8ef0425c363270b52c85a90a4f8253e8f774d41b` (unchanged, no local path) |

- **Executor:** Colab CLI `colab exec -f` on a fresh Colab Tesla T4 VM (started 2026-10-25T20:24:13Z, released afterwards). Sequential CLI execution, not a browser Run all; order from the exec log, no execution counts; device `cuda:0`. 15/15 code cells logged (the carried modules print nothing), 0 errors, 123.6 s wall; isolated environment 56 s.
- **Printed results:** COCO scene 3/4 matched (stop sign 0.977, clock 0.965, traffic light 0.931; IoU 0.877 / 0.965 / 0.983, sports ball 0.000); degenerate probes blank 1 at 0.3 (`train` 0.333) / 87 at 0.05, noise 1 at 0.3 (`cat` 0.391) / 254 at 0.05; baseline AP 0.0; fine-tune 19,259,121 parameters trained, backbone frozen, epoch losses 39.5561 / 25.1187 / 20.2263, `train_seconds` 11.6; held-out AP 0.9626 / AP50 0.9736 / AP75 0.9736 (per-class AP50 stop-sign 0.967, yield-sign 0.9538, speed-limit-sign 1.0); zero-training reference on the held-out stop signs: unadapted COCO detector AP 1.0; unseen images: image 0 two yield signs (IoU 0.965, 0.986), image 1 yield and speed-limit signs, image 2 stop sign not returned; adapter 812 tensors, 174,748,756 bytes; fresh reload matched raw outputs on 13 images (max abs logit and box difference 0.0).
- **Status:** blob `8e4a0fb7` is **Candidate**; this PR's runs cover the default path only. Remaining gates: REL12 (a hosted BYOD run with one accepted and one refused input) and REL2 (a browser Run all). Earlier blobs' runs are history (REL14).

**Drift recorded, not fixed (2026-10-26, Kurt's 2026-10-26 "historical wording, then stop" ruling).** The notebook prose of blob `8e4a0fb7` labels every run-derived number with the named recorded runs that printed it, so these differences are not prose errors. This confirming run printed these values outside the recorded values and spans the prose names. The notebook is not edited:

- Section 8 worked answer: epoch losses 39.5561 / 25.1187 / 20.2263; the prose records 39.57–39.65, 25.17–25.27 and 19.16–20.12 for the four named Colab T4 runs (all three of this run's values fall just outside; the loss still fell at every epoch, by about half).
- Section 9 worked answer: the adapted stop-sign AP50 was 0.967, below the unadapted detector's 1.000, as in one of the four named Colab runs (0.934, `e2ab4517`); the prose records this direction per run and says adaptation can cost a little. Not outside a stated value; noted for completeness.

### Colab CLI execution of `da680b8` (blob `7579885c`) — 2026-10-26

Queued and executed with the Colab CLI serial suite as the confirming run of the final prose pass; the files below are its durable copies. `run_summary.json` records `verdict` `PASS`, `exec_exit_code` 0, `input_unchanged` true and no overrides. The notebook bytes were fetched at the full SHA `da680b8787e445ce6847d7ef200966154d896d94` and their Git blob checked against `7579885cd623a53fd7b0d27cb01c6b32c29cf883` before the session.

Evidence files for the 2026-10-26 default run of blob `7579885c` (confirming run of the final prose pass), copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-26-da680b8/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `rtdetr_detection_colab_da680b8_colab-cli-t4_default.ipynb` | `28fb0f2e38008732b37784ade8ee1933271e2edfe25cc7bae65d7324aff29b43` | `28fb0f2e38008732b37784ade8ee1933271e2edfe25cc7bae65d7324aff29b43` (unchanged, no local path) |
| `rtdetr_detection_colab_da680b8_colab-cli-t4_default_exec.log` | `b897e7510d1341593f76fb9a27a8c194872f4d60375d4c75a37d71bdc4bc4040` | `7db55caee09cf141a2919189f99522eac97f1917fe4ca0e6ab680a969c241400` (2 local-path replacements) |
| `rtdetr_detection_colab_da680b8_colab-cli-t4_default_run_summary.json` | `4baef50f763450aae44e44d0796c3dc74629f6132169c1558f5e466cd8059cef` | `4baef50f763450aae44e44d0796c3dc74629f6132169c1558f5e466cd8059cef` (unchanged, no local path) |

- **Executor:** Colab CLI `colab exec -f` on a fresh Colab Tesla T4 VM (started 2026-10-26T01:26:52Z, released afterwards). Sequential CLI execution, not a browser Run all; order from the exec log, no execution counts; device `cuda:0`. 15/15 code cells logged, 0 errors, one pass, no restart, 109.0 s wall; isolated environment built in 46 s (48 locked packages, CPython 3.12.12).
- **Printed results:** COCO scene 3/4 matched (stop sign 0.977, clock 0.965, traffic light 0.931; IoU 0.877 / 0.965 / 0.983, sports ball 0.000); degenerate probes blank 1 at 0.3 (`train` 0.333) / 87 at 0.05, noise 1 at 0.3 (`cat` 0.391) / 254 at 0.05; baseline AP 0.0; fine-tune 19,259,121 of 42,733,137 parameters, epoch losses 39.5894 → 24.8445 → 19.8980, `train_seconds` 11.6; adapted AP 0.972, AP50 0.979, AP75 0.979 (per-class AP50 stop-sign 1.0, yield-sign 0.9371, speed-limit-sign 1.0); unadapted COCO detector on the 5 held-out stop signs AP 1.000, adapted AP 1.000; unseen images: image 0 two `yield-sign` boxes (IoU 0.971, 0.979), image 1 two `yield-sign` boxes (one at IoU 0.989, the speed-limit sign missed), image 2 no detection; adapter `rtdetr_adapter.safetensors` 174,748,756 bytes, 812 tensors; fresh reload on 13 images `max_abs_logit_difference` 0.0, `max_abs_box_difference` 0.0 (tolerance 0.001).
- **Status:** blob `7579885c` is **Candidate**; this PR's runs cover the default path only. Remaining gates: REL12 (a hosted BYOD run with one accepted and one refused input) and REL2 (a browser Run all). Earlier blobs' runs are history (REL14).

**Final prose pass: drift record (2026-10-26).** Under Kurt's 2026-10-26 ruling the notebook prose of blob `7579885c` is not changed again: a value this run printed that differs from the values the prose attributes to the named earlier runs is recorded here as drift, not fixed. Checked against this run's output: the COCO scene 3/4 matched, the degenerate probes (blank 1 at 0.3, `train` 0.333, 87 at 0.05; noise 1 at 0.3, `cat` 0.391, 254 at 0.05), the 19,259,121 of 42,733,137 trainable parameters and the re-headed baseline AP 0.000 equal the prose's values; `train_seconds` 11.6 lies in the named runs' 11.3–11.6 s; the adapted AP 0.972 lies in the prose's hosted span 0.9161–0.979 (it equals none of the five named values, which the prose gives as a span); the per-class AP50 (stop-sign 1.000, equal to the unadapted detector's 1.000; yield-sign 0.9371; speed-limit-sign 1.000) lies inside the named Colab runs' 0.937–0.991 and 0.835–1.000; on the unseen images this run found 3 of the 5 signs (both yield signs on image 0, the yield sign on image 1 with one duplicate box; image 1's speed-limit sign and image 2's stop sign missed), which the prose does not state for every run.

Drift (this run versus the prose's named-run values):

- Section 8 worked answer (md28): epoch losses 39.5894 / 24.8445 / 19.8980; the prose records 39.57–39.65, 25.17–25.27 and 19.16–20.12 for the four named Colab T4 runs. The epoch-2 loss falls below the recorded span; epochs 1 and 3 lie inside.

Rule check: no expectation range, timing promise, run-status sentence or unlabelled run-derived number was found in the notebook prose.
