# Weight provenance and DIMER hosting

- Upstream: `PekingU/rtdetr_r50vd`
- Immutable revision: `df939e661d8c52e80608d1ec566561aabd25a4e7` (the Hub's `main` resolved to this commit on 2026-09-14)
- Weight format: SafeTensors (`model.safetensors`, 172,175,856 bytes, float32); the upstream repository hosts no pickle checkpoint at this revision.
- Manifest: `weights/rtdetr-r50vd/dimer-base-manifest.json` (4 files: `README.md`, `config.json`, `model.safetensors`, `preprocessor_config.json`; 172,190,863 bytes total, per-file SHA-256)
- Upstream weight license: Apache-2.0 (the checkpoint's `README.md` front matter and the Hub's licence tag)
- DIMER hosting: Apache-2.0 permits use, modification, distribution, and commercial use subject to preservation of the licence and notices. The Git repository does not vendor the checkpoint (`weights/**/*.safetensors` is git-ignored); DIMER may mirror the pinned snapshot in its model store under the upstream license.
- Fresh clone: `stage_missing_files(allow_download=True)` fetches only the manifest-listed files absent on disk, at the pinned revision, into the snapshot directory; `verify_snapshot()` then checks every file before any load. `weights/**` is marked `-text` in `.gitattributes` so Windows `core.autocrlf` cannot rewrite the committed small files and break their digests.
- Loader trust boundary: Transformers `RTDetrForObjectDetection` / `AutoImageProcessor` (`RTDetrImageProcessor`, the slow processor the snapshot declares) with `trust_remote_code=False`, `local_files_only=True` from the verified directory; the backbone is the in-library `RTDetrResNet` described by the config's `backbone_config` (`use_timm_backbone` false, `use_pretrained_backbone` false — passed explicitly anyway), so nothing is fetched at construction (the smoke run loaded with `HF_HUB_OFFLINE=1`).
