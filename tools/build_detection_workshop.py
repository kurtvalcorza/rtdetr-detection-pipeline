# ruff: noqa: E501,I001
"""Generate the DIMER closed-set object detection workshop notebook.

The cells of ``detection_workshop_source.CELLS`` keep their numbered ids. Since 0.2.0 (2026-10-03) the four cells of
``detection_workshop_uv.uv_cells()`` (the uv isolated environment: a bootstrap cell carrying the hash lock
``tools/detection-workshop-requirements.lock`` and a router cell) are inserted before section 1 under their own ids,
and every code line longer than 2,000 characters is rewritten as parenthesised runs of string pieces of at most
1,000 characters (``split_long_lines``, ``ast.literal_eval`` equality asserted). ``--check`` fails when the notebook
differs from the cells or from the carried lock.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from detection_workshop_source import CELLS
from detection_workshop_uv import LINE_LIMIT, LOCK_PATH, lock_sha256, lock_text, split_long_lines, uv_cells

NOTEBOOK_NAME = "DIMER_MultiModel_Closed_Set_Object_Detection_Workshop.ipynb"
UV_INSERT_AT = 2  # after the title and the learning goals, before "## 1. Notebook controls"


def ordered_cells():
    numbered = [{**cell, "id": f"dimer-detection-workshop-{index:02d}"} for index, cell in enumerate(CELLS)]
    return numbered[:UV_INSERT_AT] + uv_cells() + numbered[UV_INSERT_AT:]


def build_notebook():
    rendered = []
    for cell in ordered_cells():
        source = split_long_lines(cell["source"]) if cell["kind"] == "code" else cell["source"]
        if cell["kind"] == "code" and max(len(line) for line in source.split("\n")) > LINE_LIMIT:
            raise SystemExit(f"{cell['id']}: a code line is longer than {LINE_LIMIT} characters")
        base = {
            "id": cell["id"],
            "metadata": cell.get("metadata", {}),
            "source": source.splitlines(keepends=True),
        }
        if cell["kind"] == "markdown":
            rendered.append({"cell_type": "markdown", **base})
        else:
            rendered.append({"cell_type": "code", "execution_count": None, "outputs": [], **base})
    return {
        "cells": rendered,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "dimer": {
                "candidate_gate": "fresh STANDARD/FULL T4 and real-model BYOD qualification",
                "canonical_runtime": "NVIDIA Tesla T4",
                "capability": "multi-model-closed-set-object-detection",
                "carrier": "comparative bounded-adaptation object-detection workshop",
                "clean_runtime_evidence": "pending",
                "credentials_required": False,
                "dataset": "60 deterministic 640x640 three-class synthetic traffic-sign scenes",
                "default_tier": "STANDARD",
                "notebook_mode": "WORKSHOP",
                "notebook_profile": "E2E",
                "notebook_spec": "2.1",
                "release_status": "candidate",
                "standalone": True,
                "worker_required": False,
                "generated_from": {
                    "repository": "kurtvalcorza/rtdetr-detection-pipeline",
                    "source": "tools/detection_workshop_source.py",
                    "generator": "tools/build_detection_workshop.py",
                },
                "carried_files": {
                    "requirements.lock.txt": {
                        "source": f"tools/{LOCK_PATH.name}",
                        "sha256": lock_sha256(lock_text()),
                    },
                },
                "runtime_environment": {
                    "platform": "Linux x86_64 only (Google Colab, Kaggle, Linux Jupyter)",
                    "python": "CPython 3.12.12 (uv managed)",
                    "uv": "0.12.15 (wheel pinned by size and SHA-256)",
                    "install": "uv pip install --require-hashes --only-binary :all: -r requirements.lock.txt",
                    "execution": "every code cell after the two kernel cells runs in one persistent worker process on the venv Python",
                    "kernel_installs": "none; no restart",
                },
                "revisions": [
                    {
                        "revision": "0.2.0-candidate",
                        "date": "2026-10-03",
                        "base_commit": "b7bb099bd05215b3969be1004ca4c9aedb7a298d",
                        "base_notebook_blob": "395cb93508166de25ae567f778018e723a0e1c61",
                        "change": "uv isolated environment: no kernel install and no restart guard; a hash-locked CPython 3.12.12 venv and a persistent worker that runs every later cell; carried literals split into pieces of at most 1,000 characters; Linux x86_64 only",
                        "fixes_kept": ["DET-01", "DET-02", "DET-03", "DET-04", "DET-05", "DET-06", "DET-07"],
                    },
                ],
            },
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "workshop_revision": "0.2.0-candidate",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def serialized():
    return json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = args.out or repo / "tutorials" / NOTEBOOK_NAME
    content = serialized()
    if args.check:
        if not out.exists() or out.read_text(encoding="utf-8") != content:
            raise SystemExit(f"STALE: {out}; regenerate the workshop notebook")
        print(f"OK: {out}")
        return 0
    out.write_text(content, encoding="utf-8")
    print(out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
