# ruff: noqa: E501,I001
"""Static contract tests for the closed-set object detection workshop notebook."""
from __future__ import annotations
import ast
import contextlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "tutorials" / "DIMER_MultiModel_Closed_Set_Object_Detection_Workshop.ipynb"
MANIFEST = REPO / "weights" / "rtdetr-r50vd" / "dimer-base-manifest.json"
YOLOX_DIGESTS = {
    "yolox_s": (72_089_125, "f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30"),
    "yolox_x": (793_463_373, "5652330b6ae860043f091b8f550a60c10e1129f416edfdb65c259be6caf355cf"),
}


def load():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells():
    return ["".join(cell["source"]) for cell in load()["cells"] if cell["cell_type"] == "code"]


def body():
    return "\n".join("".join(cell.get("source", [])) for cell in load()["cells"])


def notebook_constants():
    out = {}
    for cell in code_cells():
        for node in ast.parse(cell).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                with contextlib.suppress(ValueError):
                    out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def test_generator_parity():
    subprocess.run([sys.executable, str(REPO / "tools" / "build_detection_workshop.py"), "--check"], cwd=REPO, check=True)


def test_metadata():
    meta = load()["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.1"
    assert meta["notebook_profile"] == "E2E"
    assert meta["notebook_mode"] == "WORKSHOP"
    assert meta["standalone"] is True
    assert meta["default_tier"] == "STANDARD"
    assert meta["worker_required"] is False
    assert meta["credentials_required"] is False
    assert meta["clean_runtime_evidence"] == "pending"


def test_code_cells_compile():
    for cell in code_cells():
        compile(cell, "cell", "exec")


def test_rtdetr_snapshot_matches_the_repository_manifest():
    spec = notebook_constants()["RTDETR_SPEC"]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert spec["model_id"] == manifest["modelId"]
    assert spec["revision"] == manifest["revision"]
    assert spec["files"] == {f["path"]: (f["bytes"], f["sha256"]) for f in manifest["files"]}


def test_yolox_checkpoints_are_pinned():
    variants = notebook_constants()["YOLOX_VARIANTS"]
    for key, (size, digest) in YOLOX_DIGESTS.items():
        assert (variants[key]["bytes"], variants[key]["sha256"]) == (size, digest)
        assert "/releases/download/0.1.1rc0/" in variants[key]["url"]


def test_contract_present():
    text = body()
    for literal in [
        'WORKSHOP_TIER = "STANDARD"',
        "USE_BYOD = False",
        '"stop-sign"', '"yield-sign"', '"speed-limit-sign"',
        "YOLOX_EVAL_NMS = 0.65",
        "frozen_experiment.json",
        "rtdetr-adapter-v1",
        "dimer-yolox-detection-adapter/1",
        "weights_only=True",
        "workshop_summary.json",
        "experiment_manifest.json",
    ]:
        assert literal in text


def test_no_runtime_repo_dependency():
    text = body()
    for forbidden in ["git clone ", "pip install -e", "pip install yolox", "raw.githubusercontent.com/kurtvalcorza", "import rtdetr_detection_pipeline", "dimer-backend", "load_state_dict_from_url"]:
        assert forbidden not in text


def test_clean_notebook():
    for cell in load()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_json_is_imported_before_first_use():
    cells = code_cells()
    first_use = next(i for i, cell in enumerate(cells) if "json." in cell)
    assert any("import json" in cell for cell in cells[: first_use + 1])


def test_scipy_is_pinned_for_rtdetr_training():
    # transformers' RTDetrHungarianMatcher (the RT-DETR training loss) requires SciPy.
    assert any(pin.startswith("scipy==") for pin in notebook_constants()["PINS"])
