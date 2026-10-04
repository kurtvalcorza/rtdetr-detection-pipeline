# ruff: noqa: E501
"""uv isolated environment of the closed-set detection workshop (2026-10-03).

Static checks on the generated notebook plus two executions: the bootstrap cell's platform guard, and (POSIX only,
because the router passes pipe descriptors to the worker) the router's persistent worker on this interpreter.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
NOTEBOOK = ROOT / "tutorials" / "DIMER_MultiModel_Closed_Set_Object_Detection_Workshop.ipynb"
LOCK = TOOLS / "detection-workshop-requirements.lock"
REQUIREMENTS_IN = TOOLS / "detection-workshop-requirements.in"
sys.path.insert(0, str(TOOLS))
from detection_workshop_source import CELLS  # noqa: E402
from detection_workshop_uv import PIECE_LIMIT, pieces, split_long_lines  # noqa: E402


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def cell_by_id(cell_id):
    return "".join(next(c for c in notebook()["cells"] if c["id"] == cell_id)["source"])


def code_sources():
    return [(c["id"], "".join(c["source"])) for c in notebook()["cells"] if c["cell_type"] == "code"]


def assigned(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def lock_direct_pins():
    lines = REQUIREMENTS_IN.read_text(encoding="utf-8").splitlines()
    return [line.split("#")[0].strip() for line in lines if line.split("#")[0].strip()]


def test_no_kernel_install_and_no_restart_guard():
    for _, source in code_sources():
        assert not re.search(r"""['"]-m['"],\s*['"]pip['"]""", source)
        assert "pip install" not in source.replace("uv pip install", "")
        assert "Restart session" not in source and "stale_loaded_packages" not in source


def test_only_the_two_leading_cells_run_in_the_kernel():
    ids = [cell_id for cell_id, source in code_sources() if "# dimer: kernel cell" in source]
    assert ids == ["dimer-detection-workshop-uv-env", "dimer-detection-workshop-uv-route"]
    assert [cell_id for cell_id, _ in code_sources()][:2] == ids
    # Every DET review cell keeps its numbered id after the inserted infrastructure cells.
    numbered = [c["id"] for c in notebook()["cells"] if not c["id"].startswith("dimer-detection-workshop-uv")]
    assert numbered == [f"dimer-detection-workshop-{i:02d}" for i in range(len(CELLS))]


def test_lock_has_hashes_for_every_package():
    text = LOCK.read_text(encoding="utf-8")
    assert "\r" not in text
    blocks = re.split(r"\n(?=[a-z0-9])", text)
    packages = [b for b in blocks if re.match(r"[a-z0-9][a-z0-9._-]*==", b)]
    assert len(packages) == 57
    for block in packages:
        assert "--hash=sha256:" in block, block.splitlines()[0]
    assert "--generate-hashes" in text and "x86_64-manylinux_2_28" in text and "--python-version 3.12" in text


def test_carried_lock_matches_the_file_and_its_digest():
    source = cell_by_id("dimer-detection-workshop-uv-env")
    text = LOCK.read_text(encoding="utf-8")
    assert assigned(source, "LOCK_TEXT") == text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert assigned(source, "LOCK_SHA256") == digest
    assert notebook()["metadata"]["dimer"]["carried_files"]["requirements.lock.txt"]["sha256"] == digest
    assert assigned(source, "LOCKED_PACKAGES") == 57


def test_install_is_hash_locked_wheels_only_into_the_venv():
    source = cell_by_id("dimer-detection-workshop-uv-env")
    assert '"--require-hashes", "--only-binary", ":all:"' in source
    assert '"--python", str(ISOLATED_PYTHON)' in source
    assert '"--managed-python", "--python", MANAGED_PYTHON' in source
    assert assigned(source, "MANAGED_PYTHON") == "3.12.12"
    assert assigned(source, "UV_BYTES") == 20081404
    assert "uv-0.12.15-py3-none-manylinux" in assigned(source, "UV_URL")
    assert "hashlib.sha256(wheel).hexdigest() != UV_SHA256" in source


def test_pins_agree_across_lock_bootstrap_and_runtime_cell():
    pins = lock_direct_pins()
    assert pins == assigned(cell_by_id("dimer-detection-workshop-uv-env"), "PINS")
    assert pins == assigned(CELLS[5]["source"], "PINS")
    text = LOCK.read_text(encoding="utf-8")
    for pin in pins:
        assert f"\n{pin} \\\n" in text, pin
    # Former exact pins are unchanged; the two former ranges are fixed inside their ranges.
    for pin in ["torch==2.14.0", "torchvision==0.29.0", "torchaudio==2.11.0", "transformers==4.57.6",
                "safetensors==0.8.0", "numpy==2.1.3", "pillow==11.3.0", "huggingface-hub==0.36.2", "scipy==1.18.1"]:
        assert pin in pins


def test_stages_run_on_the_venv_python():
    source = cell_by_id("dimer-detection-workshop-uv-route")
    assert "IsolatedRuntime(ISOLATED_PYTHON)" in source
    assert "[str(python), \"-c\", _WORKER_SOURCE" in source
    assert 'MPLBACKEND="Agg"' in source and 'DIMER_ISOLATED_WORKER="1"' in source
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        assert f'"{name}"' in source
    assert 'ISOLATED_PYTHON = ISOLATED_ENV / "bin" / "python"' in cell_by_id("dimer-detection-workshop-uv-env")


def test_router_names_are_bound_by_the_two_kernel_cells():
    # The router uses os, subprocess and sys from the bootstrap cell; a missing import only shows on Linux.
    boot = ast.parse(cell_by_id("dimer-detection-workshop-uv-env"))
    route = ast.parse(cell_by_id("dimer-detection-workshop-uv-route"))
    imported = {(a.asname or a.name).split(".")[0] for tree in (boot, route) for n in tree.body
                if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    assigned_names = {t.id for tree in (boot, route) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      for t in n.targets if isinstance(t, ast.Name)}
    defined = {n.name for n in route.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    import builtins
    for node in route.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_WORKER_SOURCE":
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id in {"os", "sys", "subprocess", "signal"}:
                assert sub.value.id in imported, sub.value.id
    for name in ("SKIP_INSTALL", "ISOLATED_PYTHON"):
        assert name in assigned_names
    assert {"IsolatedRuntime", "_route_to_isolated_runtime"} <= defined and "print" in dir(builtins)


def test_no_notebook_line_over_2000_characters():
    for cell in notebook()["cells"]:
        for line in cell["source"]:
            assert len(line.rstrip("\n")) <= 2000, cell["id"]


def test_split_literals_equal_the_carried_originals():
    # Cell 18 carried nine YOLOX files on one 49,902-character line; the notebook now carries pieces of at most
    # 1,000 characters that evaluate to the same literals.
    original, generated = CELLS[18]["source"], cell_by_id("dimer-detection-workshop-18")
    assert max(len(line) for line in original.split("\n")) > 2000
    for name in ("YOLOX_SOURCE_FILES", "YOLOX_LICENSE"):
        assert assigned(generated, name) == assigned(original, name)
    for node in ast.walk(ast.parse(generated)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            segment = ast.get_source_segment(generated, node)
            for line in segment.split("\n"):
                assert len(line.strip()) <= PIECE_LIMIT
    # Everything outside the two literals is unchanged.
    def without_literals(source):
        tree = ast.parse(source)
        tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in {"YOLOX_SOURCE_FILES", "YOLOX_LICENSE"})]
        return ast.dump(tree)
    assert without_literals(generated) == without_literals(original)


def test_split_helpers():
    text = "a" * 2500 + "\n" + "b\\c'\n" * 400
    assert "".join(pieces(text)) == text and all(len(repr(p)) <= PIECE_LIMIT for p in pieces(text))
    line = "X=" + repr({"k": text, "j": "short"})
    out = split_long_lines("print(1)\n" + line)
    assert assigned(out, "X") == {"k": text, "j": "short"}
    assert max(len(part) for part in out.split("\n")) <= PIECE_LIMIT + 8
    with pytest.raises(SystemExit):
        split_long_lines("f(" + "1," * 1500 + ")")


def test_bootstrap_refuses_a_non_linux_runtime(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(a))
    with pytest.raises(RuntimeError, match="Linux x86_64"):
        exec(compile(cell_by_id("dimer-detection-workshop-uv-env"), "uv-env", "exec"), {})
    assert called == [] and not any(tmp_path.iterdir())


def test_runtime_cell_refuses_a_drifted_isolated_environment(monkeypatch):
    import importlib.metadata

    tree = ast.parse(CELLS[5]["source"])
    prefix = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == "numpy" for alias in node.names):
            break
        prefix.append(node)
    module = compile(ast.Module(body=prefix, type_ignores=[]), "runtime-prefix", "exec")
    versions = {pin.split("==")[0]: pin.split("==")[1] for pin in lock_direct_pins()}
    versions["torch"] += "+cu130"
    monkeypatch.setattr(importlib.metadata, "version", lambda name: versions[name])
    monkeypatch.setenv("DIMER_ISOLATED_WORKER", "1")
    exec(module, {})
    versions["numpy"] = "2.5.3"
    with pytest.raises(RuntimeError, match="locked versions"):
        exec(module, {})
    monkeypatch.delenv("DIMER_ISOLATED_WORKER")
    ns = {}
    exec(module, ns)  # an executor's own kernel records the drift instead
    assert ns["mismatches"] == [("numpy", "2.5.3", "2.1.3")]


def router_runtime():
    """The router cell's worker and IsolatedRuntime, without IPython (not installed in CI)."""
    source = cell_by_id("dimer-detection-workshop-uv-route")
    tree = ast.parse(source)
    keep = [n for n in tree.body if (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_WORKER_SOURCE")
            or (isinstance(n, ast.ClassDef))]
    ns = {"__name__": "router"}
    exec("import os, signal, subprocess, sys\nfrom multiprocessing.connection import Connection", ns)
    exec(compile(ast.Module(body=keep, type_ignores=[]), "router", "exec"), ns)
    return ns


@pytest.mark.skipif(os.name != "posix", reason="the router passes pipe descriptors to the worker (POSIX only)")
def test_worker_keeps_state_streams_output_and_stops_on_errors(capsys):
    ns = router_runtime()
    shown = []
    runtime = ns["IsolatedRuntime"](sys.executable, display=lambda bundle, raw: shown.append(bundle))
    try:
        runtime.run("import os\nvalue = 41\nprint('isolated', os.environ.get('DIMER_ISOLATED_WORKER'), os.environ.get('MPLBACKEND'))")
        runtime.run("value += 1\nvalue")
        assert shown[-1]["text/plain"] == "42"
        with pytest.raises(ns["IsolatedCellError"], match="ZeroDivisionError"):
            runtime.run("1/0")
        runtime.run("print(value)")
    finally:
        runtime.close()
    out = capsys.readouterr()
    assert "isolated 1 Agg" in out.out and "42" in out.out
    assert "ZeroDivisionError" in out.err
