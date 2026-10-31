"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both from the generated notebook:
- the router (install cell, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as
  one line, while every other exception keeps its traceback;
- the BYOD cell turns every user-input problem into `BYODRefusal` and removes the earlier run's
  results before it stops.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import types
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "rtdetr_detection_colab.ipynb"
BYOD_CELL_ID = "rtdetr_detection-41"


@pytest.fixture(scope="module")
def cells() -> list[dict]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _router_cell(cells: list[dict]) -> str:
    return next(_source(c) for c in cells if '_WORKER_SOURCE = r"""' in _source(c))


def _router_namespace(cells: list[dict], start: str) -> dict:
    source = _router_cell(cells)
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(cells, messages):
    namespace = _router_namespace(cells, "class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def test_router_relays_a_refusal_as_one_line_without_a_traceback(cells, capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime(cells, [("refusal", "etc: not accepted")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: etc: not accepted"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(cells, capsys):
    namespace, runtime = _kernel_runtime(cells, [("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(cells, capsys):
    _, runtime = _kernel_runtime(cells, [("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")
    assert runtime._send.sent == [("run", "print('hello')\n")]


def test_router_has_the_refusal_channel_before_the_error_branch(cells):
    source = _router_cell(cells)
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    assert "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/." in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


# --- router, end to end with a real worker (POSIX: the router uses pass_fds and start_new_session) ----------------


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(cells, capsys):
    namespace = _router_namespace(cells, '_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('etc: not accepted')\n")
        assert caught.value._render_traceback_() == ["Refused: etc: not accepted"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


# --- BYOD cell ----------------------------------------------------------------------------------------------------

CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _byod_cell(cells: list[dict]) -> dict:
    (cell,) = [c for c in cells if c.get("id") == BYOD_CELL_ID]
    return cell


def test_byod_refusal_is_carried_verbatim_and_the_cell_id_is_stable(cells):
    assert CARRIED in _source(_byod_cell(cells))


def _one_line(exc: BaseException, file_name: str, rule: str) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert line.startswith(f"Refused: {file_name}: "), line
    assert rule in line and "\n" not in line and "Traceback" not in line, line
    assert "again" in line, line
    return line


def _colab(monkeypatch, uploaded):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    monkeypatch.setitem(sys.modules, "google.colab", colab)


# --- rtdetr: Section 13, both branches ----------------------------------------------------------------------------

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import rtdetr_detection_pipeline as package  # noqa: E402

EARLIER_IMAGE = {"byod_source": "old", "byod_image": "old", "byod_res": "old"}
EARLIER_DATASET = {"byod_records": "old", "byod_manifest": "old", "byod_adapted": "old", "byod_descriptor": "old", "byod_max_diff": "old"}
DATASET_FILES = ("byod_result.json", "byod_adapter.safetensors", "byod_adapter.json")


def _cell_code(cells, *, image: str | None = None, dataset: str | None = None, classes: str = "") -> str:
    source = _source(_byod_cell(cells))
    if image is not None:
        source = source.replace("USE_BYOD_IMAGE = False", "USE_BYOD_IMAGE = True").replace("BYOD_IMAGE_PATH = ''", f"BYOD_IMAGE_PATH = {image!r}")
    if dataset is not None:
        source = source.replace("USE_BYOD_DATASET = False", "USE_BYOD_DATASET = True").replace("BYOD_DATASET_PATH = ''", f"BYOD_DATASET_PATH = {dataset!r}")
        source = source.replace("BYOD_CLASS_NAMES = ''", f"BYOD_CLASS_NAMES = {classes!r}")
    return source


def _namespace(tmp_path: Path, monkeypatch) -> dict:
    monkeypatch.chdir(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir(exist_ok=True)
    for name in DATASET_FILES:
        (outputs / name).write_text("earlier run", encoding="utf-8")
    ns = {name: getattr(package, name) for name in package.__all__}
    ns.update({"os": os, "json": json, "np": np, "Path": Path, "Image": Image, "OUTPUTS": Path("outputs"), "threshold": 0.3, "EPOCHS": 1, "SEED": 0, "HOLDOUT": 0.25, "pipe": "kept"})
    ns.update(EARLIER_IMAGE)
    ns.update(EARLIER_DATASET)
    return ns


def _run_refused(code: str, ns: dict) -> BaseException:
    with pytest.raises(ValueError) as caught:
        exec(compile(code, "<section 13>", "exec"), ns)
    return caught.value


def _dataset_refused(ns: dict, tmp_path: Path) -> None:
    assert not set(EARLIER_DATASET) & set(ns), "an earlier run's dataset results survived the refusal"
    assert not any((tmp_path / "outputs" / name).exists() for name in DATASET_FILES)
    assert set(EARLIER_IMAGE) <= set(ns) and ns["pipe"] == "kept"  # the other branch is untouched


def test_scripted_refusal_dataset_path_is_not_a_zip(cells, tmp_path, monkeypatch):
    """The hosted refusal run sets USE_BYOD_DATASET=True;BYOD_DATASET_PATH='/etc/hostname': a file in every fresh runtime that is not a .zip."""
    monkeypatch.setitem(sys.modules, "google.colab", None)
    hostname = tmp_path / "etc" / "hostname"
    hostname.parent.mkdir()
    hostname.write_text("runtime\n", encoding="utf-8")
    ns = _namespace(tmp_path, monkeypatch)
    _one_line(_run_refused(_cell_code(cells, dataset=str(hostname)), ns), "hostname", "is neither a folder nor a .zip file")
    _dataset_refused(ns, tmp_path)


@pytest.mark.skipif(not Path("/etc/hostname").is_file(), reason="needs a POSIX /etc/hostname, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc_hostname(cells, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    _one_line(_run_refused(_cell_code(cells, dataset="/etc/hostname"), _namespace(tmp_path, monkeypatch)), "hostname", "'/etc/hostname' is neither a folder nor a .zip file")


def _zip(path: Path, members: dict) -> Path:
    import zipfile

    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


@pytest.mark.parametrize(
    ("make", "file_name", "rule"),
    [
        (lambda d: _zip(d / "noannotations.zip", {"a.png": b"x"}), "noannotations.zip", "expected one annotations.csv"),
        (lambda d: _zip(d / "badheader.zip", {"annotations.csv": "name,label\n"}), "badheader.zip", "header must start with"),
        (lambda d: _zip(d / "noimage.zip", {"annotations.csv": "file,label,x0,y0,x1,y1\nmissing.png,stop-sign,1,1,20,20\n"}), "noimage.zip", "missing.png"),
        (lambda d: d / "nope.zip", "nope.zip", "neither a folder nor a .zip file"),
    ],
    ids=["no annotations.csv", "wrong header", "missing image", "missing path"],
)
def test_dataset_problems_are_one_refused_line(cells, tmp_path, monkeypatch, make, file_name, rule):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    path = make(tmp_path)
    ns = _namespace(tmp_path, monkeypatch)
    _one_line(_run_refused(_cell_code(cells, dataset=str(path)), ns), file_name, rule)
    _dataset_refused(ns, tmp_path)


@pytest.mark.parametrize(
    ("uploaded", "file_name", "rule"),
    [(None, "(no file)", "received 0"), ({"a.zip": b"", "b.zip": b""}, "(no file)", "received 2"), ({"data.tar": b""}, "data.tar", "ending in .zip")],
)
def test_cancelled_or_wrong_dataset_upload_is_one_refused_line(cells, tmp_path, monkeypatch, uploaded, file_name, rule):
    _colab(monkeypatch, uploaded)
    ns = _namespace(tmp_path, monkeypatch)
    _one_line(_run_refused(_cell_code(cells, dataset=""), ns), file_name, rule)
    _dataset_refused(ns, tmp_path)


@pytest.mark.parametrize(("make", "file_name", "rule"), [("missing.png", "missing.png", "does not exist or is not a file"), ("notes.png", "notes.png", "cannot identify image file")], ids=["missing", "not an image"])
def test_image_problems_are_one_refused_line(cells, tmp_path, monkeypatch, make, file_name, rule):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    (tmp_path / "notes.png").write_text("not an image", encoding="utf-8")
    ns = _namespace(tmp_path, monkeypatch)
    _one_line(_run_refused(_cell_code(cells, image=str(tmp_path / make)), ns), file_name, rule)
    assert not set(EARLIER_IMAGE) & set(ns), "an earlier run's image results survived the refusal"
    assert set(EARLIER_DATASET) <= set(ns) and (tmp_path / "outputs" / "byod_result.json").exists()


def test_image_upload_off_colab_is_one_refused_line(cells, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    _one_line(_run_refused(_cell_code(cells, image=""), _namespace(tmp_path, monkeypatch)), "(no file)", "only in Google Colab")


def test_genuine_internal_error_keeps_its_traceback(cells, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    ns = _namespace(tmp_path, monkeypatch)

    def broken(_path):
        raise RuntimeError("internal bug")

    ns["load_detection_dataset"] = broken
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        exec(compile(_cell_code(cells, dataset=str(tmp_path)), "<section 13>", "exec"), ns)
    assert not getattr(caught.value, "dimer_refusal", False)
    assert ns["byod_adapted"] == "old" and (tmp_path / "outputs" / "byod_result.json").exists()  # nothing is cleared for an internal error


def test_default_path_does_not_refuse(cells, tmp_path, monkeypatch, capsys):
    ns = _namespace(tmp_path, monkeypatch)
    exec(compile(_cell_code(cells), "<section 13>", "exec"), ns)
    out = capsys.readouterr().out
    assert "BYOD image branch is off" in out and "BYOD dataset branch is off" in out and "Refused:" not in out
    assert ns["byod_res"] == "old"
