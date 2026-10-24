# ruff: noqa: E501
"""ARTIFACT_SPEC 2.0 SER7/SER10/SER11 for the closed-set detection workshop's adapters.

These exec the notebook's own generated cells: the adapter-files cell (19a) and the two wrappers. Models are
the carried YOLOX architecture with random weights or a tiny stand-in module; none of this is
pretrained-model evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from detection_workshop_source import CELLS, INSERTED_CELLS  # noqa: E402

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("torchvision")

CLASSES = ("stop-sign", "yield-sign", "speed-limit-sign")
ADAPTER_CELL = next(c for c in INSERTED_CELLS if (c["after"], c["suffix"]) == (19, "a"))["source"]
RTDETR_TAG = "dimer-workshop-rtdetr-adapter/1"
YOLOX_TAG = "dimer-workshop-yolox-adapter/1"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def namespace():
    import gc
    import io
    import math
    import time

    import numpy as np
    import pandas as pd
    from PIL import Image

    ns = {"torch": torch, "json": json, "hashlib": hashlib, "Path": Path, "DEVICE": "cpu", "sha256_file": sha256_file,
          "np": np, "pd": pd, "Image": Image, "math": math, "time": time, "gc": gc, "io": io, "display": lambda *a, **k: None}
    exec(compile(ADAPTER_CELL, "cell-19a", "exec"), ns)
    return ns


class Tiny(torch.nn.Module):
    """Stand-in with a tied weight (like RT-DETR's heads) and BatchNorm (int64 num_batches_tracked)."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.a = torch.nn.Linear(4, 4)
        self.b = torch.nn.Linear(4, 4)
        self.b.weight = self.a.weight
        self.bn = torch.nn.BatchNorm1d(4)

    def forward(self, x):
        return self.bn(self.b(self.a(x)))


def identity():
    return {"model_id": "x", "revision": "y"}


@pytest.fixture
def ns():
    return namespace()


def save(ns, path, model=None, tag=RTDETR_TAG):
    return ns["save_adapter_files"](path, (model or Tiny()).state_dict(), tag, identity(), CLASSES, True)


def test_round_trip_gives_identical_tensors(ns, tmp_path):
    model = Tiny(seed=3)
    model.bn.running_mean += 0.5
    model.bn.num_batches_tracked += 7
    path = tmp_path / "adapter.safetensors"
    descriptor = save(ns, path, model)
    assert descriptor["sha256"] == sha256_file(path) and descriptor["sidecar_sha256"] == sha256_file(tmp_path / "adapter.json")
    sidecar, state = ns["read_adapter_files"](path, RTDETR_TAG)
    expected = model.state_dict()
    assert set(state) == set(expected)
    for key, value in expected.items():
        assert state[key].dtype == value.dtype and torch.equal(state[key], value), key
    fresh = Tiny(seed=99)
    fresh.load_state_dict(state, strict=True)
    assert fresh.b.weight is fresh.a.weight  # the rebuilt module keeps its own tie
    assert sidecar["class_names"] == list(CLASSES) and sidecar["identity"] == identity()
    assert not list(tmp_path.glob("*.partial"))


def test_files_hold_no_pickle_and_only_plain_data(ns, tmp_path):
    path = tmp_path / "adapter.safetensors"
    save(ns, path)
    data = path.read_bytes()
    assert not data.startswith((b"\x80", b"PK"))  # neither a raw pickle nor torch.save's zip container
    (header_len,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8:8 + header_len])
    assert header.pop("__metadata__") == {"format": RTDETR_TAG}  # exactly one metadata key
    assert all(set(entry) == {"dtype", "shape", "data_offsets"} for entry in header.values())
    sidecar = json.loads((tmp_path / "adapter.json").read_text(encoding="utf-8"))

    def plain(value):
        if isinstance(value, dict):
            return all(isinstance(k, str) and plain(v) for k, v in value.items())
        if isinstance(value, list):
            return all(plain(v) for v in value)
        return value is None or isinstance(value, (str, int, float, bool))

    assert plain(sidecar)
    assert sidecar["format"] == RTDETR_TAG and sidecar["weights_file"] == path.name


def test_same_weights_give_the_same_bytes(ns, tmp_path):
    first, second = tmp_path / "a" / "adapter.safetensors", tmp_path / "b" / "adapter.safetensors"
    save(ns, first, Tiny(seed=5))
    save(ns, second, Tiny(seed=5))
    assert first.read_bytes() == second.read_bytes()
    assert (tmp_path / "a" / "adapter.json").read_bytes() == (tmp_path / "b" / "adapter.json").read_bytes()


def test_bytes_do_not_depend_on_the_process_hash_seed(tmp_path):
    script = tmp_path / "write.py"
    script.write_text(textwrap.dedent(f"""
        import hashlib, json, sys
        from pathlib import Path
        import torch
        ns = {{"torch": torch, "json": json, "hashlib": hashlib, "Path": Path}}
        exec(compile({ADAPTER_CELL!r}, "cell-19a", "exec"), ns)
        g = torch.Generator().manual_seed(0)
        state = {{f"layer{{i}}.weight": torch.rand(3, 3, generator=g) for i in range(12)}}
        out = Path(sys.argv[1])
        d = ns["save_adapter_files"](out, state, "{YOLOX_TAG}", {{"variant": "yolox_s", "b": 1, "a": 2}}, ["z", "a"], True)
        print(d["sha256"], d["sidecar_sha256"])
    """), encoding="utf-8")
    digests = set()
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "CUDA_VISIBLE_DEVICES": "-1"}
        result = subprocess.run([sys.executable, str(script), str(tmp_path / seed / "adapter.safetensors")],
                                capture_output=True, text=True, env=env, check=True)
        digests.add(result.stdout.strip())
    assert len(digests) == 1


def tamper_weights(path):
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01
    path.write_bytes(bytes(data))


def edit_sidecar(path, **changes):
    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar.update(changes)
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")


@pytest.mark.parametrize("case,match", [
    ("weights_bytes", "digest mismatch"),
    ("sidecar_digest", "digest mismatch"),
    ("missing_sidecar", "missing"),
    ("pipeline_rtdetr_format", "foreign adapter artifact"),
    ("pipeline_yolox_format", "foreign adapter artifact"),
    ("other_workshop_model", "foreign adapter artifact"),
    ("weights_file", "sidecar names"),
    ("class_names", "class_names"),
    ("tensor_count", "tensor count"),
    ("header_tag", "header"),
])
def test_tampered_or_mismatched_files_are_refused(ns, tmp_path, case, match):
    path = tmp_path / "adapter.safetensors"
    save(ns, path)
    if case == "weights_bytes":
        tamper_weights(path)
    elif case == "sidecar_digest":
        edit_sidecar(path, weights_sha256="0" * 64)
    elif case == "missing_sidecar":
        path.with_suffix(".json").unlink()
    elif case == "pipeline_rtdetr_format":
        edit_sidecar(path, format="rtdetr-adapter-v2")
    elif case == "pipeline_yolox_format":
        edit_sidecar(path, format="dimer-yolox-detection-adapter/2")
    elif case == "other_workshop_model":
        edit_sidecar(path, format=YOLOX_TAG)
    elif case == "weights_file":
        edit_sidecar(path, weights_file="other.safetensors")
    elif case == "class_names":
        edit_sidecar(path, class_names="stop-sign")
    elif case == "tensor_count":
        edit_sidecar(path, tensors=1)
    elif case == "header_tag":
        from safetensors.torch import save_file
        save_file(Tiny().state_dict() | {"b.weight": Tiny().a.weight.detach().clone()}, str(path), metadata={"format": YOLOX_TAG})
        edit_sidecar(path, weights_sha256=sha256_file(path))
    with pytest.raises(ValueError, match=match):
        ns["read_adapter_files"](path, RTDETR_TAG)


def test_pickle_paths_are_neither_written_nor_read(ns, tmp_path):
    with pytest.raises(ValueError, match="never pickled"):
        save(ns, tmp_path / "adapter.pt")
    legacy = tmp_path / "adapter.pt"
    legacy.write_bytes(b"\x80\x04K\x01.")  # a pickle: never parsed
    with pytest.raises(ValueError, match="pickle"):
        ns["read_adapter_files"](legacy, RTDETR_TAG)
    with pytest.raises(TypeError, match="not a tensor"):
        ns["save_adapter_files"](tmp_path / "x.safetensors", {"w": [1, 2]}, RTDETR_TAG, identity(), CLASSES, True)


# ---------------------------------------------------------------- the two wrappers, strict reload


@pytest.fixture(scope="module")
def yolox(tmp_path_factory):
    root = tmp_path_factory.mktemp("yolox")
    ns = namespace()
    ns.update(WORK_ROOT=root, CLASS_NAMES=CLASSES, EVAL_SCORE_THRESHOLD=0.01, MAX_EVAL_DETECTIONS=100,
              IOU_THRESHOLDS=(0.5,), YOLOX_EVAL_NMS=0.65, YOLOX_VARIANTS={"yolox_s": {"sha256": "0" * 64}})
    exec(compile(CELLS[14]["source"].split("empty_predictions=")[0], "cell-14", "exec"), ns)
    exec(compile(CELLS[18]["source"], "cell-18", "exec"), ns)
    exec(compile(CELLS[21]["source"], "cell-21", "exec"), ns)

    class OfflineYOLOX(ns["YOLOXWorkshop"]):
        @classmethod
        def from_pretrained(cls, key, class_names=CLASSES, seed=0, device="cpu"):
            torch.manual_seed(seed + 1)  # different random init: the strict load must overwrite all of it
            return cls(key, ns["build_yolox"](0.33, 0.50, len(class_names)), class_names, device)

    ns["OfflineYOLOX"] = OfflineYOLOX
    return ns


def test_yolox_wrapper_round_trip_is_strict_and_exact(yolox, tmp_path):
    torch.manual_seed(0)
    pipe = yolox["YOLOXWorkshop"]("yolox_s", yolox["build_yolox"](0.33, 0.50, 3), CLASSES, "cpu", adapted=True)
    path = tmp_path / "adapter.safetensors"
    pipe.save_artifact(path)
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["format"] == YOLOX_TAG and sidecar["identity"]["architecture"]["batchnorm"] == {"eps": 1e-3, "momentum": 0.03}
    fresh = yolox["OfflineYOLOX"].load_artifact(path, "cpu")
    before, after = pipe.model.state_dict(), fresh.model.state_dict()
    assert set(before) == set(after) and all(torch.equal(before[k], after[k]) for k in before)
    pipe.model.eval()
    pixels = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(pipe.model(pixels), fresh.model(pixels))
    assert fresh.adapted and fresh.class_names == CLASSES


@pytest.mark.parametrize("field,value,match", [
    ("base_sha256", "1" * 64, "base digest"),
    ("variant", "yolox_q", "unknown variant"),
    ("upstream_code_revision", "0" * 40, "code revision"),
])
def test_yolox_wrapper_refuses_identity_mismatch(yolox, tmp_path, field, value, match):
    path = tmp_path / "adapter.safetensors"
    good = {"variant": "yolox_s", "release": "0.1.1rc0", "upstream_code_revision": yolox["YOLOX_CODE_REVISION"],
            "base_sha256": "0" * 64, "architecture": {"batchnorm": {"eps": 1e-3, "momentum": 0.03}}}
    yolox["save_adapter_files"](path, {"w": torch.zeros(2)}, YOLOX_TAG, good | {field: value}, CLASSES, True)
    with pytest.raises(ValueError, match=match):
        yolox["OfflineYOLOX"].load_artifact(path, "cpu")


@pytest.fixture(scope="module")
def rtdetr():
    pytest.importorskip("transformers")
    ns = namespace()
    ns.update(CLASS_NAMES=CLASSES, EVAL_SCORE_THRESHOLD=0.01, MAX_EVAL_DETECTIONS=100,
              RTDETR_ROOT=Path("."), IOU_THRESHOLDS=(0.5,), RTDETR_SPEC={"model_id": "x", "revision": "y"}, YOLOX_EVAL_NMS=0.65)
    exec(compile(CELLS[14]["source"].split("empty_predictions=")[0], "cell-14", "exec"), ns)
    exec(compile(CELLS[20]["source"], "cell-20", "exec"), ns)

    class OfflineRTDETR(ns["RTDETRWorkshop"]):
        @classmethod
        def from_pretrained(cls, class_names=CLASSES, seed=0, device="cpu"):
            return cls(Tiny(seed=seed + 1), None, class_names, device)

    ns["OfflineRTDETR"] = OfflineRTDETR
    return ns


def test_rtdetr_wrapper_round_trip_and_revision_check(rtdetr, tmp_path):
    pipe = rtdetr["OfflineRTDETR"](Tiny(seed=7), None, CLASSES, "cpu", adapted=True)
    path = tmp_path / "adapter.safetensors"
    descriptor = pipe.save_artifact(path)
    assert descriptor["format"] == RTDETR_TAG
    fresh = rtdetr["OfflineRTDETR"].load_artifact(path, "cpu")
    before, after = pipe.model.state_dict(), fresh.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before) and fresh.adapted
    rtdetr["RTDETR_SPEC"]["revision"] = "z"
    try:
        with pytest.raises(ValueError, match="revision mismatch"):
            rtdetr["OfflineRTDETR"].load_artifact(path, "cpu")
    finally:
        rtdetr["RTDETR_SPEC"]["revision"] = "y"
    with pytest.raises(ValueError, match="foreign adapter artifact"):
        rtdetr["read_adapter_files"](path, YOLOX_TAG)


def test_no_adapter_is_pickled_anywhere_in_the_workshop():
    sources = [c["source"] for c in CELLS + INSERTED_CELLS if c["kind"] == "code"]
    everything = "\n".join(sources)
    assert "torch.save(" not in everything and "adapter.pt" not in everything
    # The only torch.load left reads the upstream YOLOX .pth checkpoint, after its size and SHA-256 check.
    loads = [line.strip() for line in everything.splitlines() if "torch.load(" in line]
    assert loads == ['ckpt=torch.load(path,map_location="cpu",weights_only=True)']
    wrapper = CELLS[21]["source"]
    assert wrapper.index("path=stage_yolox_checkpoint(key)") < wrapper.index("torch.load(")
