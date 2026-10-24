"""The adapter artifact is a safetensors state dict plus a plain-JSON sidecar.

ARTIFACT_SPEC 2.0 SER7, SER10 and SER11.

Covers the round trip into a fresh model, byte-reproducible saves (in one process and from a fresh
interpreter), refusal of tampered, truncated and wrong-format files before anything is unpickled, and the
absence of pickle data in the saved file.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from rtdetr_detection_pipeline import pipeline as pipeline_module  # noqa: E402
from rtdetr_detection_pipeline.pipeline import (  # noqa: E402
    ARTIFACT_FORMAT,
    MODEL_KEY,
    RTDetrDetectionPipeline,
)

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("stop-sign", "yield-sign", "speed-limit-sign")

# Shared by the in-process tests and the fresh-interpreter save, so both build the same weights. The tied
# module reproduces RT-DETR's shared class/box heads: two state_dict keys over one storage.
BUILD_SOURCE = '''
import torch
from rtdetr_detection_pipeline.pipeline import RTDetrDetectionPipeline


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Conv2d(3, 4, 3)
        self.norm = torch.nn.BatchNorm2d(4)
        self.head = torch.nn.Linear(4, 3)
        self.decoder_head = self.head

    def forward(self, pixel_values):
        features = self.norm(self.backbone(pixel_values)).mean(dim=(2, 3))
        return self.head(features)


def build(seed):
    torch.manual_seed(seed)
    model = TinyDetector().eval()
    model.norm.running_mean.uniform_(-1.0, 1.0)
    return RTDetrDetectionPipeline(
        model=model,
        processor=None,
        device="cpu",
        class_names=("stop-sign", "yield-sign", "speed-limit-sign"),
        adapted=True,
        base_state_digest="0" * 64,
    )
'''
_NS: dict = {}
exec(compile(BUILD_SOURCE, "tiny_detector", "exec"), _NS)
build = _NS["build"]


def _save(tmp_path: Path, seed: int = 0, name: str = "adapter.safetensors") -> tuple[Path, dict]:
    pipe = build(seed)
    path = tmp_path / name
    training = {"seed": seed, "epoch_losses": [1.5, 0.75], "where": tmp_path}
    return path, pipe.save_artifact(path, notes="test", training=training)


@pytest.fixture
def fresh_model(monkeypatch):
    """load_artifact rebuilds through from_pretrained; serve a differently initialised tiny model instead."""
    calls = []

    def from_pretrained(cls, device=None, weights_dir=None, class_names=None, **_):
        calls.append(tuple(class_names))
        return build(seed=123)

    monkeypatch.setattr(RTDetrDetectionPipeline, "from_pretrained", classmethod(from_pretrained))
    return calls


@pytest.fixture
def no_rebuild_no_pickle(monkeypatch):
    """A refused artifact must stop before the model is rebuilt and never reach torch.load."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("refused artifact reached the model rebuild or torch.load")

    monkeypatch.setattr(RTDetrDetectionPipeline, "from_pretrained", classmethod(refuse))
    monkeypatch.setattr(torch, "load", refuse)


def _rewrite_sidecar(path: Path, **changes) -> None:
    sidecar = path.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload.update(changes)
    sidecar.write_text(json.dumps(payload), encoding="utf-8")


def test_round_trip_restores_identical_tensors_and_outputs(tmp_path, fresh_model):
    path, descriptor = _save(tmp_path)
    original = build(0)
    reloaded = RTDetrDetectionPipeline.load_artifact(path)
    assert fresh_model == [CLASSES]
    before, after = original.model.state_dict(), reloaded.model.state_dict()
    assert before.keys() == after.keys()
    for key in before:
        assert before[key].dtype == after[key].dtype and torch.equal(before[key], after[key]), key
    assert reloaded.model.decoder_head is reloaded.model.head  # the tie survives the shared-storage clone
    sample = torch.linspace(-1.0, 1.0, 2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    with torch.no_grad():
        assert torch.equal(original.model(sample), reloaded.model(sample))
    assert reloaded.adapted is True and reloaded.class_names == CLASSES
    assert reloaded.source == "artifact:adapter.safetensors"
    assert reloaded.training == descriptor["training"]
    assert descriptor["format"] == ARTIFACT_FORMAT == "rtdetr-adapter-v2"
    assert descriptor["tensors"] == len(before)
    assert not list(tmp_path.glob("*.partial")) and not list(tmp_path.glob("*.pt"))


def test_sidecar_is_plain_json_with_the_weights_digest(tmp_path):
    path, descriptor = _save(tmp_path)
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["format"] == ARTIFACT_FORMAT and sidecar["model_key"] == MODEL_KEY
    assert sidecar["weights_file"] == path.name
    assert sidecar["weights_sha256"] == descriptor["sha256"] == pipeline_module._sha256(path)
    assert sidecar["class_names"] == list(CLASSES) and sidecar["adapted"] is True
    assert sidecar["training"] == {"seed": 0, "epoch_losses": [1.5, 0.75], "where": str(tmp_path)}
    assert descriptor["sidecar_sha256"] == pipeline_module._sha256(path.with_suffix(".json"))


def test_saved_file_holds_no_pickle_data(tmp_path):
    path, _ = _save(tmp_path)
    data = path.read_bytes()
    assert not data.startswith(b"PK") and not data.startswith(b"\x80")  # not a torch.save zip or raw pickle
    (header_len,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + header_len].decode("utf-8"))
    assert header.pop("__metadata__") == {"format": ARTIFACT_FORMAT}  # one key: hash order cannot vary
    for name, entry in header.items():
        assert set(entry) == {"dtype", "shape", "data_offsets"}, name
    # Every byte after the header is tensor data the header accounts for: no room for a pickle.
    assert len(data) == 8 + header_len + max(entry["data_offsets"][1] for entry in header.values())
    assert set(header) == set(build(0).model.state_dict())


def test_two_saves_give_identical_bytes_in_process_and_from_a_fresh_interpreter(tmp_path):
    first, _ = _save(tmp_path, name="first.safetensors")
    second, _ = _save(tmp_path, name="second.safetensors")
    assert first.read_bytes() == second.read_bytes()

    fresh = tmp_path / "fresh.safetensors"
    script = BUILD_SOURCE + f"\nbuild(0).save_artifact({str(fresh)!r})\n"
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert fresh.read_bytes() == first.read_bytes()


def test_pt_paths_are_refused_on_save_and_load(tmp_path, no_rebuild_no_pickle):
    with pytest.raises(ValueError, match="must end in .safetensors"):
        build(0).save_artifact(tmp_path / "adapter.pt")
    legacy = tmp_path / "adapter.pt"
    legacy.write_bytes(b"\x80\x04N.")
    with pytest.raises(ValueError, match="pickle"):
        RTDetrDetectionPipeline.load_artifact(legacy)


def test_missing_sidecar_is_refused(tmp_path, no_rebuild_no_pickle):
    path, _ = _save(tmp_path)
    path.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="sidecar"):
        RTDetrDetectionPipeline.load_artifact(path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"format": "rtdetr-adapter-v1"}, "artifact format"),
        ({"model_revision": "0" * 40}, "package pins"),
        ({"model_key": "rtdetr-r101vd"}, "package pins"),
        ({"weights_sha256": None}, "weights_sha256"),
        ({"class_names": "stop-sign"}, "class_names"),
        ({"class_names": []}, "non-empty class_names"),
        ({"weights_file": "other.safetensors"}, "names weights file"),
    ],
)
def test_sidecar_identity_mismatches_are_refused(tmp_path, no_rebuild_no_pickle, changes, message):
    path, _ = _save(tmp_path)
    _rewrite_sidecar(path, **changes)
    with pytest.raises(ValueError, match=message):
        RTDetrDetectionPipeline.load_artifact(path)


def test_tampered_weights_are_refused_by_digest(tmp_path, no_rebuild_no_pickle):
    path, _ = _save(tmp_path)
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="digest mismatch"):
        RTDetrDetectionPipeline.load_artifact(path)


def test_truncated_weights_are_refused(tmp_path, no_rebuild_no_pickle):
    path, _ = _save(tmp_path)
    path.write_bytes(path.read_bytes()[:-16])
    with pytest.raises(ValueError, match="digest mismatch"):
        RTDetrDetectionPipeline.load_artifact(path)
    # Even with a sidecar rewritten to match, the safetensors reader rejects the short file itself.
    _rewrite_sidecar(path, weights_sha256=pipeline_module._sha256(path))
    with pytest.raises(Exception, match="(?i)incomplete|invalid|header|metadata|size"):
        RTDetrDetectionPipeline.load_artifact(path)


def test_header_format_tag_must_match(tmp_path, no_rebuild_no_pickle):
    # A safetensors file whose own header names another format is refused even when the sidecar's
    # digest has been rewritten to match it.
    path, _ = _save(tmp_path)
    raw = path.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_len])
    header["__metadata__"]["format"] = "other"
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    assert len(encoded) <= header_len
    path.write_bytes(raw[:8] + encoded + b" " * (header_len - len(encoded)) + raw[8 + header_len :])
    _rewrite_sidecar(path, weights_sha256=pipeline_module._sha256(path))
    with pytest.raises(ValueError, match="header does not carry format"):
        RTDetrDetectionPipeline.load_artifact(path)


def test_a_pickle_renamed_to_safetensors_is_never_unpickled(tmp_path, monkeypatch):
    path = tmp_path / "adapter.safetensors"
    real_save = torch.save
    real_save({"state_dict": build(0).model.state_dict()}, path)
    _save(tmp_path, name="genuine.safetensors")
    sidecar = json.loads((tmp_path / "genuine.json").read_text(encoding="utf-8"))
    sidecar.update(weights_file=path.name, weights_sha256=pipeline_module._sha256(path))
    path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")

    def refuse(*_args, **_kwargs):
        raise AssertionError("refused artifact reached the model rebuild or torch.load")

    monkeypatch.setattr(RTDetrDetectionPipeline, "from_pretrained", classmethod(refuse))
    monkeypatch.setattr(torch, "load", refuse)
    with pytest.raises(Exception) as caught:
        RTDetrDetectionPipeline.load_artifact(path)
    assert not isinstance(caught.value, AssertionError)


def test_rtdetr_architecture_round_trip_with_tied_heads(tmp_path, monkeypatch):
    """The real RT-DETR module (random init from the committed config) shares its heads with the decoder."""
    transformers = pytest.importorskip("transformers")
    config = transformers.RTDetrConfig.from_json_file(str(ROOT / "weights" / MODEL_KEY / "config.json"))
    config.num_labels = len(CLASSES)
    config.use_pretrained_backbone = False

    def make(seed):
        torch.manual_seed(seed)
        model = transformers.RTDetrForObjectDetection(config).eval()
        return RTDetrDetectionPipeline(
            model=model, processor=None, device="cpu", class_names=CLASSES, adapted=True
        )

    original = make(0)
    shared = [k for k in original.model.state_dict() if k.startswith("model.decoder.class_embed.")]
    assert shared, "expected the decoder to expose the tied class heads"
    path = tmp_path / "rtdetr_adapter.safetensors"
    original.save_artifact(path)
    monkeypatch.setattr(RTDetrDetectionPipeline, "from_pretrained", classmethod(lambda cls, **_: make(1)))
    reloaded = RTDetrDetectionPipeline.load_artifact(path)
    before, after = original.model.state_dict(), reloaded.model.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)
    # 160 px gives 20² + 10² + 5² = 525 encoder anchors, enough for the 300 queries.
    sample = torch.linspace(0.0, 1.0, 3 * 160 * 160).reshape(1, 3, 160, 160)
    with torch.no_grad():
        a, b = original.model(pixel_values=sample), reloaded.model(pixel_values=sample)
    assert torch.equal(a.logits, b.logits) and torch.equal(a.pred_boxes, b.pred_boxes)
