# ruff: noqa: E501
"""Regression tests for the closed-set detection workshop review, round 2 (DET2-*, 2026-10-14).

These exec the notebook's own generated cells with explicit synthetic adapters and hand-built detections.
None of this is pretrained-model evidence.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from detection_workshop_source import CELLS, INSERTED_CELLS  # noqa: E402

import test_detection_workshop_review_fixes as det1  # noqa: E402

Adapter, byod_zip, frozen, run = det1.Adapter, det1.byod_zip, det1.frozen, det1.run
notebook = det1.notebook  # the DET-01..07 fixture: controls, specs, data and evaluator cells executed in tmp_path


def run_inserted(after, suffix, ns):
    cell = next(c for c in INSERTED_CELLS if c["after"] == after and c["suffix"] == suffix)
    exec(compile(cell["source"], f"cell-{after:02d}{suffix}", "exec"), ns)


def exact(records, score):
    return [[{"box": list(map(float, b)), "label": lab, "score": score}
             for b, lab in zip(r["boxes"], r["labels"], strict=True)] for r in records]


FALSE_DETECTION = {"box": [0.0, 0.0, 3.0, 3.0], "label": "stop-sign"}


# ---------------------------------------------------------------- DET2-M1: per-model operating point


def test_operating_point_is_chosen_per_model_on_validation(notebook):
    ns = notebook
    val = ns["validation_records"]
    low = [dets + [{**FALSE_DETECTION, "score": 0.08}] for dets in exact(val, 0.25)]
    ns.update(pipes={}, post_validation_predictions={"rtdetr": low, "yolox_s": exact(val, 0.9)})
    run(28, ns)
    # RT-DETR: 0.10 and 0.20 tie on F1 = 1 (0.05 keeps the false detections, 0.30 drops every box); ties go higher.
    assert ns["VALIDATION_OPERATING_POINTS"] == {"rtdetr": 0.20, "yolox_s": 0.70}
    assert (ns["OUTPUT_ROOT"] / "validation" / "operating_points.csv").is_file()


def test_freeze_records_operating_points_and_multi_seed_declaration(notebook):
    ns = frozen(notebook)
    saved = json.loads(ns["freeze_path"].read_text())
    assert saved["validation_operating_points"] == {"rtdetr": 0.3, "yolox_s": 0.3}
    assert saved["multi_seed_repeat"]["enabled"] is False and saved["multi_seed_repeat"]["seeds"] == [1, 2, 3]
    assert saved["notebook_spec"] == "2.5"
    assert "fetching them at runtime" not in json.dumps(saved)
    run(32, ns)
    assert ns["TEST_OPERATING_POINTS"] == {"rtdetr": 0.3, "yolox_s": 0.3}


def test_size_density_table_keeps_shared_threshold_and_adds_per_model_operating_point(notebook):
    ns = frozen(notebook)
    run(32, ns)
    records = ns["test_records"]
    ns["test_predictions"] = {("rtdetr", "adapted"): exact(records, 0.25), ("yolox_s", "adapted"): exact(records, 0.9)}
    ns["TEST_OPERATING_POINTS"] = {"rtdetr": 0.20, "yolox_s": 0.30}  # as if validation had chosen these
    run(36, ns)
    shared, own = ns["diagnostics"], ns["operating_point_diagnostics"]
    assert list(shared.columns) == ["model", "dimension", "group", "recall50", "objects"]  # unchanged table
    assert (shared[shared["model"] == "rtdetr"]["recall50"].fillna(0) == 0).all()
    rt = own[own["model"] == "rtdetr"]
    assert (rt["operating_point"] == 0.20).all() and (rt["recall50"].dropna() == 1.0).all()
    density = own[own["dimension"] == "scene_density"]
    assert density["ap50_threshold_free"].notna().all()
    assert (ns["OUTPUT_ROOT"] / "test" / "size_density_operating_point_metrics.csv").is_file()


# ---------------------------------------------------------------- DET2-M5: class confusion


def test_confusion_table_counts_cross_class_errors_misses_and_false_detections(notebook):
    ns = notebook
    records = ns["test_records"]
    rt = exact(records, 0.9)
    original = rt[0][0]["label"]
    wrong = next(name for name in ns["CLASS_NAMES"] if name != original)
    rt[0][0] = {**rt[0][0], "label": wrong}
    rt[1] = rt[1] + [{**FALSE_DETECTION, "score": 0.9}]
    rt[2] = rt[2][1:]  # one missed object
    ns.update(MODEL_KEYS=["rtdetr", "yolox_s"], TEST_OPERATING_POINTS={"rtdetr": 0.5, "yolox_s": 0.5},
              test_predictions={("rtdetr", "adapted"): rt, ("yolox_s", "adapted"): exact(records, 0.9)})
    run_inserted(34, "b", ns)
    table = pd.read_csv(ns["OUTPUT_ROOT"] / "test" / "class_confusion.csv")
    r = table[table["model"] == "rtdetr"].set_index(["reference", "predicted"])["count"]
    assert r[(original, wrong)] == 1
    assert sum(r[("no object", c)] for c in ns["CLASS_NAMES"]) == 1
    assert sum(r[(c, "missed")] for c in ns["CLASS_NAMES"]) == 1
    y = table[table["model"] == "yolox_s"]
    off = y[(y["reference"] != y["predicted"]) & y["reference"].isin(ns["CLASS_NAMES"]) & y["predicted"].isin(ns["CLASS_NAMES"])]
    assert off["count"].sum() == 0


# ---------------------------------------------------------------- DET2-M4: class-agnostic display


def test_pre_adaptation_class_agnostic_ap_is_display_only(notebook):
    ns = notebook
    ns.update(RTDETRWorkshop=Adapter, YOLOXWorkshop=Adapter)
    run(23, ns)
    assert [row["class_agnostic_ap"] for row in ns["pre_val_agnostic"]] == [1.0, 1.0]
    assert not any("agnostic" in c for c in ns["pre_val_table"].columns)
    section10 = CELLS[22]["source"]
    assert "enc_score_head" in section10 and "objectness" in section10
    assert "localization structure transfers" not in section10


# ---------------------------------------------------------------- DET2-M2: FULL-only multi-seed repeat


class SeedAdapter(Adapter):
    seeds = []

    def finetune(self, records, epochs, batch_size, learning_rate, seed):
        SeedAdapter.seeds.append((self.key, seed))
        return {}


def multi_seed_ns(ns, tier):
    ns.update(WORKSHOP_TIER=tier, RTDETRWorkshop=SeedAdapter, YOLOXWorkshop=SeedAdapter, TEST_SCORE_CUTOFF=0.01,
              train_records=ns["validation_records"],
              test_table=pd.DataFrame([{"model": "rtdetr", "state": "adapted", "ap": 0.9},
                                       {"model": "yolox_s", "state": "adapted", "ap": 0.8}]))
    SeedAdapter.seeds.clear()
    Adapter.calls.clear()
    run_inserted(39, "b", ns)
    return ns


def test_multi_seed_repeat_does_nothing_on_standard(notebook):
    ns = multi_seed_ns(notebook, "STANDARD")
    assert SeedAdapter.seeds == [] and Adapter.calls == []
    assert "multi_seed" not in ns
    assert not (ns["OUTPUT_ROOT"] / "test" / "multi_seed_repeat.csv").exists()


def test_multi_seed_repeat_reports_every_seed_for_both_models_in_full(notebook):
    ns = multi_seed_ns(notebook, "FULL")
    assert SeedAdapter.seeds == [("rtdetr", s) for s in (1, 2, 3)] + [("yolox_s", s) for s in (1, 2, 3)]
    assert len(ns["multi_seed"]) == 6
    assert {"mean", "min", "max"} <= {c[1] for c in ns["multi_seed_summary"].columns}
    assert (ns["OUTPUT_ROOT"] / "test" / "multi_seed_repeat.csv").is_file()


# ---------------------------------------------------------------- DET2-m1: BYOD refusal


def byod_ready(ns, tmp_path, path):
    freeze = ns["OUTPUT_ROOT"] / "frozen" / "frozen_experiment.json"
    freeze.write_text("{}")
    ns.update(USE_BYOD=True, WORK_ROOT=tmp_path, BYOD_ZIP_PATH=str(path), byod_metrics="stale")


@pytest.mark.parametrize("make,expected", [
    (lambda t: byod_zip(t, "image_id,file,label,x0,y0,x1,y1", ["0,0.png,cat,1,1,10,10"]),
     r"annotations\.csv line 2 \(image_id '0', file '0\.png'\): Unsupported class 'cat'"),
    (lambda t: byod_zip(t, "image_id,file,label,x0,y0,x1,y1", ["0,0.png,stop-sign,1,1,10"]),
     r"annotations\.csv line 2: Annotation row width"),
    (lambda t: (t / "not.zip").write_text("x") and t / "not.zip", r"is not a readable ZIP archive"),
    (lambda t: "", r"BYOD_ZIP_PATH is empty"),
], ids=["bad_label", "ragged_row", "not_a_zip", "empty_path"])
def test_byod_refusal_prints_one_named_message_without_raising(notebook, tmp_path, capsys, make, expected):
    byod_ready(notebook, tmp_path, make(tmp_path))
    run(47, notebook)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 1 and lines[0].startswith("BYOD refused before any model was loaded")
    assert re.search(expected, lines[0]), lines[0]
    assert notebook["byod_records"] is None and "byod_metrics" not in notebook


def test_every_load_byod_refusal_names_its_item():
    source = CELLS[47]["source"]
    body = source.split("def load_byod", 1)[1].split("\ndef ", 1)[0]
    raises = re.findall(r"raise ValueError\(([^\n]*)\)", body)
    assert raises and all(r.strip().startswith(("f'", 'f"')) for r in raises)


# ---------------------------------------------------------------- DET2-m2 / m3 / m5 / m6, DET2-M2 / M3 wording


def test_static_wording_and_records():
    markdown = "\n".join(c["source"] for c in CELLS + INSERTED_CELLS if c["kind"] == "markdown")
    assert "Deterministic disagreement" not in markdown and "Keep `STANDARD` and BYOD disabled" not in markdown
    assert "transfer differently" not in CELLS[0]["source"]
    assert "not tuned on validation" in markdown and "27" in CELLS[24]["source"] and "108" in CELLS[24]["source"]
    assert "+0.143" in CELLS[37]["source"] and "+0.037" in CELLS[37]["source"]
    assert "172 MB" in CELLS[0]["source"]
    for i in (30, 52, 55):
        assert '"notebook_spec":"2.5"' in CELLS[i]["source"]
    assert all("fetching them at runtime" not in c["source"] for c in CELLS)
    load = CELLS[20]["source"].split("def load_artifact", 1)[1]
    assert load.index("set_verbosity_error") < load.index("cls.from_pretrained") < load.index("set_verbosity(previous_verbosity)")


def test_notebook_metadata_declares_spec_and_kind():
    nb = json.loads((Path(__file__).resolve().parents[1] / "tutorials" / "DIMER_MultiModel_Closed_Set_Object_Detection_Workshop.ipynb").read_text(encoding="utf-8"))
    meta = nb["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.5" and meta["notebook_kind"] == "WORKSHOP"
    assert meta["revisions"][-1]["base_notebook_blob"] == "929be36ca61620e7e33cd8dace8fcf4ba03ed09d"
