"""Tier C review findings T77-1..T77-3 (2026-10-25): worked-answer sources, DAT9 overlap, spec 2.5 items."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "rtdetr_detection_colab.ipynb"


def _nb():
    return json.loads(NB.read_text(encoding="utf-8"))


def _markdown():
    return "\n".join("".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "markdown")


def test_t77_1_contradicted_worked_answers_are_labelled_and_softened():
    md = _markdown()
    stale_phrases = (
        "Around 0.85, not higher",
        "adaptation added nothing measurable",
        'about 0.85" shows',
        "record 0.9161",
        "differ in the third decimal",
        "the same as the adapted model (stop-sign AP 1.000)",
    )
    for stale in stale_phrases:
        assert stale not in md, stale
    assert "quote one local CPU run of an earlier revision" in md
    assert "0.934, `e2ab4517`" in md and "found all five signs" in md


def test_t77_2_dat9_overlap_statement():
    md = _markdown()
    assert "**Overlap with the training data:**" in md
    assert "verified by construction" in md and "COCO 2017" in md


def test_t77_3_spec_items():
    nb = _nb()
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    head = "\n".join("".join(c["source"]) for c in nb["cells"][:first_code])
    assert "Notice.** For research, training and evaluation only." in head
    md = _markdown()
    assert "[1] Y. Zhao" in md and "[Online]. Available:" in md
    assert "`DIMER_ISOLATED_ENV`" in md
    assert "recorded Kaggle run, 15 September 2026" not in md
