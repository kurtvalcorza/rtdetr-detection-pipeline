"""Run-derived prose is a record of named runs, never an expectation (Kurt's ruling, 2026-10-26).

Runs: local CPU worked run (2026-10-09); Kaggle T4 of 315909a9 (2026-09-15); Colab T4 of blobs
e2ab4517 (2026-10-09), 0dd61e64, becde2c0 and e921b62e (2026-10-25).
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "tutorials" / "rtdetr_detection_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_expectations_and_run_status_phrases_are_gone():
    md = _markdown()
    for stale in ("about 12 s on a Tesla T4", "they were not re-run for this revision", "not re-run here",
                  "**Expect one honest failure.**",
                  "read your own printed line against a span of about 0.92 to 0.98",
                  "a GPU runtime makes it seconds"):
        assert stale not in md, stale


def test_confirming_run_of_e921b62e_is_recorded():
    md = _markdown()
    assert "0.9774 (`e921b62e`, 2026-10-25)" in md
    assert "11.5, 11.6 and 11.3 in the hosted Colab T4 runs" in md
    assert "three of the four Colab T4 runs (1.000, `0dd61e64`, `becde2c0` and `e921b62e`)" in md
