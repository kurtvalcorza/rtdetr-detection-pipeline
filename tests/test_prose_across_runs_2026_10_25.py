"""Prose numbers stay true across every recorded hosted run (Kurt's ruling, 2026-10-25).

Hosted runs: Kaggle T4 of blob 315909a9 (2026-09-15); Colab T4 of blobs e2ab4517 (2026-10-09),
0dd61e64 and becde2c0 (2026-10-25). The 2026-10-09 local CPU worked run stays labelled as CPU.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "tutorials" / "rtdetr_detection_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_range_that_excluded_a_hosted_run_is_gone():
    md = _markdown()
    for stale in ("has ranged from about 0.85 to 0.95 between runs", "hosted T4 runs of earlier blobs",
                  "fell below it in one hosted T4 run (0.934, `e2ab4517`).", "print different decimals"):
        assert stale not in md, stale


def test_every_hosted_ap_is_listed():
    md = _markdown()
    assert "0.9161 (Kaggle, `315909a9`" in md and "0.979 (`becde2c0`" in md
    assert "0.916 to 0.979 on hosted T4s" in md
    assert "0.937–0.991 and 0.835–1.000 in the three Colab T4 runs" in md
