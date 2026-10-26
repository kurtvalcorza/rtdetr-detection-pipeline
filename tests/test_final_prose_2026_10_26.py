"""Final prose pass (Kurt's ruling, 2026-10-26): the flagged expectation sentences are removed or
rewritten as records of named runs.
rtdetr: md0 timing and spread promises, md1 unnamed T4 timing. Follow-up: md0 relabels the local CPU run
to the earlier revision. Troubleshooting AP claim softened.
Fails on PR head 7f7dedf.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / 'tutorials/rtdetr_detection_colab.ipynb'


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


GONE = [
    'Budget about ten minutes',
    'differ from the third decimal on',
    'about 11 s on a Tesla T4 in a recorded hosted run',
    'a GPU makes it seconds',
    'a local CPU run of this revision',
    'a very different result means a different runtime',
]
PRESENT = [
    'the recorded Tesla T4 times are listed under **Runtime**',
    'The worked answers quote a local CPU run of an earlier revision and the recorded hosted runs',
    'If yours differs widely on the default settings, check the `torch` and `transformers` versions',
]


def test_flagged_expectations_are_gone():
    md = _markdown()
    for text in GONE:
        assert text not in md, text


def test_replacement_records_are_present():
    md = _markdown()
    for text in PRESENT:
        assert text in md, text
