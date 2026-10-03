"""Colab's already imported NumPy is left alone: the kernel cells install nothing into the kernel.

Before 2026-10-03 this test executed the in-kernel pip install against a simulated Colab kernel
with NumPy 2.1.3 loaded. The notebook now builds a uv isolated environment instead, so the regression
is that the two kernel cells import only the standard library (plus IPython for the router and
google.colab for the BYOD upload forwarding) and that the locked NumPy is the 2.1.3 the 2026-09-30
Colab run kept.
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_MultiModel_Closed_Set_Object_Detection_Workshop.ipynb"


def kernel_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in notebook["cells"]
            if c["cell_type"] == "code" and "# dimer: kernel cell" in "".join(c["source"])]


def test_kernel_cells_import_only_the_standard_library():
    cells = kernel_cells()
    assert len(cells) == 2
    allowed = set(sys.stdlib_module_names) | {"IPython", "google"}
    for cell in cells:
        for node in ast.walk(ast.parse(cell)):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert name.split(".")[0] in allowed, name
            assert not {"numpy", "torch", "pandas"} & {n.split(".")[0] for n in names}


def test_lock_keeps_the_numpy_colab_ran_with():
    lock = (ROOT / "tools" / "detection-workshop-requirements.lock").read_text(encoding="utf-8")
    assert "\nnumpy==2.1.3 \\\n" in lock
