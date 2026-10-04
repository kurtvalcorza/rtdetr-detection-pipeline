# ruff: noqa: E501
"""uv isolated environment of the closed-set detection workshop notebook (2026-10-03).

The generator (``build_detection_workshop.py``) inserts the four cells of :func:`uv_cells` after the learning goals,
before section 1. The two code cells are marked ``# dimer: kernel cell`` and are the only cells that run in the
notebook kernel:

* the bootstrap cell verifies a pinned ``uv`` wheel by size and SHA-256, creates a uv-managed CPython 3.12.12 venv
  and installs the carried hash lock (``tools/detection-workshop-requirements.lock``) with
  ``--require-hashes --only-binary :all:``;
* the router cell starts one persistent worker process with that venv's Python and routes every later code cell to
  it, so the cells of ``detection_workshop_source.CELLS`` run unchanged in the isolated environment and no package is
  installed into the kernel (no restart).

This is the fleet mechanism of ``ast-audio-classification-pipeline`` (pinned uv, managed CPython, hash lock) with the
persistent cell worker of ``bart-mnli-zero-shot-classification-pipeline`` (``ee128d2``).

The carried lock and any other literal longer than :data:`LINE_LIMIT` characters are written as parenthesised runs
of string pieces of at most :data:`PIECE_LIMIT` characters (:func:`split_long_lines`), so no notebook line is long
enough to stall the Colab editor; ``ast.literal_eval`` equality is asserted for every split.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
LOCK_PATH = TOOLS / "detection-workshop-requirements.lock"
LOCK_NAME = "requirements.lock.txt"
LINE_LIMIT = 2000
PIECE_LIMIT = 1000
MANAGED_PYTHON = "3.12.12"
UV_VERSION = "0.12.15"
UV_URL = "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
UV_BYTES = 20081404
UV_SHA256 = "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60"

# Direct pins of the isolated environment: the versions the notebook pinned for its former in-kernel install
# (numpy 2.1.3 is the version the 2026-09-30 Colab run kept); the two former ranges are fixed to the versions the lock
# resolved (matplotlib >=3.9,<3.11 -> 3.10.9; pandas >=2.2,<3.1 -> 3.0.6).
PINS = [
    "torch==2.14.0",
    "torchvision==0.29.0",
    "torchaudio==2.11.0",
    "transformers==4.57.6",
    "safetensors==0.8.0",
    "numpy==2.1.3",
    "pillow==11.3.0",
    "huggingface-hub==0.36.2",
    "scipy==1.18.1",
    "matplotlib==3.10.9",
    "pandas==3.0.6",
]


def pieces(text, limit=PIECE_LIMIT):
    """Split ``text`` into consecutive pieces whose ``repr`` is at most ``limit`` characters, at line ends if possible."""
    out = []
    current = ""
    for line in text.splitlines(keepends=True):
        while len(repr(line)) > limit:
            cut = limit // 2
            while len(repr(line[:cut])) > limit:
                cut -= 1
            if current:
                out.append(current)
                current = ""
            out.append(line[:cut])
            line = line[cut:]
        if current and len(repr(current + line)) > limit:
            out.append(current)
            current = ""
        current += line
    if current:
        out.append(current)
    assert "".join(out) == text and all(len(repr(p)) <= limit for p in out)
    return out


def string_run(value, indent):
    """``value`` as a parenthesised run of string pieces (implicit concatenation)."""
    pad = " " * indent
    return "(\n" + "".join(f"{pad}    {piece!r}\n" for piece in pieces(value)) + f"{pad})"


def split_long_lines(source, limit=LINE_LIMIT):
    """Rewrite every top-level ``NAME=<str or dict of str>`` line longer than ``limit`` as runs of short pieces."""
    out = []
    for line in source.split("\n"):
        if len(line) <= limit:
            out.append(line)
            continue
        tree = ast.parse(line)
        node = tree.body[0] if len(tree.body) == 1 else None
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            raise SystemExit(f"cannot split a {len(line)}-character line that is not a literal assignment")
        name = node.targets[0].id
        value = ast.literal_eval(node.value)
        if isinstance(value, str):
            text = f"{name}={string_run(value, 0)}"
        elif isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
            body = "".join(f"    {key!r}: {string_run(item, 4)},\n" for key, item in value.items())
            text = f"{name}={{\n{body}}}"
        else:
            raise SystemExit(f"cannot split the {len(line)}-character literal {name}")
        rebuilt = ast.parse(text).body[0]
        assert isinstance(rebuilt, ast.Assign) and ast.literal_eval(rebuilt.value) == value
        assert max(len(part) for part in text.split("\n")) <= PIECE_LIMIT + 8
        out.append(text)
    return "\n".join(out)


def lock_text():
    data = LOCK_PATH.read_bytes()
    if b"\r" in data:
        raise SystemExit(f"{LOCK_PATH} must use LF line endings")
    return data.decode("utf-8")


def lock_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def locked_packages(text):
    return sum(1 for line in text.splitlines() if line[:1].isalnum() and "==" in line)


def bootstrap_source(text):
    head = [
        "# @title Infrastructure: build the locked isolated environment",
        "# dimer: kernel cell (runs in the notebook kernel, not in the isolated environment)",
        "import hashlib",
        "import io",
        "import os",
        "import platform",
        "import subprocess",
        "import sys",
        "import time",
        "import urllib.error",
        "import urllib.request",
        "import zipfile",
        "from pathlib import Path",
        "",
        "# Direct pins of the isolated environment (tools/detection-workshop-requirements.in in the repository). The lock",
        "# below fixes every other package, with hashes. Nothing is installed into this kernel.",
        "PINS = [",
        *(f"    {pin!r}," for pin in PINS),
        "]",
        f"MANAGED_PYTHON = {MANAGED_PYTHON!r}",
        f"UV_URL = {UV_URL!r}",
        f"UV_BYTES = {UV_BYTES}",
        f"UV_SHA256 = {UV_SHA256!r}",
        f"LOCK_NAME = {LOCK_NAME!r}",
        f"LOCK_SHA256 = {lock_sha256(text)!r}",
        f"LOCKED_PACKAGES = {locked_packages(text)}",
        "# The hash lock (tools/detection-workshop-requirements.lock), compiled from PINS with `uv pip compile",
        "# --generate-hashes` for CPython 3.12 on manylinux x86_64, carried in pieces of at most 1,000 characters.",
        f"LOCK_TEXT = {string_run(text, 0)}",
    ]
    return "\n".join(head) + "\n" + BOOTSTRAP_TAIL.strip("\n")


def uv_cells():
    text = lock_text()
    return [
        {"id": "dimer-detection-workshop-uv-env-md", "kind": "markdown", "source": ENV_MARKDOWN},
        {"id": "dimer-detection-workshop-uv-env", "kind": "code", "source": bootstrap_source(text),
         "metadata": {"cellView": "form"}},
        {"id": "dimer-detection-workshop-uv-route-md", "kind": "markdown", "source": ROUTE_MARKDOWN},
        {"id": "dimer-detection-workshop-uv-route", "kind": "code", "source": ROUTER_SOURCE.strip("\n"),
         "metadata": {"cellView": "form"}},
    ]


ENV_MARKDOWN = """## Infrastructure — isolated environment

The model code of this notebook runs in its own hash-locked Python environment instead of the notebook kernel, so nothing is installed into the kernel and **Run all needs no session restart**.

The next cell downloads a pinned `uv` (checked by size and SHA-256), uses it to create a CPython 3.12.12 environment, and installs the carried lock with `--require-hashes --only-binary :all:`: wheels only, every file checked against its hash. The direct pins are torch 2.14.0, torchvision 0.29.0, torchaudio 2.11.0, Transformers 4.57.6, safetensors 0.8.0, NumPy 2.1.3, Pillow 11.3.0, huggingface-hub 0.36.2, SciPy 1.18.1, matplotlib 3.10.9 and pandas 3.0.6.

**Linux x86_64 only** (Google Colab, Kaggle or Linux Jupyter): the lock is built for manylinux x86_64, and the cell stops with a clear message on any other platform. The first run downloads several GB of wheels (PyTorch with its CUDA libraries); a re-run in the same session reuses the environment."""

ROUTE_MARKDOWN = """The next cell starts one Python process in that environment and routes **every later code cell** to it. Printed output, tables and figures come back to the notebook as usual, variables persist from cell to cell, and an error stops **Run all** as it would in the kernel. These two cells are marked `# dimer: kernel cell` and are the only cells that run in the kernel. If you re-run a single cell later, it still runs in the isolated environment with the variables created so far; to start over, choose **Run all** (the router starts a fresh worker) or restart the session. `DIMER_NOTEBOOK_CI_PREINSTALLED=1` lets an executor that has already installed exactly these pins run every cell in its own kernel instead."""

# Kernel half of the bootstrap cell (after the carried lock): the fleet mechanism, unchanged.
BOOTSTRAP_TAIL = r'''
SKIP_INSTALL = os.environ.get("DIMER_NOTEBOOK_CI_PREINSTALLED") == "1"
ISOLATED_ENV = Path(os.environ.get("DIMER_ISOLATED_ENV", "dimer_isolated_env")).resolve()
ISOLATED_PYTHON = ISOLATED_ENV / "bin" / "python"
ISOLATED_TOOLS = ISOLATED_ENV.with_name(ISOLATED_ENV.name + "_tools")

if SKIP_INSTALL:
    print("DIMER_NOTEBOOK_CI_PREINSTALLED=1: the pins are already installed; the notebook runs in this kernel.")
else:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This notebook needs a Linux x86_64 runtime (Google Colab, Kaggle or Linux Jupyter): its locked environment is built for manylinux x86_64.")
    setup_started = time.perf_counter()
    if hashlib.sha256(LOCK_TEXT.encode("utf-8")).hexdigest() != LOCK_SHA256:
        raise RuntimeError("The carried lock does not match its digest: regenerate the notebook from the repository instead of editing this cell.")
    ISOLATED_TOOLS.mkdir(parents=True, exist_ok=True)
    lock_path = ISOLATED_TOOLS / LOCK_NAME
    lock_path.write_text(LOCK_TEXT, encoding="utf-8", newline="\n")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(UV_URL, timeout=90) as response:
                wheel = response.read(UV_BYTES + 1)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    if len(wheel) != UV_BYTES or hashlib.sha256(wheel).hexdigest() != UV_SHA256:
        raise RuntimeError("The pinned uv wheel failed its size/SHA-256 check: refusing to run it. Run this cell again; if it repeats, the download is being altered.")
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        member = next(name for name in archive.namelist() if name.endswith(".data/scripts/uv"))
        uv = ISOLATED_TOOLS / "uv"
        uv.write_bytes(archive.read(member))
    uv.chmod(0o700)
    # uv gets no kernel Python path; the managed interpreter is downloaded once and reused on a re-run.
    uv_env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        uv_env.pop(name, None)
    if not ISOLATED_PYTHON.is_file():
        subprocess.run([str(uv), "venv", "--quiet", "--managed-python", "--python", MANAGED_PYTHON, str(ISOLATED_ENV)], env=uv_env, check=True)
    isolated_version = subprocess.run([str(ISOLATED_PYTHON), "-c", "import platform; print(platform.python_version())"], env=uv_env, check=True, capture_output=True, text=True).stdout.strip()
    if isolated_version != MANAGED_PYTHON:
        raise RuntimeError(f"{ISOLATED_ENV} holds Python {isolated_version}, not {MANAGED_PYTHON}: delete that folder (or start a fresh runtime) and run this cell again.")
    subprocess.run([str(uv), "pip", "install", "--quiet", "--python", str(ISOLATED_PYTHON), "--require-hashes", "--only-binary", ":all:", "--index-url", "https://pypi.org/simple", "-r", str(lock_path)], env=uv_env, check=True)
    print({"isolated_environment": str(ISOLATED_ENV), "isolated_python": isolated_version, "kernel_python": platform.python_version(), "locked_packages": LOCKED_PACKAGES, "setup_seconds": round(time.perf_counter() - setup_started)})
'''

# Router cell: bart-mnli-zero-shot-classification-pipeline ee128d2, with the worker flagged DIMER_ISOLATED_WORKER=1.
ROUTER_SOURCE = r'''
# @title Route the remaining cells to the isolated environment
# dimer: kernel cell (runs in the notebook kernel, not in the isolated environment)
import signal
from multiprocessing.connection import Connection

from IPython import get_ipython as _kernel_shell

# The worker runs in the isolated environment. It executes each routed cell in one persistent namespace and sends
# back printed text, displayed objects and matplotlib figures, so every cell behaves as it would in the kernel.
_WORKER_SOURCE = r"""
import ast, base64, builtins, io, linecache, os, signal, sys, traceback, types
from multiprocessing.connection import Connection

_send = Connection(int(sys.argv[1]), readable=False)
_recv = Connection(int(sys.argv[2]), writable=False)


class _Stream(io.TextIOBase):
    def __init__(self, name):
        self._name = name

    @property
    def encoding(self):
        return "utf-8"

    def writable(self):
        return True

    def isatty(self):
        return False

    def write(self, text):
        if text:
            _send.send(("stream", self._name, str(text)))
        return len(text)


sys.stdout, sys.stderr = _Stream("stdout"), _Stream("stderr")


def _figure_bundle(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    return {"image/png": base64.b64encode(buffer.getvalue()).decode("ascii"), "text/plain": repr(fig)}


def _flush_figures():
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return
    for number in plt.get_fignums():
        _send.send(("display", _figure_bundle(plt.figure(number))))
    plt.close("all")


def _mimebundle(obj):
    if hasattr(obj, "savefig"):
        return _figure_bundle(obj)
    data = {"text/plain": repr(obj)}
    for method, mime in (("_repr_html_", "text/html"), ("_repr_markdown_", "text/markdown"), ("_repr_png_", "image/png")):
        render = getattr(obj, method, None)
        if callable(render):
            try:
                value = render()
            except Exception:
                value = None
            if isinstance(value, bytes):
                value = base64.b64encode(value).decode("ascii")
            if value is not None:
                data[mime] = value
    return data


def display(*objects, **kwargs):
    for obj in objects:
        _send.send(("display", _mimebundle(obj)))


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot

    matplotlib.pyplot.show = lambda *args, **kwargs: _flush_figures()
except ImportError:
    pass

if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":
    # google.colab only exists in the kernel; forward the BYOD upload dialog to it.
    def _upload():
        _send.send(("upload",))
        reply = _recv.recv()
        if reply[1] is None:
            raise RuntimeError("The notebook kernel could not open the upload dialog.")
        return reply[1]

    try:
        import google
    except ImportError:
        google = types.ModuleType("google")
        google.__path__ = []
        sys.modules["google"] = google
    _colab = types.ModuleType("google.colab")
    _files = types.ModuleType("google.colab.files")
    _files.upload = _upload
    _colab.files = _files
    google.colab = _colab
    sys.modules["google.colab"] = _colab
    sys.modules["google.colab.files"] = _files

_main = types.ModuleType("__main__")
_main.__dict__.update(__builtins__=builtins, display=display)
sys.modules["__main__"] = _main
_count = 0
while True:
    # An interrupt only lands inside a running cell; between cells it is ignored.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        message = _recv.recv()
    except EOFError:
        break
    if message[0] != "run":
        continue
    _count += 1
    filename = f"<isolated cell {_count}>"
    source = message[1]
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        tree = ast.parse(source, filename)
        tail = ast.Expression(tree.body.pop().value) if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        exec(compile(tree, filename, "exec"), _main.__dict__)
        if tail is not None:
            value = eval(compile(tail, filename, "eval"), _main.__dict__)
            if value is not None:
                display(value)
        _flush_figures()
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        _send.send(("done",))
    except BaseException as exc:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        frames = exc.__traceback__.tb_next if exc.__traceback__ is not None else None
        _send.send(("error", "".join(traceback.format_exception(type(exc), exc, frames)), f"{type(exc).__name__}: {exc}"))
"""


class IsolatedCellError(RuntimeError):
    """A routed cell raised inside the isolated environment; its traceback is printed above."""


class IsolatedRuntime:
    """One persistent worker process in the isolated environment, fed one cell at a time."""

    def __init__(self, python, display=None):
        to_kernel_r, to_kernel_w = os.pipe()
        to_worker_r, to_worker_w = os.pipe()
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1", DIMER_ISOLATED_WORKER="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
        for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            env.pop(name, None)
        env["DIMER_KERNEL_IS_COLAB"] = "1" if "google.colab" in sys.modules else "0"
        self.proc = subprocess.Popen(
            [str(python), "-c", _WORKER_SOURCE, str(to_kernel_w), str(to_worker_r)],
            pass_fds=(to_kernel_w, to_worker_r),
            env=env,
            start_new_session=True,  # interrupts reach the worker only through run(), exactly once
        )
        os.close(to_kernel_w)
        os.close(to_worker_r)
        self._recv = Connection(to_kernel_r, writable=False)
        self._send = Connection(to_worker_w, readable=False)
        if display is None:
            from IPython.display import display
        self._display = display

    def _exited(self):
        return RuntimeError(
            f"The isolated environment's Python process exited (code {self.proc.wait()}); a crash of this kind is "
            "usually running out of memory. Restart the session and choose Run all again."
        )

    def run(self, source):
        try:
            self._send.send(("run", source))
        except OSError:
            raise self._exited() from None
        while True:
            try:
                message = self._recv.recv()
            except EOFError:
                raise self._exited() from None
            except KeyboardInterrupt:
                self.proc.send_signal(signal.SIGINT)
                continue
            kind = message[0]
            if kind == "stream":
                (sys.stdout if message[1] == "stdout" else sys.stderr).write(message[2])
            elif kind == "display":
                self._display(message[1], raw=True)
            elif kind == "upload":
                self._send.send(("upload", self._colab_upload()))
            elif kind == "error":
                sys.stderr.write(message[1])
                raise IsolatedCellError(message[2]) from None
            elif kind == "done":
                return

    @staticmethod
    def _colab_upload():
        try:
            from google.colab import files
        except ImportError:
            return None
        return files.upload()

    def close(self):
        self._send.close()
        self.proc.wait(timeout=30)


def _is_user_cell():
    # ipykernel transforms a cell before executing it; its caller knows whether this is a silent frontend request.
    frame = sys._getframe(2)
    while frame is not None:
        local = frame.f_locals
        if "silent" in local and "store_history" in local:
            return bool(local["store_history"]) and not bool(local["silent"])
        frame = frame.f_back
    return True


def _route_to_isolated_runtime(lines):
    source = "".join(lines)
    if not source.strip() or "# dimer: kernel cell" in source or not _is_user_cell():
        return lines
    return [f"_DIMER_ISOLATED_RUNTIME.run({source!r})\n"]


if SKIP_INSTALL:
    print("Routing disabled: the notebook runs in this kernel.")
else:
    _ip = _kernel_shell()
    _ip.input_transformers_cleanup[:] = [
        t for t in _ip.input_transformers_cleanup if getattr(t, "__name__", "") != "_route_to_isolated_runtime"
    ]
    if isinstance(globals().get("_DIMER_ISOLATED_RUNTIME"), IsolatedRuntime):
        _DIMER_ISOLATED_RUNTIME.close()
    _DIMER_ISOLATED_RUNTIME = IsolatedRuntime(ISOLATED_PYTHON)
    _ip.input_transformers_cleanup.append(_route_to_isolated_runtime)
    print(f"Every later code cell now runs in {ISOLATED_PYTHON} (pid {_DIMER_ISOLATED_RUNTIME.proc.pid}).")
'''
