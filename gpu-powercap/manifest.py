"""Experiment manifest and session records.

manifest.json is written once, when an experiment starts. Its "measurement" half is everything
that changes what the numbers mean: GPU UUID and driver, software versions, model revision,
workload shapes, concurrency, timing, the run plan, and a hash of the measurement code.
sweep.py refuses to resume if any of it differs, so a resumed run can't quietly mix old and new
conditions. Change something on purpose? Use a new --experiment name. For a code fix you've
decided is safe to mix in, --allow-code-change lets the resume go ahead.

Every sweep.py session also writes sessions/<session_id>.json (never rewritten) with the code
and software it actually ran, and every results row carries session_id and code_version, so
it's always possible to tell which rows came from which implementation.
"""
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent
# code that decides what gets measured and how. config.py is checked value by value instead
MEASUREMENT_CODE = ["sweep.py", "gpu.py", "loadgen.py", "train_bench.py", "manifest.py"]
OTHER_CODE = ["config.py", "aggregate.py", "preflight.py"]
# flashinfer-python decides vLLM's attention backend on Blackwell; the rest is what vLLM's server
# and tokenizer path run on
PACKAGES = ["vllm", "torch", "transformers", "tokenizers", "huggingface-hub", "flashinfer-python", "openai",
            "fastapi", "aiohttp", "uvloop", "psutil", "nvidia-ml-py"]


def _norm(x):
    return json.loads(json.dumps(x))  # tuples -> lists etc, so comparisons after a reload work


def _version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def code_version():
    """Short hash over all the measurement code."""
    h = hashlib.sha256()
    for f in MEASUREMENT_CODE:
        h.update(f.encode() + b"\0" + (ROOT / f).read_bytes())
    return h.hexdigest()[:12]


def code_hashes():
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()[:16]
            for f in MEASUREMENT_CODE + OTHER_CODE if (ROOT / f).exists()}


def model_revision():
    """Commit hash of the cached model snapshot, or None if it isn't cached."""
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(config.MODEL, local_files_only=True)).name
    except Exception:
        return None


def _cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _git():
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def software():
    return {p: _version(p) for p in PACKAGES} | {"python": sys.version.split()[0]}


def build(gpu, fam, phase, plan, extra=None):
    info = gpu.info()
    spec = config.GPU_FAMILIES[fam]
    return _norm({
        "measurement": {
            "phase": phase,
            "gpu_family": fam,
            "gpu": {k: info.get(k) for k in ("name", "uuid", "driver", "default_w", "min_w", "max_w", "mem_total")},
            "model_revision": model_revision(),
            "software": software(),
            "code_version": code_version(),
            "config": {k: getattr(config, k) for k in config.MEASUREMENT_KEYS},
            "family_config": {k: spec[k] for k in config.FAMILY_KEYS},
            "plan": plan,
            **(extra or {}),
        },
        "record": {
            "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "hostname": socket.gethostname(),
            "cpu": _cpu_model(),
            "platform": platform.platform(),
            "gpu_info": info,
            "code_sha256": code_hashes(),
        },
    })


def _diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += _diff(a.get(k), b.get(k), f"{path}.{k}" if path else k)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: was {json.dumps(a)[:200]}, now {json.dumps(b)[:200]}"]


def check_or_create(path, new, allow_code_change=False):
    """Write the manifest if it's new. Otherwise return (blocking differences, allowed differences)."""
    if not path.exists():
        path.write_text(json.dumps(new, indent=2))
        return [], []
    old = json.loads(path.read_text())
    diffs = _diff(old["measurement"], new["measurement"])
    code = [d for d in diffs if d.startswith("code_version:")]
    if code:  # say which files, not just that the combined hash moved
        code += [d for d in _diff(old["record"].get("code_sha256"), new["record"]["code_sha256"], "code_sha256")
                 if d.split(".", 1)[1].split(":")[0] in MEASUREMENT_CODE]
    other = [d for d in diffs if not d.startswith("code_version:")]
    if allow_code_change:
        return other, code
    return other + code, []


def new_session_id():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"


def write_session(out, session_id, phase, args):
    """One immutable record per sweep.py invocation."""
    d = out / "sessions"
    d.mkdir(exist_ok=True)
    (d / f"{session_id}.json").write_text(json.dumps(_norm({
        "session_id": session_id,
        "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "phase": phase,
        "argv": sys.argv,
        "args": vars(args),
        "code_version": code_version(),
        "code_sha256": code_hashes(),
        "git": _git(),
        "software": software(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }), indent=2))
