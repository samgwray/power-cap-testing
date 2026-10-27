"""Run this first on a new box. Makes sure the machine is what we think it is and that
power caps actually work on it, before we burn hours of GPU time.

Exits 1 if anything FAILs. Log and environment dump go to results/<gpu>/host-<hostname>/, so two
rentals of the same GPU type never overwrite each other's.
"""
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pynvml as nv

import config
from gpu import Gpu, Sampler, as_root, box_lock, child_env, window_stats

ROOT = Path(__file__).resolve().parent

# torch.compile -> Triton compiles a C launcher against Python.h with gcc at runtime; vLLM and the
# training probe both go through this, so a missing compiler or header should fail here, not mid-smoke
COMPILE_CHECK = """
import torch
f = torch.compile(lambda x: (x * 2).sin() + 1)
x = torch.randn(4096, device='cuda')
assert torch.allclose(f(x), (x * 2).sin() + 1, atol=1e-4), 'compiled result differs'
print('torch.compile ok')
"""

# vLLM's own tokenizer path, which is where an incompatible transformers breaks `vllm serve` at startup
VLLM_TOKENIZER = f"""
from vllm.transformers_utils.tokenizer import get_tokenizer
t = get_tokenizer({config.MODEL!r})
print(type(t).__name__, len(t), "tokens")
"""

LOAD = """
import sys, time, torch
a = torch.randn(8192, 8192, device='cuda', dtype=torch.bfloat16)
end = time.time() + float(sys.argv[1])
while time.time() < end:
    for _ in range(20): a @ a
    torch.cuda.synchronize()
"""

CUDA_CHECK = """
import torch
assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'
a = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
err = (a @ torch.eye(1024, device='cuda') - a).abs().max().item()
assert err < 1e-3, f'matmul result wrong (max err {err})'
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
"""


def run_load(gpu, seconds):
    with Sampler(gpu, config.SAMPLE_PERIOD_S) as s:
        subprocess.run([sys.executable, "-c", LOAD, str(seconds)], check=True, env=child_env(gpu.index))
    t_end = s.rows[-1][0]
    return window_stats(s.rows, t_end - (seconds - 10), t_end)  # ignore the first 10s


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def main():
    lock = box_lock(ROOT / "results")  # a running sweep and this would fight over the cap
    os.environ["HF_HUB_OFFLINE"] = "1"  # setup.sh downloaded the model; check what's cached, don't fetch
    gpu = Gpu(0)
    fam = config.gpu_family(gpu.name)
    spec = config.GPU_FAMILIES[fam]
    out = ROOT / "results" / fam / f"host-{socket.gethostname()}"
    out.mkdir(parents=True, exist_ok=True)
    log_lines, failures = [], []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    def check(ok, name, detail="", fatal=True):
        tag = "PASS" if ok else ("FAIL" if fatal else "WARN")
        log(f"{tag} {name}" + (f": {detail}" if detail else ""))
        if not ok and fatal:
            failures.append(name)
        return ok

    (out / "env.txt").write_text("\n\n".join(sh(c)[1] for c in [
        "nvidia-smi", "nvidia-smi -q", f"{sys.executable} -m pip freeze", "vllm --version", "lscpu", "df -h",
    ]) + f"\n\n{platform.platform()}\n{sys.version}\n")
    log(f"GPU: {gpu.name}  UUID: {gpu.uuid}  driver: {gpu.driver}")
    log(f"Power limits: default {gpu.default_w} W, min {gpu.min_w} W, max {gpu.max_w} W")

    # is this the right box, and is it healthy
    check(os.geteuid() == 0 or sh("sudo -n true")[0] == 0, "root or passwordless sudo (needed for nvidia-smi -pl)")
    check(abs(gpu.default_w - spec["expected_default_w"]) <= 0.05 * spec["expected_default_w"],
          f"GPU is the expected {fam.upper()} SKU",
          f"default limit {gpu.default_w} W, expected {spec['expected_default_w']} W "
          "(PCIe/NVL H100s show 350/400 W)")
    procs = nv.nvmlDeviceGetComputeRunningProcesses(gpu.h)
    check(not procs, "no other processes on the GPU", f"{len(procs)} running" if procs else "")
    try:
        mig = nv.nvmlDeviceGetMigMode(gpu.h)[0]
        check(mig == 0, "MIG disabled", f"MIG mode {mig}")
    except nv.NVMLError:
        log("INFO MIG mode not reported")
    try:
        ecc = nv.nvmlDeviceGetTotalEccErrors(gpu.h, nv.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, nv.NVML_VOLATILE_ECC)
        check(ecc == 0, "no uncorrected ECC errors since boot", f"{ecc} errors: ask for a different machine")
    except nv.NVMLError:
        log("INFO ECC counters not reported")
    temp = gpu.sample()[6]
    check(temp < 55, "idle GPU temperature", f"{temp} C", fatal=False)
    free_gb = shutil.disk_usage(".").free / 1e9
    check(free_gb > 30, "free disk", f"{free_gb:.0f} GB (want ~30 GB for the model + telemetry)")
    # in case whoever had this box before locked the clocks
    sh(" ".join(as_root(["nvidia-smi", "-i", "0", "-rgc"])))
    sh(" ".join(as_root(["nvidia-smi", "-i", "0", "-rmc"])))

    # software
    r = subprocess.run([sys.executable, "-c", CUDA_CHECK], capture_output=True, text=True, env=child_env(gpu.index))
    rc, txt = r.returncode, (r.stdout + r.stderr).strip()
    check(rc == 0, "PyTorch sees the GPU and computes correctly", txt.splitlines()[-1] if txt else "")
    if "PyTorch sees the GPU and computes correctly" not in failures:
        r = subprocess.run([sys.executable, "-c", COMPILE_CHECK], capture_output=True, text=True,
                           env=child_env(gpu.index))
        txt = (r.stdout + r.stderr).strip()
        check(r.returncode == 0, "torch.compile works (needs gcc and Python headers)", txt.splitlines()[-1] if txt else "")
    rc, txt = sh("vllm --version")
    check(rc == 0, "vLLM installed", txt.splitlines()[-1] if txt else "")
    r = subprocess.run([sys.executable, "-c", VLLM_TOKENIZER], capture_output=True, text=True, env=child_env(gpu.index))
    txt = (r.stdout + r.stderr).strip()
    check(r.returncode == 0, "vLLM can build the model's tokenizer (needs transformers < 5)",
          txt.splitlines()[-1] if txt else "")
    rc, txt = sh(f"{sys.executable} -c 'import aiohttp, transformers; print(aiohttp.__version__)'")
    check(rc == 0, "load generator deps (aiohttp, transformers)", txt.splitlines()[-1] if txt else "")
    rc, txt = sh(f"{sys.executable} -c 'import uvloop, psutil'")
    check(rc == 0, "uvloop + psutil (faster client, API server CPU readings)", txt.splitlines()[-1] if txt else "",
          fatal=False)
    log(f"INFO instantaneous power reading: {'yes' if gpu.has_instant else 'not on this driver'}")
    rc, txt = sh(f"{sys.executable} -c 'import flashinfer; print(flashinfer.__version__)'")
    log(f"INFO flashinfer: {txt.splitlines()[-1] if rc == 0 and txt else 'not installed'}; "
        f"nvcc: {(sh('nvcc --version')[1].splitlines() or ['none'])[-1]}")
    if fam == "b200" and rc != 0 and not spec.get("attention_backend"):
        check(False, "attention backend is a deliberate choice", "no FlashInfer, so vLLM will fall back to "
              "FlashAttention 2 on Blackwell. Fine if intended (set attention_backend in config.py), "
              "otherwise FLASHINFER=1 bash setup.sh", fatal=False)
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(config.MODEL, local_files_only=True)
        check(any(Path(path).glob("*.safetensors")), "model weights cached", path)
    except Exception as e:
        check(False, "model weights cached", f"{type(e).__name__}, did setup.sh finish?")

    # the actual point: does the cap work
    subprocess.run(as_root(["nvidia-smi", "-pm", "1"]), stdout=subprocess.DEVNULL, timeout=60)
    test_w = max(gpu.min_w, int(round(0.6 * gpu.default_w, -1)))
    caps = config.caps_for(fam, gpu.min_w, gpu.max_w, gpu.default_w)
    log(f"Caps that will be swept: {caps}")
    cap_set = False
    try:  # whatever happens below, the GPU goes back to its default limit
        try:
            gpu.set_cap(test_w)
            check(True, f"set cap to {test_w} W and read it back")
            cap_set = True
        except Exception as e:
            check(False, "set power cap", str(e))
        if cap_set and "PyTorch sees the GPU and computes correctly" not in failures:
            cap_tests(gpu, test_w, min(caps), check, log)
    finally:
        if cap_set:
            gpu.set_cap(gpu.default_w)

    if shutil.which("dcgmi"):
        ok = sh("dcgmi dmon -e 1002,1004,1005 -c 3")[0] == 0
        check(ok, "DCGM profiling fields (SM/tensor/DRAM active)", fatal=False)
    else:
        log("WARN no dcgmi, that's fine, telemetry.sh just skips it")

    log("RESULT: " + ("FAILED: " + "; ".join(failures) if failures else "ALL CHECKS PASSED"))
    (out / "preflight.log").write_text("\n".join(log_lines) + "\n")
    lock.close()
    sys.exit(1 if failures else 0)


def cap_tests(gpu, test_w, floor_w, check, log):
    """Capped load, then uncapped, then the lowest cap the screen will use."""
    capped = run_load(gpu, 40)
    log(f"Under load at {test_w} W cap: avg {capped['avg_power_w']:.0f} W (energy counter), "
        f"max sample {capped['max_power_usage_w']:.0f} W, sw_power_cap active "
        f"{capped['frac_sw_power_cap']:.0%} of samples, SM {capped['sm_mhz_mean']:.0f} MHz")
    check(capped["avg_power_w"] <= test_w * config.CAP_OVERSHOOT, "cap enforced under load")
    check(capped["frac_sw_power_cap"] >= 0.5, "firmware reports power capping (sw_power_cap)",
          f"{capped['frac_sw_power_cap']:.0%} of samples", fatal=False)

    gpu.set_cap(gpu.default_w)
    free = run_load(gpu, 40)
    log(f"Under load at default {gpu.default_w} W: avg {free['avg_power_w']:.0f} W, "
        f"SM {free['sm_mhz_mean']:.0f} MHz, max temp {free['temp_c_max']} C")
    check(free["avg_power_w"] > test_w * 1.1, "power rises again when the cap is lifted",
          f"{free['avg_power_w']:.0f} W")
    check(free["sm_mhz_mean"] > capped["sm_mhz_mean"] * 1.05, "SM clocks drop under the cap",
          f"{capped['sm_mhz_mean']:.0f} vs {free['sm_mhz_mean']:.0f} MHz", fatal=False)
    slow = [k for k in ("frac_hw_slowdown", "frac_sw_thermal", "frac_hw_thermal", "frac_hw_power_brake")
            if free[k] > 0.01]
    check(not slow, "no thermal/HW slowdown at full power", ", ".join(slow))
    diff = abs(free["avg_power_w"] - free["avg_power_usage_w"]) / free["avg_power_usage_w"]
    check(diff <= 0.05, "energy counter agrees with power samples", f"{diff:.1%} apart")
    check(free["max_gap_s"] < 1, "10 Hz power sampling keeps up", f"max gap {free['max_gap_s']:.2f} s")

    if floor_w < test_w:
        # the sweep keeps runs whose cap is below what the GPU can reach under load (marked as its
        # power floor), but it's worth knowing before paying for them
        gpu.set_cap(floor_w)
        low = run_load(gpu, 40)
        ok = low["avg_power_w"] <= floor_w * config.CAP_OVERSHOOT + 5
        check(ok, f"holds the lowest swept cap ({floor_w} W) under load", f"avg {low['avg_power_w']:.0f} W"
              + ("" if ok else ", so those runs will measure the GPU's power floor instead"), fatal=False)


if __name__ == "__main__":
    main()
