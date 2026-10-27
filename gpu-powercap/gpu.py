"""NVML wrappers: read the power limits, set caps, and sample power/energy/clocks in a background thread."""
import os
import subprocess
import threading
import time

import pynvml as nv

NVMLError = nv.NVMLError  # not a RuntimeError, so callers that retry need to name it

# throttle reason bits from nvml.h
REASONS = {
    "sw_power_cap": 0x4,
    "hw_slowdown": 0x8,
    "sw_thermal": 0x20,
    "hw_thermal": 0x40,
    "hw_power_brake": 0x80,
}

# power_instant_mw is last so the older columns keep their positions; None if the driver lacks it
SAMPLE_FIELDS = ["t", "energy_mj", "power_mw", "enforced_mw", "sm_mhz", "mem_mhz", "temp_c", "reasons",
                 "power_instant_mw"]

_FI_POWER_INSTANT = getattr(nv, "NVML_FI_DEV_POWER_INSTANT", 186)  # from nvml.h
_VALUE_FIELD = {0: "dVal", 1: "uiVal", 2: "ulVal", 3: "ullVal", 4: "sllVal", 5: "siVal"}


def _str(x):
    return x.decode() if isinstance(x, bytes) else x


def as_root(cmd):
    """cmd with passwordless sudo in front, unless we're already root (some providers log you in as root)."""
    return list(cmd) if os.geteuid() == 0 else ["sudo", "-n", *cmd]


def box_lock(results_dir):
    """One sweep.py/preflight.py per box: a second one would fight over the cap and the files.
    Held while the returned file stays open, and released when the process ends, even on kill -9."""
    import fcntl
    results_dir.mkdir(parents=True, exist_ok=True)
    f = open(results_dir / ".lock", "a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        raise SystemExit("another sweep.py or preflight.py is already running on this box. tmux attach -t sweep?")
    return f


def child_env(index):
    """Env for vLLM/torch children so CUDA device 0 is the same GPU that NVML index `index` measures.
    CUDA's default order is fastest-first, NVML's is PCI bus order, and on a multi-GPU box they can differ."""
    return dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(index))


class Gpu:
    def __init__(self, index=0):
        nv.nvmlInit()
        self.index = index
        self.h = nv.nvmlDeviceGetHandleByIndex(index)
        self.name = _str(nv.nvmlDeviceGetName(self.h))
        self.uuid = _str(nv.nvmlDeviceGetUUID(self.h))
        self.driver = _str(nv.nvmlSystemGetDriverVersion())
        mn, mx = nv.nvmlDeviceGetPowerManagementLimitConstraints(self.h)
        self.min_w, self.max_w = mn // 1000, mx // 1000
        self.default_w = nv.nvmlDeviceGetPowerManagementDefaultLimit(self.h) // 1000
        self.mem_total = nv.nvmlDeviceGetMemoryInfo(self.h).total
        self._reasons = (getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons", None)
                         or nv.nvmlDeviceGetCurrentClocksThrottleReasons)
        # when the cap last actually changed, and from what: setting the same cap again doesn't count
        self.cap_w = self.prev_cap_w = self.cap_since = None
        # NVML's regular power reading is averaged over ~1 s. Use the instantaneous field if this
        # driver has it and it reads like milliwatts.
        self.has_instant = True
        mw = self._instant_mw()
        self.has_instant = mw is not None and 1_000 < mw < 5 * self.max_w * 1000

    def info(self):
        """Hardware identity for the manifest. Anything a VM doesn't expose is just left out."""
        d = {"name": self.name, "uuid": self.uuid, "driver": self.driver, "min_w": self.min_w,
             "max_w": self.max_w, "default_w": self.default_w, "mem_total": self.mem_total,
             "instant_power": self.has_instant}
        for key, fn in (("serial", nv.nvmlDeviceGetSerial), ("vbios", nv.nvmlDeviceGetVbiosVersion),
                        ("pci_bus_id", lambda h: nv.nvmlDeviceGetPciInfo(h).busId),
                        ("cuda_driver_version", lambda h: nv.nvmlSystemGetCudaDriverVersion())):
            try:
                d[key] = _str(fn(self.h))
            except nv.NVMLError:
                pass
        return d

    def compute_procs(self):
        return nv.nvmlDeviceGetComputeRunningProcesses(self.h)

    def wait_until_free(self, timeout=180):
        """Wait for every process to let go of the GPU, e.g. vLLM's engine core after the server exits."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.compute_procs():
                return
            time.sleep(2)
        pids = [p.pid for p in self.compute_procs()]
        raise RuntimeError(f"GPU still in use by pid(s) {pids} after {timeout} s")

    def enforced_w(self):
        return nv.nvmlDeviceGetEnforcedPowerLimit(self.h) / 1000

    def set_cap(self, watts):
        # timeout: a wedged driver can hang nvidia-smi forever; this turns it into a failed run
        r = subprocess.run(as_root(["nvidia-smi", "-i", str(self.index), "-pl", str(watts)]),
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"nvidia-smi -pl {watts} failed: {(r.stdout + r.stderr).strip()[-300:]}")
        time.sleep(1)
        got = self.enforced_w()
        if abs(got - watts) > 1:
            raise RuntimeError(f"Asked for {watts} W but enforced limit is {got:.0f} W")
        if watts != self.cap_w:
            self.prev_cap_w, self.cap_w, self.cap_since = self.cap_w, watts, time.time()

    def energy_mj(self):
        return nv.nvmlDeviceGetTotalEnergyConsumption(self.h)

    def _instant_mw(self):
        if not self.has_instant:
            return None
        try:
            v = nv.nvmlDeviceGetFieldValues(self.h, [_FI_POWER_INSTANT])[0]
            if v.nvmlReturn != 0:
                return None
            return getattr(v.value, _VALUE_FIELD.get(v.valueType, "uiVal"), None)
        except Exception:  # old bindings/drivers: just go without it
            return None

    def sample(self):
        return (
            time.time(),
            nv.nvmlDeviceGetTotalEnergyConsumption(self.h),
            nv.nvmlDeviceGetPowerUsage(self.h),
            nv.nvmlDeviceGetEnforcedPowerLimit(self.h),
            nv.nvmlDeviceGetClockInfo(self.h, nv.NVML_CLOCK_SM),
            nv.nvmlDeviceGetClockInfo(self.h, nv.NVML_CLOCK_MEM),
            nv.nvmlDeviceGetTemperature(self.h, nv.NVML_TEMPERATURE_GPU),
            self._reasons(self.h),
            self._instant_mw(),
        )


class Sampler:
    """Grabs a sample every `period` seconds until the with-block ends.

    If sampling fails, the error comes out of check() and out of the with-block, instead of the
    thread quietly dying and leaving the run with a few seconds of telemetry.
    """

    def __init__(self, gpu, period=0.1):
        self.gpu, self.period, self.rows, self.error = gpu, period, [], None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        try:
            while not self._stop.is_set():
                self.rows.append(self.gpu.sample())
                self._stop.wait(self.period)
        except Exception as e:
            self.error = e

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"power sampler died: {type(self.error).__name__}: {self.error}")
        if not self._thread.is_alive() and not self._stop.is_set():
            raise RuntimeError("power sampler stopped")

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, *exc):
        self._stop.set()
        self._thread.join(timeout=10)
        if exc_type is None:  # don't hide an exception that's already on its way out
            if self._thread.is_alive():
                raise RuntimeError("power sampler hung, an NVML call didn't return")
            self.check()


def window_stats(rows, t0, t1):
    """Average power over [t0, t1] from the energy counter, plus some stats for sanity checks.

    max_gap_s includes the stretches from t0 to the first sample and from the last sample to t1,
    so telemetry that only covers part of the window shows up as a gap.
    NVML's regular power reading is averaged over ~1 s, so its max/p99 understate sub-second
    peaks. The *_instant_* columns use the instantaneous reading when the driver has one.
    """
    win = [r for r in rows if t0 <= r[0] <= t1]
    if len(win) < 2:
        raise RuntimeError(f"only {len(win)} power samples in the timed window")
    first, last = win[0], win[-1]
    dt = last[0] - first[0]
    gaps = [first[0] - t0, t1 - last[0]] + [b[0] - a[0] for a, b in zip(win, win[1:])]
    power = sorted(r[2] for r in win)
    instant = sorted(r[8] for r in win if r[8] is not None)
    stats = {
        "window_s": dt,
        "energy_j": (last[1] - first[1]) / 1000,
        "avg_power_w": (last[1] - first[1]) / 1000 / dt if dt > 0 else float("nan"),
        "avg_power_usage_w": sum(power) / len(power) / 1000,
        "max_power_usage_w": power[-1] / 1000,
        "p99_power_usage_w": power[int(0.99 * (len(power) - 1))] / 1000,
        "max_power_instant_w": instant[-1] / 1000 if instant else None,
        "p99_power_instant_w": instant[int(0.99 * (len(instant) - 1))] / 1000 if instant else None,
        "enforced_w": win[len(win) // 2][3] / 1000,
        "enforced_w_min": min(r[3] for r in win) / 1000,
        "enforced_w_max": max(r[3] for r in win) / 1000,
        "sm_mhz_mean": sum(r[4] for r in win) / len(win),
        "mem_mhz_mean": sum(r[5] for r in win) / len(win),
        "temp_c_mean": sum(r[6] for r in win) / len(win),
        "temp_c_max": max(r[6] for r in win),
        "temp_c_first": first[6],  # with last: still warming up if these are far apart
        "temp_c_last": last[6],
        "max_gap_s": max(gaps),
        "samples_per_s": len(win) / (t1 - t0) if t1 > t0 else 0.0,
    }
    for name, bit in REASONS.items():
        stats[f"frac_{name}"] = sum(1 for r in win if r[7] & bit) / len(win)
    return stats
