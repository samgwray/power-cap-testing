"""Runs the power cap measurements.

    python sweep.py --smoke                              # quick end-to-end test, do this first
    python sweep.py --phase concurrency                  # throughput vs concurrency at a few caps
    python sweep.py                                      # screening sweep over all caps
    python sweep.py --phase confirm --caps 450,500,550   # balanced blocks of the promising caps
    python sweep.py --only training                      # just one workload
    python sweep.py --dry-run --gpu h100 [--phase ...]   # print the plan, no GPU needed

Each experiment lives in results/<gpu>/<experiment>/ with a manifest.json that pins down
everything that affects the numbers, including the measurement code. Rerunning the same command
resumes it and skips finished runs, but only if the manifest still matches. Otherwise it stops
and tells you what changed.

The load keeps running across cap changes: set cap, let it settle (at least SETTLE_S, then until
GPU temperature stops moving), then time a fixed window with energy and tokens read at the same
two instants. Inference then holds the cap until every request sent during the window has
finished, so latency isn't censored.

Every measurement goes in results.csv with a quality of ok, warn or rejected, plus the reasons.
Rejected runs stay in the file but get measured again on resume. A run that looks broken gets
one retry, then the sweep stops so we don't keep paying for bad data. events.jsonl keeps every
warning, rejection and abort across sessions. status.json in the experiment dir shows progress.
"""
import argparse
import asyncio
import csv
import http.client
import importlib
import itertools
import json
import math
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent

RESULT_FIELDS = [
    "experiment", "phase", "session_id", "code_version", "measurement_id", "run_id", "attempt",
    "quality", "quality_notes", "gpu", "gpu_name", "gpu_uuid", "workload", "kind", "block", "seq", "role",
    "cap_w", "default_w", "concurrency", "input_len", "output_len", "settle_s",
    "t_cap", "cap_since", "prev_cap_w", "t_start", "t_end", "window_s", "energy_j", "avg_power_w",
    "avg_power_usage_w", "max_power_usage_w", "p99_power_usage_w", "max_power_instant_w", "p99_power_instant_w",
    "enforced_w", "enforced_w_min", "enforced_w_max", "sm_mhz_mean", "mem_mhz_mean", "temp_c_mean", "temp_c_max",
    "temp_c_first", "temp_c_last",
    "frac_sw_power_cap", "frac_hw_slowdown", "frac_sw_thermal", "frac_hw_thermal", "frac_hw_power_brake",
    "max_gap_s", "samples_per_s",
    "tokens_per_s", "total_tokens_per_s", "requests_per_s", "request_errors", "server_gen_tokens_per_s",
    "server_trace_n", "server_running_mean", "server_running_min", "server_waiting_mean", "server_waiting_max",
    "server_waiting_frac", "inflight_deficit", "kv_usage_mean", "kv_usage_max", "preemptions",
    "client_cpu_cores", "client_max_lag_s", "server_cpu_cores", "server_children_cpu_cores",
    "cohort_n", "cohort_errors", "cohort_wait_s",
    "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms", "e2e_p50_ms", "e2e_p99_ms",
    "train_batch", "train_steps", "step_time_s", "mfu", "final_loss",
]
INVALIDATING = ("frac_hw_slowdown", "frac_sw_thermal", "frac_hw_thermal", "frac_hw_power_brake")
TRACE_FIELDS = ["t", "running", "waiting", "kv", "preempt", "gen", "prompt"]
PHASES = ("screen", "confirm", "concurrency")


class SweepAbort(Exception):
    """Something is wrong enough that we should stop the whole sweep."""


class RunFailed(Exception):
    """A single run looks broken. Gets one retry."""


# ---------- plans ----------
# A run is {"block", "seq", "cap_w", "role"} plus "concurrency" in the concurrency phase.
# Inference keeps one load generator running for each stretch of runs at the same concurrency.

def plan_screen(caps, default_w, passes):
    """Every cap, high->low then low->high, with the default cap at the start/middle/end of each pass."""
    caps = [c for c in caps if c != default_w]
    orders = [sorted(caps, reverse=True), sorted(caps), random.Random(0).sample(caps, len(caps))]
    runs = []
    for b in range(passes):
        order = orders[b % len(orders)]
        mid = len(order) // 2
        seq = ([(default_w, "anchor")] + [(c, "sweep") for c in order[:mid]] + [(default_w, "anchor")]
               + [(c, "sweep") for c in order[mid:]] + [(default_w, "anchor")])
        runs += [{"block": b, "seq": i, "cap_w": c, "role": r} for i, (c, r) in enumerate(seq)]
    return runs


def williams_balanced(n, blocks):
    """True if the confirm orders form complete Williams designs: every cap follows every other cap
    equally often. With CONFIRM_BLOCKS = 4 that's 2 or 4 caps (default included)."""
    return blocks % (n if n % 2 == 0 else 2 * n) == 0


def plan_confirm(caps, default_w, blocks, seed):
    """Complete blocks: the default + every candidate once per block, in orders taken from a Williams
    design (a Williams square, plus its mirror image for an odd number of caps). When the block count is
    a multiple of the design, every cap follows every other cap equally often and carryover from the
    previous cap balances out; otherwise the blocks are distinct rows of it, balanced as far as they go."""
    rng = random.Random(seed)
    units = sorted(set(caps) | {default_w}, reverse=True)
    n = len(units)
    first = [0] + [(i + 1) // 2 if i % 2 else n - i // 2 for i in range(1, n)]
    square = [[(x + r) % n for x in first] for r in range(n)]
    if n % 2:  # odd n: the square plus its mirror image is the balanced design
        square += [row[::-1] for row in square]
    labels = rng.sample(units, n)
    rows = square * (blocks // len(square)) + rng.sample(square, blocks % len(square))
    rng.shuffle(rows)
    orders = [[labels[i] for i in row] for row in rows]
    return [{"block": b, "seq": i, "cap_w": c, "role": "anchor" if c == default_w else "candidate"}
            for b, order in enumerate(orders) for i, c in enumerate(order)]


def plan_concurrency(caps, concs):
    runs = []
    for b, conc in enumerate(sorted(concs)):
        runs += [{"block": b, "seq": i, "cap_w": c, "role": "sweep", "concurrency": conc}
                 for i, c in enumerate(sorted(caps, reverse=True))]
    return runs


def conc_sweep_caps(default_w, min_w, max_w):
    caps = {min(max_w, max(min_w, int(round(f * default_w, -1)))) for f in config.CONC_SWEEP_CAP_FRACS}
    return sorted(caps | {default_w}, reverse=True)


def run_id(wl, r):
    rid = f"{wl['name']}_b{r['block']}_{r['seq']:02d}_{r['cap_w']}W"
    return rid + (f"_c{r['concurrency']}" if r.get("concurrency") else "")


def timing(phase):
    if phase == "concurrency":
        return config.CONC_SWEEP_SETTLE_S, config.CONC_SWEEP_RUN_S
    return config.SETTLE_S, config.TARGET_RUN_S


def make_plan(phase, fam, default_w, min_w, max_w, confirm_caps=None, seed=0):
    spec = config.GPU_FAMILIES[fam]
    if phase == "screen":
        return plan_screen(config.caps_for(fam, min_w, max_w, default_w), default_w, config.PASSES)
    if phase == "confirm":
        bad = [c for c in confirm_caps if not min_w <= c <= max_w]
        if bad:
            raise SystemExit(f"--caps {bad} outside this GPU's {min_w}-{max_w} W range")
        n = len(set(confirm_caps) | {default_w})
        if not williams_balanced(n, config.CONFIRM_BLOCKS):
            print(f"note: {n} caps (default included) in {config.CONFIRM_BLOCKS} blocks isn't a complete Williams "
                  "design, so carryover from the previous cap is only partly balanced. 1 or 3 candidates balance it fully.")
        return plan_confirm(confirm_caps, default_w, config.CONFIRM_BLOCKS, seed)
    return plan_concurrency(conc_sweep_caps(default_w, min_w, max_w),
                            [c for c in spec["concurrency_sweep"] if c <= spec["max_num_seqs"]])


def dry_run(fam, phase, workloads, confirm_caps, seed):
    spec = config.GPU_FAMILIES[fam]
    d = spec["expected_default_w"]
    runs = make_plan(phase, fam, d, min(spec["caps"]), d, confirm_caps, seed)
    settle, window = timing(phase)
    blocks = len({r["block"] for r in runs})
    print(f"{fam} {phase}: {len(runs)} runs per workload in {blocks} block(s), "
          f"{settle}-{config.SETTLE_MAX_S}+{window} s per run")
    total_h = 0
    for wl in workloads:
        # cap change + settle (~20 s past the minimum on average, more after big cap jumps) + window per
        # run, plus a few minutes each time the load generator / trainer starts. Inference also waits
        # ~one request duration for the latency cohort; guess ~25 tok/s per request
        per_run = settle + 20 + window + 3 + (wl["output_len"] / 25 if wl["kind"] == "inference" else 0)
        restarts = blocks if (wl["kind"] == "inference" and phase == "concurrency") else 1
        h = (len(runs) * per_run + restarts * 180) / 3600
        total_h += h
        print(f"  {wl['name']:<14} ~{h:.1f} h")
    total_h += 0.5  # vLLM startup, idle measurement, training calibration
    print(f"Total ~{total_h:.1f} h, ~${total_h * spec['price_per_hr']:.0f} at ${spec['price_per_hr']}/hr. "
          "Rough: low caps make requests (and so the latency cohorts) take longer.")
    for b in sorted({r["block"] for r in runs}):
        print(f"  block {b}:", [r["cap_w"] for r in runs if r["block"] == b],
              f"concurrency {runs[[r['block'] for r in runs].index(b)].get('concurrency', 'from config')}")


# ---------- bookkeeping ----------

def write_atomic(path, text):
    """So `cat status.json` mid-write, or a crash mid-write, never sees half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_events(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass  # a line cut short by a crash
    return out


def append_result(path, row):
    fields = None
    if path.exists():
        with open(path, newline="") as f:
            fields = next(csv.reader(f), [])
        missing = [k for k in RESULT_FIELDS if k not in fields]
        if missing:  # newer code added columns: rewrite once with them, keeping every old row as it was
            old = read_rows(path)
            fields += missing
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(old)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or RESULT_FIELDS, extrasaction="ignore")
        if fields is None:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())  # a half-written row would be counted as done on resume


def save_samples(rows, path):
    from gpu import SAMPLE_FIELDS
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SAMPLE_FIELDS)
        w.writerows(rows)


def save_trace(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TRACE_FIELDS)
        w.writerows([t] + [m.get(k) for k in TRACE_FIELDS[1:]] for t, m in rows)


# ---------- monitor ----------

def power_floor(m, cap, min_w):
    """None if power stayed within the cap. True if it went over while the GPU was power-capping, or at
    its minimum limit: that's its power floor. False if it went over with no capping: a broken cap."""
    if m["avg_power_w"] <= cap * config.CAP_OVERSHOOT + 5:
        return None
    return m["frac_sw_power_cap"] >= 0.5 or cap <= min_w


class Monitor:
    """Judges each run, decides when to bail, and keeps status.json and events.jsonl up to date.

    events.jsonl is append-only across sessions (warnings, rejected and failed runs, aborts), and
    status.json is built from it, so an abort reason survives a later session finishing cleanly.
    """

    def __init__(self, out, total_runs, phase, session_id, prior_rows, min_w):
        self.path, self.events_path = out / "status.json", out / "events.jsonl"
        self.min_w = min_w  # the GPU's lowest allowed limit
        self.total, self.completed = total_runs, 0
        self.phase, self.session = phase, session_id
        self.started = time.time()
        self.warnings = []
        self.done = defaultdict(int)
        self.invalid = defaultdict(int)
        self.first_anchor = {}
        self.since_anchor = defaultdict(list)  # (run_id, measurement_id) accepted since the last good anchor
        self.last = None
        # carry state over from earlier sessions so a resume checks against the same reference
        for r in prior_rows:
            if r.get("quality") != "rejected":
                self._accept(r["workload"], r["role"], float(r["tokens_per_s"]),
                             any(float(r[k] or 0) > 0.01 for k in INVALIDATING), r["run_id"], r.get("measurement_id"))

    def _accept(self, wl, role, tps, throttled, rid, mid):
        self.done[wl] += 1
        if throttled:
            self.invalid[wl] += 1
        if role == "anchor":
            self.first_anchor.setdefault(wl, tps)
            self.since_anchor[wl] = []
        else:
            self.since_anchor[wl].append((rid, mid))

    def event(self, kind, msg, **extra):
        e = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "t": time.time(), "session": self.session,
             "kind": kind, "msg": msg, **extra}
        with open(self.events_path, "a") as f:
            f.write(json.dumps(e) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def warn(self, msg, **extra):
        print("WARN " + msg, flush=True)
        self.warnings.append(f"{time.strftime('%H:%M:%S')} {msg}")
        self.event("warning", msg, **extra)

    def write(self, state, **extra):
        elapsed = time.time() - self.started
        left = self.total - self.completed
        eta_h = elapsed / self.completed * left / 3600 if self.completed else None
        events = read_events(self.events_path)
        write_atomic(self.path, json.dumps({
            "state": state, "phase": self.phase, "session": self.session,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runs_done_this_session": self.completed, "runs_total_this_session": self.total,
            "eta_hours": round(eta_h, 2) if eta_h is not None else None,
            "last_run": self.last,
            "n_warnings_this_session": len(self.warnings), "recent_warnings": self.warnings[-20:],
            # history from every session of this experiment
            "aborts": [e for e in events if e["kind"] in ("abort", "interrupted")][-20:],
            "rejected_or_failed_runs": [{k: e.get(k) for k in ("time", "session", "run_id", "attempt", "msg")}
                                        for e in events if e["kind"] in ("run_rejected", "run_failed")][-50:],
            "n_warnings_all_sessions": sum(1 for e in events if e["kind"] == "warning"),
            **extra,
        }, indent=2))

    def hard_problems(self, m, cap):
        """Things that mean the measurement itself is broken. The run gets one retry."""
        errs = []
        if abs(m["enforced_w_min"] - cap) > 1 or abs(m["enforced_w_max"] - cap) > 1:
            errs.append(f"enforced limit was {m['enforced_w_min']:.0f}-{m['enforced_w_max']:.0f} W, cap is {cap} W")
        # Overshoot while the firmware is power-capping, or at the GPU's own minimum limit, is the GPU's
        # power floor (NVIDIA: a B200 capped at 200 W draws more even at idle), not a cap that isn't
        # working. assess() notes it instead
        if not math.isfinite(m["avg_power_w"]) or power_floor(m, cap, self.min_w) is False:
            errs.append(f"avg power {m['avg_power_w']:.0f} W is over the {cap} W cap and the GPU isn't reporting "
                        "power capping, so the cap isn't working")
        tps = m.get("tokens_per_s")
        if not tps or not math.isfinite(tps) or tps <= 0:
            errs.append(f"bad throughput {tps}")
        if m["max_gap_s"] > config.MAX_SAMPLE_GAP_S:
            errs.append(f"power samples have a {m['max_gap_s']:.1f} s hole (window ends included)")
        if m["samples_per_s"] < 0.5 / config.SAMPLE_PERIOD_S:
            errs.append(f"only {m['samples_per_s']:.1f} power samples/s, expected ~{1 / config.SAMPLE_PERIOD_S:.0f}")
        if m.get("request_errors"):
            errs.append(f"{m['request_errors']} requests failed during the window")
        if m.get("cohort_errors"):
            errs.append(f"{m['cohort_errors']} latency-cohort requests failed")
        if "preemptions" in m and m["preemptions"] is None and self.phase != "concurrency":
            errs.append("couldn't read vLLM's preemption counter around the window (/metrics failed)")
        return errs

    def assess(self, wl, run, m, rid, tag, default_w):
        """Quality of a run whose measurement worked: "ok", "warn" (kept, with notes) or "rejected"
        (kept in results.csv but measured again on resume). Also returns why to stop, if we should."""
        rejects, warns, abort = [], [], None
        conc_phase = self.phase == "concurrency"

        def problem(msg, stop):
            """Rejected (and stop) normally; just a warning while sweeping concurrency, where it's the point."""
            nonlocal abort
            if conc_phase:
                warns.append(msg)
            else:
                rejects.append(msg)
                abort = abort or f"{rid}: {msg}. {stop}"

        throttled = [k for k in INVALIDATING if m[k] > 0.01]
        if throttled:
            warns.append(f"throttled ({', '.join(k[len('frac_'):] for k in throttled)})")
        if power_floor(m, run["cap_w"], self.min_w):
            warns.append(f"avg power {m['avg_power_w']:.0f} W is over the {run['cap_w']} W cap while the GPU is "
                         "power-capping: it can't get that low under this load, so this run is its power floor")
        if abs(m["temp_c_last"] - m["temp_c_first"]) >= config.TEMP_DRIFT_WARN_C:
            warns.append(f"GPU temperature went {m['temp_c_first']} -> {m['temp_c_last']} C during the window, "
                         "so it wasn't settled")
        cpu = m.get("client_cpu_cores") or 0
        if cpu > config.CLIENT_CPU_MAX:
            problem(f"load generator used {cpu:.2f} cores, so it may be what limits throughput",
                    "Lower concurrency or fix the client before measuring caps.")
        if (m.get("client_max_lag_s") or 0) > config.CLIENT_LAG_MAX_S:
            warns.append(f"load generator event loop lagged {m['client_max_lag_s']:.2f} s")
        srv, deficit = m.get("server_cpu_cores") or 0, m.get("inflight_deficit") or 0
        if srv > config.SERVER_CPU_WARN and deficit > config.INFLIGHT_DEFICIT_WARN:
            # a starved engine at the high caps would flatten the top of the curve and make capping look free
            problem(f"vLLM's API server used {srv:.2f} cores and vLLM held {deficit:.1%} fewer requests than the "
                    "concurrency, so the API server, not the GPU, limits throughput",
                    "Lower concurrency before measuring caps.")
        else:
            if srv > config.SERVER_CPU_WARN:
                warns.append(f"vLLM's API server used {srv:.2f} cores, it may be the bottleneck")
            if deficit > config.INFLIGHT_DEFICIT_WARN:
                warns.append(f"vLLM held {deficit:.1%} fewer requests than the concurrency on average, so "
                             "requests are slow to turn around (client or API server)")
        if m.get("preemptions"):
            problem(f"vLLM preempted {m['preemptions']:.0f} requests at concurrency {m.get('concurrency')}",
                    "The KV cache can't hold that many, lower concurrency in config.py.")
        elif "preemptions" in m and m["preemptions"] is None:  # only gets this far in the concurrency phase
            warns.append("couldn't read vLLM's preemption counter around the window")
        if (m.get("kv_usage_max") or 0) > config.KV_USAGE_WARN:
            warns.append(f"KV cache peaked at {m['kv_usage_max']:.0%}, little headroom before preemption")
        if (m.get("server_waiting_frac") or 0) > config.WAITING_FRAC_WARN:
            warns.append(f"requests were queued in vLLM in {m['server_waiting_frac']:.0%} of samples "
                         f"(max {m['server_waiting_max']:.0f}), so fewer than {m.get('concurrency')} ran")
        n = m.get("server_trace_n")
        if n is not None and n < 0.8 * m["window_s"] / config.METRICS_PERIOD_S:
            warns.append(f"only {n} /metrics samples in the window, so the queue/KV numbers cover part of it")
        if run["role"] == "anchor":
            ref = self.first_anchor.get(wl)
            if ref is None:
                ratio = m["avg_power_w"] / default_w
                if ratio < config.FULL_LOAD_RATIO:
                    warns.append(f"averages {ratio:.0%} of the default cap ({m['avg_power_w']:.0f} W, p99 "
                                 f"{m['p99_power_usage_w']:.0f} W). Caps above the average can still clip "
                                 "higher-power phases, so they aren't necessarily no-ops, but the load isn't "
                                 "using the GPU's whole power budget")
            else:
                dev = abs(m["tokens_per_s"] / ref - 1)
                if dev > config.ANCHOR_ABORT:
                    rejects.append(f"default-cap throughput is {dev:.1%} off the first anchor")
                    abort = abort or (f"{rid}: {wl} throughput at the default cap drifted {dev:.1%} from the first "
                                      "run. Something changed on the box, so the runs aren't comparable anymore.")
                    # the runs since the last good anchor may be off too: they get measured again on resume
                    suspect = self.since_anchor[wl]
                    if suspect:
                        self.event("runs_suspect", f"{wl}: measured since the last good anchor, before {rid} "
                                   f"drifted {dev:.1%}", run_ids=[a for a, _ in suspect],
                                   measurement_ids=[b for _, b in suspect])
                elif dev > config.ANCHOR_WARN:
                    warns.append(f"default cap is {dev:.1%} off the first anchor")

        quality = "rejected" if rejects else ("warn" if warns else "ok")
        if quality != "rejected":
            self._accept(wl, run["role"], m["tokens_per_s"], bool(throttled), rid, tag)
            if throttled and self.invalid[wl] >= 4 and self.invalid[wl] / self.done[wl] > config.INVALID_ABORT_FRAC:
                abort = abort or (f"{self.invalid[wl]} of {self.done[wl]} {wl} runs got throttled. "
                                  "This box runs hot, try a different one.")
        free_gb = shutil.disk_usage(ROOT).free / 1e9
        if free_gb < config.MIN_FREE_DISK_GB:
            abort = abort or f"only {free_gb:.1f} GB of disk left"
        self.completed += 1
        self.last = {"run_id": rid, "quality": quality, "tokens_per_s": round(m["tokens_per_s"], 1),
                     "avg_power_w": round(m["avg_power_w"], 1), "temp_c_max": m["temp_c_max"]}
        for n in rejects + warns:
            self.warn(f"{rid}: {n}", run_id=rid)
        return quality, rejects + warns, abort


# ---------- signals ----------

_signals = []


def _interrupt(signum, frame):
    _signals.append(signum)
    # Only the first one raises. A second (double Ctrl-C, or the two SIGHUPs an SSH drop sends <1 ms
    # apart) would cut the vLLM/trainer cleanup short and leave vLLM on the GPU. Force quit: kill -9,
    # then pkill -f 'vllm serve'
    if len(_signals) == 1:
        raise KeyboardInterrupt(f"signal {signum}")
    print("already stopping, cleaning up (vLLM can take a couple of minutes to exit)", flush=True)


async def _watch_signals():
    """uvloop and aiohttp's parser catch a KeyboardInterrupt raised inside their protocol callbacks, so
    under streaming load a signal can get lost. Raised again from a plain task, it stops the loop."""
    while True:
        await asyncio.sleep(1)
        if _signals:
            raise KeyboardInterrupt(f"signal {_signals[0]}")


# ---------- inference ----------

def port_in_use(host, port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def server_cmd(fam):
    spec = config.GPU_FAMILIES[fam]
    return ["vllm", "serve", config.MODEL, "--host", config.VLLM_HOST, "--port", str(config.VLLM_PORT),
            "--gpu-memory-utilization", str(config.GPU_MEMORY_UTILIZATION),
            "--max-num-seqs", str(spec["max_num_seqs"]),
            "--max-model-len", str(config.MAX_MODEL_LEN), "--seed", "0",
            # prompts are random and never repeat, but make sure no caching can skip prefill
            "--no-enable-prefix-caching"]


def server_env(fam, index):
    spec = config.GPU_FAMILIES[fam]
    env = child_env(index)
    if spec.get("attention_backend"):
        env["VLLM_ATTENTION_BACKEND"] = spec["attention_backend"]
    env.update(spec.get("vllm_env") or {})
    return env


class VllmServer:
    def __init__(self, gpu, fam, log_path):
        self.gpu, self.fam, self.log_path, self.proc = gpu, fam, log_path, None
        self.base = f"http://{config.VLLM_HOST}:{config.VLLM_PORT}"
        self.metric_missing = False

    def __enter__(self):
        # a leftover server from a crashed session would answer our health checks and get benchmarked
        if port_in_use(config.VLLM_HOST, config.VLLM_PORT):
            raise SweepAbort(f"something is already listening on port {config.VLLM_PORT}. "
                             "Old vllm serve? `pkill -f 'vllm serve'` and try again.")
        if self.gpu.compute_procs():
            raise SweepAbort("another process is already using the GPU. Check nvidia-smi.")
        cmd = server_cmd(self.fam)
        self.log = open(self.log_path, "a")
        self.log.write(f"\n=== {time.ctime()} {' '.join(cmd)}\n")
        self.log.flush()
        # own session so we can take down vLLM's engine-core child process with it
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     env=server_env(self.fam, self.gpu.index), start_new_session=True)
        try:
            deadline = time.time() + 1200
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    raise SweepAbort(f"vllm serve died while starting up:\n{self.log_tail()}")
                if self.healthy():
                    print("vLLM server ready", flush=True)
                    return self
                time.sleep(5)
            raise SweepAbort("vllm serve still isn't up after 20 min")
        except BaseException:  # __exit__ doesn't run when __enter__ raises, and vLLM is in its own session
            self._kill()
            self.log.close()
            raise

    def __exit__(self, exc_type, exc, tb):
        self._kill()
        self.log.close()
        try:
            self.gpu.wait_until_free()
        except RuntimeError as e:
            if exc_type is None:
                raise SweepAbort(f"vLLM didn't release the GPU: {e}")
            print(f"!! vLLM didn't release the GPU either: {e}", flush=True)  # keep the original reason

    def _kill(self):
        """SIGTERM, then SIGKILL, to vLLM's whole process group, even if the API server is already gone:
        the engine core is a separate process that would otherwise keep the GPU."""
        # an empty group gives ESRCH (ProcessLookupError) on Linux, EPERM on some other systems
        gone = (ProcessLookupError, PermissionError)
        for sig, wait_s in ((signal.SIGTERM, 120), (signal.SIGKILL, 30)):
            try:
                os.killpg(self.proc.pid, sig)
            except gone:
                return
            deadline = time.time() + wait_s
            while time.time() < deadline:
                self.proc.poll()  # reap the API server so it doesn't linger as a zombie
                try:
                    os.killpg(self.proc.pid, 0)  # anything left in the group?
                except gone:
                    return
                time.sleep(1)

    def healthy(self):
        try:
            return urllib.request.urlopen(self.base + "/health", timeout=5).status == 200
        except OSError:
            return False

    def check(self):
        if self.proc.poll() is not None or not self.healthy():
            raise SweepAbort(f"vLLM server went down:\n{self.log_tail()}")

    def metrics(self, tries=1):
        """Counters/gauges from /metrics, summed over label sets. Missing ones come back as None."""
        text = ""
        for i in range(tries):
            try:
                text = urllib.request.urlopen(self.base + "/metrics", timeout=5).read().decode()
                break
            except (OSError, http.client.HTTPException):
                if i + 1 < tries:
                    time.sleep(1)
        want = {"vllm:num_preemptions_total": "preempt", "vllm:generation_tokens_total": "gen",
                "vllm:prompt_tokens_total": "prompt", "vllm:num_requests_running": "running",
                "vllm:num_requests_waiting": "waiting", "vllm:kv_cache_usage_perc": "kv"}  # kv is a 0-1 fraction
        out = dict.fromkeys(want.values())
        for line in text.splitlines():
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name in want:
                out[want[name]] = (out[want[name]] or 0) + float(line.rsplit(" ", 1)[1])
        if out["preempt"] is None and text:
            # name differs in this vLLM version, so count log lines instead
            self.metric_missing = True
            out["preempt"] = sum(1 for l in Path(self.log_path).read_text().splitlines() if "preempt" in l.lower())
        return out

    def cpu_seconds(self):
        """CPU seconds used so far by the API server process and by its children (the engine core etc).
        (None, None) if psutil isn't around."""
        try:
            import psutil
            p = psutil.Process(self.proc.pid)
            main = sum(p.cpu_times()[:2])
            kids = 0.0
            for c in p.children(recursive=True):
                try:
                    kids += sum(c.cpu_times()[:2])
                except psutil.Error:
                    pass  # exited between listing and reading
            return main, kids
        except Exception:
            return None, None

    def startup_log(self):
        return Path(self.log_path).read_text(errors="replace").split("\n=== ")[-1]

    def kv_capacity(self):
        """What vLLM says the KV cache holds, from its startup log."""
        m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", self.startup_log())
        return int(m.group(1).replace(",", "")) if m else None

    def attention_backend(self):
        """The attention backend vLLM says it picked, whether it wanted FlashInfer but couldn't import it, and
        (FlashInfer on Blackwell) whether it used the TRT-LLM kernels or couldn't reach NVIDIA's artifactory."""
        text = self.startup_log()
        found = sorted(set(re.findall(r"Using (.+?) backend", text)))
        return {"backend": ", ".join(found) or None,
                "flashinfer_missing": "FlashInfer failed to import" in text,
                "trtllm_attention": "Using TRTLLM attention" in text,
                "artifactory_unreachable": "Failed to connect to NVIDIA artifactory" in text}

    def log_tail(self, n=40):
        self.log.flush()
        return "\n".join(Path(self.log_path).read_text().splitlines()[-n:])


class ServerTrace:
    """Polls vLLM's /metrics every `period` seconds for as long as the block runs, so queueing and KV
    pressure in the middle of a window show up, not just at its two ends."""

    def __init__(self, server, period):
        self.server, self.period, self.rows = server, period, []

    async def __aenter__(self):
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc):
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)

    async def _loop(self):
        while True:
            t = time.time()
            try:
                m = await asyncio.to_thread(self.server.metrics)
            except Exception:  # one bad response shouldn't end the trace
                m = {}
            self.rows.append(((t + time.time()) / 2, m))
            await asyncio.sleep(max(0.0, self.period - (time.time() - t)))


def trace_stats(rows, t0, t1, conc):
    win = [m for t, m in rows if t0 <= t <= t1]

    def col(k):
        return [m[k] for m in win if m.get(k) is not None]

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    running, waiting, kv = col("running"), col("waiting"), col("kv")
    out = {"server_trace_n": len(running),  # polls in the window that got an answer
           "server_running_mean": mean(running), "server_running_min": min(running, default=None),
           "server_waiting_mean": mean(waiting), "server_waiting_max": max(waiting, default=None),
           "server_waiting_frac": mean([1.0 if w > 0 else 0.0 for w in waiting]),
           "kv_usage_mean": mean(kv), "kv_usage_max": max(kv, default=None)}
    if running and waiting:
        out["inflight_deficit"] = 1 - (mean(running) + mean(waiting)) / conc
    return out


def child_env(index):
    from gpu import child_env as env
    return env(index)


def vocab_size():
    from transformers import AutoTokenizer
    # regular tokens only, no special ones. local_files_only: never touch the Hub mid-experiment
    return AutoTokenizer.from_pretrained(config.MODEL, local_files_only=True).vocab_size


async def settle(s, min_s):
    """At least min_s under load at the new cap, then until GPU temperature stops moving (the last two
    20 s means within SETTLE_FLAT_C), SETTLE_MAX_S at most. Returns how long it actually took."""
    t0 = time.time()
    await asyncio.sleep(min_s)
    while time.time() - t0 < config.SETTLE_MAX_S:
        now = time.time()
        a = [r[6] for r in s.rows if now - 40 <= r[0] < now - 20]
        b = [r[6] for r in s.rows if r[0] >= now - 20]
        if a and b and abs(sum(b) / len(b) - sum(a) / len(a)) <= config.SETTLE_FLAT_C:
            break
        await asyncio.sleep(5)
    return time.time() - t0


async def wait_cohort(lg, cohort):
    """Same cap, same background load, until everything sent in the window has finished."""
    deadline = time.time() + config.COHORT_TIMEOUT_S
    while not cohort.done.is_set():
        if time.time() > deadline:
            raise RunFailed(f"{cohort.pending} of {len(cohort.members)} latency-cohort requests still "
                            f"running {config.COHORT_TIMEOUT_S} s after the window")
        if lg.stalled(config.STALL_S):
            raise RunFailed(f"no request has succeeded for {config.STALL_S} s while waiting for the latency "
                            f"cohort ({lg.errors} errors, last: {lg.last_error})")
        await asyncio.sleep(0.5)


async def measure_inference(ctx, lg, wl, cap, tag, settle_min_s, window_s):
    from gpu import Sampler, window_stats
    from loadgen import latency_stats
    gpu, server, run_dir = ctx["gpu"], ctx["server"], ctx["run_dir"]
    s = trace = None
    try:
        # power samples and the /metrics trace cover settle, window and the cohort's follow-up
        with Sampler(gpu, config.SAMPLE_PERIOD_S) as s:
            async with ServerTrace(server, config.METRICS_PERIOD_S) as trace:
                await asyncio.to_thread(gpu.set_cap, cap)
                t_cap = time.time()
                settle_s = await settle(s, settle_min_s)
                await asyncio.to_thread(server.check)
                met0 = await asyncio.to_thread(server.metrics, 3)
                cpu0 = server.cpu_seconds()
                m0, e0 = lg.mark(), gpu.energy_mj()
                cohort = lg.open_cohort(tag)
                try:
                    await asyncio.sleep(window_s)
                    m1, e1 = lg.mark(), gpu.energy_mj()
                finally:
                    lg.close_cohort()
                cpu1 = server.cpu_seconds()
                met1 = await asyncio.to_thread(server.metrics, 3)
                s.check()
                await wait_cohort(lg, cohort)
                t_done = time.time()
        t0, t1 = m0["t"], m1["t"]
        try:
            stats = window_stats(s.rows, t0, t1)
        except RuntimeError as e:
            raise RunFailed(str(e))
    finally:
        # every attempt keeps its raw telemetry, failed ones included
        if s is not None:
            save_samples(s.rows, run_dir / f"{tag}_samples.csv")
        if trace is not None:
            save_trace(trace.rows, run_dir / f"{tag}_server.csv")
    dt = t1 - t0
    # energy straight from the counter at the same instants the token counts were taken
    stats.update(window_s=dt, energy_j=(e1 - e0) / 1000, avg_power_w=(e1 - e0) / 1000 / dt)

    def edge(k, met, t, after):
        """A counter at a window end: the direct read, or else the nearest trace sample just outside."""
        if met.get(k) is not None:
            return met[k]
        vals = [m[k] for tt, m in trace.rows if m.get(k) is not None and (tt >= t if after else tt <= t)]
        return (vals[0] if after else vals[-1]) if vals else None

    def delta(k):
        a, b = edge(k, met0, t0, False), edge(k, met1, t1, True)
        return None if a is None or b is None else b - a

    def cores(a, b):
        return None if a is None or b is None else (b - a) / dt

    gen = delta("gen")
    return dict(stats, **latency_stats(cohort, wl["output_len"]), **trace_stats(trace.rows, t0, t1, lg.conc),
                t_cap=t_cap, cap_since=gpu.cap_since, prev_cap_w=gpu.prev_cap_w, t_start=t0, t_end=t1,
                settle_s=settle_s, cohort_wait_s=t_done - t1,
                concurrency=lg.conc, input_len=wl["input_len"], output_len=wl["output_len"],
                tokens_per_s=(m1["out"] - m0["out"]) / dt,
                total_tokens_per_s=(m1["out"] - m0["out"] + m1["prompt"] - m0["prompt"]) / dt,
                requests_per_s=(m1["completed"] - m0["completed"]) / dt,
                request_errors=m1["errors"] - m0["errors"],
                client_cpu_cores=(m1["cpu"] - m0["cpu"]) / dt, client_max_lag_s=m1["max_lag_s"],
                server_cpu_cores=cores(cpu0[0], cpu1[0]), server_children_cpu_cores=cores(cpu0[1], cpu1[1]),
                server_gen_tokens_per_s=gen / dt if gen is not None else None,
                preemptions=delta("preempt"))


async def run_inference(ctx, wl, runs, phase):
    from loadgen import LoadGen
    gpu, server, monitor = ctx["gpu"], ctx["server"], ctx["monitor"]
    settle_s, window_s = timing(phase)
    default_conc = config.GPU_FAMILIES[ctx["fam"]]["concurrency"][wl["name"]]
    kv = ctx["calib"].get("vllm_kv_cache_tokens")
    # one load generator per stretch of runs at the same concurrency, so it only restarts when that changes
    for k, (conc, group) in enumerate(itertools.groupby(runs, lambda r: r.get("concurrency") or default_conc)):
        block = [r for r in group if run_id(wl, r) not in ctx["done"]]
        if not block:
            continue
        # Worst case is every request at full length at once. Staggered, they're spread through
        # their lifecycle, so this is only a heads-up. Preemptions and peak KV usage decide.
        worst = conc * (wl["input_len"] + wl["output_len"])
        if kv and worst > kv:
            typical = conc * (wl["input_len"] + wl["output_len"] / 2)
            monitor.warn(f"{wl['name']} at concurrency {conc}: {worst:,} tokens if every request were at full "
                         f"length at once, vs vLLM's {kv:,}-token KV cache. Staggered they average ~{typical:,.0f}, "
                         "so going by observed preemptions and peak KV usage.")
        await asyncio.to_thread(gpu.set_cap, block[0]["cap_w"])
        lg = LoadGen(server.base, config.MODEL, wl["input_len"], wl["output_len"], conc, ctx["vocab"],
                     ctx["run_dir"] / f"{wl['name']}_requests.csv", f"{ctx['session']}/c{conc}", seed=k)
        print(f"[{wl['name']}] starting {conc} workers, waiting for them to spread out", flush=True)
        monitor.write("staggering load", current=wl["name"], concurrency=conc)
        await lg.start()
        try:
            try:
                await lg.wait_staggered(config.STAGGER_TIMEOUT_S, config.STALL_S)
            except RuntimeError as e:
                await asyncio.to_thread(server.check)
                raise SweepAbort(f"{wl['name']}: {e}")
            await sweep_runs(ctx, wl, block, lambda cap, tag: measure_inference(
                ctx, lg, wl, cap, tag, settle_s, window_s))
        finally:
            await lg.stop()


# ---------- training ----------

def train_cmd(batch, *extra):
    # the TRAIN this session pinned in the manifest, so an edit to config.py mid-session can't leak in
    return [sys.executable, str(ROOT / "train_bench.py"), "--batch", str(batch),
            "--train-config", json.dumps(config.TRAIN), *extra]


def run_train_probe(gpu, batch, log_path):
    r = subprocess.run(train_cmd(batch, "--probe-steps", "10"), capture_output=True, text=True, timeout=1800,
                       cwd=ROOT, env=child_env(gpu.index))
    with open(log_path, "a") as f:  # the whole output, the terminal only shows the tail
        f.write(f"\n=== {time.ctime()} calibration probe, batch {batch}, exit {r.returncode}\n{r.stdout}{r.stderr}")
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        raise RunFailed(f"train_bench failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return json.loads(line[len("RESULT "):])


def mfu(res, fam):
    peak = config.GPU_FAMILIES[fam]["peak_bf16_tflops"] * 1e12
    return 6 * res["n_params_nonembed"] * res["tokens_per_s"] / peak


def calibrate_training(gpu, fam, monitor, log_path):
    gpu.set_cap(gpu.default_w)
    skipped = []  # why each batch was passed over, kept in events.jsonl and calibration.json

    def skip(b, why, **extra):
        skipped.append({"batch": b, "why": why, **extra})
        monitor.event("calibration", f"training batch {b}: {why}", **skipped[-1])

    for b in config.GPU_FAMILIES[fam]["train_batch_candidates"]:
        if _signals:  # Ctrl-C or kill: don't go on to the next candidate
            raise KeyboardInterrupt(f"signal {_signals[0]}")
        try:
            res = run_train_probe(gpu, b, log_path)
        except (RunFailed, subprocess.TimeoutExpired) as e:
            print(f"batch {b}: failed ({str(e)[-300:]})")
            skip(b, "failed", error=str(e)[-2000:])
            continue
        if res.get("oom"):
            print(f"batch {b}: OOM")
            skip(b, "oom")
            continue
        frac = res["peak_mem_bytes"] / gpu.mem_total
        if frac > 0.85:
            print(f"batch {b}: uses {frac:.0%} of memory, too tight")
            skip(b, "too tight", mem_frac=frac)
            continue
        u = mfu(res, fam)
        print(f"calibrated training: batch {b} ({frac:.0%} memory), {res['tokens_per_s']:.0f} tok/s, "
              f"MFU ~{u:.0%}", flush=True)
        if not 0.15 <= u <= 0.85:
            monitor.warn(f"training MFU is {u:.0%}, which is weird (expected 15-85%), worth a look")
        return {"batch": b, "mem_frac": frac, "mfu_at_default": u, "skipped": skipped}
    crashes = [x for x in skipped if x["why"] == "failed"]
    if crashes:  # not just memory, so don't send anyone off shrinking the batch
        raise SweepAbort("no training batch size worked (" + ", ".join(f"{x['batch']}: {x['why']}" for x in skipped)
                         + f"). Last crash, more in train_bench.log:\n{crashes[-1]['error']}")
    raise SweepAbort("none of the training batch sizes fit, add smaller ones in config.py")


class Trainer:
    """train_bench.py in persistent mode: compiles once, keeps training while we change caps."""

    def __init__(self, gpu, batch, log_path):
        self.gpu, self.batch, self.log_path, self.proc = gpu, batch, log_path, None

    async def start(self):
        self.log = open(self.log_path, "a")
        self.proc = subprocess.Popen(
            train_cmd(self.batch, "--nvml-index", str(self.gpu.index)),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
            cwd=ROOT, env=child_env(self.gpu.index))
        line = await self._read_until(("READY", "RESULT "), timeout=1800)
        if not line.startswith("READY"):
            raise SweepAbort(f"train_bench didn't start: {line}")

    async def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=60)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait()
        if getattr(self, "log", None):
            self.log.close()

    async def measure(self, seconds):
        if self.proc.poll() is not None:
            raise SweepAbort(f"train_bench exited (code {self.proc.returncode}), see {self.log_path}")
        self.proc.stdin.write(f"MEASURE {seconds}\n")
        self.proc.stdin.flush()
        line = await self._read_until(("RESULT ",), timeout=3 * seconds + 300)
        if not line.startswith("RESULT "):
            raise SweepAbort(f"train_bench stopped mid-measurement: {line}")
        res = json.loads(line[len("RESULT "):])
        if res.get("oom"):
            raise SweepAbort("train_bench ran out of memory mid-sweep")
        if not math.isfinite(res.get("final_loss", float("nan"))):
            raise RunFailed(f"training loss is not finite: {res.get('final_loss')}")
        return res

    async def _read_until(self, prefixes, timeout):
        async def read():
            while True:
                line = await asyncio.to_thread(self.proc.stdout.readline)
                if not line:
                    return f"(train_bench exited, code {self.proc.poll()}, see {self.log_path})"
                if line.startswith(prefixes):
                    return line.strip()
        try:
            return await asyncio.wait_for(read(), timeout)
        except asyncio.TimeoutError:
            self.proc.kill()  # also unblocks the reader thread
            raise SweepAbort(f"train_bench gave no answer within {timeout:.0f} s")


async def measure_training(ctx, trainer, cap, tag, settle_min_s, window_s):
    from gpu import Sampler, window_stats
    gpu = ctx["gpu"]
    s = None
    try:
        with Sampler(gpu, config.SAMPLE_PERIOD_S) as s:
            await asyncio.to_thread(gpu.set_cap, cap)
            t_cap = time.time()
            settle_s = await settle(s, settle_min_s)
            res = await trainer.measure(window_s)
            s.check()
        try:
            stats = window_stats(s.rows, res["t0"], res["t1"])  # t0/t1 are step boundaries from the trainer
        except RuntimeError as e:
            raise RunFailed(str(e))
    finally:
        if s is not None:  # every attempt keeps its raw telemetry, failed ones included
            save_samples(s.rows, ctx["run_dir"] / f"{tag}_samples.csv")
    if res.get("e0_mj") is not None and res.get("e1_mj") is not None:
        # the trainer read the energy counter itself at t0/t1, so use that over the samples
        dt = res["t1"] - res["t0"]
        joules = (res["e1_mj"] - res["e0_mj"]) / 1000
        stats.update(window_s=dt, energy_j=joules, avg_power_w=joules / dt)
    return dict(stats, t_cap=t_cap, cap_since=gpu.cap_since, prev_cap_w=gpu.prev_cap_w,
                t_start=res["t0"], t_end=res["t1"], settle_s=settle_s,
                tokens_per_s=res["tokens_per_s"], total_tokens_per_s=res["tokens_per_s"],
                train_batch=res["batch"], train_steps=res["steps"], step_time_s=res["step_time_s"],
                mfu=mfu(res, ctx["fam"]), final_loss=res["final_loss"])


async def run_training(ctx, wl, runs, phase):
    import manifest
    pending = [r for r in runs if run_id(wl, r) not in ctx["done"]]
    if not pending:
        return
    # train_bench.py starts from disk hours into a session, so make sure it's still the code we pinned
    if manifest.code_version() != ctx["base"]["code_version"]:
        raise SweepAbort("measurement code changed on disk during this session. Rerun so the manifest "
                         "check sees it.")
    settle_s, window_s = timing(phase)
    await asyncio.to_thread(ctx["gpu"].wait_until_free)
    log_path = ctx["out"] / "train_bench.log"
    if "training" not in ctx["calib"]:
        ctx["monitor"].write("calibrating", current="training")
        ctx["calib"]["training"] = await asyncio.to_thread(calibrate_training, ctx["gpu"], ctx["fam"],
                                                           ctx["monitor"], log_path)
        ctx["save_calib"]()
    await asyncio.to_thread(ctx["gpu"].set_cap, pending[0]["cap_w"])
    trainer = Trainer(ctx["gpu"], ctx["calib"]["training"]["batch"], log_path)
    ctx["monitor"].write("starting trainer", current="training")
    try:
        await trainer.start()
        await sweep_runs(ctx, wl, pending, lambda cap, tag: measure_training(
            ctx, trainer, cap, tag, settle_s, window_s))
    finally:
        await trainer.stop()


# ---------- the run loop ----------

def measure_idle(gpu, seconds=30):
    from gpu import Sampler, window_stats
    if gpu.compute_procs() or port_in_use(config.VLLM_HOST, config.VLLM_PORT):
        raise SweepAbort(f"something is still on the GPU or on port {config.VLLM_PORT} (old vllm serve?), so idle "
                         "power would be wrong. Check nvidia-smi.")
    gpu.set_cap(gpu.default_w)
    time.sleep(5)
    with Sampler(gpu, config.SAMPLE_PERIOD_S) as s:
        time.sleep(seconds)
    w = window_stats(s.rows, s.rows[0][0], s.rows[-1][0])
    print(f"idle power: {w['avg_power_w']:.0f} W")
    if w["avg_power_w"] > 0.3 * gpu.default_w:
        raise SweepAbort(f"idle power is {w['avg_power_w']:.0f} W, way too high. Is something else on the GPU?")
    return w


def write_row(ctx, wl, r, m, rid, tag, attempt, quality, notes):
    append_result(ctx["results"], dict(
        ctx["base"], **m, measurement_id=tag, run_id=rid, attempt=attempt, quality=quality,
        quality_notes="; ".join(notes), workload=wl["name"], kind=wl["kind"], block=r["block"], seq=r["seq"],
        role=r["role"], cap_w=r["cap_w"]))


async def sweep_runs(ctx, wl, runs, measure):
    from gpu import NVMLError
    monitor = ctx["monitor"]
    for r in runs:
        rid = run_id(wl, r)
        if rid in ctx["done"]:
            continue
        print(f"[{wl['name']}] block {r['block']} cap {r['cap_w']} W ({r['role']})", flush=True)
        monitor.write("running", current=rid)
        for attempt in (1, 2):
            tag = f"{rid}.{ctx['session']}.{attempt}"  # names this measurement's files and latency cohort
            m = None
            try:
                m = await measure(r["cap_w"], tag)
                problems = monitor.hard_problems(m, r["cap_w"])
            except (RunFailed, RuntimeError, subprocess.TimeoutExpired, NVMLError) as e:
                problems = [f"{type(e).__name__}: {e}" if isinstance(e, NVMLError) else str(e)]
            if problems:
                if m is not None:  # keep the raw numbers, marked rejected
                    write_row(ctx, wl, r, m, rid, tag, attempt, "rejected", problems)
                monitor.event("run_rejected" if m is not None else "run_failed", "; ".join(problems)[:4000],
                              run_id=rid, attempt=attempt, measurement_id=tag)
                if attempt == 2:
                    raise SweepAbort(f"{rid} failed twice: {'; '.join(problems)}")
                monitor.warn(f"{rid} failed, trying once more: {'; '.join(problems)[:500]}", run_id=rid)
                continue
            quality, notes, abort = monitor.assess(wl["name"], r, m, rid, tag, ctx["gpu"].default_w)
            write_row(ctx, wl, r, m, rid, tag, attempt, quality, notes)
            if quality == "rejected":
                monitor.event("run_rejected", "; ".join(notes), run_id=rid, attempt=attempt, measurement_id=tag)
            else:
                ctx["done"].add(rid)
            print(f"    {m['tokens_per_s']:.0f} tok/s at {m['avg_power_w']:.0f} W, max {m['temp_c_max']} C "
                  f"[{quality}]", flush=True)
            if abort:
                raise SweepAbort(abort)
            break


# ---------- smoke test ----------

def smoke_verdict(results, default_w, low):
    """The low cap should actually lower power, and aggregate.py should be able to read the results."""
    import aggregate
    rows = [r for r in read_rows(results) if r["quality"] != "rejected"]
    problems = []
    for wl in dict.fromkeys(r["workload"] for r in rows):
        g = [r for r in rows if r["workload"] == wl]
        d = next((r for r in g if int(r["cap_w"]) == default_w), None)
        lo = next((r for r in g if int(r["cap_w"]) == low), None)
        if d is None or lo is None:
            problems.append(f"{wl}: no accepted run at {default_w} W and {low} W")
            continue
        dp, dt, lp, lt = (float(d["avg_power_w"]), float(d["tokens_per_s"]),
                          float(lo["avg_power_w"]), float(lo["tokens_per_s"]))
        print(f"  {wl:<14} default: {dt:8.0f} tok/s at {dp:5.0f} W | {low} W cap: {lt:8.0f} tok/s at {lp:5.0f} W")
        if dp > low * 1.05 and lp > dp * 0.95:
            problems.append(f"{wl}: power didn't drop under the {low} W cap")
        if lt > dt * 1.05:
            problems.append(f"{wl}: throughput went UP >5% with a lower cap, something's off")
    aggregate.collect([results.parent])  # blows up here if the outputs are malformed
    return problems


# ---------- main ----------

async def run_all(ctx, workloads, runs, phase, fam):
    gpu, monitor, calib = ctx["gpu"], ctx["monitor"], ctx["calib"]
    watch = ctx["signal_watch"] = asyncio.create_task(_watch_signals())  # keep a reference, tasks are held weakly
    # it may fire again while shutdown is already under way; that's expected, so don't log it at exit
    watch.add_done_callback(lambda t: t.cancelled() or t.exception())
    if "idle_w" not in calib:
        w = await asyncio.to_thread(measure_idle, gpu)
        calib.update(idle_w=w["avg_power_w"], idle_temp_c=w["temp_c_mean"], idle_session=ctx["session"],
                     idle_time=time.time())
        ctx["save_calib"]()

    inference = [w for w in workloads if w["kind"] == "inference"
                 and any(run_id(w, r) not in ctx["done"] for r in runs)]
    if inference:
        monitor.write("starting vLLM")
        with VllmServer(gpu, fam, ctx["out"] / "vllm_server.log") as server:
            ctx["server"] = server
            kv = server.kv_capacity()
            if kv:
                calib["vllm_kv_cache_tokens"] = kv
            att = server.attention_backend()
            calib["vllm_attention_backend"] = att["backend"]
            calib["vllm_attention"] = att
            ctx["save_calib"]()
            if att["flashinfer_missing"]:
                monitor.warn(f"vLLM wanted FlashInfer on this GPU but couldn't import it, so it's using {att['backend']}. "
                             "Fine if that's a deliberate choice (see attention_backend in config.py), otherwise "
                             "FLASHINFER=1 bash setup.sh.")
            if att["artifactory_unreachable"]:
                monitor.warn("vLLM couldn't reach NVIDIA's artifactory, so FlashInfer's kernel choice may differ from "
                             "other sessions. See vllm_env in config.py.")
            ctx["vocab"] = await asyncio.to_thread(vocab_size)
            for wl in inference:
                await run_inference(ctx, wl, runs, phase)
            if server.metric_missing:
                monitor.warn("vLLM's preemption counter isn't on /metrics, counted 'preempt' lines in the "
                             "server log instead. Worth a glance at vllm_server.log.")

    for wl in (w for w in workloads if w["kind"] == "training"):
        await run_training(ctx, wl, runs, phase)


def run_async(coro):
    """uvloop if it's installed: a faster event loop gives the load generator more headroom."""
    try:
        import uvloop
    except ImportError:
        return asyncio.run(coro)
    if hasattr(uvloop, "run"):
        return uvloop.run(coro)
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    return asyncio.run(coro)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=PHASES, default="screen")
    ap.add_argument("--experiment", help="results subfolder name, defaults to the phase name")
    ap.add_argument("--caps", help="comma-separated candidate caps for --phase confirm")
    ap.add_argument("--seed", type=int, default=0, help="run order seed for --phase confirm")
    ap.add_argument("--only", nargs="*", help="workload names to run")
    ap.add_argument("--smoke", action="store_true", help="short end-to-end check at 2 caps")
    ap.add_argument("--allow-code-change", action="store_true",
                    help="resume even though the measurement code changed. Rows record the code version they ran")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpu", choices=list(config.GPU_FAMILIES), help="family for --dry-run")
    args = ap.parse_args()
    workloads = [w for w in config.WORKLOADS if not args.only or w["name"] in args.only]
    if args.only and len(workloads) != len(args.only):
        raise SystemExit(f"unknown workload in --only {args.only}")
    if args.phase == "concurrency" and not args.smoke:
        skipped = [w["name"] for w in workloads if w["kind"] != "inference"]
        workloads = [w for w in workloads if w["kind"] == "inference"]
        if skipped:
            print(f"--phase concurrency only sweeps inference, skipping {', '.join(skipped)}")
        if not workloads:
            raise SystemExit("nothing to run: --phase concurrency only applies to inference workloads")
    confirm_caps = [int(c) for c in args.caps.split(",")] if args.caps else None
    if args.phase == "confirm" and not confirm_caps:
        raise SystemExit("--phase confirm needs --caps, e.g. --caps 450,500,550")
    if args.phase != "confirm" and (args.caps or args.seed):
        raise SystemExit("--caps and --seed only apply to --phase confirm. Did you forget --phase confirm?")

    if args.dry_run:
        dry_run(args.gpu or "h100", args.phase, workloads, confirm_caps, args.seed)
        return

    from gpu import Gpu, as_root, box_lock
    lock = box_lock(ROOT / "results")  # held until exit: one sweep.py/preflight.py per box
    # offline before anything imports huggingface_hub (it reads this once, at import): from here on only
    # the snapshot setup.sh cached gets used, by this process and by vLLM
    os.environ["HF_HUB_OFFLINE"] = "1"
    importlib.import_module("loadgen")  # load the measurement code now, so what runs is what the manifest hashed
    import manifest
    rev = manifest.model_revision()
    if not rev:
        raise SystemExit(f"{config.MODEL} isn't in the HF cache (HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')}). "
                         "Run setup.sh, or export the HF_HOME you ran it with.")
    gpu = Gpu(0)
    fam = config.gpu_family(gpu.name)
    expected = config.GPU_FAMILIES[fam]["expected_default_w"]
    if abs(gpu.default_w - expected) > 0.05 * expected:
        raise SystemExit(f"{gpu.name} has a {gpu.default_w} W default limit, expected {expected} W. "
                         "Wrong SKU (PCIe/NVL)?")
    try:
        subprocess.run(as_root(["nvidia-smi", "-i", str(gpu.index), "-pm", "1"]),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    except subprocess.TimeoutExpired:
        raise SystemExit("nvidia-smi -pm 1 hung for 60 s. The driver looks stuck, check nvidia-smi.")

    phase = "smoke" if args.smoke else args.phase
    out = ROOT / "results" / fam / ("smoke" if args.smoke else (args.experiment or args.phase))
    if args.smoke:
        config.TARGET_RUN_S, config.SETTLE_S, config.SETTLE_MAX_S = 30, 10, 60
        shutil.rmtree(out, ignore_errors=True)  # start fresh every time
        low = max(gpu.min_w, int(round(0.6 * gpu.default_w, -1)))
        runs = [{"block": 0, "seq": 0, "cap_w": gpu.default_w, "role": "anchor"},
                {"block": 0, "seq": 1, "cap_w": low, "role": "sweep"}]
        print(f"SMOKE TEST: caps {gpu.default_w} W and {low} W, {config.TARGET_RUN_S} s windows")
    else:
        runs = make_plan(args.phase, fam, gpu.default_w, gpu.min_w, gpu.max_w, confirm_caps, args.seed)
        print(f"{gpu.name}: default {gpu.default_w} W, {args.phase}, caps {sorted({r['cap_w'] for r in runs})}")
    (out / "runs").mkdir(parents=True, exist_ok=True)

    extra = {"confirm_caps": confirm_caps, "seed": args.seed, "server_cmd": server_cmd(fam)}
    blocking, allowed = manifest.check_or_create(out / "manifest.json", manifest.build(gpu, fam, phase, runs, extra),
                                                 args.allow_code_change)
    if blocking:
        code_only = all(d.startswith("code_") for d in blocking)
        raise SystemExit(f"{out} was started under different conditions, so resuming would mix experiments:\n  "
                         + "\n  ".join(blocking) + "\nUse a new --experiment name (or put things back)."
                         + ("\nOnly measurement code changed. If mixing is OK, --allow-code-change resumes and "
                            "tags each row with its code version." if code_only else ""))
    session_id = manifest.new_session_id()
    manifest.write_session(out, session_id, phase, args)
    code_version = manifest.code_version()

    results = out / "results.csv"
    # runs measured between a good anchor and one that drifted get measured again
    suspect = {x for e in read_events(out / "events.jsonl") if e["kind"] == "runs_suspect"
               for x in e.get("measurement_ids", [])}
    prior = [r for r in read_rows(results) if r.get("measurement_id") not in suspect]
    done = {r["run_id"] for r in prior if r.get("quality") != "rejected"}
    todo = sum(1 for wl in workloads for r in runs if run_id(wl, r) not in done)
    monitor = Monitor(out, todo, phase, session_id, prior, gpu.min_w)
    monitor.event("session_start", " ".join(sys.argv), code_version=code_version)
    for d in allowed:
        monitor.warn(f"measurement code changed since this experiment started (--allow-code-change): {d}")
    monitor.write("starting")
    calib_path = out / "calibration.json"
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else {}
    ctx = {
        "gpu": gpu, "fam": fam, "out": out, "run_dir": out / "runs", "results": results, "done": done,
        "monitor": monitor, "calib": calib, "session": session_id, "lock": lock,
        "save_calib": lambda: write_atomic(calib_path, json.dumps(calib, indent=2)),
        "base": {"experiment": out.name, "phase": phase, "session_id": session_id, "code_version": code_version,
                 "gpu": fam, "gpu_name": gpu.name, "gpu_uuid": gpu.uuid, "default_w": gpu.default_w},
    }

    # so Ctrl-C, `kill` or a dropped SSH session still stops vLLM and puts the cap back. Under nohup,
    # SIGHUP is already ignored and stays that way
    handled = [sig for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
               if signal.getsignal(sig) is not signal.SIG_IGN]
    for sig in handled:
        signal.signal(sig, _interrupt)
    try:
        run_async(run_all(ctx, workloads, runs, phase, fam))
        if args.smoke:
            problems = smoke_verdict(results, gpu.default_w, runs[1]["cap_w"])
            for w in monitor.warnings:
                print("  warning: " + w)
            if problems:
                monitor.event("abort", "smoke test failed: " + "; ".join(problems))
                monitor.write("smoke FAILED", problems=problems)
                print("SMOKE TEST FAILED:\n  " + "\n  ".join(problems))
                sys.exit(1)
            monitor.event("finished", "smoke PASSED")
            monitor.write("smoke PASSED")
            print("SMOKE TEST PASSED" + (f" ({len(monitor.warnings)} warnings, read them above)"
                                         if monitor.warnings else ""))
        else:
            monitor.event("finished", f"{monitor.completed} runs this session")
            monitor.write("finished")
            print(f"Done! {len(monitor.warnings)} warnings, see {monitor.path}")
    except (SweepAbort, RunFailed) as e:
        monitor.event("abort", str(e)[:4000])
        monitor.write("ABORTED", reason=str(e))
        print(f"\nABORTED: {e}\nStopping here. Fix it and rerun, finished runs are kept.")
        sys.exit(2)
    except KeyboardInterrupt as e:
        monitor.event("interrupted", str(e) or "Ctrl-C")
        monitor.write("INTERRUPTED", reason=str(e) or "Ctrl-C")
        print("\nINTERRUPTED. Finished runs are kept, rerun the same command to resume.")
        sys.exit(130)
    except Exception as e:  # e.g. pynvml.NVMLError or OSError: still leave a record, and the traceback
        monitor.event("abort", f"crashed: {type(e).__name__}: {e}"[:4000])
        monitor.write("CRASHED", reason=f"{type(e).__name__}: {e}")
        raise
    finally:
        for sig in handled:
            signal.signal(sig, signal.default_int_handler if sig == signal.SIGINT else signal.SIG_DFL)
        try:
            gpu.set_cap(gpu.default_w)
            print(f"Restored default cap {gpu.default_w} W")
        except Exception as e:
            print(f"!! Couldn't restore the default cap ({e}). Run: sudo nvidia-smi -pl {gpu.default_w}")


if __name__ == "__main__":
    main()
