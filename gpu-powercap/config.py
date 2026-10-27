# Pretty much every knob lives here.
# Everything that affects a measurement is copied into each experiment's manifest.json, and
# sweep.py refuses to resume an experiment if any of it changed (see manifest.py).

MODEL = "Qwen/Qwen3-8B"  # Apache 2.0, no HF sign-up needed. The cached HF commit goes in the manifest
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.90
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000

TARGET_RUN_S = 120        # length of each timed window
SETTLE_S = 60             # load keeps running at the new cap at least this long before the window starts,
SETTLE_MAX_S = 240        # ...then until GPU temperature stops moving (or this long, whichever is first):
SETTLE_FLAT_C = 0.5       # the last two 20 s temperature means within this many degrees
PASSES = 2                # screening: high->low, then low->high, with default-cap anchors in each pass
SAMPLE_PERIOD_S = 0.1     # read power 10x a second
METRICS_PERIOD_S = 1.0    # read vLLM's /metrics (queue, running, KV usage) once a second
STAGGER_TIMEOUT_S = 1800  # how long the load generator gets to spread its workers out
STALL_S = 120             # give up on the load generator if no request has succeeded for this long
COHORT_TIMEOUT_S = 1200   # how long requests sent during a window get to finish (same cap, same load)

# Each run gets quality "ok", "warn" or "rejected" (see Monitor in sweep.py). Rejected runs stay
# in results.csv but get measured again on resume. Some problems also stop the sweep.
FULL_LOAD_RATIO = 0.90    # warn if a load can't get to 90% of the default cap
CLIENT_CPU_MAX = 0.85     # the load generator is one asyncio thread, so ~1 core means it's maxed out
CLIENT_LAG_MAX_S = 0.25   # event loop waking up this late means the client is falling behind
SERVER_CPU_WARN = 0.90    # vLLM's API server is one Python process too; near 1 core it's the bottleneck
INFLIGHT_DEFICIT_WARN = 0.03  # running + waiting more than 3% below concurrency: requests are slow to turn around
WAITING_FRAC_WARN = 0.05  # vLLM had requests queued in more than 5% of the window's samples
KV_USAGE_WARN = 0.95      # peak KV cache usage above this leaves little headroom before preemption
MAX_SAMPLE_GAP_S = 2.0    # longest allowed hole in the power samples, including the window's two ends
TEMP_DRIFT_WARN_C = 3     # GPU temperature moving this much during a window means it wasn't settled
CAP_OVERSHOOT = 1.03      # power more than 3% over the cap means the cap isn't working
ANCHOR_WARN = 0.03        # default-cap reruns drifting from the first one
ANCHOR_ABORT = 0.08
INVALID_ABORT_FRAC = 0.25 # stop if this many runs of a workload get thermal throttled
MIN_FREE_DISK_GB = 10

# 3 inference loads + training
WORKLOADS = [
    {"name": "batch-offline", "kind": "inference", "input_len": 1024, "output_len": 1024},
    {"name": "prefill-heavy", "kind": "inference", "input_len": 4096, "output_len": 128},
    {"name": "decode-heavy", "kind": "inference", "input_len": 256, "output_len": 2048},
    {"name": "training", "kind": "training"},
]

# ~1.5B Llama-ish model, random weights, random tokens
TRAIN = {
    "seq_len": 2048,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "warmup_steps": 30,
}

# `concurrency` is what the screening/confirm sweeps use. Fitting the KV cache is a constraint,
# not proof that throughput has plateaued, so pick these from a `--phase concurrency` run first.
# Qwen3-8B uses ~147 KB of KV per token (36 layers x 8 KV heads x 128 dim x 2 x 2 bytes);
# vLLM logs the real KV capacity at startup and it ends up in calibration.json.
# Caps outside what the GPU reports as min/max just get dropped.
GPU_FAMILIES = {
    "h100": {
        "match": "H100",
        "expected_default_w": 700,    # SXM. PCIe is 350 W and NVL is 400 W, we don't want those
        "caps": [700, 650, 600, 550, 500, 450, 400, 300, 200],
        "price_per_hr": 4.41,         # your provider's hourly price (DigitalOcean 1x H100); only for --dry-run
        "attention_backend": None,    # None lets vLLM pick (FlashAttention 3 on Hopper). Recorded either way
        "vllm_env": {},               # extra environment for vllm serve, pinned in the manifest
        "max_num_seqs": 256,
        "concurrency": {"batch-offline": 192, "prefill-heavy": 64, "decode-heavy": 256},  # from --phase concurrency
        "concurrency_sweep": [32, 64, 128, 192, 256],
        "train_batch_candidates": [16, 12, 8, 4],
        "peak_bf16_tflops": 989,      # dense, from the spec sheet. Only for the MFU sanity check
    },
    "b200": {
        "match": "B200",
        "expected_default_w": 1000,   # HGX B200
        # 50 W steps through 500-750 W: the H100's peak and cliff were at 57-71% of its default
        "caps": [1000, 900, 800, 750, 700, 650, 600, 550, 500, 400, 300, 200],
        "price_per_hr": 8.44,         # Verda 1x B200; only for --dry-run
        # vLLM 0.11.0 wants FlashInfer on Blackwell and falls back to FlashAttention 2 if it isn't
        # installed (FLASHINFER=1 bash setup.sh installs it). Set "FLASHINFER" or "FLASH_ATTN" here
        # to make the choice explicit; the backend vLLM actually used goes in calibration.json.
        "attention_backend": "FLASHINFER",
        # With FlashInfer, vLLM 0.11.0 decides per server start whether to use its TRT-LLM kernels by
        # pinging NVIDIA's artifactory (5 s timeout). This makes that the same every start (TRT-LLM kernels
        # for small batches, downloaded once and cached). "VLLM_USE_TRTLLM_ATTENTION": "0" instead never
        # uses them and needs no network. Does nothing without FlashInfer.
        "vllm_env": {"VLLM_HAS_FLASHINFER_CUBIN": "1"},
        "max_num_seqs": 512,
        "concurrency": {"batch-offline": 256, "prefill-heavy": 128, "decode-heavy": 256},  # from --phase concurrency
        "concurrency_sweep": [64, 128, 256, 384, 512],
        "train_batch_candidates": [32, 24, 16, 8],
        "peak_bf16_tflops": 2250,
    },
}

# --phase concurrency: every value in concurrency_sweep, at the default cap and these
# fractions of it, with shorter windows. Enough to see where throughput flattens out and
# whether that moves at lower caps.
CONC_SWEEP_CAP_FRACS = [1.0, 0.75, 0.55]
CONC_SWEEP_RUN_S = 60
CONC_SWEEP_SETTLE_S = 30

# --phase confirm: complete blocks. Each block runs the default cap and every --caps value once, so
# later analysis can compare against the default within each block. The orders come from a Williams
# design; with 1 or 3 candidates (2 or 4 caps counting the default) and CONFIRM_BLOCKS = 4 it's
# complete, so every cap follows every other cap equally often (balances carryover).
CONFIRM_BLOCKS = 4

# Settings that go in the manifest and must match to resume
MEASUREMENT_KEYS = [
    "MODEL", "MAX_MODEL_LEN", "GPU_MEMORY_UTILIZATION", "TARGET_RUN_S", "SETTLE_S", "SETTLE_MAX_S",
    "SETTLE_FLAT_C", "PASSES", "SAMPLE_PERIOD_S", "METRICS_PERIOD_S", "WORKLOADS", "TRAIN",
    "CONC_SWEEP_CAP_FRACS", "CONC_SWEEP_RUN_S", "CONC_SWEEP_SETTLE_S", "CONFIRM_BLOCKS",
]
FAMILY_KEYS = ["max_num_seqs", "concurrency", "concurrency_sweep", "train_batch_candidates", "attention_backend",
               "vllm_env"]


def gpu_family(name):
    for fam, spec in GPU_FAMILIES.items():
        if spec["match"] in name:
            return fam
    raise SystemExit(f"Unknown GPU '{name}'. Add it to GPU_FAMILIES in config.py.")


def caps_for(fam, min_w, max_w, default_w):
    caps = [c for c in GPU_FAMILIES[fam]["caps"] if min_w <= c <= max_w]
    if default_w not in caps:
        caps.append(default_w)
    if min_w not in caps:
        caps.append(min_w)
    return sorted(set(caps), reverse=True)
