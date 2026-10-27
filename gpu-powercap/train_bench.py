"""Training throughput benchmark. ~1.5B Llama-ish model with random weights and random tokens.

The loss is meaningless but a dense model does the same math on random tokens as on real text,
so throughput is legit. The optimizer runs with lr=0, so every step is the identical computation
(same kernels, FLOPs and bytes, and the weights never drift), and a run early in the sweep does
exactly the same work as one hours later. This is a single-GPU compute probe: it says nothing
about multi-GPU scaling efficiency or time to convergence.

    python train_bench.py --batch 16 --probe-steps 10   # calibration: run 10 steps, print RESULT, exit
    python train_bench.py --batch 16                    # persistent: used by sweep.py

In persistent mode it compiles once, prints READY, then trains nonstop so the GPU stays loaded
while sweep.py changes caps. Each "MEASURE <seconds>" line on stdin times whole steps for at
least that long and prints a RESULT line with JSON. t0/t1 are wall-clock times at step
boundaries. With --nvml-index it also reads the GPU's energy counter at those same two instants,
so energy and tokens cover exactly the same interval. "QUIT" or EOF exits.
"""
import argparse
import json
import queue
import sys
import threading
import time

import torch
from transformers import LlamaConfig, LlamaForCausalLM

import config


def emit(obj):
    print("RESULT " + json.dumps(obj), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--probe-steps", type=int, help="time this many steps after warmup, then exit")
    p.add_argument("--warmup-steps", type=int)
    p.add_argument("--nvml-index", type=int, help="NVML index of the GPU, to read its energy counter")
    p.add_argument("--train-config", help="TRAIN settings as JSON (sweep.py passes the ones it pinned in "
                                          "the manifest), instead of reading config.py")
    args = p.parse_args()
    t = json.loads(args.train_config) if args.train_config else config.TRAIN
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else t["warmup_steps"]

    energy_mj = None
    if args.nvml_index is not None:
        import pynvml as nv
        nv.nvmlInit()
        h = nv.nvmlDeviceGetHandleByIndex(args.nvml_index)
        energy_mj = lambda: nv.nvmlDeviceGetTotalEnergyConsumption(h)  # noqa: E731

    seq = t["seq_len"]
    torch.manual_seed(0)
    cfg = LlamaConfig(
        hidden_size=t["hidden_size"], intermediate_size=t["intermediate_size"],
        num_hidden_layers=t["num_hidden_layers"], num_attention_heads=t["num_attention_heads"],
        num_key_value_heads=t["num_key_value_heads"], vocab_size=t["vocab_size"],
        max_position_embeddings=seq,
    )
    model = LlamaForCausalLM(cfg).to("cuda", dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    n_embed = model.get_input_embeddings().weight.numel()  # a lookup, not a matmul, so no 6N FLOPs
    model = torch.compile(model)
    # lr=0: the fused AdamW kernel still does all its work, but the weights stay put, so every step
    # is the same computation (a fixed batch at a real lr would get memorized and keep changing)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0, fused=True)
    x = torch.randint(0, cfg.vocab_size, (args.batch, seq), device="cuda")

    def step():
        loss = model(input_ids=x, labels=x).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    def timed(n_steps=None, seconds=None):
        torch.cuda.synchronize()
        t0, n = time.time(), 0
        e0 = energy_mj() if energy_mj else None
        while (n < n_steps) if n_steps else (n == 0 or time.time() - t0 < seconds):
            loss = step()
            torch.cuda.synchronize()
            n += 1
        t1 = time.time()
        e1 = energy_mj() if energy_mj else None
        tokens = n * args.batch * seq
        return {"t0": t0, "t1": t1, "e0_mj": e0, "e1_mj": e1, "steps": n, "batch": args.batch, "seq_len": seq,
                "n_params": n_params, "n_params_nonembed": n_params - n_embed,
                "tokens_per_s": tokens / (t1 - t0), "step_time_s": (t1 - t0) / n,
                "peak_mem_bytes": torch.cuda.max_memory_allocated(),
                "final_loss": loss.item()}  # after t1 so the sync doesn't count

    try:
        for _ in range(warmup_steps):  # torch.compile happens here, not timed
            step()
        torch.cuda.synchronize()
        if args.probe_steps:
            emit(timed(n_steps=args.probe_steps))
            return

        cmds = queue.Queue()
        threading.Thread(target=lambda: [cmds.put(l.strip()) for l in sys.stdin] + [cmds.put("QUIT")],
                         daemon=True).start()
        print("READY", flush=True)
        while True:
            try:
                cmd = cmds.get_nowait()
            except queue.Empty:
                step()  # keep the GPU busy between measurements
                torch.cuda.synchronize()
                continue
            if cmd == "QUIT":
                return
            if cmd.startswith("MEASURE "):
                emit(timed(seconds=float(cmd.split()[1])))
    except torch.cuda.OutOfMemoryError:
        emit({"oom": True})
        sys.exit(3)


if __name__ == "__main__":
    main()
