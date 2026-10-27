"""Closed-loop load generator for the vLLM server, replacing `vllm bench serve`.

It keeps `concurrency` requests in flight the whole time, across cap changes, so there's no
ramp-up, no drain, no cooling between runs, and no guessing where the benchmark started.
sweep.py calls mark() at the start and end of each timed window and reads the energy counter
at the same moment, so tokens and joules cover exactly the same interval.

Latency comes from a cohort: every request sent during the window, followed until it finishes
while the cap and the background load stay the same. Membership depends only on when a request
was sent, not how long it took, so slow requests can't drop out of the percentiles.

Every request ever sent is written to the request log (sent, first token, end, tokens, and
whether it finished, errored, or was cancelled when the load stopped), cohort tagged.

Prompts are random token IDs sent straight to /v1/completions, so input lengths are exact
(no detokenize/retokenize drift). Every worker's first request has a shortened output so the
workers end up spread evenly through the request lifecycle. Otherwise all requests (same
length) would prefill together and finish together, and a fixed window would catch a
different mix of prefill and decode depending on where it started.
"""
import asyncio
import csv
import random
import re
import time

import aiohttp

_COMPLETION = re.compile(rb'"completion_tokens":\s*(\d+)')
_PROMPT = re.compile(rb'"prompt_tokens":\s*(\d+)')

LOG_FIELDS = ["session", "req", "worker", "cohort", "t_send", "t_first_token", "t_end",
              "prompt_tokens", "output_tokens", "max_tokens", "status", "error"]


class Request:
    __slots__ = ("id", "worker", "max_tokens", "cohort", "t_send", "t_first", "t_end", "n_prompt", "n_out",
                 "status", "error")

    def __init__(self, id, worker, max_tokens, cohort):
        self.id, self.worker, self.max_tokens, self.cohort = id, worker, max_tokens, cohort
        self.t_send, self.t_first, self.t_end = time.time(), None, None
        self.n_prompt = self.n_out = 0
        self.status, self.error = "running", ""


class Cohort:
    """The requests sent while it was open. `done` fires once it's closed and they've all ended."""

    def __init__(self, tag):
        self.tag, self.open, self.members, self.pending = tag, True, [], 0
        self.done = asyncio.Event()

    def _ended(self):
        self.pending -= 1
        if not self.open and not self.pending:
            self.done.set()

    def close(self):
        if self.open:
            self.open = False
            if not self.pending:
                self.done.set()


class LoadGen:
    def __init__(self, base_url, model, input_len, output_len, concurrency, vocab_size, log_path, session,
                 seed=0):
        self.url = base_url + "/v1/completions"
        self.model, self.input_len, self.output_len = model, input_len, output_len
        self.conc, self.vocab = concurrency, vocab_size
        self.log_path, self.session = log_path, session
        self.rng = random.Random(seed)
        # running totals, only ever read as differences between two mark() calls
        self.out_tokens = self.prompt_tokens = self.completed = self.errors = 0
        self.last_error = None
        self.last_ok = time.time()  # when a request last finished cleanly
        self.staggered = 0       # workers that finished their shortened first request
        self.cohort = None
        self.max_lag = 0.0
        self._next_id = 0
        self._tasks = []

    # ---- lifecycle ----

    async def start(self):
        new = not self.log_path.exists()
        self._log_file = open(self.log_path, "a", newline="")
        self._log = csv.writer(self._log_file)
        if new:
            self._log.writerow(LOG_FIELDS)
        self.last_ok = time.time()
        self.http = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0),  # default limit is 100 connections
            timeout=aiohttp.ClientTimeout(total=None, sock_read=600))
        self._tasks = [asyncio.create_task(self._worker(i)) for i in range(self.conc)]
        self._tasks.append(asyncio.create_task(self._lag_probe()))

    async def stop(self):
        """Cancels whatever is in flight. Those requests go in the log as cancelled."""
        self.close_cohort()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.http.close()
        self._log_file.close()

    def stalled(self, after_s):
        """Requests are failing and none has finished cleanly in after_s seconds (dead server, bad request)."""
        return self.errors > 0 and time.time() - self.last_ok > after_s

    async def wait_staggered(self, timeout, stall_s):
        """Block until every worker is past its first (shortened) request."""
        deadline = time.time() + timeout
        while self.staggered < self.conc:
            if time.time() > deadline:
                raise RuntimeError(f"only {self.staggered}/{self.conc} workers got going in {timeout:.0f} s "
                                   f"({self.errors} errors, last: {self.last_error})")
            if self.stalled(stall_s):
                raise RuntimeError(f"no request has succeeded for {stall_s:.0f} s ({self.errors} errors, "
                                   f"last: {self.last_error})")
            await asyncio.sleep(1)

    # ---- measurement ----

    def mark(self):
        """Snapshot of the running totals. Diff two of these to get a window."""
        self._log_file.flush()
        lag, self.max_lag = self.max_lag, 0.0
        return {"t": time.time(), "out": self.out_tokens, "prompt": self.prompt_tokens,
                "completed": self.completed, "errors": self.errors, "cpu": time.process_time(),
                "max_lag_s": lag}

    def open_cohort(self, tag):
        self.close_cohort()
        self.cohort = Cohort(tag)
        return self.cohort

    def close_cohort(self):
        if self.cohort:
            self.cohort.close()
        self.cohort = None

    # ---- internals ----

    def _prompt(self):
        return self.rng.choices(range(self.vocab), k=self.input_len)

    async def _worker(self, i):
        # first request ends at (i+1)/conc of the full length, so the workers' phases spread out
        max_tokens = max(1, round(self.output_len * (i + 1) / self.conc))
        staggered = False
        while True:
            ok = await self._request(i, max_tokens)
            if ok and not staggered:
                staggered = True
                self.staggered += 1
                max_tokens = self.output_len
            if not ok:
                await asyncio.sleep(1)  # don't spin if the server is unhappy

    async def _request(self, worker, max_tokens):
        prompt = self._prompt()
        cohort = self.cohort
        r = Request(self._next_id, worker, max_tokens, cohort)
        self._next_id += 1
        if cohort:
            cohort.members.append(r)
            cohort.pending += 1
        # Sampling is pinned (plain sampling at temperature 1, no top-k/top-p) so the work per token doesn't
        # depend on the model's generation_config defaults or on whether FlashInfer's sampler is installed
        body = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "min_tokens": max_tokens,
                "ignore_eos": True, "temperature": 1.0, "top_k": 0, "top_p": 1.0, "stream": True,
                "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
        try:
            async with self.http.post(self.url, json=body) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
                buf = b""
                async for data in resp.content.iter_any():
                    buf += data
                    events, sep, buf = buf.rpartition(b"\n\n")  # keep any half-received event in buf
                    if not sep:
                        continue
                    # usage in each chunk is cumulative, so the last one in this read is all we need
                    m = None
                    for m in _COMPLETION.finditer(events):
                        pass
                    if m is None:
                        continue
                    n = int(m.group(1))
                    if n > r.n_out:
                        if r.t_first is None:
                            r.t_first = time.time()
                            p = _PROMPT.search(events)
                            r.n_prompt = int(p.group(1)) if p else len(prompt)
                            self.prompt_tokens += r.n_prompt
                        self.out_tokens += n - r.n_out
                        r.n_out = n
            if r.n_out != max_tokens:
                raise RuntimeError(f"got {r.n_out} output tokens, asked for {max_tokens}")
            r.status = "ok"
            self.completed += 1
            self.last_ok = time.time()
        except asyncio.CancelledError:
            r.status = "cancelled"
            raise
        except Exception as e:
            r.status, r.error = "error", f"{type(e).__name__}: {e}"[:300]
            self.errors += 1
            self.last_error = r.error
        finally:
            r.t_end = time.time()
            self._write(r)
            if cohort:
                cohort._ended()
        return r.status == "ok"

    def _write(self, r):
        self._log.writerow([self.session, r.id, r.worker, r.cohort.tag if r.cohort else "",
                            f"{r.t_send:.6f}", f"{r.t_first:.6f}" if r.t_first else "", f"{r.t_end:.6f}",
                            r.n_prompt, r.n_out, r.max_tokens, r.status, r.error])

    async def _lag_probe(self, period=0.05):
        """How late the event loop wakes up. Big lag means the client, not the GPU, is the bottleneck."""
        while True:
            t = time.perf_counter()
            await asyncio.sleep(period)
            self.max_lag = max(self.max_lag, time.perf_counter() - t - period)


def percentile(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def latency_stats(cohort, output_len):
    """Percentiles over a finished cohort. Every member ended before this is called, so nothing is
    censored. Only full-length requests count, so a shortened stagger request can't skew TPOT."""
    ok = [r for r in cohort.members if r.status == "ok" and r.max_tokens == output_len]
    ttft = [(r.t_first - r.t_send) * 1000 for r in ok]
    tpot = [(r.t_end - r.t_first) * 1000 / (r.n_out - 1) for r in ok if r.n_out > 1]
    e2e = [(r.t_end - r.t_send) * 1000 for r in ok]
    return {"ttft_p50_ms": percentile(ttft, 50), "ttft_p99_ms": percentile(ttft, 99),
            "tpot_p50_ms": percentile(tpot, 50), "tpot_p99_ms": percentile(tpot, 99),
            "e2e_p50_ms": percentile(e2e, 50), "e2e_p99_ms": percentile(e2e, 99),
            "cohort_n": len(cohort.members),
            "cohort_errors": sum(1 for r in cohort.members if r.status != "ok")}
