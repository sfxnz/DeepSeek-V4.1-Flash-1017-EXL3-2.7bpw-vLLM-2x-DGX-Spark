"""Engram prefill read-ahead: gather the next prefill chunk's rows while this chunk computes.

Every EngramDiskStager.stage() call larger than the native stage's limit (prefill and
mixed batches) hashes on the GPU, waits in hashes_ready.synchronize() until the previous
step has drained, then reads and dequantizes ~98k rows per table (8192 tokens x 12 local
hash heads) with the stock gather v2 while the GPU has nothing queued; only then is the
chunk's forward launched. 1017 [gv2-census]: 13-20 us/row, ~1.4 s per 8192-token chunk.

DSV41_ENGRAM_PREFILL_AHEAD=1 (default 0: nothing is patched, today's code path):

- Read-ahead. After a stage() call whose batch holds a request that is still prefilling
  after this step, the rows of its next chunk (positions [this chunk's end, + up to
  DSV41_ENGRAM_PREFILL_AHEAD_TOKENS)) are gathered on a background thread while this chunk
  runs on the GPU. A position's hash ids depend only on the tokens at positions p-3..p (the
  lookback window holds the same tokens), so hash_segment(), a numpy transcription of
  _hash_ids_kernel, computes them from the prompt in the V2 runner's all_token_ids (a UVA
  host buffer). The rows are read and dequantized by engram_native.c's eng_gather_bf16 on a
  thread pool (the GIL is released inside) into a pinned read-ahead buffer.
- Consume. The next stage() call still runs the stock GPU hash, DtoH and sync. Batch rows
  whose (request, position) the read-ahead covered have their GPU hash ids compared with
  the read-ahead's, int32 for int32, for every table. A row is a function of its hash id
  (same file bytes, same dequant), so equal ids give the rows the stock gather would read;
  those rows are copied host-to-device from the read-ahead buffer. Unequal ids disarm the
  lever (one LOG_DISARMED line) and those rows take the sync path.
- Sync path. Rows no read-ahead covers (a request's first chunk, decode rows of a mixed
  batch) are gathered by the same parallel native gather into rows_host and copied as the
  stock stage copies them.

Exactness: staged_rows[t] for t < n is the stock row of hash_host[t] on both paths; rows
>= n are not written (as in the stock stage). The first DSV41_ENGRAM_PREFILL_AHEAD_VERIFY
calls (default 4, minimum 1) also re-gather up to 512 rows per table with the stock gather
and compare raw bf16 bits; a mismatch refills the step from the stock gather and disarms.
Any error disarms: that step is redone by the stock stage, later steps take it.

Page cache: with _DONTNEED=1 (default) the prefill rows' reads leave the page cache as
they found it: pages a read brought in (its RWF_NOWAIT probe missed) are dropped with
posix_fadvise(DONTNEED) once the bytes are copied out; pages cached before stay cached
(but for a cached neighbour of a missed row that straddles two pages, 1 row in 16).
The stock stage leaves ~1 GB of Engram pages per 8192-token chunk and rank behind, and on
GB10 the driver allocates from MemFree (drop_page_cache.py). _DONTNEED=0 keeps them, as
today. Decode rows of a mixed batch are never dropped.

Knobs: _TOKENS read-ahead rows (default and cap = the stager's max_tokens, i.e.
--max-num-batched-tokens); _THREADS gather threads (default 8); _VERIFY (above);
_DONTNEED (above); _CPUS pins the gather threads (e.g. "0-4,10-14", the A725 cores;
empty = no pinning). Memory: 2 tables x TOKENS x 12 x 256 x 2 B pinned (96 MiB at 8192,
48 MiB at 4096, allocated at the first read-ahead) + TOKENS x 96 B of hash ids + THREADS x
0.8 MB scratch. DSV41_ENGRAM_CENSUS=1 prints a [prefill-ahead] line per call that carries
a prefill. With DSV41_SCHED_PREFILL_CAP=X (sched_prefill_cap.py: the scheduler gives each
prefilling request at most X tokens per step while it holds more than one request), a
batch of more than one request reads each request at most X tokens ahead, the chunk the
scheduler will give it; a read-ahead of up to TOKENS would make the next stage wait for
rows it does not use yet (and read them again a step later).

Hook points: wraps EngramDiskStager.__init__ / stage() (same signature) and
DeepseekV41ModelState.prepare_inputs (records the batch stage() is called for). Installed
before the native stage, so stage() calls of at most DSV41_ENGRAM_NATIVE_MAX_TOKENS tokens
never reach this wrapper while that stage is armed. Top-level imports are stdlib only.
"""

from __future__ import annotations

import ctypes
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dsv41: engram prefill ahead armed"
LOG_DISARMED = "dsv41: engram prefill ahead DISABLED ->"

DEFAULT_THREADS = 8
DEFAULT_VERIFY = 4
TOK_BLOCK = 256  # tokens per eng_gather_bf16 call: 3072 rows, ~0.8 MB of scratch
MIN_AHEAD = 64  # shorter continuations are left to their own stage() call
VERIFY_ROWS = 512  # token rows per verified call
FLAG_NOWAIT, FLAG_DONTNEED = 1, 2  # engram_native.c eng_gather_bf16 flags
NSTATS = 5  # engram_native.c ENG_NSTATS

_STATE = {"armed": True, "verify_left": DEFAULT_VERIFY, "engaged": False}


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return env.get("DSV41_ENGRAM_PREFILL_AHEAD", "0") == "1"


def cap_tokens(max_tokens: int, env=None) -> int:
    """Read-ahead buffer rows (tokens); unset/0 = the stager's max_tokens, never more."""
    env = os.environ if env is None else env
    n = int(env.get("DSV41_ENGRAM_PREFILL_AHEAD_TOKENS", "") or 0)
    if n < 0:
        raise ValueError(f"DSV41_ENGRAM_PREFILL_AHEAD_TOKENS={n} must be >= 0")
    return min(n, int(max_tokens)) if n else int(max_tokens)


def share_tokens(env=None) -> int:
    """The scheduler's per-request prefill cap with co-tenants (sched_prefill_cap.py; 0 = off)."""
    env = os.environ if env is None else env
    return max(0, int(env.get("DSV41_SCHED_PREFILL_CAP", "") or 0))


def threads(env=None) -> int:
    env = os.environ if env is None else env
    n = int(env.get("DSV41_ENGRAM_PREFILL_AHEAD_THREADS", "") or DEFAULT_THREADS)
    if n < 1:
        raise ValueError(f"DSV41_ENGRAM_PREFILL_AHEAD_THREADS={n} must be >= 1")
    return n


def verify_calls(env=None) -> int:
    env = os.environ if env is None else env
    return max(1, int(env.get("DSV41_ENGRAM_PREFILL_AHEAD_VERIFY", "") or DEFAULT_VERIFY))


def dontneed(env=None) -> bool:
    env = os.environ if env is None else env
    return (env.get("DSV41_ENGRAM_PREFILL_AHEAD_DONTNEED", "") or "1") == "1"


def cpus(env=None) -> set | None:
    """CPU set for the gather threads ("0-4,10-14"); None = no pinning."""
    env = os.environ if env is None else env
    spec = (env.get("DSV41_ENGRAM_PREFILL_AHEAD_CPUS", "") or "").strip()
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.strip().partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    if not out:
        raise ValueError(f"DSV41_ENGRAM_PREFILL_AHEAD_CPUS={spec!r} names no CPU")
    return out


def _pin(cpu_set) -> None:
    """ThreadPoolExecutor initializer: this thread runs on cpu_set only."""
    if cpu_set:
        os.sched_setaffinity(0, cpu_set)


def census_on(env=None) -> bool:
    env = os.environ if env is None else env
    return env.get("DSV41_ENGRAM_CENSUS", "0") == "1"


def _disarm(exc: BaseException) -> None:
    if _STATE["armed"]:
        _STATE["armed"] = False
        print("dsv41: engram prefill ahead DISABLED -> stock stage: %r" % (exc,), flush=True)


# ---------------------------------------------------------------- pure helpers (numpy)


def hash_consts(token_map, multipliers, primes, offsets, pad_id: int, dead_ids) -> dict:
    """int64 host copies of NgramHashState's buffers for hash_segment()."""
    import numpy as np

    mult = np.asarray(multipliers, dtype=np.int64)
    nl = mult.shape[0]
    return {
        "tm": np.asarray(token_map, dtype=np.int64).reshape(-1),
        "mult": mult,
        "primes": np.asarray(primes, dtype=np.int64).reshape(nl, -1),
        "offs": np.asarray(offsets, dtype=np.int64).reshape(nl, -1),
        "pad": int(pad_id),
        "dead": np.asarray(dead_ids, dtype=np.int64).reshape(-1),
    }


def hash_segment(c: dict, ids, window, pos0: int, h0: int, h1: int):
    """_hash_ids_kernel for one request segment that starts at position pos0.

    ids: [L] token ids at positions pos0..pos0+L-1; window: the ids at pos0-1, pos0-2, ...
    (newest first, -1 before the sequence start), as _gather_lookback_kernel fills it.
    Returns int32 [L, layers, h1 - h0], hash columns h0..h1 (this rank's heads).
    """
    import numpy as np

    tm, mult, primes, offs, pad = c["tm"], c["mult"], c["primes"], c["offs"], c["pad"]
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    window = np.asarray(window, dtype=np.int64).reshape(-1)
    n = ids.shape[0]
    nl, ngram = mult.shape
    heads = primes.shape[1] // (ngram - 1)
    nv = tm.shape[0]

    def source(tok):
        # compressed id; image span tokens are DEAD (-1); out-of-range ids read pad
        src = np.where((tok >= 0) & (tok < nv), tm[np.clip(tok, 0, nv - 1)], pad)
        return np.where(np.isin(tok, c["dead"]), np.int64(-1), src)

    t = np.arange(n, dtype=np.int64)
    batch_src = source(ids)
    win_src = np.where(window >= 0, source(window), np.int64(pad))  # unknown -> pad fallback
    rolling = np.zeros((n, nl), dtype=np.int64)
    blocked = np.zeros(n, dtype=bool)
    out = np.empty((n, nl, primes.shape[1]), dtype=np.int64)
    for shift in range(ngram):
        in_batch = t >= shift
        col = np.clip(shift - t - 1, 0, window.shape[0] - 1)  # chunk_start - 1 - lookback
        src = np.where(in_batch, batch_src[np.maximum(t - shift, 0)], win_src[col])
        blocked |= (pos0 + t - shift < 0) | (src == -1)
        value = np.where(blocked, np.int64(pad), src)
        rolling ^= value[:, None] * mult[None, :, shift]
        if shift:
            b = (shift - 1) * heads
            out[:, :, b : b + heads] = (
                rolling[:, :, None] % primes[None, :, b : b + heads] + offs[None, :, b : b + heads]
            )
    return out[:, :, h0:h1].astype(np.int32)


def plan_ahead(req_ids, idx_mapping, num_computed, num_scheduled, prefill_len, is_prefilling,
               cap: int, min_tokens: int = MIN_AHEAD, per_req: int = 0) -> list:
    """Next-chunk read-ahead: [(req_id, req_idx, start, length, slot)], slots packed from 0.

    A request still prefilling after this step resumes at num_computed + num_scheduled
    (a chunked prefill continues where it stopped) and gets up to
    min(prefill_len - start, cap - used, per_req or cap) tokens.
    """
    plan, used = [], 0
    for i, req_id in enumerate(req_ids):
        if used >= cap:
            break
        if not bool(is_prefilling[i]):
            continue
        start = int(num_computed[i]) + int(num_scheduled[i])
        length = min(int(prefill_len[i]) - start, cap - used, per_req or cap)
        if length < min_tokens:
            continue
        plan.append((req_id, int(idx_mapping[i]), start, length, used))
        used += length
    return plan


def covered_rows(plan, req_ids, num_computed, is_prefilling, qsl) -> list:
    """[(q0, q1, slot)]: batch rows [q0, q1) of a request at the positions the plan read ahead."""
    by_id = {e[0]: e for e in plan}
    out = []
    for i, req_id in enumerate(req_ids):
        e = by_id.get(req_id)
        if e is None or not bool(is_prefilling[i]):
            continue
        q0, q1, p0 = int(qsl[i]), int(qsl[i + 1]), int(num_computed[i])
        lo, hi = max(p0, e[2]), min(p0 + q1 - q0, e[2] + e[3])
        if hi > lo:
            out.append((q0 + lo - p0, q0 + hi - p0, e[4] + lo - e[2]))
    return out


def segment_runs(qsl, covered, is_prefilling) -> list:
    """[(a, b, prefilling)]: the rows of each request segment that `covered` leaves out."""
    covs = sorted(covered)
    out, j = [], 0
    for i in range(len(qsl) - 1):
        a, b = int(qsl[i]), int(qsl[i + 1])
        cur = a
        while j < len(covs) and covs[j][0] < b:
            if covs[j][0] > cur:
                out.append((cur, covs[j][0], bool(is_prefilling[i])))
            cur = max(cur, covs[j][1])
            j += 1
        if cur < b:
            out.append((cur, b, bool(is_prefilling[i])))
    return out


def batch_meta_ok(qsl, num_reqs: int, n: int) -> bool:
    """The GPU query_start_loc is the batch the CPU fields describe (else: no read-ahead use)."""
    if len(qsl) != num_reqs + 1 or int(qsl[0]) != 0 or int(qsl[-1]) != n:
        return False
    return all(int(qsl[i]) <= int(qsl[i + 1]) for i in range(num_reqs))


def token_window(rows, start: int, length: int, depth: int):
    """(ids at [start, start+length), lookback window newest first) from one all_token_ids row."""
    import numpy as np

    lo = max(0, start - depth)
    seg = np.asarray(rows[lo : start + length], dtype=np.int64)
    window = np.full(depth, -1, dtype=np.int64)
    prev = seg[: start - lo][::-1]
    window[: prev.shape[0]] = prev
    return seg[start - lo :].copy(), window


# --------------------------------------------------- the parallel native gather (ctypes)


class Gather:
    """eng_gather_bf16 over token-row runs, TOK_BLOCK tokens per call, on a thread pool.

    tables: [(w_fd, w_off, s_fd, s_off, dim, sb, vocab_start, vocab_end, layer)].
    Hash ids are int32 [tokens, layers, heads_valid] rows; outputs are bf16
    [tokens, local_heads, dim] buffers given by address.
    """

    def __init__(self, lib, tables, nl: int, heads_valid: int, local_heads: int, dim: int,
                 lut_ptr: int, nthreads: int, cpu_set=None):
        import numpy as np

        self.lib, self.tables = lib, list(tables)
        self.nl, self.hv, self.lh, self.dim = nl, heads_valid, local_heads, dim
        self.lut_ptr = lut_ptr
        self.pool = ThreadPoolExecutor(max_workers=nthreads, thread_name_prefix="dsv41-pa-gather",
                                       initializer=_pin, initargs=(cpu_set,))
        self.scratch = queue.SimpleQueue()
        rows, sb = TOK_BLOCK * local_heads, max(t[5] for t in self.tables)
        for _ in range(nthreads):
            self.scratch.put((np.empty((rows, dim), np.uint8), np.empty((rows, sb), np.uint8),
                              (ctypes.c_int64 * NSTATS)()))

    def run(self, hash_ptr: int, runs, outs, cancel=None) -> list:
        """runs: [(a, b, flags)] token ranges; outs: one output address per table.

        Returns summed stats (engram_native.c order) and raises the first worker error.
        A set `cancel` skips blocks not yet started (stats[0] then falls short).
        """
        futs = [
            self.pool.submit(self._one, t, b0, min(b, b0 + TOK_BLOCK), flags, hash_ptr, outs[t], cancel)
            for t in range(len(self.tables))
            for a, b, flags in runs
            for b0 in range(a, b, TOK_BLOCK)
        ]
        total, err = [0] * NSTATS, None
        for f in futs:
            try:
                st = f.result()
            except Exception as exc:  # noqa: BLE001 - every future is drained first
                err = err or exc
                continue
            if st is not None:
                total = [x + y for x, y in zip(total, st)]
        if err is not None:
            raise err
        return total

    def _one(self, t, b0, b1, flags, hash_ptr, out_ptr, cancel):
        if cancel is not None and cancel.is_set():
            return None
        sw, ss, st = self.scratch.get()
        try:
            w_fd, w_off, s_fd, s_off, dim, sb, v0, v1, layer = self.tables[t]
            stride = self.nl * self.hv
            rc = self.lib.eng_gather_bf16(
                w_fd, w_off, s_fd, s_off, dim, sb, v0, v1,
                hash_ptr + (b0 * stride + layer * self.hv) * 4, stride, b1 - b0, self.hv, self.lh,
                self.lut_ptr, out_ptr + b0 * self.lh * dim * 2, sw.ctypes.data, ss.ctypes.data, flags, st,
            )
            if rc:
                raise OSError(-rc, f"eng_gather_bf16 layer {layer} rows {b0}:{b1}: {os.strerror(-rc)}")
            return list(st)
        finally:
            self.scratch.put((sw, ss, st))

    def close(self) -> None:
        self.pool.shutdown(wait=True)


# ------------------------------------------------------------- read-ahead core (no torch)


class Job:
    """One background read-ahead: plan, cancel flag, future, timings."""

    def __init__(self, plan):
        self.plan = plan
        self.cancel = threading.Event()
        self.future = None
        self.complete = False
        self.ms = 0.0


class AheadCore:
    """Plan, background job and consume logic over numpy buffers and addresses.

    pf_hash: int32 [cap, layers, heads_valid]; pf_rows_ptrs: one address per table of a
    bf16 [cap, local_heads, dim] buffer; wait_h2d(): blocks until the last host-to-device
    copies out of the read-ahead buffer are done (the job overwrites it).
    """

    def __init__(self, gather: Gather, consts: dict, h0: int, h1: int, depth: int, cap: int,
                 pf_hash, pf_rows_ptrs, wait_h2d, flags_prefill: int, cpu_set=None, share: int = 0):
        self.gather, self.consts = gather, consts
        self.h0, self.h1, self.depth, self.cap = h0, h1, depth, cap
        self.share = share  # share_tokens(): per-request read-ahead when the batch holds > 1 request
        self.pf_hash, self.pf_rows_ptrs = pf_hash, list(pf_rows_ptrs)
        self.wait_h2d, self.flags_prefill = wait_h2d, flags_prefill
        self.coord = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dsv41-pa-job",
                                        initializer=_pin, initargs=(cpu_set,))
        self.job: Job | None = None

    def plan(self, batch) -> list:
        """The read-ahead of the next chunk of every request that keeps prefilling."""
        nreq = int(batch.num_reqs)
        return plan_ahead(batch.req_ids[:nreq], batch.idx_mapping_np, batch.num_computed_tokens_np,
                          batch.num_scheduled_tokens, batch.prefill_len_np, batch.is_prefilling_np, self.cap,
                          per_req=self.share if nreq > 1 else 0)

    def start(self, plan, token_rows) -> Job:
        """Submit `plan`; token_rows[req_idx] is that request's all_token_ids row."""
        self.drop()
        # Copied now: the slot of a finished request may be rewritten by a later step.
        items = [(slot, start) + token_window(token_rows[idx], start, length, self.depth)
                 for _, idx, start, length, slot in plan]
        job = Job(plan)
        job.future = self.coord.submit(self._run, job, items)
        self.job = job
        return job

    def _run(self, job: Job, items) -> None:
        t0 = time.perf_counter()
        self.wait_h2d()
        total = 0
        for slot, start, ids, window in items:
            if job.cancel.is_set():
                return
            self.pf_hash[slot : slot + ids.shape[0]] = hash_segment(
                self.consts, ids, window, start, self.h0, self.h1)
            total = max(total, slot + ids.shape[0])
        st = self.gather.run(self.pf_hash.ctypes.data, [(0, total, self.flags_prefill)],
                             self.pf_rows_ptrs, job.cancel)
        job.complete = not job.cancel.is_set() and st[0] == total * self.gather.lh * len(self.gather.tables)
        job.ms = 1e3 * (time.perf_counter() - t0)

    def drop(self) -> None:
        """Cancel and join the pending job (bounded by one block per gather thread)."""
        job, self.job = self.job, None
        if job is not None:
            job.cancel.set()
            job.future.result()

    def consume(self, batch, host_hash, qsl, n: int):
        """(covered [(q0, q1, slot)], runs [(a, b, flags)], wait ms, job ms) after the GPU sync.

        host_hash: int32 [n, layers, heads_valid], this step's GPU hash ids. Covered rows
        have equal read-ahead ids; a covered row with other ids disarms the lever.
        """
        import numpy as np

        nreq = int(batch.num_reqs)
        ok = batch_meta_ok(qsl, nreq, n)
        pref = batch.is_prefilling_np[:nreq]
        job, self.job = self.job, None
        covered, wait_ms, job_ms = [], 0.0, 0.0
        if job is not None:
            if ok:
                covered = covered_rows(job.plan, batch.req_ids[:nreq], batch.num_computed_tokens_np, pref, qsl)
            if covered:
                t0 = time.perf_counter()
                job.future.result()
                wait_ms, job_ms = 1e3 * (time.perf_counter() - t0), job.ms
                if not job.complete:
                    covered = []
            else:
                job.cancel.set()
                job.future.result()
        good = []
        for a, b, s in covered:
            if np.array_equal(host_hash[a:b], self.pf_hash[s : s + b - a]):
                good.append((a, b, s))
            else:
                _disarm(RuntimeError(f"read-ahead hash ids != GPU hash ids (rows {a}:{b} of {n})"))
        if not ok:
            return [], [(0, n, FLAG_NOWAIT)], wait_ms, job_ms
        runs = [(a, b, self.flags_prefill if p else FLAG_NOWAIT) for a, b, p in segment_runs(qsl, good, pref)]
        return good, runs, wait_ms, job_ms

    def close(self) -> None:
        self.drop()
        self.coord.shutdown(wait=True)


# ------------------------------------------------------------------------ torch glue


def _lut(torch):
    """torch's float8_e4m3fn -> float32 table (the native stage builds the same one)."""
    return (torch.arange(256, dtype=torch.int32).to(torch.uint8)
            .view(torch.float8_e4m3fn).to(torch.float32).contiguous())


def _stock_rows(engram, host_rows, local_heads, dim):
    """The stock per-table gather (engram_stage_fast._fast_stage_one body) of host_rows ids."""
    import torch

    local = host_rows[:, engram.layer_hash_index, :].to(torch.int64)
    file_rows, owned = engram.embed_tokens.disk_file_rows_owned(local)
    rows = engram.embed_tokens.disk.gather_dequant(file_rows, owned)
    return rows.view(host_rows.shape[0], local_heads, dim)


class Ahead:
    """Per-stager state: buffers, the core, the stage() body for prefill-size calls."""

    def __init__(self, stager, torch, lib, dead_ids, env=None):
        import numpy as np

        import engram_native_stage as ens

        env = os.environ if env is None else env
        self.torch = torch
        hh = stager.hash_host
        if hh.dtype != torch.int32 or hh.dim() != 3 or not hh.is_contiguous():
            raise ValueError(f"hash_host layout {tuple(hh.shape)} {hh.dtype}")
        self.nl, self.hv = int(hh.shape[1]), int(hh.shape[2])
        self.lh, self.dim = int(stager.local_heads), int(stager.dim)
        if self.hv != stager.head_end - stager.head_start or not 0 < self.hv <= self.lh:
            raise ValueError(f"heads {self.hv} / {stager.head_end - stager.head_start} / {self.lh}")
        tables = []
        for engram, buf in zip(stager.engrams, stager.rows_host):
            emb, disk = engram.embed_tokens, engram.embed_tokens.disk
            if buf.dtype != torch.bfloat16 or not buf.is_contiguous() or tuple(buf.shape[1:]) != (self.lh, self.dim):
                raise ValueError(f"rows_host layout {tuple(buf.shape)} {buf.dtype}")
            if disk.dim != self.dim or disk.dim % disk.sb or not 0 <= engram.layer_hash_index < self.nl:
                raise ValueError(f"table geometry dim={disk.dim} sb={disk.sb}")
            tables.append((disk.w_fd, disk.w_off, disk.s_fd, disk.s_off, disk.dim, disk.sb,
                           emb.vocab_start_idx, emb.vocab_end_idx, engram.layer_hash_index))
        self.small = ens.max_tokens(env)  # the native stage takes calls up to this size
        self.cap = cap_tokens(stager.max_tokens, env)
        self.lut = _lut(torch)
        cpu_set = cpus(env)
        self.gather = Gather(lib, tables, self.nl, self.hv, self.lh, self.dim, self.lut.data_ptr(), threads(env),
                             cpu_set)
        hs = stager.hash_state
        consts = hash_consts(hs.token_map.cpu(), hs.multipliers.cpu(), hs.primes.cpu(), hs.offsets.cpu(),
                             hs.pad_id, dead_ids)
        self.h2d_done = torch.cuda.Event()
        # Allocated at the first read-ahead: a serve without multi-chunk prefills pays nothing.
        self.pf_rows = None
        self.pf_hash = np.zeros((self.cap, self.nl, self.hv), dtype=np.int32)
        self.core = AheadCore(self.gather, consts, stager.head_start, stager.head_end, int(hs.lookback_depth),
                              self.cap, self.pf_hash, [], self.h2d_done.synchronize,
                              FLAG_NOWAIT | (FLAG_DONTNEED if dontneed(env) else 0), cpu_set,
                              share=share_tokens(env))
        self.qsl_host = None
        self.census = census_on(env)
        self.mib = len(tables) * self.cap * self.lh * self.dim * 2 / 2**20
        self.last_ahead = 0  # rows of the last stage() call staged from the read-ahead

    def _qsl(self, nq: int):
        if self.qsl_host is None or self.qsl_host.shape[0] < nq:
            self.qsl_host = self.torch.empty(max(nq, 64), dtype=self.torch.int32, pin_memory=True)
        return self.qsl_host[:nq]

    def launch(self, batch, req_states) -> None:
        plan = self.core.plan(batch)
        if not plan:
            self.core.drop()
            return
        uva = getattr(req_states.all_token_ids, "_uva_buf", None)
        rows = getattr(uva, "np", None)
        if rows is None or rows.ndim != 2:
            raise RuntimeError("req_states.all_token_ids has no UVA host view (_uva_buf.np)")
        if self.pf_rows is None:
            self.pf_rows = [
                self.torch.empty((self.cap, self.lh, self.dim), dtype=self.torch.bfloat16, pin_memory=True)
                for _ in self.gather.tables
            ]
            self.core.pf_rows_ptrs = [t.data_ptr() for t in self.pf_rows]
            print("dsv41: engram prefill ahead buffer %.1f MiB pinned (%d tokens)" % (self.mib, self.cap), flush=True)
        self.core.start(plan, rows)

    def stage(self, st, input_ids, positions, query_start_loc, lookback, n, batch, mask) -> int:
        """The stock stage() for one prefill-size batch, with read-ahead rows where they match."""
        torch = self.torch
        t0 = time.perf_counter()
        # Stock hash + DtoH + sync (engram_stage_fast STAGE_FAST), plus the batch's
        # query_start_loc in the same sync.
        ids = input_ids[:n]
        hashes = st.hash_state(
            ids,
            positions[:n],
            query_start_loc,
            mask(ids),
            lookback,
            mask(lookback),
            None,
            None,
        )
        host = st.hash_host[:n]
        host.copy_(hashes[:, :, st.head_start : st.head_end], non_blocking=True)
        qh = self._qsl(int(query_start_loc.shape[0]))
        qh.copy_(query_start_loc, non_blocking=True)
        st.hashes_ready.record()
        st.hashes_ready.synchronize()
        qsl = qh.numpy().tolist()
        t1 = time.perf_counter()
        covered, runs, wait_ms, job_ms = self.core.consume(batch, host.numpy(), qsl, n)
        t2 = time.perf_counter()
        if runs:
            self.gather.run(host.data_ptr(), runs, [b.data_ptr() for b in st.rows_host])
        t3 = time.perf_counter()
        if _STATE["verify_left"] > 0 and not self.verify(st, host, n, covered):
            # Refill the whole step from the stock gather (the lever is now off).
            for engram, buf in zip(st.engrams, st.rows_host):
                buf[:n].copy_(_stock_rows(engram, host, self.lh, self.dim))
            covered, runs = [], [(0, n, FLAG_NOWAIT)]
        for t, (engram, buf) in enumerate(zip(st.engrams, st.rows_host)):
            dest = engram._staged_rows_for_ubatch()
            for a, b, s in covered:
                dest[a:b].copy_(self.pf_rows[t][s : s + b - a], non_blocking=True)
            for a, b, _ in runs:
                dest[a:b].copy_(buf[a:b], non_blocking=True)
        self.h2d_done.record()
        st.num_staged = n
        self.last_ahead = ahead = sum(b - a for a, b, _ in covered)
        if self.census and any(bool(x) for x in batch.is_prefilling_np[: int(batch.num_reqs)]):
            print(
                "[prefill-ahead] n=%d ahead=%d sync=%d wait=%.1fms gather=%.1fms stage=%.1fms "
                "hash_sync=%.1fms job=%.1fms"
                % (n, ahead, n - ahead, wait_ms, 1e3 * (t3 - t2), 1e3 * (time.perf_counter() - t0),
                   1e3 * (t1 - t0), job_ms),
                flush=True,
            )
        return n

    def verify(self, st, host, n, covered) -> bool:
        """Raw bf16 bits of up to VERIFY_ROWS staged rows per table vs the stock gather."""
        torch = self.torch
        step = max(1, n // VERIFY_ROWS)
        rows = sorted(set(range(0, n, step)) | {n - 1} | {x for a, b, _ in covered for x in (a, b - 1)})
        slot = {}
        for a, b, s in covered:
            slot.update((r, s + r - a) for r in rows if a <= r < b)
        idx = torch.tensor(rows, dtype=torch.int64)
        try:
            for t, (engram, buf) in enumerate(zip(st.engrams, st.rows_host)):
                ref = _stock_rows(engram, host[idx], self.lh, self.dim)
                got = torch.stack([self.pf_rows[t][slot[r]] if r in slot else buf[r] for r in rows])
                if not torch.equal(ref.view(torch.int16), got.view(torch.int16)):
                    raise RuntimeError(f"staged rows != stock gather (layer {engram.layer_hash_index}, n={n})")
        except Exception as exc:  # noqa: BLE001
            _disarm(exc)
            return False
        _STATE["verify_left"] -= 1
        if not _STATE["engaged"]:
            _STATE["engaged"] = True
            print("dsv41: engram prefill ahead self-check bit-exact (%d tables, n=%d, %d rows ahead)"
                  % (len(st.engrams), n, sum(b - a for a, b, _ in covered)), flush=True)
        return True

    def close(self) -> None:
        try:
            self.core.drop()
        except Exception:  # noqa: BLE001 - a broken job is dropped with the lever
            pass


def _make_stage(orig_stage, torch, image_sentinel_mask):
    @torch.inference_mode()
    def stage(self, input_ids, positions, query_start_loc, lookback_token_ids, num_tokens):
        pa = getattr(self, "_eng_pa", None)
        n = min(int(num_tokens), self.max_tokens)
        cur = getattr(self, "_pa_cur", None)
        batch = cur[0] if cur is not None else None
        if (
            pa is None
            or not _STATE["armed"]
            or n <= pa.small
            or batch is None
            or batch.input_ids is not input_ids
            or int(batch.num_tokens) != int(num_tokens)
            or not self.hash_state.ensure_cache()
        ):
            return orig_stage(self, input_ids, positions, query_start_loc, lookback_token_ids, num_tokens)
        try:
            ret = pa.stage(self, input_ids, positions, query_start_loc, lookback_token_ids, n, batch,
                           image_sentinel_mask)
        except Exception as exc:  # noqa: BLE001 - the stock stage redoes this step
            _disarm(exc)
            pa.close()
            return orig_stage(self, input_ids, positions, query_start_loc, lookback_token_ids, num_tokens)
        if _STATE["armed"]:
            try:
                pa.launch(batch, cur[1])
            except Exception as exc:  # noqa: BLE001 - nothing was staged from it yet
                _disarm(exc)
                pa.close()
        return ret

    stage._dsv41_prefill_ahead = True
    return stage


def install() -> str:
    """Wrap EngramDiskStager (__init__ builds the state, stage() reads ahead) and
    DeepseekV41ModelState.prepare_inputs (records the batch stage() is called for)."""
    import torch
    from vllm.models.deepseek_v4_1.common import engram as eng
    from vllm.models.deepseek_v4_1.common.mm_preprocess import (
        IMAGE_PAD_ID,
        IMAGE_SENTINEL_BASE_ID,
        image_sentinel_mask,
    )
    from vllm.models.deepseek_v4_1.nvidia import model_state as ms_mod

    import engram_native_stage as ens

    cls = eng.EngramDiskStager
    if getattr(cls.stage, "_dsv41_prefill_ahead", False):
        return "already installed"
    conflict = ens.stage_conflict(cls)
    if conflict is not None:
        _disarm(RuntimeError(conflict))
        return "not installed (stage() is not the stock one)"
    _STATE["verify_left"] = verify_calls()
    cap_tokens(1), threads(), dontneed(), cpus()  # validate the env before arming
    orig_init, orig_stage = cls.__init__, cls.stage
    dead = (IMAGE_SENTINEL_BASE_ID, IMAGE_PAD_ID)

    def __init__(self, hash_state, engrams):
        orig_init(self, hash_state, engrams)
        self._eng_pa = None
        if not _STATE["armed"]:
            return
        try:
            if getattr(hash_state, "use_slot_cache", False):
                raise RuntimeError("V1 slot-cache hash state: lookbacks are not the prompt tokens")
            self._eng_pa = Ahead(self, torch, ens._lib(), dead)
            print(
                "dsv41: engram prefill ahead armed (%d tables, read-ahead %d tokens = %.1f MiB pinned "
                "at first use, threads=%d, cpus=%s, verify=%d, dontneed=%d, above %d tokens)"
                % (len(self._eng_pa.gather.tables), self._eng_pa.cap, self._eng_pa.mib, threads(),
                   os.environ.get("DSV41_ENGRAM_PREFILL_AHEAD_CPUS") or "any", _STATE["verify_left"],
                   int(dontneed()), self._eng_pa.small),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            _disarm(exc)

    msc = ms_mod.DeepseekV41ModelState
    orig_inputs = msc.prepare_inputs

    def prepare_inputs(self, input_batch, req_states):
        stager = self.engram_stager
        if stager is None:
            return orig_inputs(self, input_batch, req_states)
        stager._pa_cur = (input_batch, req_states)
        try:
            return orig_inputs(self, input_batch, req_states)
        finally:
            stager._pa_cur = None

    cls.__init__ = __init__
    cls.stage = _make_stage(orig_stage, torch, image_sentinel_mask)
    msc.prepare_inputs = prepare_inputs
    return (f"EngramDiskStager wrapped (read-ahead tokens={os.environ.get('DSV41_ENGRAM_PREFILL_AHEAD_TOKENS') or 'max'}, "
            f"threads={threads()}, verify={_STATE['verify_left']})")
