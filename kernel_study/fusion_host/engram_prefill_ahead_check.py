#!/usr/bin/env python3
"""DSV41_ENGRAM_PREFILL_AHEAD in the serving image, one rank's view, on a GPU.

  docker run --rm --gpus all --network none --memory 16g -v <repo>:/repo [-v <hf>:/hf:ro] \
      -e DSV41_ENGRAM_GATHER_V2=1 -e DSV41_ENGRAM_WILLNEED=1 --entrypoint python3 <image> \
      /repo/kernel_study/fusion_host/engram_prefill_ahead_check.py --mode check

Do not set DSV41_ENGRAM_PREFILL_AHEAD in the container: the baked sitecustomize would wrap
EngramDiskStager.stage at import and the stock arm would not be stock. The lever is built
here from docker/patch/engram_prefill_ahead.py with an explicit env.

--mode check (no disk window: synthetic tables of ~12k rows per table in a tmp dir):
  1. mirror: epa.hash_segment vs the image's _hash_ids_kernel (NgramHashState.forward on a
     stand-in with the real multiplier formula, an EngramLayout prime draw and a random
     token map): prefill segments inside mixed batches, image tokens, positions 0..2.
  2. stage: the image's EngramDiskStager.stage (engram_stage_fast + gather v2 text, real
     DiskEngramTable, real Triton hash) stock vs wrapped by engram_prefill_ahead, scripted
     scheduler steps per scenario (chunked prefill, 4-sequence mixed batches, prefix-cache
     start, preemption + re-admission, abort, two prefills, a shorter chunk), both TP ranks:
     staged device rows [:n] compared bit for bit after every step; coverage per scenario.
  3. sabotage: a corrupted read-ahead hash id -> one disarm line, rows still the stock ones.
--mode bench --snapshot <pack> (DISK WINDOW: serve down, nothing else reading the NVMe):
  real layout and tables, one novel prompt per arm in --chunk-tokens chunks. After each
  stage() a --compute-ms GPU sleep stands for that chunk's forward, which the read-ahead
  overlaps as in the serve; CUDA events from the previous sleep's end to this stage's H2D
  end = the GPU idle the stage causes. Arms: stock, ahead (DONTNEED=1), ahead (DONTNEED=0).
  /proc/meminfo Cached and MemFree per chunk; the table files' pages dropped between arms.
--mode decode-probe --snapshot <pack> (DISK WINDOW): the native decode gather (both tables,
  --decode-tokens tokens) on fresh random rows: back to back vs one call every --gap-ms,
  optionally while holding a PM QoS request (/dev/cpu_dma_latency, --pm-qos-us).
Prints one JSON line; exit 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import struct
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docker" / "patch"))
import engram_native_stage as ens  # noqa: E402
import engram_prefill_ahead as epa  # noqa: E402

DIM = 256
VOCAB = 129280


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def summarize(v):
    if not v:
        return {"n": 0}
    return dict(n=len(v), median=statistics.median(v), p10=pct(v, 0.10), p90=pct(v, 0.90),
                mean=statistics.fmean(v), min=min(v), max=max(v))


def meminfo() -> dict:
    out = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, v = line.split(":", 1)
            if k in ("MemFree", "MemAvailable", "Cached"):
                out[k] = int(v.split()[0]) // 1024  # MiB
    return out


def synthetic_config():
    return SimpleNamespace(
        engram_layer_ids=[1, 14], engram_num_embeddings=[0, 0], engram_max_ngram_size=4, engram_n_heads=8,
        engram_head_dim=DIM, engram_compressed_vocab_size=99092, engram_pad_token_id=2, engram_vocab_size=500,
    )


def snapshot_config(snapshot: str):
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    return SimpleNamespace(**cfg.get("text_config", cfg))


def write_pack(d: Path, layout, seed: int) -> None:
    """A minimal pack: index + one safetensors file holding both Engram tables."""
    rng = np.random.default_rng(seed)
    header, blobs, off = {}, [], 0
    for li, lid in enumerate(layout.layer_ids):
        rows = sum(p for order in layout.primes[li] for p in order)
        for name, arr, dt in (
            (f"layers.{lid}.engram.embed.weight", rng.integers(0, 256, (rows, DIM), dtype=np.uint8), "F8_E4M3"),
            (f"layers.{lid}.engram.embed.scale", rng.integers(100, 140, (rows, DIM // 32), dtype=np.uint8), "U8"),
        ):
            b = arr.tobytes()
            header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
            blobs.append(b)
            off += len(b)
    hb = json.dumps(header).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    fname = "model-00001-of-00001.safetensors"
    with open(d / fname, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for b in blobs:
            fh.write(b)
    with open(d / "model.safetensors.index.json", "w", encoding="utf-8") as fh:
        json.dump({"weight_map": {k: fname for k in header}}, fh)


class MiniHash:
    """NgramHashState's buffers on the GPU; calls the image's NgramHashState.forward."""

    def __init__(self, eng, layout, dev, seed: int):
        self.use_slot_cache, self._cache, self.block_size = False, None, 64
        self.lookback_depth = layout.max_ngram_size - 1
        self.multipliers = eng.compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, layout.compressed_vocab_size).to(dev)
        self.primes = torch.tensor(layout.primes).to(dev)
        self.offsets = layout.offsets.to(dev)
        g = torch.Generator().manual_seed(seed)
        self.token_map = torch.randint(0, layout.compressed_vocab_size, (VOCAB,), dtype=torch.int32,
                                       generator=g).to(dev)
        self.pad_id = int(self.token_map[layout.pad_token_id])
        self._fwd = eng.NgramHashState.forward

    def ensure_cache(self) -> bool:
        return True

    def __call__(self, *args):
        return self._fwd(self, *args)


def make_stager(eng, layout, disks, hash_state, rank: int, max_tokens: int, dev):
    heads = layout.n_hash_cols
    part = -(-heads // 2)
    h0, h1 = rank * part, min(rank * part + part, heads)
    engrams = []
    for li, disk in enumerate(disks):
        sizes = [p for order in layout.primes[li] for p in order]
        emb = SimpleNamespace(part_n_hash_cols=part, head_start=h0, n_hash_cols=heads, dim=DIM,
                              vocab_start_idx=sum(sizes[:h0]), vocab_end_idx=sum(sizes[: h0 + part]), disk=disk)
        emb.disk_file_rows_owned = MethodType(eng.ParallelEngramEmbedding.disk_file_rows_owned, emb)
        staged = torch.zeros((max_tokens, part, DIM), dtype=torch.bfloat16, device=dev)
        e = SimpleNamespace(layer_hash_index=li, embed_tokens=emb, staged=staged)
        e._staged_rows_for_ubatch = (lambda s=staged: s)
        engrams.append(e)
    st = object.__new__(eng.EngramDiskStager)
    st.hash_state, st.engrams = hash_state, engrams
    st.local_heads, st.head_start, st.head_end, st.dim, st.max_tokens = part, h0, h1, DIM, max_tokens
    st.hash_host = torch.empty((max_tokens, len(disks), h1 - h0), dtype=torch.int32, pin_memory=True)
    st.rows_host = [torch.empty((max_tokens, part, DIM), dtype=torch.bfloat16, pin_memory=True) for _ in disks]
    st.hashes_ready = torch.cuda.Event()
    st.num_staged = 0
    st.prefetch_on = False
    return st


class Sched:
    """Requests, the V2 runner's batch fields and all_token_ids, a FCFS chunked-prefill
    scheduler with DSpark verify steps (1 + K tokens) for decoding requests."""

    K = 3

    def __init__(self, dev, seed: int, max_len: int):
        self.dev, self.rng = dev, random.Random(seed)
        self.uva = np.zeros((8, max_len), np.int32)
        self.free = list(range(8))[::-1]
        self.running = []
        self.req_states = SimpleNamespace(all_token_ids=SimpleNamespace(_uva_buf=SimpleNamespace(np=self.uva)))
        self.pending = None

    def prompt(self, n: int, dead) -> list:
        toks = [self.rng.randrange(VOCAB) for _ in range(n)]
        for _ in range(max(1, n // 300)):
            a = self.rng.randrange(n)
            toks[a : a + 5] = [dead[0]] * len(toks[a : a + 5])
        return toks

    def add(self, rid: str, prompt: list, computed: int = 0) -> None:
        idx = self.free.pop()
        self.uva[idx, : len(prompt)] = prompt
        self.running.append(SimpleNamespace(rid=rid, idx=idx, tokens=list(prompt), prefill_len=len(prompt),
                                            computed=computed))

    def remove(self, rid: str) -> None:
        r = next(x for x in self.running if x.rid == rid)
        self.running.remove(r)
        self.free.append(r.idx)

    def prefilling(self) -> bool:
        return any(r.computed < r.prefill_len for r in self.running)

    def next(self, budget: int):
        dec, pre, left = [], [], budget
        for r in self.running:
            if r.computed >= r.prefill_len and left >= 1 + self.K:
                dec.append((r, 1 + self.K))
                left -= 1 + self.K
        for r in self.running:
            if r.computed < r.prefill_len and left > 0:
                n = min(r.prefill_len - r.computed, left)
                pre.append((r, n))
                left -= n
        segs = dec + sorted(pre, key=lambda x: x[1])
        ids, pos, qsl, look, drafts = [], [], [0], [], {}
        for r, n in segs:
            p0 = r.computed
            if p0 < r.prefill_len:
                ids += r.tokens[p0 : p0 + n]
            else:
                drafts[r.rid] = [self.rng.randrange(VOCAB) for _ in range(self.K)]
                ids += [r.tokens[-1]] + drafts[r.rid]
            pos += list(range(p0, p0 + n))
            qsl.append(qsl[-1] + n)
            look.append([r.tokens[p0 - 1 - j] if p0 - 1 - j >= 0 else -1 for j in range(3)])
        n_tok = qsl[-1]
        input_ids = torch.tensor(ids, dtype=torch.int32, device=self.dev)
        batch = SimpleNamespace(
            req_ids=[r.rid for r, _ in segs], num_reqs=len(segs),
            idx_mapping_np=np.array([r.idx for r, _ in segs], np.intp),
            num_computed_tokens_np=np.array([r.computed for r, _ in segs], np.int32),
            num_scheduled_tokens=np.array([n for _, n in segs], np.int32),
            prefill_len_np=np.array([r.prefill_len for r, _ in segs], np.int32),
            is_prefilling_np=np.array([r.computed < r.prefill_len for r, _ in segs]),
            num_tokens=n_tok, input_ids=input_ids,
        )
        self.pending = (segs, drafts)
        return SimpleNamespace(
            batch=batch, n=n_tok, input_ids=input_ids,
            positions=torch.tensor(pos, dtype=torch.int64, device=self.dev),
            qsl=torch.tensor(qsl, dtype=torch.int32, device=self.dev),
            lookback=torch.tensor(look, dtype=torch.int32, device=self.dev).reshape(len(segs), 3),
            segs=[(r.rid, n) for r, n in segs],
            prefill_rows=sum(n for r, n in segs if r.computed < r.prefill_len),
        )

    def commit(self) -> None:
        segs, drafts = self.pending
        for r, n in segs:
            if r.computed < r.prefill_len:
                r.computed += n
                if r.computed == r.prefill_len:
                    r.tokens.append(self.rng.randrange(VOCAB))
            else:
                a = self.rng.randrange(self.K + 1)
                r.tokens += drafts[r.rid][:a] + [self.rng.randrange(VOCAB)]
                r.computed += 1 + a
            self.uva[r.idx, : len(r.tokens)] = r.tokens
        self.pending = None


def lever_env(**kw) -> dict:
    env = {"DSV41_ENGRAM_PREFILL_AHEAD": "1", "DSV41_ENGRAM_CENSUS": "0"}
    env.update({f"DSV41_ENGRAM_PREFILL_AHEAD_{k.upper()}": str(v) for k, v in kw.items()})
    return env


def build_arms(eng, layout, disks, hs, rank, max_tokens, dev, dead, envs):
    """[(name, stager, stage fn)]: stock first, then one lever arm per env."""
    from vllm.models.deepseek_v4_1.common.mm_preprocess import image_sentinel_mask

    if getattr(eng, "_ENG_STAGE_POOL", None) is None:
        eng._ENG_STAGE_POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("DSV41_ENGRAM_STAGE_THREADS", "16")))
    stock = eng.EngramDiskStager.stage
    if getattr(stock, "_dsv41_prefill_ahead", False) or getattr(stock, "_dsv41_native", False):
        raise SystemExit("EngramDiskStager.stage is already wrapped: unset the lever envs in the container")
    lever = epa._make_stage(stock, torch, image_sentinel_mask)
    arms = [("stock", make_stager(eng, layout, disks, hs, rank, max_tokens, dev), stock)]
    lib = ens._lib()
    for name, env in envs:
        st = make_stager(eng, layout, disks, hs, rank, max_tokens, dev)
        st._eng_pa = epa.Ahead(st, torch, lib, dead, env=env)
        arms.append((name, st, lever))
    return arms


def run_step(arms, b, req_states):
    for _, st, fn in arms:
        st._pa_cur = (b.batch, req_states)
        try:
            fn(st, b.input_ids, b.positions, b.qsl, b.lookback, b.n)
        finally:
            st._pa_cur = None


def staged_equal(arms, n) -> bool:
    torch.cuda.synchronize()
    ref = [e.staged[:n].view(torch.int16) for e in arms[0][1].engrams]
    return all(torch.equal(r, e.staged[:n].view(torch.int16)) for _, st, _ in arms[1:] for r, e in zip(ref, st.engrams))


def shutdown(arms) -> None:
    for _, st, _ in arms[1:]:
        st._eng_pa.close()
        st._eng_pa.core.close()
        st._eng_pa.gather.close()


def mode_check(args, eng, dev, dead) -> dict:
    from vllm.models.deepseek_v4_1.common.mm_preprocess import image_sentinel_mask

    out = {"mirror": {}, "scenarios": {}, "sabotage": {}}
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        layout = eng.EngramLayout(synthetic_config())
        write_pack(Path(tmp), layout, seed=3)
        from vllm.models.deepseek_v4_1.common.engram_disk import DiskEngramTable

        disks = [DiskEngramTable(tmp, lid, DIM, 32) for lid in layout.layer_ids]
        hs = MiniHash(eng, layout, dev, seed=5)
        consts = epa.hash_consts(hs.token_map.cpu(), hs.multipliers.cpu(), hs.primes.cpu(), hs.offsets.cpu(),
                                 hs.pad_id, dead)
        # 1. mirror vs the Triton kernel
        sched = Sched(dev, seed=7, max_len=8192)
        bad = 0
        for trial in range(40):
            nreq = 1 + trial % 4
            ids, pos, qsl, look, segs = [], [], [0], [], []
            for i in range(nreq):
                p0 = [0, 1, 2, 3, 77, 4000][(trial + i) % 6]
                n = [1, 3, 64, 257][(trial * 3 + i) % 4]
                hist = sched.prompt(p0 + n, dead)
                ids += hist[p0:]
                pos += list(range(p0, p0 + n))
                qsl.append(qsl[-1] + n)
                look.append([hist[p0 - 1 - j] if p0 - 1 - j >= 0 else -1 for j in range(3)])
                segs.append((hist[p0:], look[-1], p0))
            input_ids = torch.tensor(ids, dtype=torch.int32, device=dev)
            lb = torch.tensor(look, dtype=torch.int32, device=dev)
            got = hs(input_ids, torch.tensor(pos, dtype=torch.int64, device=dev),
                     torch.tensor(qsl, dtype=torch.int32, device=dev),
                     image_sentinel_mask(input_ids), lb, image_sentinel_mask(lb), None, None).cpu().numpy()
            for i, (seg_ids, win, p0) in enumerate(segs):
                mine = epa.hash_segment(consts, seg_ids, win, p0, 0, layout.n_hash_cols)
                bad += int(not np.array_equal(mine, got[qsl[i] : qsl[i + 1]]))
        out["mirror"] = {"trials": 40, "segment_mismatches": bad}
        ok &= bad == 0

        # 2. scripted scenarios, both ranks
        def scenario(seed, rank, script):
            envs = [("ahead", lever_env(threads=4, dontneed=1)), ("ahead_keep", lever_env(threads=4, dontneed=0))]
            arms = build_arms(eng, layout, disks, hs, rank, 512, dev, dead, envs)
            s = Sched(dev, seed=seed, max_len=8192)
            stats = {"steps": 0, "mismatch_steps": 0, "ahead": [], "prefill_rows": []}

            def step(budget=384):
                b = s.next(budget)
                for _, st, _ in arms[1:]:
                    st._eng_pa.last_ahead = 0  # a delegated (stock) call leaves it at 0
                run_step(arms, b, s.req_states)
                stats["steps"] += 1
                stats["mismatch_steps"] += int(not staged_equal(arms, b.n))
                stats["prefill_rows"].append(b.prefill_rows)
                stats["ahead"].append([st._eng_pa.last_ahead for _, st, _ in arms[1:]])
                s.commit()

            script(s, step)
            shutdown(arms)
            return stats

        def chunked(s, step):
            s.add("a", s.prompt(1950, dead))
            while s.prefilling():
                step()

        def mixed(s, step):
            for rid in ("d1", "d2", "d3"):
                s.add(rid, s.prompt(50, dead))
            step()
            s.add("p", s.prompt(1600, dead))
            while s.prefilling():
                step()
            for _ in range(3):
                step()

        def prefix(s, step):
            s.add("a", s.prompt(1500, dead), computed=640)
            while s.prefilling():
                step()

        def preempt(s, step):
            p = s.prompt(1600, dead)
            s.add("a", p)
            step()
            step()
            s.remove("a")
            s.add("b", s.prompt(200, dead))
            step()
            s.add("a", p)
            while s.prefilling():
                step()

        def abort(s, step):
            s.add("a", s.prompt(2000, dead))
            step()
            s.remove("a")
            s.add("b", s.prompt(900, dead))
            step()
            step()

        def two(s, step):
            s.add("a", s.prompt(500, dead))
            s.add("b", s.prompt(1500, dead))
            while s.prefilling():
                step()

        def shorter(s, step):
            s.add("a", s.prompt(1500, dead))
            step()
            step(100)
            step()

        for k, (name, fn) in enumerate((("chunked", chunked), ("mixed_c4", mixed), ("prefix_hit", prefix),
                                        ("preempt", preempt), ("abort", abort), ("two_prefills", two),
                                        ("shorter_chunk", shorter))):
            for rank in (0, 1):
                epa._STATE.update(armed=True, verify_left=4, engaged=False)
                with redirect_stdout(StringIO()) as log:
                    st = scenario(100 * k + rank, rank, fn)
                st["armed_after"] = epa._STATE["armed"]
                st["disarm_lines"] = log.getvalue().count(epa.LOG_DISARMED)
                out["scenarios"][f"{name}_r{rank}"] = st
                ok &= st["mismatch_steps"] == 0 and st["armed_after"] and st["disarm_lines"] == 0

        # 3. sabotage: one wrong read-ahead id -> disarm, rows still the stock ones
        epa._STATE.update(armed=True, verify_left=4, engaged=False)
        real = epa.hash_segment

        def wrong(*a, **k):
            r = real(*a, **k)
            r[5, 0, 2] ^= 1
            return r

        arms = build_arms(eng, layout, disks, hs, 0, 512, dev, dead, [("ahead", lever_env(threads=4))])
        s = Sched(dev, seed=99, max_len=8192)
        s.add("a", s.prompt(1500, dead))
        eq = []
        with redirect_stdout(StringIO()) as log:
            for i in range(4):
                if i == 1:
                    epa.hash_segment = wrong
                b = s.next(384)
                run_step(arms, b, s.req_states)
                eq.append(staged_equal(arms, b.n))
                s.commit()
        epa.hash_segment = real
        shutdown(arms)
        out["sabotage"] = {"rows_equal_stock": eq, "disarmed": not epa._STATE["armed"],
                           "disarm_lines": log.getvalue().count(epa.LOG_DISARMED)}
        ok &= all(eq) and not epa._STATE["armed"] and out["sabotage"]["disarm_lines"] == 1
    out["ok"] = bool(ok)
    return out


def drop_tables(disks) -> None:
    for d in disks:
        for fd in (d.w_fd, d.s_fd):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def mode_bench(args, eng, dev, dead) -> dict:
    from vllm.models.deepseek_v4_1.common.engram_disk import DiskEngramTable

    layout = eng.EngramLayout(snapshot_config(args.snapshot))
    disks = [DiskEngramTable(args.snapshot, lid, layout.head_dim, 32) for lid in layout.layer_ids]
    hs = MiniHash(eng, layout, dev, seed=11)
    envs = [("ahead", lever_env(threads=args.threads, dontneed=1)),
            ("ahead_keep", lever_env(threads=args.threads, dontneed=0))]
    arms_all = build_arms(eng, layout, disks, hs, args.rank, args.chunk_tokens, dev, dead, envs)
    torch.cuda._sleep(1000)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    torch.cuda._sleep(20_000_000)
    e1.record()
    torch.cuda.synchronize()
    cycles = int(args.compute_ms * 1e3 * 20_000_000 / (e0.elapsed_time(e1) * 1e3))
    out = {"args": vars(args), "arms": {}}
    for name, st, fn in arms_all:
        drop_tables(disks)
        epa._STATE.update(armed=True, verify_left=1, engaged=False)
        s = Sched(dev, seed=1000 + len(out["arms"]), max_len=args.prompt_tokens + 64)
        s.add("a", s.prompt(args.prompt_tokens, dead))
        mem0 = meminfo()
        idle, host, mem, ahead = [], [], [], []
        evs = []
        while s.prefilling():
            b = s.next(args.chunk_tokens)
            a0 = torch.cuda.Event(enable_timing=True)
            a1 = torch.cuda.Event(enable_timing=True)
            pa = getattr(st, "_eng_pa", None)
            if pa is not None:
                pa.last_ahead = 0
            a0.record()  # completes when the previous chunk's sleep ends
            t0 = time.perf_counter()
            run_step([(name, st, fn)], b, s.req_states)
            host.append(1e3 * (time.perf_counter() - t0))
            ahead.append(pa.last_ahead if pa is not None else 0)
            a1.record()
            torch.cuda._sleep(cycles)  # this chunk's forward; the read-ahead runs under it
            evs.append((a0, a1))
            s.commit()
            mem.append(meminfo())
        torch.cuda.synchronize()
        idle = [a.elapsed_time(z) for a, z in evs]
        pa = getattr(st, "_eng_pa", None)
        if pa is not None:
            pa.close()
        out["arms"][name] = {
            "chunk_idle_ms": [round(x, 1) for x in idle], "host_stage_ms": [round(x, 1) for x in host],
            "rows_ahead": ahead,
            "first_chunk_idle_ms": round(idle[0], 1), "later_chunks_idle_ms": summarize(idle[1:]),
            "cached_mib_delta": [m["Cached"] - mem0["Cached"] for m in mem],
            "memfree_mib_min": min(m["MemFree"] for m in mem), "armed_after": epa._STATE["armed"],
        }
    # bit-exactness on real tables: the three arms' rows for one shared prompt, 2 chunks
    epa._STATE.update(armed=True, verify_left=1, engaged=False)
    s = Sched(dev, seed=4242, max_len=3 * args.chunk_tokens)
    s.add("a", s.prompt(2 * args.chunk_tokens, dead))
    eq = []
    while s.prefilling():
        b = s.next(args.chunk_tokens)
        run_step(arms_all, b, s.req_states)
        eq.append(staged_equal(arms_all, b.n))
        s.commit()
    shutdown(arms_all)
    out["rows_equal_stock"] = eq
    out["ok"] = all(eq) and all(a["armed_after"] for a in out["arms"].values())
    return out


def mode_decode_probe(args, eng, dev, dead) -> dict:
    from vllm.models.deepseek_v4_1.common.engram_disk import DiskEngramTable

    layout = eng.EngramLayout(snapshot_config(args.snapshot))
    disks = [DiskEngramTable(args.snapshot, lid, layout.head_dim, 32) for lid in layout.layer_ids]
    hs = MiniHash(eng, layout, dev, seed=13)
    st = make_stager(eng, layout, disks, hs, args.rank, 64, dev)
    nat = ens._NativeStager(st, torch)
    heads, part = layout.n_hash_cols, -(-layout.n_hash_cols // 2)
    h0 = args.rank * part
    rng = random.Random(17)
    sizes = [[p for order in layout.primes[li] for p in order] for li in range(len(disks))]
    offs = [[sum(s[:c]) for c in range(heads)] for s in sizes]

    def fill(n):
        hh = st.hash_host[:n]
        for li in range(len(disks)):
            for t in range(n):
                for c in range(st.head_end - st.head_start):
                    hh[t, li, c] = offs[li][h0 + c] + rng.randrange(sizes[li][h0 + c])

    qos = None
    if args.pm_qos_us is not None:
        qos = os.open("/dev/cpu_dma_latency", os.O_WRONLY)
        os.write(qos, struct.pack("i", args.pm_qos_us))
    out = {"args": vars(args), "patterns": {}}
    try:
        for label, gap in (("back_to_back", 0.0), (f"gap_{args.gap_ms}ms", args.gap_ms / 1e3)):
            us = []
            for i in range(args.iters):
                fill(args.decode_tokens)
                if gap:
                    time.sleep(gap)
                t0 = time.perf_counter()
                nat.gather(st, args.decode_tokens)
                us.append(1e6 * (time.perf_counter() - t0))
            out["patterns"][label] = summarize(us)
    finally:
        if qos is not None:
            os.close(qos)
    out["ok"] = True
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("check", "bench", "decode-probe"), default="check")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--chunk-tokens", type=int, default=8192)
    ap.add_argument("--prompt-tokens", type=int, default=65536)
    ap.add_argument("--compute-ms", type=float, default=8000.0, help="GPU sleep per chunk (its forward)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--decode-tokens", type=int, default=16)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--gap-ms", type=float, default=150.0)
    ap.add_argument("--pm-qos-us", type=int, default=None)
    args = ap.parse_args()
    if args.mode != "check" and not args.snapshot:
        ap.error("--snapshot is required for bench and decode-probe")
    for k in os.environ:
        if k.startswith("DSV41_ENGRAM_PREFILL_AHEAD"):
            raise SystemExit(f"unset {k}: the baked sitecustomize would wrap the stock stage")

    from vllm.models.deepseek_v4_1.common import engram as eng
    from vllm.models.deepseek_v4_1.common.mm_preprocess import IMAGE_PAD_ID, IMAGE_SENTINEL_BASE_ID

    dev = torch.device("cuda")
    dead = (IMAGE_SENTINEL_BASE_ID, IMAGE_PAD_ID)
    # No inference_mode here: the stager buffers must be normal tensors, as EngramDiskStager
    # makes them at model init (the stock stage's pool threads write rows_host outside
    # inference mode, which is thread-local); both stage() functions enter it themselves.
    out = {"check": mode_check, "bench": mode_bench, "decode-probe": mode_decode_probe}[args.mode](
        args, eng, dev, dead)
    out["torch"] = torch.__version__
    print(json.dumps(out))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
