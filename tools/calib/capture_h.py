#!/usr/bin/env python3
"""Hessian capture (dsv41-hessian-v1, decision B-3) over the native stream, with stream checkpoints (B-4).

Runs the native layer-streaming forward (native.py) over calib.jsonl and writes, for each layer in
--layers, every selected expert's w13_in / w2_in in the builder's format (hooks.HessianV1). The forward
starts at layer 0, or at L+1 when --state holds a checkpoint taken after layer L, runs through
--stop-after, then saves the stream state (per row: hyper-connection stream h, pre_mix, Engram hashes and
the cross-layer attention state r.shared) as a new checkpoint and exits. A resumed run continues
bit-identically: same rows, same per-layer seeds, same GEMM shapes.

  python3 tools/calib/capture_h.py --out HESS --state STATE --layers 1,3 --stop-after 3
      [--experts all|strat16|0-7|1,5] [--stats S.npz] [--dump-factors N --factors-out D]
      [--holdout-mod 5 --train-out D] [--limit ROWS] [--fresh] [--summary J]

Checkpoint layout: STATE/ckpt-LL/row_RRRR.safetensors + meta.json; STATE/CURRENT.json names the newest
(written last, atomically). Older checkpoints this tool wrote are removed after the switch.
Run inside dsv41-quant151 (GPU). Full calib in one pass (spark2, data/dsv41-1017/hess/RUNS.txt): peak GPU
62.5 GiB, MemAvailable >= 36 GiB; ~40 s per layer (Engram layers ~150 s) + ~30 s per captured layer
(48.4 GB written); checkpoint 17 GB, saved in ~15 s, loaded in ~5 s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cli  # noqa: E402
from hooks import HESSIAN_FORMAT, HessianV1, RoutingStats, drop_page_cache, save_safetensors  # noqa: E402
from native import Hooks  # noqa: E402

CALIB = Path("/home/sfxnz/projects/data/dsv41-1017/calib/calib.jsonl")
CKPT_FORMAT = "dsv41-stream-ckpt-v1"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return float("nan")


def sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def code_hash() -> str:
    """Numerics-relevant harness code: a resume must run the same forward as the segment before it."""
    h = hashlib.sha256()
    for name in ("native.py", "refkernel.py", "stio.py"):
        h.update((HERE / name).read_bytes())
    return h.hexdigest()[:16]


def parse_experts(s: str):
    if s in ("all", ""):
        return None
    if s == "strat16":
        return "strat16"
    return set(cli.parse_layers(s, 384))


class Fan(Hooks):
    """Fan out to several hooks; per-expert calls only to hooks that want the layer (as cli.Multi)."""

    def __init__(self, hooks):
        self.h = hooks

    def begin(self, hz, rows):
        for h in self.h:
            h.begin(hz, rows)

    def routing(self, *a):
        for h in self.h:
            h.routing(*a)

    def wants_expert(self, layer):
        return any(h.wants_expert(layer) for h in self.h)

    def prior_tokens(self, layer):
        for h in self.h:
            if h.wants_expert(layer):
                p = h.prior_tokens(layer)
                if p is not None:
                    return p
        return None

    def _per(self, name, layer, *a):
        for h in self.h:
            if h.wants_expert(layer):
                getattr(h, name)(layer, *a)

    def layer_tokens(self, layer, *a):
        self._per("layer_tokens", layer, *a)

    def expert_prior(self, layer, *a):
        self._per("expert_prior", layer, *a)

    def expert(self, layer, *a):
        self._per("expert", layer, *a)

    def end_layer(self, layer):
        for h in self.h:
            h.end_layer(layer)

    def finish(self):
        for h in self.h:
            h.finish()


# ------------------------------------------------------------------------------------- checkpoints

def fingerprint(a, rows) -> dict:
    return {"calib": str(Path(a.calib).resolve()), "calib_sha256": a.calib_sha, "limit": a.limit,
            "rows": len(rows), "tokens": sum(r.n for r in rows), "row_ids": hashlib.sha256(
                "\n".join(r.id for r in rows).encode()).hexdigest()[:16],
            "ckpt": Path(a.ckpt).name, "engram": not a.no_engram, "code": code_hash()}


class Staging:
    """One growing pinned host buffer for stream checkpoint I/O. Pageable D2H/H2D copies of the stream
    measured ~160-180 MB/s on the GB10 inside the capture (17 GB checkpoint in 93-107 s); pinned is DMA."""

    def __init__(self):
        self.buf = None

    def views(self, tensors: dict) -> dict:
        """Pinned CPU views (64-byte aligned slots) shaped like `tensors`; valid until the next call."""
        offs, n = {}, 0
        for k, v in tensors.items():
            offs[k] = n
            n += (v.numel() * v.element_size() + 63) // 64 * 64
        if self.buf is None or self.buf.numel() < n:
            self.buf = torch.empty(max(n, 1 << 30), dtype=torch.uint8, pin_memory=True)
        return {k: self.buf[offs[k] : offs[k] + v.numel() * v.element_size()].view(v.dtype).view(v.shape)
                for k, v in tensors.items()}


def current(state: Path) -> dict | None:
    p = state / "CURRENT.json"
    return json.loads(p.read_text()) if p.is_file() else None


def save_checkpoint(state: Path, layer_done: int, rows, fp: dict) -> dict:
    t0 = time.time()
    name = f"ckpt-{layer_done:02d}"
    tmp = state / f".{name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    files, nbytes = {}, 0
    stage = Staging()
    for i, r in enumerate(rows):
        t = {"h": r.h, "pre_mix": r.pre_mix}
        if r.hashes is not None:
            t["hashes"] = r.hashes
        for k, v in r.shared.items():
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"row {r.id}: shared[{k}] is {type(v).__name__}, not a tensor")
            t[f"shared.{k}"] = v
        pv = stage.views(t)
        for k, v in t.items():
            pv[k].copy_(v)
        t = pv
        f = f"row_{i:04d}.safetensors"
        save_safetensors(tmp / f, t, {"id": r.id})
        files[f] = (tmp / f).stat().st_size
        nbytes += files[f]
    meta = {"format": CKPT_FORMAT, "layer_done": layer_done, "next_layer": layer_done + 1, "fingerprint": fp,
            "row_ids": [r.id for r in rows], "files": files, "bytes": nbytes,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    final = state / name
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    cur = state / ".CURRENT.json.tmp"
    cur.write_text(json.dumps({"dir": name, "layer_done": layer_done, "fingerprint": fp}, indent=1) + "\n")
    os.replace(cur, state / "CURRENT.json")
    for old in state.glob("ckpt-*"):  # only checkpoints this tool wrote live here
        if old.name != name:
            shutil.rmtree(old, ignore_errors=True)
    log(f"checkpoint after layer {layer_done}: {final} {nbytes / 1e9:.2f} GB in {time.time() - t0:.0f}s")
    return meta


def load_checkpoint(state: Path, rows, fp: dict, dev, allow_code_change=False) -> int:
    from safetensors import safe_open
    cur = current(state)
    d = state / cur["dir"]
    meta = json.loads((d / "meta.json").read_text())
    want = dict(fp)
    have = dict(meta["fingerprint"])
    if allow_code_change:
        want.pop("code"), have.pop("code")
    if have != want:
        raise SystemExit(f"checkpoint {d} fingerprint {have} != this run {want}")
    if meta["row_ids"] != [r.id for r in rows]:
        raise SystemExit("checkpoint row order differs")
    t0 = time.time()
    stage = Staging()
    for i, r in enumerate(rows):
        p = d / f"row_{i:04d}.safetensors"
        with safe_open(str(p), framework="pt", device="cpu") as fh:
            cpu = {k: fh.get_tensor(k) for k in fh.keys()}
        pv = stage.views(cpu)
        g = {}
        for k, v in cpu.items():
            pv[k].copy_(v)
            g[k] = pv[k].to(dev)  # synchronous: the staging buffer is reused by the next row
        del cpu, v  # get_tensor is mmap-backed: a live tensor keeps the file mapped, the copies re-fault pages
        drop_page_cache(p)
        r.h, r.pre_mix, r.hashes = g["h"], g["pre_mix"], g.get("hashes")
        r.shared = {k[len("shared."):]: v for k, v in g.items() if k.startswith("shared.")}
    log(f"resumed from {d} (after layer {meta['layer_done']}) in {time.time() - t0:.0f}s")
    return int(meta["layer_done"])


# ----------------------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(cli.SNAP), help="1017 snapshot dir")
    ap.add_argument("--index", default=None)
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--attn-chunk", type=int, default=256)
    ap.add_argument("--cfg-override", default=None)
    ap.add_argument("--calib", default=str(CALIB))
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (tests)")
    ap.add_argument("--out", help="dsv41-hessian-v1 root (layer_LL/...)")
    ap.add_argument("--layers", default="", help="layers to capture, e.g. 1,3 or 0-3; empty = forward only")
    ap.add_argument("--experts", default="all", help="all | strat16 | list/ranges of expert ids")
    ap.add_argument("--wpow", type=float, default=2.0)
    ap.add_argument("--m0-mult", type=float, default=256.0)
    ap.add_argument("--prior-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-factors", type=int, default=0, help="routed tokens kept per selected expert (r_eff)")
    ap.add_argument("--factors-out")
    ap.add_argument("--holdout-mod", type=int, default=0)
    ap.add_argument("--train-out")
    ap.add_argument("--stats", help="routing stats .npz (layers forwarded in this run)")
    ap.add_argument("--state", required=True, help="stream checkpoint dir")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint (start at layer 0)")
    ap.add_argument("--stop-after", type=int, default=None, help="last layer to run (default: max of --layers)")
    ap.add_argument("--no-save", action="store_true", help="do not write a checkpoint at the end")
    ap.add_argument("--allow-code-change", action="store_true")
    ap.add_argument("--summary", help="JSON summary path")
    a = ap.parse_args(argv)
    log("argv", sys.argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    layers = cli.parse_layers(a.layers) if a.layers else []
    stop = a.stop_after if a.stop_after is not None else (max(layers) if layers else None)
    if stop is None:
        ap.error("--stop-after or --layers needed")
    state = Path(a.state)
    state.mkdir(parents=True, exist_ok=True)
    a.calib_sha = sha256_path(Path(a.calib))
    t_all = time.time()
    with torch.inference_mode():
        hz = cli.make_harness(a)
        rows = cli.rows_from_jsonl(hz, a.calib, a.limit, score_all=True)
        fp = fingerprint(a, rows)
        cur = None if a.fresh else current(state)
        start = 0 if cur is None else cur["layer_done"] + 1
        if any(L < start or L > stop for L in layers):
            raise SystemExit(f"--layers {layers} outside this run's range {start}..{stop}")
        if start > stop:
            raise SystemExit(f"checkpoint is after layer {start - 1} >= --stop-after {stop}")
        info = {"calib": fp["calib"], "calib_sha256": a.calib_sha, "calib_rows": fp["rows"],
                "code": fp["code"], "snapshot": fp["ckpt"]}
        hooks = []
        hess = None
        if layers:
            if not a.out:
                ap.error("--out needed with --layers")
            hess = HessianV1(layers, a.out, experts=parse_experts(a.experts), wpow=a.wpow, m0_mult=a.m0_mult,
                             prior_tokens=a.prior_tokens, seed=a.seed, dump_factors=a.dump_factors,
                             factors_out=a.factors_out, holdout_mod=a.holdout_mod, train_out=a.train_out,
                             info=info, log=log)
            hooks.append(hess)
        if a.stats:
            hooks.append(RoutingStats(a.stats))
        fan = Fan(hooks)
        log(f"{len(rows)} rows, {fp['tokens']} tokens; layers {start}..{stop}; capture {layers}; "
            f"MemAvailable {mem_available_gib():.1f} GiB")
        hz.pref.submit(start, hz.layer_names(start))
        if start == 0:
            hz.start(rows, fan)
        else:
            load_checkpoint(state, rows, fp, hz.dev, a.allow_code_change)
            hz.dom_ids = {}
            for r in rows:
                hz.dom_ids.setdefault(r.domain, len(hz.dom_ids))
            hz.domains = list(hz.dom_ids)
            fan.begin(hz, rows)
        per_layer = {}
        for layer in range(start, stop + 1):
            tl = time.time()
            W = hz.load_layer(layer)
            hz.layer_forward(layer, W, rows, fan)
            del W
            per_layer[layer] = {"s": round(time.time() - tl, 1), "mem_avail_gib": round(mem_available_gib(), 1),
                                "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            log(f"[layer {layer:2d}] {per_layer[layer]['s']:6.1f}s peak={per_layer[layer]['peak_gpu_gib']}GiB "
                f"MemAvailable={per_layer[layer]['mem_avail_gib']}GiB")
        fan.finish()
        hz.pref.pending.clear()  # the prefetch of stop+1 is not needed
        ck = None if a.no_save else save_checkpoint(state, stop, rows, fp)
    out = {"start": start, "stop": stop, "layers": layers, "tokens": fp["tokens"], "rows": fp["rows"],
           "wall_s": round(time.time() - t_all, 1), "timing_s": {k: round(v, 1) for k, v in hz.timing.items()},
           "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "per_layer": per_layer,
           "checkpoint": None if ck is None else {"layer_done": ck["layer_done"], "bytes": ck["bytes"]},
           "hessian_hook_s": None if hess is None else round(hess.t_hess, 1), "format": HESSIAN_FORMAT}
    if a.summary:
        Path(a.summary).write_text(json.dumps(out, indent=1) + "\n")
    log("done", json.dumps(out)[:800])


if __name__ == "__main__":
    main()
