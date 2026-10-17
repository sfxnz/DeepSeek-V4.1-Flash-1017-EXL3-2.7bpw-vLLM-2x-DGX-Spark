#!/usr/bin/env python3
"""K-bits routed-MoE microbench: p2b SORT=0 (v0) vs coop DSV41_P2B_COOP=1 (v1) vs dataflow =2 (v2)
for (K_gu, K_d) in {2, 3, 4}^2, on real layer-20 experts encoded at each K (encode_layer.py).

GPU only (spark2 via spark2.sh), in a serve image (JIT-builds build/bench_kbits.cu = the e15 p2b TU
plus kernel_study/p2b_coop/make_bench_r3's runtime variant switch). Data under /data:
  tensors/L<layer>_tp<rank>_K<K>.safetensors   (encode_layer.py; gate/up from K_gu, down from K_d)
  census_npy/                                  (the s10 decode routing census of the 40 routed layers)

  python3 kernel_study/p2b_kbits/bench_kbits.py --pairs 2:2,3:3,4:4,3:2 --mode check,capture,stress,time \
      --out /data/runs/<name>.json

The modes are kernel_study/p2b_coop/bench_r3.py's (same gates, fixed before any result):
  check    one-hot routing weights: every variant bitwise equal to v0 of the same (K_gu, K_d);
           real weights: max|v - v0| / max|v0| <= 1e-3, v2 bitwise equal to v1, repeat bitwise,
           finite; --ref adds the fp64 reference error of v0 and of each variant (exllamav3
           reconstruct at each projection's K).
  capture  CUDA graph per variant and m, replays after new ids / weights, each bitwise equal to eager.
  stress   v2 vs v1 on --stress-calls fresh routings per m, eager (L2 flush every 4th) and graph.
  time     cold (L2 flush) and warm per call, arms alternating per iteration, median / p10 / p90, GB/s
           over the unique experts' bytes at this (K_gu, K_d), paired savings.
--lb3 runs the bench-only build where the dataflow kernel keeps __launch_bounds__(256, 3) for every K
(before widen_p2b_kbits moved K != 2 to 2 blocks/SM; the first timing round, kb-time-lb2.json, ran the
opposite pairing: production at 3 blocks/SM and --lb2 at 2).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kernel_study/p2b_coop"))
sys.path.insert(0, str(HERE))

import bench_r3 as R  # noqa: E402  (pops DSV41_P2B_* from the env, imports torch)
import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

DATA = Path(os.environ.get("KB_DATA", "/data"))
R.CENSUS = DATA / "census_npy"
HIDDEN, INTER, EXPERTS, TOPK = R.HIDDEN, R.INTER, R.EXPERTS, R.TOPK
SWIGLU_LIMIT = R.SWIGLU_LIMIT
log = R.log


def build_ext(lb3: bool = False):
    import make_bench_kbits

    make_bench_kbits.main()
    name = "bench_kbits_lb3" if lb3 else "bench_kbits"
    return load(
        name=f"p2b_{name}",
        sources=[str(make_bench_kbits.OUT / f"{name}.cu")],
        extra_include_paths=[R.EXL, os.path.join(R.EXL, "quant")],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def expert_bytes(kgu: int, kd: int) -> int:
    """Bytes one unique expert costs per call on one TP rank: 3 trellis + suh/svh vectors."""
    words = (HIDDEN // 16) * (INTER // 16) * 16
    return 2 * words * kgu * 2 + words * kd * 2 + 3 * (HIDDEN + INTER) * 2


class Layer:
    """One routed layer's TP-rank slice from the encode_layer.py files, in the serve's param layout."""

    def __init__(self, layer: int, tp_rank: int, kgu: int, kd: int, dev: str) -> None:
        from safetensors import safe_open

        kt, nt = HIDDEN // 16, INTER // 16
        self.layer, self.tp_rank, self.kgu, self.kd = layer, tp_rank, kgu, kd
        self.w13 = torch.empty((EXPERTS, 2, kt, nt, 16 * kgu), dtype=torch.int16, device=dev)
        self.w2 = torch.empty((EXPERTS, nt, kt, 16 * kd), dtype=torch.int16, device=dev)
        self.w13_suh = torch.empty((EXPERTS, 2, HIDDEN), dtype=torch.half, device=dev)
        self.w13_svh = torch.empty((EXPERTS, 2, INTER), dtype=torch.half, device=dev)
        self.w2_suh = torch.empty((EXPERTS, INTER), dtype=torch.half, device=dev)
        self.w2_svh = torch.empty((EXPERTS, HIDDEN), dtype=torch.half, device=dev)
        marks = set()
        f13 = DATA / "tensors" / f"L{layer}_tp{tp_rank}_K{kgu}.safetensors"
        f2 = DATA / "tensors" / f"L{layer}_tp{tp_rank}_K{kd}.safetensors"
        with safe_open(str(f13), "pt") as fh:
            for e in range(EXPERTS):
                for j, w in ((0, "w1"), (1, "w3")):
                    self.w13[e, j].copy_(fh.get_tensor(f"{e}.{w}.trellis"))
                    self.w13_suh[e, j].copy_(fh.get_tensor(f"{e}.{w}.suh"))
                    self.w13_svh[e, j].copy_(fh.get_tensor(f"{e}.{w}.svh"))
                    marks.add(int(fh.get_tensor(f"{e}.{w}.mcg").item()))
            self.norm = fh.get_tensor("ffn_norm.weight").float().to(dev)
        with safe_open(str(f2), "pt") as fh:
            for e in range(EXPERTS):
                self.w2[e].copy_(fh.get_tensor(f"{e}.w2.trellis"))
                self.w2_suh[e].copy_(fh.get_tensor(f"{e}.w2.suh"))
                self.w2_svh[e].copy_(fh.get_tensor(f"{e}.w2.svh"))
                marks.add(int(fh.get_tensor(f"{e}.w2.mcg").item()))
        if marks != {R.MCG_MARK}:
            raise SystemExit(f"expected MCG markers only, got {marks}")

        def ptrs(ts):
            return torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)

        E = range(EXPERTS)
        self.tables = [
            ptrs([self.w13[e, 0] for e in E]), ptrs([self.w13_suh[e, 0] for e in E]), ptrs([self.w13_svh[e, 0] for e in E]),
            ptrs([self.w13[e, 1] for e in E]), ptrs([self.w13_suh[e, 1] for e in E]), ptrs([self.w13_svh[e, 1] for e in E]),
            ptrs([self.w2[e] for e in E]), ptrs([self.w2_suh[e] for e in E]), ptrs([self.w2_svh[e] for e in E]),
        ]


class Bench(R.Bench):
    def raw(self, x, out, ids, rw) -> None:
        L = self.L
        self.ext.p2b_fused_moe(x, out, *L.tables, ids, rw, L.kgu, L.kgu, L.kd, True, INTER, SWIGLU_LIMIT)

    def call(self, v: int, x, ids, rw, out):
        self.ext.set_variant(v)
        self.raw(x, out, ids, rw)
        return out


class Reference(R.Reference):
    """fp64 MoE from the decoded weights, each projection at its own K (trellis words / 16)."""

    def _w(self, trellis, k, n):
        key = trellis.data_ptr()
        if key not in self.cache:
            w = torch.empty((k, n), dtype=torch.half, device=trellis.device)
            self.X.reconstruct(w, trellis, int(trellis.shape[-1]) // 16, True, False)
            self.cache[key] = w.double()
        return self.cache[key]


def capture(b: Bench, routes, variants, ms, replays: int) -> dict:
    res = {}
    for v in variants:
        if v == 0:
            continue  # p2b's slot sum is atomicAdd: not bitwise run to run (bench_r3 skips it too)
        for m in ms:
            x, ids, rw = b.x(m), b.ids(routes.draw("census", m)), b.rw(m)
            out = torch.empty_like(x)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    b.call(v, x, ids, rw, out)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            b.ext.set_variant(v)
            with torch.cuda.graph(g):
                b.raw(x, out, ids, rw)
            ok = True
            for _ in range(replays):
                ids.copy_(b.ids(routes.draw("census", m)))
                rw.copy_(b.rw(m))
                g.replay()
                torch.cuda.synchronize()
                ok &= R.bitwise(out.clone(), b.once(v, x, ids, rw))
            res[f"v{v}/m{m}"] = {"replays": replays, "replay_equals_eager": ok}
            log(f"[capture] v{v} m={m}: replay_equals_eager={ok} ({replays} replays)")
            del g
    return res


def stress(b: Bench, routes, variants, ms, calls: int, flush) -> dict:
    res = {}
    for v in variants:
        if v < 2:
            continue
        for m in ms:
            x, rw = b.x(m), b.rw(m)
            out1, out2, gout = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
            ids = b.ids(routes.draw("census", m))
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    b.call(v, x, ids, rw, gout)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            b.ext.set_variant(v)
            with torch.cuda.graph(g):
                b.raw(x, gout, ids, rw)
            eager_bad = graph_bad = 0
            for i in range(calls):
                ids.copy_(b.ids(routes.draw("census", m)))
                if i % 4 == 0:
                    flush()
                b.call(1, x, ids, rw, out1)
                b.call(v, x, ids, rw, out2)
                g.replay()
                torch.cuda.synchronize()
                eager_bad += not R.bitwise(out1, out2)
                graph_bad += not R.bitwise(out1, gout)
            res[f"v{v}/m{m}"] = {"calls": calls, "eager_mismatch": eager_bad, "graph_mismatch": graph_bad}
            log(f"[stress] v{v} m={m}: {calls} calls, eager mismatches {eager_bad}, graph mismatches {graph_bad}")
            del g
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="2:2,3:3,4:4,3:2,2:3,3:4,4:3,4:2,2:4", help="K_gu:K_d list")
    ap.add_argument("--mode", default="check,capture,stress,time")
    ap.add_argument("--variants", default="0,1,2")
    ap.add_argument("--ms", default="1,3,4,6,8")
    ap.add_argument("--sources", default="census")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--tp-rank", type=int, default=0)
    ap.add_argument("--check-routings", type=int, default=8)
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--replays", type=int, default=32)
    ap.add_argument("--stress-calls", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lb3", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    modes = set(args.mode.split(","))
    variants = [int(v) for v in args.variants.split(",")]
    if variants[0] != 0:
        variants = [0] + [v for v in variants if v != 0]
    ms = [int(m) for m in args.ms.split(",")]
    sources = args.sources.split(",")
    pairs = [tuple(int(k) for k in p.split(":")) for p in args.pairs.split(",")]
    dev = "cuda"
    props = torch.cuda.get_device_properties(0)
    t0 = time.time()
    ext = build_ext(args.lb3)
    log(f"extension ready in {time.time() - t0:.1f} s (lb3={args.lb3})")
    flush = R.Flusher(dev)
    result: dict = {"device": {"name": props.name, "sms": props.multi_processor_count, "l2_bytes": props.L2_cache_size},
                    "args": vars(args) | {"out": str(args.out)}, "variants": variants,
                    "tolerance": {"onehot": "bitwise vs v0", "full_max_rel": R.REL_TOL, "v2_vs_v1": "bitwise",
                                  "repeat": "bitwise", "capture": "bitwise", "stress": "bitwise"},
                    "pairs": {}}
    ok = True
    for kgu, kd in pairs:
        torch.manual_seed(args.seed)
        rng = random.Random(args.seed)
        t0 = time.time()
        layer = Layer(args.layer, args.tp_rank, kgu, kd, dev)
        log(f"=== K_gu={kgu} K_d={kd}: loaded layer {args.layer} rank {args.tp_rank} in {time.time() - t0:.1f} s")
        routes = R.Routings(args.layer, rng)
        b = Bench(ext, layer, dev, args.seed)
        R.EXPERT_BYTES = expert_bytes(kgu, kd)  # R.timing / R.stats read it per call
        entry: dict = {"expert_bytes": R.EXPERT_BYTES, "census_captures": len(routes.caps)}
        launch = {}
        for v in variants:
            x = b.x(4)
            b.call(v, x, b.ids(routes.draw("census", 4)), b.rw(4), torch.empty_like(x))
            launch[f"v{v}"] = {"grid": ext.last_grid(), "blocks_per_sm": ext.last_grid() / props.multi_processor_count}
        entry["launch"] = launch
        log(f"[setup] K_gu={kgu} K_d={kd} launch {json.dumps(launch)}")
        if "check" in modes:
            ref = Reference(layer) if args.ref else None
            entry["check"] = R.check(b, routes, variants, ms, sources + (["census_seq"] if max(ms) > 4 else []),
                                     args.check_routings, ref)
        if "capture" in modes:
            entry["capture"] = capture(b, routes, variants, ms, args.replays)
        if "stress" in modes:
            entry["stress"] = stress(b, routes, variants, ms, args.stress_calls, flush)
        if "time" in modes:
            entry["time"] = R.timing(b, routes, variants, ms, sources, args.iters, args.warmup, flush)
        p_ok = all(r["pass"] for r in entry.get("check", {}).values()) and all(
            r["replay_equals_eager"] for r in entry.get("capture", {}).values()) and all(
            r["eager_mismatch"] == 0 and r["graph_mismatch"] == 0 for r in entry.get("stress", {}).values())
        entry["pass"] = p_ok
        ok &= p_ok
        result["pairs"][f"{kgu}:{kd}"] = entry
        log(f"=== K_gu={kgu} K_d={kd}: pass={p_ok}")
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result | {"pass": ok, "partial": True}, indent=1) + "\n")
        del b, layer
        torch.cuda.empty_cache()
    result["pass"] = ok
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    log(f"pass={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
