#!/usr/bin/env python3
"""b12x MXFP8 dense GEMM at 9..24 rows: the e15 tile vs the m <= 8 decode tiles ("rows16"), real weights.

e15-kbits picks (32, 128) for 8 < m <= 128 (MXFP8, n > 1536); the rows16 candidate keeps the m <= 8 tiles,
(16, 64) for n <= 8192 else (16, 128), up to m = 16. Both selections are produced here by swapping FlashInfer's
_select_default_mma_tiler_mn (no image patch needed). Result 2026-10-19 (spark2, 1017 weights): bitwise
everywhere, rows16 within -2.4..+2.0 % of the e15 tile at M 12..24, so the candidate patch was dropped; the
GEMV in <= 8-row slices loses 19-65 % above 8 rows (kernel_study/p2b_rows/README.txt).

Correctness (gate, fixed up front), per shape, M and activation distribution (correctness.DISTS), on real
weights: b12x at the rows16 tile == b12x at the e15 tile == b12x on the same rows in <= 8-row slices (the
m <= 8 path) bitwise (bf16 as int16); for the shapes the production dense GEMV serves, the GEMV in <= 8-row
slices == the same. CUDA-graph replay of the rows16 path == eager.
Timing (benchutil protocol: rotation over >= 512 MiB of real-layer copies, double hashed 64 MiB flush before
every cold call, ABBA, --iters timed calls): b12x_e15 and b12x_rows16 (FlashInfer quant + mm_mxfp8, the
serve's bf16-input path), gemv_slices (the production GEMV in <= 8-row launches; at m <= 8 the one launch).

  kernel_study/p2b_kbits/spark2.sh rows16 dsv41-flash-exl3-sm121:e15-kbits kernel_study/dense_gemv/rows16_check.py \
      --pack /hf1017/snapshots/2.7bpw-mcg-hq --out /data/runs/rows16.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "docker", "patch"))
import weights  # noqa: E402
from benchutil import Flusher, Rotation, copies_for, load_ext, stats, time_arms  # noqa: E402
from correctness import DISTS, b12x_ref, make_x, unswizzle  # noqa: E402
from timing import build_copies  # noqa: E402

SHAPES = "qkv_a,wq_b,wo_b,shared_gate_up,shared_down,engram_wkv,main_proj,lm_head"


def tilers():
    import flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x as bk

    orig = getattr(bk._select_default_mma_tiler_mn, "_c4_orig", bk._select_default_mma_tiler_mn)

    def e15(m, n, sm_count, *, is_mxfp8=False, expected_m=None, k=None):
        if is_mxfp8 and n > 1536 and expected_m is not None and 8 < expected_m <= 128:
            return (32, 128)
        return orig(m, n, sm_count, is_mxfp8=is_mxfp8, expected_m=expected_m, k=k)

    def rows16(m, n, sm_count, *, is_mxfp8=False, expected_m=None, k=None):
        if is_mxfp8 and n > 1536 and expected_m is not None and 8 < expected_m <= 16:
            return (16, 64) if n <= 8192 else (16, 128)
        return orig(m, n, sm_count, is_mxfp8=is_mxfp8, expected_m=expected_m, k=k)

    for f in (e15, rows16):
        f._c4_orig = orig

    @contextlib.contextmanager
    def using(fn):
        saved = bk._select_default_mma_tiler_mn
        bk._select_default_mma_tiler_mn = fn
        try:
            yield
        finally:
            bk._select_default_mma_tiler_mn = saved

    return {"e15": e15, "rows16": rows16}, using


def gemv_for(name: str, N: int, K: int, s2d: torch.Tensor):
    """The production dense GEMV (docker/patch/dense_gemv.py) armed for (K, N), or None if it does not serve it."""
    import dense_gemv as dg

    cfg = dg.CONFIGS.get((K, N))
    if cfg is None:
        return None
    _, kc, smode, buckets = cfg
    ext = dg._ext()
    plans = [(max_m, w, s, mr, int(ext.plan_grid(w, s, kc, mr, smode, 0, N, K))) for max_m, w, s, mr in buckets]
    return dg._Armed((K, N), name, kc, smode, dg.build_scales(s2d, N, K, kc, smode), plans)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default=SHAPES)
    ap.add_argument("--ms", default="1,4,8,9,12,16,17,20,24")
    ap.add_argument("--time-ms", default="8,12,16,24")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--min-total-mib", type=int, default=512)
    ap.add_argument("--pack", default=weights.PACK, help="MXFP8 dense weights (any V4.1 EXL3 pack: same shapes)")
    args = ap.parse_args()
    weights.PACK = args.pack
    import dense_gemv as dg

    ext = load_ext("dgemv_v1", ["gemv_ext.cu"])  # flush kernels
    flusher = Flusher(ext)
    tiles, using = tilers()
    gen = torch.Generator(device="cuda").manual_seed(16)
    res = {"protocol": "rotation+double-hashed-flush, eager events, ABBA", "shapes": {}, "bad": []}
    for name in [s for s in args.shapes.split(",") if s]:
        kind, N, K, _ = weights.SHAPES[name]
        R = copies_for(N * K, min_total=args.min_total_mib << 20)
        copies = build_copies(name, R)
        w0, wsw0, _ = copies[0]
        s2d = unswizzle(wsw0, N, K // 32).contiguous()
        armed = gemv_for(name, N, K, s2d)
        entry = {"N": N, "K": K, "copies": R, "gemv": armed is not None, "M": {}}

        def b12x(x, c, tile):
            with using(tiles[tile]):
                return b12x_ref(x, c[0].view(torch.float8_e4m3fn), c[1])[2]

        def gemv(x, c):
            layer = SimpleNamespace(weight=c[0].view(torch.float8_e4m3fn))
            return torch.cat([dg._run(armed, layer, x2d=x[i : i + 8]) for i in range(0, x.shape[0], 8)])

        for M in [int(m) for m in args.ms.split(",")]:
            row = {"bitwise": True}
            for dist in DISTS:
                x = make_x(M, K, dist, gen)
                y16 = b12x(x, copies[0], "rows16")
                y15 = b12x(x, copies[0], "e15")
                ys = torch.cat([b12x(x[i : i + 8], copies[0], "e15") for i in range(0, M, 8)])
                ok = torch.equal(y16.view(torch.int16), y15.view(torch.int16)) and torch.equal(
                    y16.view(torch.int16), ys.view(torch.int16))
                if armed is not None and armed.plan(min(M, 8)) is not None:
                    ok &= torch.equal(gemv(x, copies[0]).view(torch.int16), ys.view(torch.int16))
                if not ok:
                    res["bad"].append(f"{name} M={M} {dist}")
                row["bitwise"] &= ok
            if 8 < M <= 16:  # graph replay of the rows16 path
                x = make_x(M, K, "normal", gen)
                with using(tiles["rows16"]):
                    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
                    from vllm.utils import flashinfer as vfi

                    def call():
                        q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
                        return vfi.mm_mxfp8(q, w0.view(torch.float8_e4m3fn).t(), s, wsw0, out_dtype=torch.bfloat16,
                                            backend="auto")

                    eager = call()
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        gy = call()
                    x.copy_(make_x(M, K, "outlier", gen))
                    g.replay()
                    eager = call()
                    torch.cuda.synchronize()
                    row["graph_replay_bitwise"] = bool(torch.equal(gy.view(torch.int16), eager.view(torch.int16)))
                    if not row["graph_replay_bitwise"]:
                        res["bad"].append(f"{name} M={M} graph")
                    del g
            if M in [int(m) for m in args.time_ms.split(",")]:
                x = make_x(M, K, "normal", gen)
                fns = {"b12x_e15": lambda c: b12x(x, c, "e15"), "b12x_rows16": lambda c: b12x(x, c, "rows16")}
                if armed is not None and armed.plan(min(M, 8)) is not None:
                    fns["gemv_slices"] = lambda c: gemv(x, c)
                names = list(fns)
                rot = Rotation(copies, narms=len(names))
                arms = {n: (lambda n=n, k=k: fns[n](rot.get(k))) for k, n in enumerate(names)}
                cold = time_arms(arms, iters=args.iters, pre=flusher)
                wbytes = N * K
                row["cold_us"] = {n: stats(v) for n, v in cold.items()}
                row["cold_weight_GBps"] = {n: wbytes / row["cold_us"][n]["median"] / 1e3 for n in names}
                base = row["cold_us"]["b12x_e15"]["median"]
                print(f"{name:14s} N={N:5d} K={K:5d} M={M:2d} bitwise={row['bitwise']} cold us: " + "  ".join(
                    f"{n} {row['cold_us'][n]['median']:8.2f} ({row['cold_weight_GBps'][n]:5.0f} GB/s, "
                    f"{100 * (base / row['cold_us'][n]['median'] - 1):+5.1f}%)" for n in names), flush=True)
            else:
                print(f"{name:14s} N={N:5d} K={K:5d} M={M:2d} bitwise={row['bitwise']}", flush=True)
            entry["M"][M] = row
        res["shapes"][name] = entry
        del copies
        torch.cuda.empty_cache()
    res["pass"] = not res["bad"]
    json.dump(res, open(args.out, "w"), indent=1)
    print("rows16_check:", "OK" if res["pass"] else f"FAIL {res['bad'][:8]}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
