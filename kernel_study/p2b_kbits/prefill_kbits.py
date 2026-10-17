#!/usr/bin/env python3
"""Prefill routed-MoE throughput and correctness per (K_gu, K_d) through the installed plugin.

GPU only, in the e15-kbits image (its vllm_exl3/exl3.py carries widen_p2b_kbits). One layer-20 TP-rank
slice from /data/tensors (encode_layer.py), held as LinearEXL3 handles exactly like
Exl3MoEMethod.process_weights_after_loading builds them, then vllm_exl3.exl3.apply_exl3_fused_moe:
rows > 8 skip native p2b, so this is exllamav3 exl3_moe (vllm_exl3_c.exl3_moe_kmix when K_gu != K_d;
arms k0_* force exllamav3's runtime-K K = 0 instance instead)
plus the fat path for experts with > VLLM_EXL3_FAT_THRESHOLD (256) rows (exl3_fat_gemm per
projection when that K is 4, else reconstruct + hgemm).

Routing: m token rows drawn with replacement from the layer's own s10 decode census (real expert
popularity skew). Activations: RMS-normalised Gaussian x ffn_norm weight. Routing weights as bench_r3.

  check   at m = 512: fused (default threshold) and fused with every routed expert fat (threshold 0)
          against the per-expert LinearEXL3 loop (apply_exl3_python_loop: each projection's own K,
          an independent code path), max|d| / max|ref| (gate 1e-2: different fp16 GEMM paths).
  time    m in --ms: cold-ish (one call per iteration, nothing flushed: 2.6+ GB of experts per call is
          far beyond L2), median / p10 / p90 ms and tokens/s; arms: default threshold, no fat path.

  python3 kernel_study/p2b_kbits/prefill_kbits.py --pairs 2:2,3:3,4:4 --ms 512,2048,8192 --out /data/runs/x.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kernel_study/p2b_coop"))
sys.path.insert(0, str(HERE))

import bench_r3 as R  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

DATA = Path(os.environ.get("KB_DATA", "/data"))
HIDDEN, INTER, EXPERTS, TOPK = R.HIDDEN, R.INTER, R.EXPERTS, R.TOPK
log = R.log


def load_layer(P, layer: int, tp_rank: int, kgu: int, kd: int, dev: str):
    from safetensors import safe_open

    mark = torch.tensor(R.MCG_MARK, dtype=torch.int32, device=dev)
    raw = {}
    for k, ws in ((kgu, ("w1", "w3")), (kd, ("w2",))):
        with safe_open(str(DATA / "tensors" / f"L{layer}_tp{tp_rank}_K{k}.safetensors"), "pt") as fh:
            for e in range(EXPERTS):
                for w in ws:
                    raw[(e, w)] = tuple(fh.get_tensor(f"{e}.{w}.{s}").to(dev) for s in ("trellis", "suh", "svh"))
            if kgu == k:
                norm = fh.get_tensor("ffn_norm.weight").float().to(dev)
    inners = []
    for e in range(EXPERTS):
        pack = {}
        for name, w in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
            t, su, sv = raw[(e, w)]
            pack[name] = P.make_linear_exl3(t, su, sv, mark, None)
        inners.append(pack)
    lay = SimpleNamespace(
        w13_trellis=raw[(0, "w1")][0], _exl3_hidden_size=HIDDEN, _exl3_intermediate_local=INTER,
        _exl3_bits=kgu, _exl3_bits_d=kd, _exl3_codebook_flags=(True, False, True, False, True, False),
        expert_map=None,
    )
    P.build_exl3_fused_state(lay, inners)
    assert (lay._exl3_k, lay._exl3_k_d) == (kgu, kd), (lay._exl3_k, lay._exl3_k_d)
    for pack, (e, w) in ((inners[0]["gate"], (0, "w1")), (inners[0]["down"], (0, "w2"))):
        assert int(pack.K) == (kgu if w == "w1" else kd)
    return lay, inners, norm


def routing(census_rows: np.ndarray, m: int, rng: random.Random, dev: str) -> torch.Tensor:
    idx = [rng.randrange(census_rows.shape[0]) for _ in range(m)]
    return torch.tensor(census_rows[idx], dtype=torch.long, device=dev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="2:2,3:3,4:4,3:2,2:3,3:4,4:3,4:2,2:4")
    ap.add_argument("--ms", default="512,2048,8192")
    ap.add_argument("--mode", default="check,time")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--tp-rank", type=int, default=0)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--thresholds", default="", help="extra fat-threshold arms, e.g. 128,64,32")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    import vllm_exl3.exl3 as P

    dev = "cuda"
    modes = set(args.mode.split(","))
    ms = [int(m) for m in args.ms.split(",")]
    R.CENSUS = DATA / "census_npy"
    census = np.concatenate(R.Routings(args.layer, random.Random(args.seed)).caps, 0)
    limit = R.SWIGLU_LIMIT
    default_fat = P.FAT_EXPERT_THRESHOLD
    result = {"args": vars(args) | {"out": str(args.out)}, "fat_threshold_default": default_fat,
              "temp_rows_fused": P.TEMP_ROWS_FUSED, "census_rows": int(census.shape[0]), "pairs": {}}
    ok = True
    for pair in args.pairs.split(","):
        kgu, kd = (int(k) for k in pair.split(":"))
        rng = random.Random(args.seed)
        gen = torch.Generator(device=dev)
        gen.manual_seed(args.seed)
        t0 = time.time()
        lay, inners, norm = load_layer(P, args.layer, args.tp_rank, kgu, kd, dev)
        log(f"=== K_gu={kgu} K_d={kd}: built {len(inners)} experts in {time.time() - t0:.1f} s")
        entry: dict = {}

        def x_of(m):
            h = torch.randn(m, HIDDEN, device=dev, generator=gen)
            return ((h / h.pow(2).mean(dim=1, keepdim=True).sqrt()) * norm).to(torch.bfloat16)

        def rw_of(m):
            z = torch.randn(m, TOPK, device=dev, generator=gen)
            s = torch.nn.functional.softplus(z).sqrt()
            return s / s.sum(dim=1, keepdim=True) * R.ROUTED_SCALE

        native = P._load_native_exl3_ext()

        class _NoKmix:  # the installed vllm_exl3_c without exl3_moe_kmix: mixed K runs exl3_moe's K = 0
            def __getattr__(self, name):
                if name == "exl3_moe_kmix":
                    raise AttributeError(name)
                return getattr(native, name)

        def fused(x, ids, w, fat, k0=False):
            P.FAT_EXPERT_THRESHOLD = fat
            if k0:
                P._load_native_exl3_ext = lambda: _NoKmix()
            try:
                return P.apply_exl3_fused_moe(x, ids, w, lay, inners, None, limit)
            finally:
                P.FAT_EXPERT_THRESHOLD = default_fat
                P._load_native_exl3_ext = lambda: native

        if "check" in modes:
            m = 512
            x, ids, w = x_of(m), routing(census, m, rng, dev), rw_of(m)
            ref = P.apply_exl3_python_loop(x, ids, w, inners, None, limit).double()
            chk = {}
            mixed = kgu != kd and hasattr(native, "exl3_moe_kmix")
            cases = [("exl3_moe_plus_default_fat", default_fat, False), ("all_fat", 0, False)]
            if mixed:
                cases.append(("exl3_moe_k0_plus_default_fat", default_fat, True))
            for name, fat, k0 in cases:
                y = fused(x, ids, w, fat, k0).double()
                rel = float((y - ref).abs().max() / ref.abs().max())
                rms = float((y - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
                chk[name] = {"max_rel_vs_loop": rel, "rms_rel_vs_loop": rms, "finite": bool(torch.isfinite(y).all())}
                ok &= rel <= 1e-2 and chk[name]["finite"]
            chk["pass"] = all(v["max_rel_vs_loop"] <= 1e-2 and v["finite"] for v in chk.values() if isinstance(v, dict))
            entry["check"] = chk
            log(f"[check] K_gu={kgu} K_d={kd} m={m}: {json.dumps(chk)}")
        if "time" in modes:
            tim = {}
            for m in ms:
                x, w = x_of(m), rw_of(m)
                pool = [routing(census, m, rng, dev) for _ in range(args.iters + 2)]
                counts = torch.stack([torch.bincount(p.reshape(-1), minlength=EXPERTS) for p in pool]).float()
                fat_experts = float((counts > default_fat).float().sum(1).mean())
                fat_rows = float((counts * (counts > default_fat)).sum(1).mean() / (m * TOPK))
                row = {"fat_experts_mean": fat_experts, "fat_row_share": fat_rows, "max_count_mean": float(counts.max(1).values.mean())}
                arms = [("default", default_fat, False), ("no_fat", 1 << 30, False)]
                if kgu != kd and hasattr(native, "exl3_moe_kmix"):
                    arms += [("k0_default", default_fat, True), ("k0_no_fat", 1 << 30, True)]
                arms += [(f"fat{t}", int(t), False) for t in args.thresholds.split(",") if t]
                for arm, fat, k0 in arms:
                    ts = []
                    for i, ids in enumerate(pool):
                        torch.cuda.synchronize()
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record()
                        fused(x, ids, w, fat, k0)
                        e1.record()
                        torch.cuda.synchronize()
                        if i >= 2:
                            ts.append(e0.elapsed_time(e1))
                    med = statistics.median(ts)
                    row[arm] = {"median_ms": med, "p10_ms": R.pct(ts, 0.1), "p90_ms": R.pct(ts, 0.9),
                                "tok_per_s": m / (med / 1e3), "n": len(ts)}
                tim[f"m{m}"] = row
                log(f"[time] K_gu={kgu} K_d={kd} m={m}: default {row['default']['median_ms']:.2f} ms "
                    f"({row['default']['tok_per_s']:.0f} tok/s), no_fat {row['no_fat']['median_ms']:.2f} ms; "
                    f"fat experts {fat_experts:.1f} rows {100 * fat_rows:.1f}% max count {row['max_count_mean']:.0f}"
                    + "".join(f"; {a} {row[a]['median_ms']:.2f} ms" for a, _, _ in arms[2:]))
            entry["time"] = tim
        result["pairs"][pair] = entry
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result | {"pass": ok, "partial": True}, indent=1) + "\n")
        del lay, inners
        torch.cuda.empty_cache()
    result["pass"] = ok
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    log(f"pass={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
