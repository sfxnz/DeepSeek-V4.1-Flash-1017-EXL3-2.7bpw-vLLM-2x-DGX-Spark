#!/usr/bin/env python3
"""Routed-MoE prefill: docker/patch/moe_prefill_tc (DSV41_MOE_PREFILL_TC) against the stock vllm_exl3 path.

GPU, in the e15-kbits (or e16p-moe) image, one layer of the served 1017 pack (TP rank slice of the real
routed experts, the plugin's own LinearEXL3 handles and pointer tables, exactly as
Exl3MoEMethod.process_weights_after_loading builds them). Pairs and layers: 2/2 layer 5, 2/3 layer 8,
3/3 layer 20, 3/4 layer 19 (tools/calib/alloc/expert_bits_2.70.json).

  stock   vllm_exl3.exl3.apply_exl3_fused_moe as the serve runs it today (exllamav3 exl3_moe, or
          vllm_exl3_c.exl3_moe_kmix when K_gu != K_d, plus the fat path above 256 rows per expert)
  tc      moe_prefill_tc: argsort grouping + worklist + gu_kernel + dn_kernel (JIT-built from
          docker/patch unless the image has the prebuilt module)
  ref     fp32 reference: per expert, ExLlamaV3 reconstruct (the same fp16 weights), fp32 GEMMs (TF32
          off) and fp32 Hadamards, with fp16 rounding only where every EXL3 path has it (the rotated
          GEMM inputs) and the fat path's activation; all arms are scored against it.

Routings: uniform (distinct top-6 per token, ~128 rows per expert at 8192 tokens) and census (rows drawn
from the layer's s10 decode census, the skew kernel_study/p2b_kbits/prefill_kbits.py measured).

Modes (comma list):
  check   m = 512 and 2048 (+ edge cases): tc vs stock, tc vs ref, stock vs ref (max |d| / max |ref|,
          rms(d) / rms(ref)); tc's A2 intermediate vs the reference's; tc run twice (atomic order only).
          Gate: tc rms_rel vs ref <= max(1.05 x stock's, 2e-4) and max_rel <= 1e-2, finite, A2 rms <= 2e-3.
  time    m in --ms: CUDA-event median / p10 / p90 per call over --iters routings, both arms.
  profile torch.profiler kernel breakdown per call, both arms, at the largest m, per routing.
  syncs   torch.cuda.set_sync_debug_mode('warn'): synchronizing ops per call, both arms.
  memory  per call at each m: peak allocated above the pre-call level, memory left allocated, reserved
          growth (allocator segments), cold (stock fat scratch cleared) and warm, both arms.

  python3 kernel_study/moe_prefill/bench_prefill_tc.py --pack /pack --census /census \
      --pairs 2:2,2:3,3:3,3:4 --ms 2048,4096,8192 --mode check,time,syncs --out /data/x.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "docker/patch"))

import moe_prefill_tc as tc  # noqa: E402

PAIR_LAYER = {"2:2": 5, "2:3": 8, "3:3": 20, "3:4": 19}
HIDDEN, INTER_FULL, EXPERTS, TOPK = 5120, 2304, 384, 6
SWIGLU_LIMIT = 10.0
ROUTED_SCALE = 1.5
R_SCALE = 0.088388347648


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


# ------------------------------------------------------------------------------------------- weights
def load_layer(P, pack: Path, layer: int, tp_rank: int, tp: int, dev: str):
    """One layer's routed experts from the pack, sharded like the plugin's loader."""
    import torch
    from safetensors import safe_open

    idx = json.loads((pack / "model.safetensors.index.json").read_text())["weight_map"]
    stem = f"layers.{layer}.ffn.experts"
    files = sorted({idx[f"{stem}.{e}.{w}.{s}"] for e in range(EXPERTS) for w in ("w1", "w2", "w3")
                    for s in ("trellis", "suh", "svh", "mcg")})
    handles = {f: safe_open(str(pack / f), "pt") for f in files}

    def get(name):
        return handles[idx[name]].get_tensor(name)

    inners, keep = [], []
    for e in range(EXPERTS):
        pack_e = {}
        for name, w, shard in (("gate", "w1", P.shard_exl3_col), ("up", "w3", P.shard_exl3_col),
                               ("down", "w2", P.shard_exl3_row)):
            t, su, sv = (shard(get(f"{stem}.{e}.{w}.{s}"), s, tp_rank, tp).to(dev) for s in ("trellis", "suh", "svh"))
            mcg = get(f"{stem}.{e}.{w}.mcg").to(dev)
            keep += [t, su, sv, mcg]
            pack_e[name] = P.make_linear_exl3(t, su, sv, mcg, None)
        inners.append(pack_e)
    norm = get(f"layers.{layer}.ffn_norm.weight").float().to(dev)
    k_gu = int(inners[0]["gate"].trellis.shape[-1]) // 16
    k_d = int(inners[0]["down"].trellis.shape[-1]) // 16
    lay = SimpleNamespace(
        w13_trellis=inners[0]["gate"].trellis, _exl3_hidden_size=HIDDEN, _exl3_intermediate_local=INTER_FULL // tp,
        _exl3_bits=k_gu, _exl3_bits_d=k_d, _exl3_codebook_flags=(True, False, True, False, True, False),
        expert_map=None,
    )
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    P.build_exl3_fused_state(lay, inners)
    torch.cuda.synchronize()
    lay._bench_state_mib = (torch.cuda.memory_allocated() - before) / 2**20
    assert (lay._exl3_k, lay._exl3_k_d) == (k_gu, k_d)
    return lay, inners, norm, keep


# ------------------------------------------------------------------------------------------- inputs
def census_rows(census: Path, layer: int):
    import numpy as np

    caps = [np.load(p, allow_pickle=False)[:, layer, :].astype("int64") for p in sorted(census.glob("*.npy"))]
    return np.concatenate(caps, 0) if caps else None


def routing(kind: str, m: int, rng: random.Random, gen, dev: str, census=None):
    import torch

    if kind == "uniform":
        return torch.rand(m, EXPERTS, device=dev, generator=gen).topk(TOPK, dim=1).indices
    if kind == "census":
        rows = [rng.randrange(census.shape[0]) for _ in range(m)]
        return torch.tensor(census[rows], dtype=torch.long, device=dev)
    raise ValueError(kind)


def edge_routings(m: int, dev: str):
    """Crafted: one expert in every row (m rows, 16 gate/up chunks), 1-row and 17-row experts, invalid ids."""
    import torch

    g = torch.Generator(device="cpu")
    g.manual_seed(5)
    base = torch.rand(m, EXPERTS, generator=g).topk(TOPK, dim=1).indices
    hot = base.clone()
    hot[:, 0] = 7
    hot[:, 1:] = torch.where(hot[:, 1:] == 7, torch.full_like(hot[:, 1:], 8), hot[:, 1:])
    sparse = torch.stack([torch.arange(TOPK) + 6 * (t % 3) for t in range(m)])  # 18 experts only
    sparse[0, 0] = 100   # a 1-row expert
    sparse[1:18, 1] = 200  # a 17-row expert
    inval = base.clone()
    inval[::5, -1] = -1
    return {"hot": hot.to(dev), "sparse": sparse.to(dev), "invalid": inval.to(dev)}


def make_x(m: int, norm, gen, dev: str):
    import torch

    h = torch.randn(m, HIDDEN, device=dev, generator=gen)
    return ((h / h.pow(2).mean(dim=1, keepdim=True).sqrt()) * norm).to(torch.bfloat16)


def make_w(m: int, gen, dev: str):
    import torch

    z = torch.randn(m, TOPK, device=dev, generator=gen)
    s = torch.nn.functional.softplus(z).sqrt()
    return s / s.sum(dim=1, keepdim=True) * ROUTED_SCALE


# ------------------------------------------------------------------------------------------- arms
def tc_call(mod, P, x, ids, w, lay, inners, keep_a2: bool = False):
    """moe_prefill_tc.run, keeping the A2 intermediate and the sort for the checks."""
    import torch

    tokens = x.shape[0]
    n_exp = len(inners)
    inter = lay._exl3_intermediate_local
    local = P.map_topk_to_local(ids, n_exp, None).reshape(-1)
    order = torch.argsort(local, stable=True)
    counts = torch.zeros(n_exp + 1, dtype=torch.long, device=x.device)
    counts.scatter_add_(0, local, torch.ones_like(local))
    tok = torch.div(order, TOPK, rounding_mode="floor").to(torch.int32)
    ws = w.reshape(-1).to(torch.float16)[order].contiguous()
    slots = tokens * TOPK
    out = torch.zeros(tokens, HIDDEN, dtype=torch.float32, device=x.device)
    a2 = torch.empty(slots, inter, dtype=torch.float16, device=x.device)
    it1 = torch.empty(4 * (-(-slots // mod.MT1) + n_exp), dtype=torch.int32, device=x.device)
    it2 = torch.empty(4 * (-(-slots // mod.MT2) + n_exp), dtype=torch.int32, device=x.device)
    ctrl = torch.empty(4, dtype=torch.int32, device=x.device)
    p = lay._exl3_ptrs
    mod.moe_prefill_tc(x.contiguous(), out, tok, ws, counts, *(p[n] for n in tc.PTR_NAMES), a2, it1, it2, ctrl,
                       lay._exl3_k, lay._exl3_k_d, SWIGLU_LIMIT)
    if keep_a2:
        return out, SimpleNamespace(a2=a2, order=order, counts=counts, ctrl=ctrl)
    return out


def had128(v):
    """[..., n * 128] fp32 rows, 128-point Sylvester Hadamard per block (unnormalised)."""
    import torch

    shp = v.shape
    y = v.reshape(-1, 128).clone()
    h = 1
    while h < 128:
        y = y.view(-1, 128 // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2)
        h *= 2
    return y.reshape(shp)


def ref_call(ext, x, ids, w, lay, inners):
    """fp32 reference (see the module doc); also A2 in tc's sorted-row order."""
    import torch

    tokens = x.shape[0]
    n_exp = len(inners)
    local = ids.reshape(-1).clone()
    local[(local < 0) | (local >= n_exp)] = n_exp
    order = torch.argsort(local, stable=True)
    sorted_e = local[order]
    counts = torch.bincount(sorted_e, minlength=n_exp + 1).tolist()
    w16 = w.reshape(-1).to(torch.float16)
    x16 = x.to(torch.float16)
    out = torch.zeros(tokens, HIDDEN, dtype=torch.float32, device=x.device)
    a2 = torch.zeros(tokens * TOPK, lay._exl3_intermediate_local, dtype=torch.float16, device=x.device)
    k_gu, k_d = lay._exl3_k, lay._exl3_k_d
    pos = 0
    for e in range(n_exp):
        n = counts[e]
        if n == 0:
            continue
        flat = order[pos:pos + n]
        rows = torch.div(flat, TOPK, rounding_mode="floor")
        gate, up, down = inners[e]["gate"], inners[e]["up"], inners[e]["down"]
        wts = {}
        for name, lin, k in (("g", gate, k_gu), ("u", up, k_gu), ("d", down, k_d)):
            W = torch.empty(lin.in_features, lin.out_features, dtype=torch.float16, device=x.device)
            ext.reconstruct(W, lin.trellis, k, True, False)
            wts[name] = W.float()
        xr = x16[rows]

        def rot_in(v16, suh):
            return (had128((v16 * suh).float()) * R_SCALE).to(torch.float16)

        g = had128(rot_in(xr, gate.suh).float() @ wts["g"]) * R_SCALE * gate.svh.float()
        u = had128(rot_in(xr, up.suh).float() @ wts["u"]) * R_SCALE * up.svh.float()
        g = g.clamp(max=SWIGLU_LIMIT)
        u = u.clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
        act = (torch.sigmoid(g) * g * u).to(torch.float16)
        h2 = rot_in(act, down.suh)
        a2[pos:pos + n] = h2
        y = had128(h2.float() @ wts["d"]) * R_SCALE * down.svh.float() * w16[flat].float().unsqueeze(1)
        out.index_add_(0, rows, y)
        pos += n
    return out, a2


def score(a, b):
    """max |a - b| / max |b|, rms(a - b) / rms(b), all finite."""
    import torch

    a, b = a.double(), b.double()
    fin = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    d = a - b
    return {"max_rel": float(d.abs().max() / b.abs().max().clamp_min(1e-30)),
            "rms_rel": float(d.pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30)), "finite": fin}


# ------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True, help="pack snapshot dir (model.safetensors.index.json)")
    ap.add_argument("--census", type=Path, default=None, help="dir of s10 census .npy (census routing)")
    ap.add_argument("--pairs", default="2:2,2:3,3:3,3:4")
    ap.add_argument("--ms", default="2048,4096,8192")
    ap.add_argument("--routings", default="uniform,census")
    ap.add_argument("--mode", default="check,time,syncs")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--tp-rank", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lazy-temps", action="store_true",
                    help="moe_prefill_tc's lazy exl3_moe scratch: rewrite build_exl3_fused_state before loading, "
                         "the stock arm allocates the scratch at its first call (ensure_temps)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    import torch
    import exllamav3_ext as ext
    import vllm_exl3.exl3 as P

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    modes = set(args.mode.split(","))
    ms = [int(m) for m in args.ms.split(",")]
    t0 = time.time()
    tag = tc.source_tag()
    mod = tc._load_prebuilt(tag)  # the e16p image's module for these sources, else a JIT build kept in the worktree
    if mod is None:
        mod = tc.build(HERE / "build" / tag, tc.MODULE_PREFIX + tag)
    log(f"moe_prefill_tc module {mod.__name__} ready in {time.time() - t0:.1f} s (tag {tag})")
    result = {"args": {k: str(v) for k, v in vars(args).items()}, "tag": tc.source_tag(), "pairs": {}}
    ok = True
    if args.lazy_temps:
        log(f"exl3_moe scratch: {tc.install_lazy_temps(P)}")

    def save(partial: bool):
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result | {"pass": ok, "partial": partial}, indent=1) + "\n")

    for pair in args.pairs.split(","):
        layer = PAIR_LAYER[pair]
        t0 = time.time()
        lay, inners, norm, keep = load_layer(P, args.pack, layer, args.tp_rank, 2, dev)
        log(f"=== {pair} layer {layer}: K_gu/K_d {lay._exl3_k}/{lay._exl3_k_d}, {len(inners)} experts in {time.time() - t0:.1f} s; "
            f"build_exl3_fused_state allocated {lay._bench_state_mib:.1f} MiB (exl3_moe scratch "
            f"{'lazy' if getattr(lay, '_dsv41_temps_key', None) else 'eager'}, "
            f"{sum(t.numel() * t.element_size() for v in P._FUSED_TEMP_CACHE.values() for t in v) / 2**20:.1f} MiB held)")
        census = census_rows(args.census, layer) if args.census else None
        gen = torch.Generator(device=dev)
        gen.manual_seed(args.seed)
        rng = random.Random(args.seed)
        entry = {"layer": layer, "k": [lay._exl3_k, lay._exl3_k_d]}

        def stock(x, ids, w):
            if args.lazy_temps:
                tc.ensure_temps(P, lay, int(x.shape[0]))
            return P.apply_exl3_fused_moe(x, ids, w, lay, inners, None, SWIGLU_LIMIT)

        if "check" in modes:
            chk = {}
            cases = []
            for kind in args.routings.split(","):
                if kind == "census" and census is None:
                    continue
                for m in (512, 2048):
                    cases.append((f"{kind}_m{m}", routing(kind, m, rng, gen, dev, census), m))
            for name, ids in edge_routings(1000, dev).items():
                cases.append((f"edge_{name}_m1000", ids, 1000))
            for name, ids, m in cases:
                x, w = make_x(m, norm, gen, dev), make_w(m, gen, dev)
                out, aux = tc_call(mod, P, x, ids, w, lay, inners, keep_a2=True)
                out2 = tc_call(mod, P, x, ids, w, lay, inners)
                st = stock(x, ids, w)
                ref, a2_ref = ref_call(ext, x, ids, w, lay, inners)
                torch.cuda.synchronize()
                n_valid = int(aux.counts[:-1].sum())
                c = {"tc_vs_ref": score(out, ref), "stock_vs_ref": score(st, ref), "tc_vs_stock": score(out, st),
                     "tc_repeat_maxabs": float((out - out2).abs().max()),
                     "a2_vs_ref": score(aux.a2[:n_valid].float(), a2_ref[:n_valid].float()),
                     "a2_equal_frac": float((aux.a2[:n_valid] == a2_ref[:n_valid]).float().mean()),
                     "items": [int(v) for v in aux.ctrl.tolist()[:2]]}
                tv, sv = c["tc_vs_ref"], c["stock_vs_ref"]
                c["pass"] = bool(tv["finite"] and sv["finite"] and tv["max_rel"] <= 1e-2
                                 and tv["rms_rel"] <= max(1.05 * sv["rms_rel"], 2e-4)
                                 and c["a2_vs_ref"]["rms_rel"] <= 2e-3)
                ok &= c["pass"]
                chk[name] = c
                log(f"[check] {pair} {name}: tc/ref max {tv['max_rel']:.2e} rms {tv['rms_rel']:.2e} | "
                    f"stock/ref max {sv['max_rel']:.2e} rms {sv['rms_rel']:.2e} | tc/stock rms "
                    f"{c['tc_vs_stock']['rms_rel']:.2e} | A2 rms {c['a2_vs_ref']['rms_rel']:.2e} eq "
                    f"{c['a2_equal_frac']:.4f} | repeat {c['tc_repeat_maxabs']:.2e} | items {c['items']} | "
                    f"{'PASS' if c['pass'] else 'FAIL'}")
            entry["check"] = chk
            save(True)

        if "syncs" in modes:
            m = max(ms)
            ids = routing("uniform", m, rng, gen, dev)
            x, w = make_x(m, norm, gen, dev), make_w(m, gen, dev)
            syn = {}
            for arm, fn in (("stock", lambda: stock(x, ids, w)), ("tc", lambda: tc_call(mod, P, x, ids, w, lay, inners))):
                fn()
                torch.cuda.synchronize()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    torch.cuda.set_sync_debug_mode("warn")
                    try:
                        fn()
                    finally:
                        torch.cuda.set_sync_debug_mode("default")
                msgs = [str(wr.message).split("\n")[0][:120] for wr in caught if "synchroniz" in str(wr.message)]
                syn[arm] = {"syncs": len(msgs), "first": msgs[:8]}
                log(f"[syncs] {pair} m={m} {arm}: {len(msgs)} synchronizing ops per call")
            entry["syncs"] = syn
            save(True)

        if "time" in modes:
            tim = {}
            for kind in args.routings.split(","):
                if kind == "census" and census is None:
                    continue
                for m in ms:
                    x, w = make_x(m, norm, gen, dev), make_w(m, gen, dev)
                    pool = [routing(kind, m, rng, gen, dev, census) for _ in range(args.iters + 2)]
                    counts = torch.stack([torch.bincount(p.reshape(-1), minlength=EXPERTS) for p in pool]).float()
                    row = {"max_count_mean": float(counts.max(1).values.mean()),
                           "fat_experts_mean": float((counts > P.FAT_EXPERT_THRESHOLD).float().sum(1).mean())}
                    for arm in ("stock", "tc"):
                        ts = []
                        for i, ids in enumerate(pool):
                            torch.cuda.synchronize()
                            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            e0.record()
                            if arm == "stock":
                                stock(x, ids, w)
                            else:
                                tc_call(mod, P, x, ids, w, lay, inners)
                            e1.record()
                            torch.cuda.synchronize()
                            if i >= 2:
                                ts.append(e0.elapsed_time(e1))
                        med = statistics.median(ts)
                        row[arm] = {"median_ms": med, "p10_ms": pct(ts, 0.1), "p90_ms": pct(ts, 0.9), "n": len(ts),
                                    "tok_per_s": m / (med / 1e3)}
                    row["speedup"] = row["stock"]["median_ms"] / row["tc"]["median_ms"]
                    flops = 2 * 3 * HIDDEN * lay._exl3_intermediate_local * m * TOPK
                    row["tc_tflops"] = flops / (row["tc"]["median_ms"] / 1e3) / 1e12
                    tim[f"{kind}_m{m}"] = row
                    log(f"[time] {pair} {kind} m={m}: stock {row['stock']['median_ms']:.2f} ms, tc "
                        f"{row['tc']['median_ms']:.2f} ms (p10 {row['tc']['p10_ms']:.2f} p90 {row['tc']['p90_ms']:.2f}), "
                        f"x{row['speedup']:.2f}, tc {row['tc_tflops']:.1f} TFLOPS; max count {row['max_count_mean']:.0f}, "
                        f"fat experts {row['fat_experts_mean']:.1f}")
            entry["time"] = tim
            save(True)

        if "memory" in modes:
            # Transient peak of one call above what was allocated before it (torch.cuda.max_memory_allocated;
            # the call's own fp32 output is part of it), what it leaves allocated besides that output (the stock
            # fat path's persistent scratch), and the caching allocator's reserved growth (new segments, i.e.
            # cuMemCreate). cold: stock fat-scratch cache cleared and free blocks released first; warm: next call.
            mem = {}
            for kind in args.routings.split(","):
                if kind == "census" and census is None:
                    continue
                for m in ms:
                    x, w = make_x(m, norm, gen, dev), make_w(m, gen, dev)
                    ids = routing(kind, m, rng, gen, dev, census)
                    row = {}
                    for arm, fn in (("stock", lambda: stock(x, ids, w)), ("tc", lambda: tc_call(mod, P, x, ids, w, lay, inners))):
                        for phase in ("cold", "warm"):
                            if phase == "cold":
                                getattr(P, "_FAT_SCRATCH_CACHE", {}).clear()
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            base = torch.cuda.memory_allocated()
                            base_res = torch.cuda.memory_reserved()
                            torch.cuda.reset_peak_memory_stats()
                            out = fn()
                            torch.cuda.synchronize()
                            peak = torch.cuda.max_memory_allocated() - base
                            res = torch.cuda.memory_reserved() - base_res
                            out_mib = out.numel() * out.element_size() / 2**20
                            del out
                            held = torch.cuda.memory_allocated() - base
                            row[f"{arm}_{phase}"] = {"peak_mib": peak / 2**20, "held_after_mib": held / 2**20,
                                                     "reserved_growth_mib": res / 2**20, "output_mib": out_mib}
                    getattr(P, "_FAT_SCRATCH_CACHE", {}).clear()
                    torch.cuda.empty_cache()
                    mem[f"{kind}_m{m}"] = row
                    log(f"[memory] {pair} {kind} m={m}: " + "; ".join(
                        f"{k} peak {v['peak_mib']:.0f} MiB held {v['held_after_mib']:.0f} reserved+ {v['reserved_growth_mib']:.0f}"
                        for k, v in row.items()) + f" (each peak includes the {row['tc_cold']['output_mib']:.0f} MiB output)")
            entry["memory"] = mem
            save(True)

        if "profile" in modes:
            from torch.profiler import ProfilerActivity, profile

            m = max(ms)
            prof_out = {}
            for kind in args.routings.split(","):
                if kind == "census" and census is None:
                    continue
                ids = routing(kind, m, rng, gen, dev, census)
                x, w = make_x(m, norm, gen, dev), make_w(m, gen, dev)
                for arm, fn in (("tc", lambda: tc_call(mod, P, x, ids, w, lay, inners)), ("stock", lambda: stock(x, ids, w))):
                    fn()
                    torch.cuda.synchronize()
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        for _ in range(3):
                            fn()
                        torch.cuda.synchronize()
                    rows = []
                    for evt in prof.key_averages():
                        dt = getattr(evt, "device_time_total", None)
                        if dt is None:
                            dt = getattr(evt, "cuda_time_total", 0)
                        if dt > 0:
                            rows.append((evt.key[:90], dt / 3 / 1e3, evt.count // 3))
                    rows.sort(key=lambda r: -r[1])
                    prof_out[f"{kind}_{arm}"] = rows[:30]
                    total = sum(t for _, t, _ in rows)
                    log(f"[profile] {pair} {kind} m={m} {arm}: kernel sum {total:.2f} ms/call; "
                        + "; ".join(f"{k} {t:.2f} ms x{c}" for k, t, c in rows[:12]))
            entry["profile"] = prof_out
            save(True)

        result["pairs"][pair] = entry
        del lay, inners, keep
        torch.cuda.empty_cache()

    save(False)
    log(f"pass={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
