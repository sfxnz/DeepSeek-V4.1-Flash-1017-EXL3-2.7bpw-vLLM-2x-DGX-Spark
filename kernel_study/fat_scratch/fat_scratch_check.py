#!/usr/bin/env python3
"""fat_scratch_shared on the real kernels: one GPU, real 1017 routed experts, stock vllm_exl3 fused MoE.

For each layer (K_gu/K_d 2/2 layer 5, 2/3 layer 8, 3/3 layer 20, 3/4 layer 19; TP rank 0 shards, loaded and
packed like Exl3MoEMethod.process_weights_after_loading) and each routing (uniform, census) at each m:
  fat     vllm_exl3.exl3.apply_exl3_batched_fat (the fat-expert path, every expert above 256 rows) on each
          2048-token slice, as apply_exl3_fused_moe calls it, once with the image's _fat_scratch (one cached set
          per bucket and K) and once with docker/patch/fat_scratch_shared.py's; before each shared call every
          shared buffer is poisoned (NaN / 12345), so a read the expert did not write shows up.
          PASS needs torch.equal on every case (the path is deterministic: cuBLAS, Hadamard, reconstruct, and
          index_add_ over distinct tokens per expert).
  full    apply_exl3_fused_moe (exl3_moe + fat path): stock twice, then shared. Informational: equal to stock
          when stock repeats bit-exactly (exl3_moe's own determinism is not this lever's).
Also reports the fat scratch each arm holds at the end (stock grows with the buckets and K seen; shared is one set).

  docker run --rm --gpus all --network none --memory 24g -v "$PWD":/repo -v <pack>:/pack:ro \
    -v /home/sfxnz/projects/data/dsv41-1017/kernels/census_npy:/census:ro -v <out dir>:/data -w /repo \
    --entrypoint python3 dsv41-flash-exl3-sm121:e15-kbits kernel_study/fat_scratch/fat_scratch_check.py \
    --pack /pack --census /census --out /data/fat-scratch-check.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docker/patch"))

import fat_scratch_shared as fss  # noqa: E402

LAYERS = {5: "2/2", 8: "2/3", 20: "3/3", 19: "3/4"}
HIDDEN, INTER_FULL, EXPERTS, TOPK, LIMIT = 5120, 2304, 384, 6, 10.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def load_layer(P, torch, pack: Path, layer: int, dev: str):
    from safetensors import safe_open

    idx = json.loads((pack / "model.safetensors.index.json").read_text())["weight_map"]
    stem = f"layers.{layer}.ffn.experts"
    names = [f"{stem}.{e}.{w}.{s}" for e in range(EXPERTS) for w in ("w1", "w2", "w3") for s in ("trellis", "suh", "svh", "mcg")]
    handles = {f: safe_open(str(pack / f), "pt") for f in sorted({idx[n] for n in names})}
    get = lambda n: handles[idx[n]].get_tensor(n)  # noqa: E731
    inners = []
    for e in range(EXPERTS):
        ent = {}
        for name, w, shard in (("gate", "w1", P.shard_exl3_col), ("up", "w3", P.shard_exl3_col), ("down", "w2", P.shard_exl3_row)):
            t, su, sv = (shard(get(f"{stem}.{e}.{w}.{s}"), s, 0, 2).to(dev) for s in ("trellis", "suh", "svh"))
            ent[name] = P.make_linear_exl3(t, su, sv, get(f"{stem}.{e}.{w}.mcg").to(dev), None)
        inners.append(ent)
    norm = get(f"layers.{layer}.ffn_norm.weight").float().to(dev)
    k_gu = int(inners[0]["gate"].trellis.shape[-1]) // 16
    k_d = int(inners[0]["down"].trellis.shape[-1]) // 16
    lay = SimpleNamespace(w13_trellis=inners[0]["gate"].trellis, _exl3_hidden_size=HIDDEN,
                          _exl3_intermediate_local=INTER_FULL // 2, _exl3_bits=k_gu, _exl3_bits_d=k_d,
                          _exl3_codebook_flags=(True, False, True, False, True, False), expert_map=None)
    P.build_exl3_fused_state(lay, inners)
    return lay, inners, norm


def held_mib(cache: dict) -> float:
    seen, total = set(), 0
    for entry in cache.values():
        for t in (entry.values() if isinstance(entry, dict) else ()):
            if hasattr(t, "untyped_storage"):
                s = t.untyped_storage()
                if s.data_ptr() not in seen:
                    seen.add(s.data_ptr())
                    total += s.nbytes()
    return total / 2**20


def poison(cache: dict, torch) -> None:
    for key, entry in cache.items():
        if isinstance(key, tuple) and key and key[0] == fss.BASE_TAG:
            for t in entry.values():
                if hasattr(t, "fill_"):
                    t.fill_(12345 if t.dtype == torch.int16 else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--census", type=Path, default=None)
    ap.add_argument("--layers", default="5,8,20,19")
    ap.add_argument("--ms", default="2048,8192")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    import numpy as np
    import torch
    import vllm_exl3.exl3 as P

    dev = "cuda"
    stock_fn = P._fat_scratch
    why = fss.stock_mismatch(__import__("inspect").getsource(stock_fn))
    assert why is None, why
    shared_cache: dict = {}
    shared_fn = fss.make_fat_scratch(SimpleNamespace(_FAT_SCRATCH_CACHE=shared_cache, TEMP_ROWS_FUSED=P.TEMP_ROWS_FUSED), torch)
    stock_cache = P._FAT_SCRATCH_CACHE
    res = {"cases": [], "pass": True}
    for layer in [int(x) for x in a.layers.split(",")]:
        t0 = time.time()
        lay, inners, norm = load_layer(P, torch, a.pack, layer, dev)
        census = None
        if a.census:
            caps = [np.load(p)[:, layer, :].astype("int64") for p in sorted(a.census.glob("*.npy"))]
            census = np.concatenate(caps, 0) if caps else None
        log(f"layer {layer} K {LAYERS.get(layer)} loaded in {time.time() - t0:.0f} s")
        gen = torch.Generator(device=dev)
        gen.manual_seed(layer)
        rng = random.Random(layer)
        for m in [int(x) for x in a.ms.split(",")]:
            for kind in ("uniform", "census"):
                if kind == "census" and census is None:
                    continue
                for seed in range(a.seeds):
                    if kind == "uniform":
                        ids = torch.rand(m, EXPERTS, device=dev, generator=gen).topk(TOPK, dim=1).indices
                    else:
                        ids = torch.tensor(census[[rng.randrange(census.shape[0]) for _ in range(m)]], device=dev)
                    h = torch.randn(m, HIDDEN, device=dev, generator=gen)
                    x = ((h / h.pow(2).mean(dim=1, keepdim=True).sqrt()) * norm).to(torch.bfloat16)
                    s = torch.nn.functional.softplus(torch.randn(m, TOPK, device=dev, generator=gen)).sqrt()
                    w = s / s.sum(dim=1, keepdim=True) * 1.5
                    case = {"layer": layer, "m": m, "routing": kind, "seed": seed}
                    # fat path alone, per 2048-token slice (what apply_exl3_fused_moe does above 2048 rows)
                    eq_fat, n_fat = True, 0
                    for a0 in range(0, m, P.TEMP_ROWS_FUSED):
                        sl = slice(a0, min(a0 + P.TEMP_ROWS_FUSED, m))
                        xs, ids_s, ws = x[sl], ids[sl], w[sl]
                        local = P.map_topk_to_local(ids_s, len(inners), None)
                        counts = torch.bincount(local.reshape(-1).long(), minlength=len(inners) + 1)[: len(inners)]
                        n_fat += int((counts > P.FAT_EXPERT_THRESHOLD).sum())
                        order = local.reshape(-1).argsort()
                        tok = torch.arange(xs.shape[0], device=dev).repeat_interleave(ids_s.shape[-1])[order]
                        wt = ws.reshape(-1).to(torch.float16)[order]
                        xh = xs.contiguous().half()
                        outs = []
                        for arm in ("stock", "shared"):
                            P._fat_scratch = stock_fn if arm == "stock" else shared_fn
                            if arm == "shared":
                                poison(shared_cache, torch)
                            o = torch.zeros(xs.shape[0], HIDDEN, dtype=torch.float32, device=dev)
                            P.apply_exl3_batched_fat(xh, tok, wt, counts.tolist(), inners, LIMIT, P.FAT_EXPERT_THRESHOLD, o,
                                                     use_kernel=P._fat_kernel_available())
                            outs.append(o)
                        P._fat_scratch = stock_fn
                        eq_fat &= bool(torch.equal(outs[0], outs[1])) and bool(torch.isfinite(outs[0]).all())
                    # full fused call: stock, stock again, shared
                    full = []
                    for arm in ("stock", "stock", "shared"):
                        P._fat_scratch = stock_fn if arm == "stock" else shared_fn
                        full.append(P.apply_exl3_fused_moe(x, ids, w, lay, inners, None, LIMIT).clone())
                    P._fat_scratch = stock_fn
                    torch.cuda.synchronize()
                    case.update({
                        "fat_experts": n_fat, "fat_equal": eq_fat,
                        "full_stock_repeat_equal": bool(torch.equal(full[0], full[1])),
                        "full_shared_equal": bool(torch.equal(full[0], full[2])),
                        "full_shared_max_abs_diff": float((full[0] - full[2]).abs().max()),
                        "full_stock_repeat_max_abs_diff": float((full[0] - full[1]).abs().max()),
                        "stock_held_mib": round(held_mib(stock_cache), 1), "shared_held_mib": round(held_mib(shared_cache), 1),
                    })
                    res["cases"].append(case)
                    res["pass"] &= eq_fat
                    log(f"layer {layer} m={m} {kind} seed {seed}: fat experts {n_fat}, fat path equal {eq_fat}; full: "
                        f"stock repeat equal {case['full_stock_repeat_equal']}, shared equal {case['full_shared_equal']}; "
                        f"held stock {case['stock_held_mib']} / shared {case['shared_held_mib']} MiB")
        del lay, inners
        torch.cuda.empty_cache()
    res["stock_keys"] = sorted([list(k[1:]) for k in stock_cache if isinstance(k, tuple) and len(k) == 5])
    res["final_held_mib"] = {"stock": round(held_mib(stock_cache), 1), "shared": round(held_mib(shared_cache), 1)}
    log(f"PASS {res['pass']}: {len(res['cases'])} cases; held stock {res['final_held_mib']['stock']} MiB "
        f"({len(res['stock_keys'])} sets) vs shared {res['final_held_mib']['shared']} MiB")
    if a.out:
        a.out.write_text(json.dumps(res, indent=1) + "\n")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
