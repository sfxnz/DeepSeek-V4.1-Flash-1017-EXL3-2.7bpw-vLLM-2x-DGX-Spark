#!/usr/bin/env python3
"""Encode one routed layer's experts at K bits (MCG, H = I) for the K-bits kernel tests.

GPU, dsv41-quant151 image (exllamav3 1.5.1). The per-tensor path of tools/requant_full.py
(the served 2.0 bpw pack's encoder): MXFP4 dequant, _quantize_fast (uncalibrated fallback,
tail-biting Viterbi, crc32-seeded sign flips), then the H = I row/column refit kept only when
it lowers relerr. Not the Hessian encode the 1017 pack will use: these tensors exist to test
kernels on realistic trellis/scale statistics, not to measure quality.

Each tensor is encoded as its TP-rank slice (gate/up w1, w3: output columns of the rank;
down w2: input rows of the rank), i.e. a [5120, 1152] / [1152, 5120] EXL3 tensor of its own,
in exactly the layout the serve's RoutedExperts params hold after shard_exl3_col/row. Half the
encode time of a full tensor; the kernels see the same shapes and formats.

  python3 kernel_study/p2b_kbits/encode_layer.py --src /src --layer 20 --tp-rank 0 \
      --bits 2,3,4 --out /out

Writes <out>/L<layer>_tp<rank>_K<K>.safetensors with keys "<e>.<w1|w3|w2>.<trellis|suh|svh|mcg>"
plus "ffn_norm.weight", and a .json with per-tensor relerr and timing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

TP = 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="source snapshot dir (MXFP4 experts)")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--tp-rank", type=int, default=0)
    ap.add_argument("--bits", default="2,3,4")
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--tile-chunk", type=int, default=4608)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch
    from exllamav3.ext import exllamav3_ext as ext
    from safetensors import safe_open
    from safetensors.torch import save_file

    from quantize_experts_exl3 import _dequant_t, _quantize_fast
    from requant_probe import refit_identity, relerr

    dev = torch.device("cuda:0")
    idx = json.loads((args.src / "model.safetensors.index.json").read_text())["weight_map"]
    pre = f"layers.{args.layer}.ffn.experts"
    r = args.tp_rank
    args.out.mkdir(parents=True, exist_ok=True)
    handles: dict[str, object] = {}

    def get(name: str):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(str(args.src / f), framework="pt")
        return handles[f].get_tensor(name)

    for K in (int(k) for k in args.bits.split(",")):
        out_st = args.out / f"L{args.layer}_tp{r}_K{K}.safetensors"
        if out_st.exists():
            print(f"exists, skip: {out_st}", flush=True)
            continue

        def recon(trellis, suh, svh, shape):
            o = torch.empty(shape, dtype=torch.half, device=dev)
            ext.reconstruct_had_slice(o, trellis.to(dev), suh.to(dev).half(), svh.to(dev).half(), K, True, False, 0)
            return o

        tensors: dict = {"ffn_norm.weight": get(f"layers.{args.layer}.ffn_norm.weight").contiguous()}
        rows = []
        h_cache: dict = {}
        t0 = time.time()
        for e in range(args.experts):
            for w in ("w1", "w3", "w2"):
                stem = f"{pre}.{e}.{w}"
                ta = time.time()
                full = _dequant_t(get(stem + ".weight"), get(stem + ".scale"), "cpu")  # (in, out) fp32
                if w == "w2":
                    n = full.shape[0] // TP
                    wt = full[r * n : (r + 1) * n, :]
                else:
                    n = full.shape[1] // TP
                    wt = full[:, r * n : (r + 1) * n]
                wt = wt.contiguous().to(dev)
                torch.manual_seed(zlib.crc32(f"{stem}.tp{r}".encode()))
                enc = _quantize_fast([wt.clone()], K, "cuda:0", h_cache, codebook="mcg", tile_chunk=args.tile_chunk)[0]
                trellis, su, sv = enc["trellis"], enc["suh"].to(dev), enc["svh"].to(dev)
                q = recon(trellis, su, sv, wt.shape)
                e_vit = relerr(q, wt)
                _, rr, cc = refit_identity(wt, q.float())
                su_f, sv_f = (su.float() * rr).half(), (sv.float() * cc).half()
                e_fit = relerr(recon(trellis, su_f, sv_f, wt.shape), wt)
                keep = math.isfinite(e_fit) and e_fit <= e_vit
                if keep:
                    su, sv = su_f, sv_f
                key = f"{e}.{w}"
                tensors[key + ".trellis"] = trellis.contiguous()
                tensors[key + ".suh"] = su.cpu().contiguous()
                tensors[key + ".svh"] = sv.cpu().contiguous()
                tensors[key + ".mcg"] = enc["mcg"].contiguous()
                rows.append({"t": key, "shape": list(wt.shape), "viterbi": e_vit, "final": e_fit if keep else e_vit,
                             "refit": keep, "s": time.time() - ta})
                del full, wt, q, enc
            if (e + 1) % 16 == 0 or e + 1 == args.experts:
                done = len(rows)
                spt = (time.time() - t0) / done
                print(f"K={K} expert {e + 1}/{args.experts} {spt:.3f} s/tensor relerr "
                      f"{sum(x['final'] for x in rows) / done:.4f} eta {(3 * args.experts - done) * spt / 60:.1f} min",
                      flush=True)
        tmp = out_st.with_name(out_st.name + ".tmp")
        save_file(tensors, str(tmp), metadata={"layer": str(args.layer), "tp_rank": str(r), "K": str(K),
                                                 "codebook": "mcg", "encoder": "requant_full per-tensor, H=I"})
        tmp.rename(out_st)
        summ = {"layer": args.layer, "tp_rank": r, "K": K, "experts": args.experts, "tensors": len(rows),
                "sec": time.time() - t0, "relerr_mean": sum(x["final"] for x in rows) / len(rows),
                "relerr_by_proj": {w: sum(x["final"] for x in rows if x["t"].endswith(w)) / (len(rows) // 3)
                                   for w in ("w1", "w3", "w2")},
                "refit_kept": sum(x["refit"] for x in rows), "src": str(args.src), "rows": rows}
        out_st.with_suffix(".json").write_text(json.dumps(summ, indent=1) + "\n")
        print(f"wrote {out_st} ({out_st.stat().st_size / 1e9:.2f} GB) relerr {summ['relerr_mean']:.4f} "
              f"in {summ['sec'] / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
