#!/usr/bin/env python3
"""Check a dsv41-hessian-v1 layer written by capture_h.py (decision B-3).

  python3 tools/calib/check_hessians_v1.py --hdir HESS --layer 0 --stats S.npz [--factors F] [--eig 24] --out J

- meta.json: format, layer, complete; every expected expert file present; Sum n_e == 6 * tokens
- per expert (all files): keys/dtype/shape, finite, exactly symmetric, positive diagonal; n_e and
  sum w_e == the routing stats (RoutingStats of the same run); the builder's own load_hessian accepts it
- eigenvalues (fp64, GPU) of --eig experts spread over routed-count ranks (+ every zero-token expert):
  min/max eigenvalue, condition number, also of the undamped and the builder-damped (2.5% mean diag) H
- with --factors (factors of experts whose routed tokens were all kept): recompute
  S_e = sum w^p x x^T, then H_all_e = (w13_in (W_e + m0) - S_e) / m0 must agree across experts (and equal
  w13_in of a zero-token expert): an independent check of the mixing formula.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from hooks import drop_page_cache, read_meta  # noqa: E402


def eig_stats(h: torch.Tensor) -> dict:
    ev = torch.linalg.eigvalsh(h.double().cuda())
    damp = 0.025 * h.double().diagonal().mean().item()
    lo, hi = ev.min().item(), ev.max().item()
    return {"min": lo, "max": hi, "cond": hi / lo if lo > 0 else math.inf,
            "cond_damped": (hi + damp) / (lo + damp), "n_neg": int((ev < 0).sum()),
            "rank_1e-6": int((ev > 1e-6 * hi).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--factors", default=None)
    ap.add_argument("--eig", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import build_pack_1017 as B

    L = a.layer
    d = Path(a.hdir) / f"layer_{L:02d}"
    meta = json.loads((d / "meta.json").read_text())
    st = np.load(a.stats)
    counts = st["counts"][L].sum(0)  # over labels
    mass = st["mass"][L].sum(0)
    rep = {"layer": L, "meta_ok": meta.get("format") == B.HESSIAN_FORMAT and meta.get("layer") == L
           and meta.get("complete") is True and B.hessian_ready(Path(a.hdir), L)}
    experts = sorted(int(e) for e in meta["experts"])
    rep["experts"] = len(experts)
    rep["files"] = len(list(d.glob("expert_*.safetensors")))
    bad, n_err, w_err, asym, negdiag = [], 0, 0.0, 0.0, 0
    for e in experts:
        p = d / f"expert_{e:03d}.safetensors"
        with safe_open(str(p), framework="pt") as fh:
            keys = set(fh.keys())
            md = read_meta(fh)
            hs = {k: fh.get_tensor(k) for k in keys}
        if keys != {"w13_in", "w2_in"} or hs["w13_in"].shape != (5120, 5120) or hs["w2_in"].shape != (2304, 2304) \
                or any(h.dtype != torch.float32 for h in hs.values()):
            bad.append((e, "layout"))
            continue
        for k, h in hs.items():
            if not torch.isfinite(h).all():
                bad.append((e, f"{k} non-finite"))
            asym = max(asym, (h - h.t()).abs().max().item())
            negdiag += int((h.diagonal() <= 0).sum())
            B.load_hessian(Path(a.hdir), L, e, "gu" if k == "w13_in" else "d", h.shape[0])  # builder's checks
        drop_page_cache(p)
        n_err = max(n_err, abs(md["n"] - int(counts[e])))
        w_err = max(w_err, abs(md["wsum"] - float(mass[e])) / max(float(mass[e]), 1e-12))
    ns = [meta["experts"][str(e)]["n"] for e in experts]
    pw = [meta["experts"][str(e)]["prior_weight"] for e in experts]
    rep.update(bad=bad, max_abs_asym=asym, nonpos_diag=negdiag, max_n_err_vs_stats=n_err, max_rel_wsum_err_vs_stats=w_err,
               sum_n=sum(ns), routed_picks_stats=float(counts.sum()), six_T=6 * meta["tokens"],
               n_quantiles={q: float(np.quantile(ns, q)) for q in (0, 0.01, 0.1, 0.5, 0.9, 0.99, 1)},
               zero_token_experts=[e for e, n in zip(experts, ns) if n == 0],
               prior_weight_quantiles={q: float(np.quantile(pw, q)) for q in (0, 0.1, 0.5, 0.9, 1)}, m0=meta["m0"])
    # eigen spectra over a spread of experts
    order = [e for _, e in sorted(zip(ns, experts))]
    pick = sorted(set([order[round(i * (len(order) - 1) / max(1, a.eig - 1))] for i in range(a.eig)]
                      + rep["zero_token_experts"]))
    eig = {}
    for e in pick:
        with safe_open(str(d / f"expert_{e:03d}.safetensors"), framework="pt") as fh:
            eig[str(e)] = {"n": meta["experts"][str(e)]["n"], "w13_in": eig_stats(fh.get_tensor("w13_in")),
                           "w2_in": eig_stats(fh.get_tensor("w2_in"))}
    rep["eig"] = eig
    rep["psd_ok"] = all(v[k]["min"] > -1e-6 * v[k]["max"] for v in eig.values() for k in ("w13_in", "w2_in"))
    rep["cond_max"] = {k: max(v[k]["cond"] for v in eig.values()) for k in ("w13_in", "w2_in")}
    rep["cond_damped_max"] = {k: max(v[k]["cond_damped"] for v in eig.values()) for k in ("w13_in", "w2_in")}
    # independent mixing check from full factors
    if a.factors:
        fd = Path(a.factors) / f"layer_{L:02d}"
        est, m0, p = {}, meta["m0"], meta["wpow"]
        for f in sorted(fd.glob("expert_*.safetensors")):
            e = int(f.stem.split("_")[1])
            with safe_open(str(f), framework="pt") as fh:
                md = read_meta(fh)
                if md["n_kept"] != md["n_routed"]:
                    continue
                x, w = fh.get_tensor("x").cuda().double(), fh.get_tensor("w").cuda().double()
            S = (x * w.pow(p).unsqueeze(-1)).t() @ x
            W = float(w.pow(p).sum())
            with safe_open(str(d / f"expert_{e:03d}.safetensors"), framework="pt") as fh:
                h13 = fh.get_tensor("w13_in").cuda().double()
            est[e] = (h13 * (W + m0) - S) / m0
        if len(est) >= 2:
            es = sorted(est)
            ref = est[es[0]]
            rep["mixing_check"] = {"experts": es, "max_rel_diff_H_all_estimates": max(
                ((est[e] - ref).norm() / ref.norm()).item() for e in es[1:])}
            if rep["zero_token_experts"]:
                z = rep["zero_token_experts"][0]
                with safe_open(str(d / f"expert_{z:03d}.safetensors"), framework="pt") as fh:
                    hz = fh.get_tensor("w13_in").cuda().double()
                rep["mixing_check"]["rel_diff_vs_zero_token_expert"] = ((ref - hz).norm() / hz.norm()).item()
    rep["ok"] = (rep["meta_ok"] and not bad and asym == 0.0 and negdiag == 0 and n_err == 0 and w_err < 1e-6
                 and rep["files"] == len(experts) and rep["psd_ok"])
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "eig"}, indent=1))
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
