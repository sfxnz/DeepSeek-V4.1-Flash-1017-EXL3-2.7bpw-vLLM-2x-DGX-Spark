#!/usr/bin/env python3
"""Collect r_eff.py outputs into r_eff.json + a text table, and fit a smooth model across layers.

  python3 tools/calib/r_eff_fit.py --dir DATA/hess/reff --out DATA/hess/r_eff.json --table DATA/hess/r_eff_table.txt

Per (group, K, eval set) the pooled r_eff of the sampled layers is fitted with
  const:  r(L) = c                      linear: r(L) = a + b * L/39      quad: r(L) = a + b*x + c*x^2
by least squares; the chosen model is the lowest-order one whose RMS residual is within 10% of the
best fit's (plain stdlib, no numpy). model[group][K][set] gives the chosen fit and its
prediction for every layer 0..39 (the allocation reads model.*.*.primary.by_layer).
primary = held-out ("out") pooled r_eff, the r_eff with the real encode's Hessian built without the
evaluated rows (what serving sees); "in" (in-sample) is reported alongside.
"""
import argparse
import json
from pathlib import Path


def lstsq(xs, ys, deg):
    """Least squares polynomial (normal equations, tiny sizes)."""
    n = deg + 1
    A = [[sum(x ** (i + j) for x in xs) for j in range(n)] for i in range(n)]
    b = [sum(y * x ** i for x, y in zip(xs, ys)) for i in range(n)]
    for c in range(n):  # Gauss-Jordan
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        A[c], A[p], b[c], b[p] = A[p], A[c], b[p], b[c]
        for r in range(n):
            if r != c and A[c][c]:
                f = A[r][c] / A[c][c]
                A[r] = [u - f * v for u, v in zip(A[r], A[c])]
                b[r] -= f * b[c]
    return [b[i] / A[i][i] for i in range(n)]


def fit(layers, ys):
    xs = [L / 39 for L in layers]
    out = {}
    for name, deg in (("const", 0), ("linear", 1), ("quad", 2)):
        if len(xs) <= deg:
            continue
        co = lstsq(xs, ys, deg)
        pred = [sum(c * x ** i for i, c in enumerate(co)) for x in xs]
        rms = (sum((p - y) ** 2 for p, y in zip(pred, ys)) / len(ys)) ** 0.5
        out[name] = {"coef": co, "rms": rms, "max_abs_res": max(abs(p - y) for p, y in zip(pred, ys))}
    order = [k for k in ("const", "linear", "quad") if k in out]
    best = min(out[k]["rms"] for k in order)
    chosen = next(k for k in order if out[k]["rms"] <= 1.1 * best + 1e-4)
    co = out[chosen]["coef"]
    by_layer = {str(L): sum(c * (L / 39) ** i for i, c in enumerate(co)) for L in range(40)}
    return {"fits": out, "chosen": chosen, "by_layer": by_layer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", required=True)
    a = ap.parse_args()
    runs = {}
    for f in sorted(Path(a.dir).glob("r_eff_L*.json")):
        d = json.loads(f.read_text())
        runs[d["layer"]] = d
    layers = sorted(runs)
    ks = runs[layers[0]]["ks"]
    sets = ["out", "in"] if any(k.endswith("/out/all") for k in runs[layers[0]]["pooled"]) else ["in"]
    table = {}
    lines = [f"r_eff (pooled over each layer's 16 sampled experts, weights W_e = sum routed w^2); layers {layers}",
             "noise = W + r*rms(W)*G (native.Harness._variant_weights), rho = w^2-weighted per-token relative output MSE",
             "real_corr = rho_real(EXL3 numerics vs native fp8 ref) - rho_ctrl(native weights, EXL3 numerics)", ""]
    hdr = f"{'grp':3s} K set layer  r_eff  r_dir  r_agg   rho_real   rho_ctrl  relerr  proxy    hot    mid   cold  s/tensor"
    lines.append(hdr)
    for g in ("GU", "D"):
        for k in ks:
            for s in sets:
                for L in layers:
                    P = runs[L]["pooled"]
                    p = P.get(f"{g}/K{k}/{s}/all")
                    if p is None:
                        continue
                    tiers = [P.get(f"{g}/K{k}/{s}/{t}", {}).get("r_eff", float("nan")) for t in ("hot", "mid", "cold")]
                    spt = runs[L]["sec_per_tensor"].get(f"K{k}-{g}-{s}", float("nan"))
                    table.setdefault(g, {}).setdefault(str(k), {}).setdefault(s, {})[str(L)] = {
                        **{x: p[x] for x in ("r_eff", "r_eff_direct", "r_eff_agg", "real_rho", "ctrl_rho", "real_corr_rho",
                                             "direct_rho", "relerr_mean", "proxy_mean", "r_eff_unweighted_median", "n_experts")},
                        "tiers": dict(zip(("hot", "mid", "cold"), tiers)), "sec_per_tensor": spt}
                    lines.append(f"{g:3s} {k} {s:3s} {L:5d} {p['r_eff']:.4f} {p['r_eff_direct']:.4f} {p['r_eff_agg']:.4f} "
                                 f"{p['real_rho']:.3e} {p['ctrl_rho']:.3e} {p['relerr_mean']:.4f} {p['proxy_mean']:.5f} "
                                 + " ".join(f"{t:.4f}" for t in tiers) + f" {spt:6.2f}")
            lines.append("")
    model = {}
    for g in table:
        for k in table[g]:
            for s in table[g][k]:
                ls = sorted(int(L) for L in table[g][k][s])
                model.setdefault(g, {}).setdefault(k, {})[s] = fit(ls, [table[g][k][s][str(L)]["r_eff"] for L in ls])
            model[g][k]["primary"] = model[g][k]["out" if "out" in model[g][k] else "in"]
    lines.append("model (x = layer/39): group K set chosen coef rms")
    for g in model:
        for k in model[g]:
            for s in sets:
                m = model[g][k][s]
                lines.append(f"  {g:3s} K{k} {s:3s} {m['chosen']:6s} " + " ".join(f"{c:+.5f}" for c in m["fits"][m["chosen"]]["coef"])
                             + f"  rms {m['fits'][m['chosen']]['rms']:.4f} (const rms {m['fits']['const']['rms']:.4f})")
    out = {"layers": layers, "ks": ks, "sets": sets, "table": table, "model": model,
           "convention": "r: W + r*rms(W)*G per tensor (w1,w3 for GU; w2 for D), as tools/calib/native.py; "
                         "primary = held-out eval set when available",
           "sources": {str(L): str(Path(a.dir) / f"r_eff_L{L:02d}.json") for L in layers}}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    Path(a.table).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
