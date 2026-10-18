#!/usr/bin/env python3
"""Fit per-(layer, group) sensitivity curves KL(r) from `cli.py sens --smooth` outputs and write curves.json
plus a summary table. Variant names are L{L}-{G}-r{r}[-s{seed}] (cli.sens_variant).

curves.json schema (also in its "doc" field):
  meta: {panel, mode, domains, n_scored_by_domain, sources, objective}
  curves: {"<layer>": {"GU"|"D": {
      params        routed-expert weights in the group (384 experts; GU = w1+w3, D = w2)
      r_values      [r ...] (seed 0)                kl_values  [token-mean KL(native||variant) ...]
      kl_dom_values {domain: [KL ...]}              kl_eq_values  [equal-domain-weight mean KL ...]
      a_total       KL_eq = a_total * r^2, fixed exponent 2, fitted in log space (geometric mean of KL_eq/r^2)
      a_tok         same for the token-weighted KL
      a_by_domain   {domain: a}
      resid         rms of log(KL_eq / (a_total r^2)) over r  (0 = exactly quadratic)
      p_free        free log-log slope d log KL_eq / d log r
      a2, a4        two-term fit KL_eq = a2 r^2 + a4 r^4 (relative least squares)
      out_rel_mse   routed-output rel. MSE (per-token mean) at the modified layer, per r
      drafter_input_move {"by_r": {r: {layer: rel MSE of drafter input h.mean(hc) vs native}}, "slope": mean_L(rel/r^2) at max r}
      seed_a        {seed: a_total} when extra seeds ran;  seed_cv  std/mean of seed_a (ddof 1)
      kl_top1pct_share  share of the KL in the top 1% tokens at the largest r
      a_by_row      {row id: a} (row-mean KL / r^2, log-space fit over r)
      top_row       {id, domain, share}: the row with the largest share of the token-weighted KL (at the largest r)
      a_total_excl_top_row  a_total recomputed from per-row KL without top_row (robustness check)
      fp8           {r, kl_eq, kl_tok} fp8-mode KL at that r, when run
  }}}
  meta.fp8_vs_smooth: {groups, spearman_kl_eq: rank correlation of fp8-mode KL_eq(r) vs smooth KL_eq at the same r}
  meta.sources: per input file: mode, wall_s, variants, tokens
"""

import argparse
import json
import math
import re
from pathlib import Path

NAME = re.compile(r"^L(\d+)-(GU|D)-r([0-9.]+)(?:-s(\d+))?$")
N_EXP, DIM, INTER = 384, 5120, 2304
PARAMS = {"GU": N_EXP * 2 * INTER * DIM, "D": N_EXP * DIM * INTER}


def eq_mean(by_dom: dict, domains) -> float:
    return sum(by_dom[d]["kl_mean"] for d in domains) / len(domains)


def fit_log(rs, ks):
    """a with k = a r^2 in log space, rms log residual, free slope."""
    pts = [(r, k) for r, k in zip(rs, ks) if r > 0 and k > 0]
    if not pts:
        return None, None, None
    la = [math.log(k / r**2) for r, k in pts]
    a = math.exp(sum(la) / len(la))
    resid = math.sqrt(sum((x - math.log(a)) ** 2 for x in la) / len(la))
    p = None
    if len(pts) >= 2:
        x = [math.log(r) for r, _ in pts]
        y = [math.log(k) for _, k in pts]
        mx, my = sum(x) / len(x), sum(y) / len(y)
        p = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / max(sum((xi - mx) ** 2 for xi in x), 1e-30)
    return a, resid, p


def fit_24(rs, ks):
    """KL = a2 r^2 + a4 r^4, least squares with weights 1/KL^2 (relative)."""
    rows = [(r**2, r**4, k) for r, k in zip(rs, ks) if r > 0 and k > 0]
    if len(rows) < 2:
        return None, None
    s11 = sum((u / k) ** 2 for u, v, k in rows)
    s12 = sum(u * v / k**2 for u, v, k in rows)
    s22 = sum((v / k) ** 2 for u, v, k in rows)
    b1 = sum(u / k for u, v, k in rows)
    b2 = sum(v / k for u, v, k in rows)
    det = s11 * s22 - s12**2
    if abs(det) < 1e-300:
        return None, None
    return (b1 * s22 - b2 * s12) / det, (s11 * b2 - s12 * b1) / det


def spearman(x, y):
    def ranks(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(o):
            r[i] = float(k)
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def load(paths):
    var, meta = {}, {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        for n, v in d["variants"].items():
            v = dict(v, _src=str(p), _mode=d.get("mode", "fp8"))
            var.setdefault(d.get("mode", "fp8"), {})[n] = v
        meta.setdefault("sources", []).append({"path": str(p), "mode": d.get("mode", "fp8"), "wall_s": d.get("wall_s"),
                                               "variants": len(d["variants"]), "tokens": d.get("tokens"),
                                               "panel": d.get("panel")})
        meta["panel"] = d.get("panel")
    return var, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="sens output JSONs (smooth and fp8 mode)")
    ap.add_argument("--out", required=True, help="curves.json")
    ap.add_argument("--table", required=True, help="summary table (markdown)")
    a = ap.parse_args()
    var, meta = load(a.inputs)
    rowlen, rowdom = {}, {}
    if meta.get("panel") and Path(meta["panel"]).exists():  # scored targets per row (score-all): n - 1 - image tokens
        for line in Path(meta["panel"]).read_text().splitlines():
            r = json.loads(line)
            rowlen[r["id"]] = r["n_tokens"] - 1 - sum(im["n_tokens"] for im in r.get("images") or [])
            rowdom[r["id"]] = r["domain"]
    sm = var.get("smooth", {})
    domains = sorted({dm for v in sm.values() for dm in v["by_domain"]})
    n_by_dom = {}
    for v in sm.values():
        for dm, x in v["by_domain"].items():
            n_by_dom[dm] = x["n"]
    groups = {}
    for n, v in sm.items():
        m = NAME.match(n)
        if not m:
            continue
        L, g, r, sd = int(m[1]), m[2], float(m[3]), int(m[4] or 0)
        groups.setdefault((L, g), {}).setdefault(sd, []).append((r, v))
    curves = {}
    for (L, g), seeds in sorted(groups.items()):
        pts = sorted(seeds.get(0, []), key=lambda t: t[0])
        pts = [(r, v) for r, v in pts if r > 0]
        if not pts:
            continue
        rs = [r for r, _ in pts]
        keq = [eq_mean(v["by_domain"], domains) for _, v in pts]
        ktok = [v["kl_mean"] for _, v in pts]
        a_eq, resid, p = fit_log(rs, keq)
        a2, a4 = fit_24(rs, keq)
        c = {
            "params": PARAMS[g], "r_values": rs, "kl_values": ktok, "kl_eq_values": keq,
            "kl_dom_values": {dm: [v["by_domain"][dm]["kl_mean"] for _, v in pts] for dm in domains},
            "a_total": a_eq, "a_tok": fit_log(rs, ktok)[0],
            "a_by_domain": {dm: fit_log(rs, [v["by_domain"][dm]["kl_mean"] for _, v in pts])[0] for dm in domains},
            "resid": resid, "p_free": p, "a2": a2, "a4": a4,
            "out_rel_mse": [v.get("routed_out_rel_mse_tokmean") for _, v in pts],
            "kl_top1pct_share": pts[-1][1].get("kl_top1pct_share"),
        }
        rows_ = sorted(pts[-1][1].get("kl_by_row", {}))
        if rows_:
            c["a_by_row"] = {rid: fit_log(rs, [v["kl_by_row"][rid] for _, v in pts])[0] for rid in rows_}
            kb = pts[-1][1]["kl_by_row"]
            ntok = {rid: rowlen.get(rid, 1) for rid in rows_}
            tot = sum(kb[rid] * ntok[rid] for rid in rows_)
            top = max(rows_, key=lambda rid: kb[rid] * ntok[rid])
            c["top_row"] = {"id": top, "domain": rowdom.get(top), "share": kb[top] * ntok[top] / tot if tot else None}
            if rowdom:
                def eq_rows(v, drop):
                    acc = {}
                    for rid in rows_:
                        if rid != drop:
                            x = acc.setdefault(rowdom[rid], [0.0, 0])
                            x[0] += v["kl_by_row"][rid] * ntok[rid]
                            x[1] += ntok[rid]
                    return sum(x[0] / x[1] for x in acc.values()) / len(acc)
                c["a_total_excl_top_row"] = fit_log(rs, [eq_rows(v, top) for _, v in pts])[0]
        dm = {str(r): v.get("drafter_in_rel_mse", {}) for r, v in pts}
        rmax = rs[-1]
        last = pts[-1][1].get("drafter_in_rel_mse", {})
        c["drafter_input_move"] = {"by_r": dm, "slope": (sum(last.values()) / len(last) / rmax**2) if last else None}
        if len(seeds) > 1:
            sa = {}
            for sd, sp in seeds.items():
                sp = sorted((t for t in sp if t[0] > 0), key=lambda t: t[0])
                sa[str(sd)] = fit_log([r for r, _ in sp], [eq_mean(v["by_domain"], domains) for _, v in sp])[0]
            vals = [x for x in sa.values() if x]
            mu = sum(vals) / len(vals)
            sdv = math.sqrt(sum((x - mu) ** 2 for x in vals) / max(1, len(vals) - 1))
            c["seed_a"] = sa
            c["seed_cv"] = sdv / mu if mu else None
        f8 = [(float(NAME.match(n)[3]), v) for n, v in var.get("fp8", {}).items()
              if NAME.match(n) and int(NAME.match(n)[1]) == L and NAME.match(n)[2] == g and not NAME.match(n)[4]]
        if f8:
            r8, v8 = max(f8, key=lambda t: t[0])
            c["fp8"] = {"r": r8, "kl_tok": v8["kl_mean"],
                        "kl_eq": eq_mean(v8["by_domain"], [x for x in domains if x in v8["by_domain"]]),
                        "smooth_kl_eq_same_r": next((k for r, k in zip(rs, keq) if abs(r - r8) < 1e-12), None)}
        curves.setdefault(str(L), {})[g] = c
    cmp = [(f"{L}:{g}", c["fp8"]["kl_eq"], c["fp8"]["smooth_kl_eq_same_r"]) for L, gs in curves.items()
           for g, c in gs.items() if "fp8" in c and c["fp8"]["smooth_kl_eq_same_r"] is not None]
    if len(cmp) >= 2:
        meta["fp8_vs_smooth"] = {"groups": [t[0] for t in cmp], "fp8_kl_eq": [t[1] for t in cmp],
                                 "smooth_kl_eq": [t[2] for t in cmp],
                                 "spearman_kl_eq": spearman([t[1] for t in cmp], [t[2] for t in cmp])}
    meta.update({"mode": "smooth", "domains": domains, "n_scored_by_domain": n_by_dom,
                 "objective": "a_total uses equal domain weights (B-6); a_tok is token-weighted"})
    out = {"doc": __doc__, "meta": meta, "curves": curves}
    Path(a.out).write_text(json.dumps(out, indent=1))

    rows = [(L, g, c) for L, gs in curves.items() for g, c in gs.items()]
    rows.sort(key=lambda t: -(t[2]["a_total"] or 0) * t[2]["params"])
    lines = ["| rank | layer | group | a_total | a·params (1e9) | resid | p_free | a_tok | seed_cv | drafter slope | top row (share) | a excl top row | "
             + " | ".join(f"a[{d}]" for d in domains) + " |",
             "|" + "---|" * (12 + len(domains))]
    f = lambda x, s="{:.4g}": "-" if x is None else s.format(x)
    for i, (L, g, c) in enumerate(rows):
        lines.append(f"| {i+1} | {L} | {g} | {f(c['a_total'])} | {f((c['a_total'] or 0) * c['params'] / 1e9)} | "
                     f"{f(c['resid'], '{:.3f}')} | {f(c['p_free'], '{:.2f}')} | {f(c['a_tok'])} | {f(c.get('seed_cv'), '{:.3f}')} | "
                     f"{f(c['drafter_input_move']['slope'])} | {c.get('top_row', {}).get('id', '-')} ({f(c.get('top_row', {}).get('share'), '{:.2f}')}) | "
                     f"{f(c.get('a_total_excl_top_row'))} | " + " | ".join(f(c["a_by_domain"][d]) for d in domains) + " |")
    tot = 0.0
    lines += ["", "| source | mode | variants | wall_s |", "|---|---|---|---|"]
    for src in meta["sources"]:
        lines.append(f"| {Path(src['path']).name} | {src['mode']} | {src['variants']} | {src['wall_s']} |")
        tot += src["wall_s"] or 0
    lines.append(f"| total | | | {tot:.0f} s = {tot / 3600:.2f} h |")
    if "fp8_vs_smooth" in meta:
        m = meta["fp8_vs_smooth"]
        lines += ["", f"fp8-mode vs smooth KL_eq at r = {curves[m['groups'][0].split(':')[0]][m['groups'][0].split(':')[1]]['fp8']['r']}: "
                  f"Spearman {m['spearman_kl_eq']:.3f}", "| group | fp8 KL_eq | smooth KL_eq |", "|---|---|---|"]
        lines += [f"| {g} | {x:.4g} | {y:.4g} |" for g, x, y in zip(m["groups"], m["fp8_kl_eq"], m["smooth_kl_eq"])]
    Path(a.table).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
