#!/usr/bin/env python3
"""r_eff calibration (decision B-2): the iid relative-noise level whose routed-output error equals a REAL
Hessian EXL3 encode's, per (layer, group, K), on sample experts' own routed calib inputs.

Inputs (from capture_h.py --experts strat16 --dump-factors N --holdout-mod M):
  --hess    dsv41-hessian-v1 root (full calib H)       --train  same from the train fold (row % M != 0)
  --factors routed tokens of the selected experts: x bf16 [n, dim], w (routing weight), row
For each selected expert e of a layer, K in --ks, group GU (w1+w3) and D (w2):
  real    build_pack_1017.encode_hessian (the builder's real-H encoder, batch --batch experts) ->
          reconstruct (exllamav3 reconstruct_had_slice); weight relerr and the builder's proxy recorded
  noise   W + r * rms(W) * G, G iid N(0,1) per tensor, exactly native.Harness._variant_weights (w1 and w3
          for GU, w2 for D), at every r of --r-grid; one draw per (expert, group) shared across r;
          evaluated in fp32 (weights and activations) against the fp32 reference by default
  control native weights through the EXL3 serving numerics (fp16 activations, no act quant)
Output error of one expert on token t, y_t = w_t * W2 (silu(clamp W1 x) * clamp W3 x):
  rho = sum_t w_t^2 |dy_t|^2/|y_t|^2 / sum_t w_t^2   (per-token relative MSE, routed-weight^2 weighted)
  agg = sum_t w_t^2 |dy_t|^2 / sum_t w_t^2 |y_t|^2   (aggregate, dominated by large-output tokens)
References: real and control vs the native reference numerics (fp8 act quant, bf16 GEMMs, as native.py
"native"); noise vs the reference in its own numerics (--noise-numerics fp32 default, smooth = B-1 bf16, or fp8).
  rho_real_corr = rho_real - rho_ctrl (quadrature), rho_direct = real vs control (same numerics).
Eval sets: "in" = full-H encode evaluated on all kept routed tokens (in-sample); "out" = train-fold-H
encode evaluated on the held-out rows (row % M == 0) only.
r_eff solves rho_noise(r) = rho_real_corr by log-log interpolation on the grid (pooled over the layer's
experts with weights W_e = sum of routed w^2 over the full calib, also per tier).

  python3 tools/calib/r_eff.py --hess H --train HT --factors F --layer 3 --out DIR/r_eff_L03.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import refkernel as rk  # noqa: E402
from native import Cfg, Harness, Variant  # noqa: E402

SNAP = Path("/home/sfxnz/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/"
            "1bae0e1088993b9b82de08f59de5be5a34e42170")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def expert_out(x, w, w13, w2, numerics: str, cfg: Cfg):
    """One expert on its tokens, weighted output fp32 [n, dim]. numerics: fp8 (native.py "native"),
    smooth (bf16 activations, no fake quant), fp32 (fp32 activations and weights), swap (EXL3 serving:
    fp16 activations and weights)."""
    inter = cfg.moe_inter_dim
    if numerics == "fp32":  # exact math: fp32 activations and weights (the cleanest iid-noise reference)
        gu = x.float() @ w13.float().t()
    elif numerics == "swap":
        gu = torch.mm(x.half(), w13.half().t(), out_dtype=torch.float32).to(torch.bfloat16).float()
    else:
        xq = rk.act_quant_dequant(x, 32) if numerics == "fp8" else x
        gu = torch.matmul(xq, w13.t()).float()
    g, u = gu[:, :inter], gu[:, inter:]
    if cfg.swiglu_limit > 0:
        u = torch.clamp(u, min=-cfg.swiglu_limit, max=cfg.swiglu_limit)
        g = torch.clamp(g, max=cfg.swiglu_limit)
    a = w.unsqueeze(-1) * (F.silu(g) * u)
    if numerics == "fp32":
        return a @ w2.float().t()
    a = a.to(torch.bfloat16)
    if numerics == "swap":
        return torch.mm(a.half(), w2.half().t(), out_dtype=torch.float32).to(torch.bfloat16).float()
    aq = rk.act_quant_dequant(a, 32) if numerics == "fp8" else a
    return torch.matmul(aq, w2.t()).float()


def err(y, y0, w):
    """(rho, agg) of y vs reference y0, routed-weight^2 weighted."""
    d2 = (y - y0).double().square().sum(-1)
    n2 = y0.double().square().sum(-1).clamp_min(1e-30)
    g = w.double().square()
    return float((g * d2 / n2).sum() / g.sum()), float((g * d2).sum() / (g * n2).sum())


def chunked_err(fn_y, fn_y0, x, w, chunk=4096):
    """err() accumulated over token chunks (fn_* map (x, w) -> outputs)."""
    s_rho = s_g = s_d = s_n = 0.0
    for a in range(0, x.size(0), chunk):
        xa, wa = x[a : a + chunk], w[a : a + chunk]
        y, y0 = fn_y(xa, wa), fn_y0(xa, wa)
        d2 = (y - y0).double().square().sum(-1)
        n2 = y0.double().square().sum(-1).clamp_min(1e-30)
        g = wa.double().square()
        s_rho += float((g * d2 / n2).sum())
        s_g += float(g.sum())
        s_d += float((g * d2).sum())
        s_n += float((g * n2).sum())
    return s_rho / max(s_g, 1e-300), s_d / max(s_n, 1e-300)


def solve_r(grid, rhos, target):
    """r with rho_noise(r) = target, log-log interpolation (extrapolated with the end slopes)."""
    if not (target > 0):
        return 0.0
    lr = [math.log(r) for r in grid]
    lp = [math.log(max(p, 1e-300)) for p in rhos]
    t = math.log(target)
    i = 0
    while i < len(grid) - 2 and t > lp[i + 1]:
        i += 1
    s = (lp[i + 1] - lp[i]) / (lr[i + 1] - lr[i])
    return math.exp(lr[i] + (t - lp[i]) / s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(SNAP))
    ap.add_argument("--hess", required=True)
    ap.add_argument("--train", default=None, help="train-fold H root (held-out eval); omit for in-sample only")
    ap.add_argument("--factors", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--ks", default="2,3,4")
    ap.add_argument("--r-grid", default="0.02,0.05,0.08,0.12,0.18,0.25,0.35,0.5")
    ap.add_argument("--holdout-mod", type=int, default=5)
    ap.add_argument("--noise-numerics", default="fp32", choices=["fp32", "smooth", "fp8"],
                    help="numerics of the noise path and its reference (fp32: noisy weights kept fp32)")
    ap.add_argument("--batch", type=int, default=8, help="experts per encode_hessian call (builder default 8)")
    ap.add_argument("--max-tokens", type=int, default=0, help="cap eval tokens per expert (0 = all kept)")
    ap.add_argument("--experts", default="", help="comma list: only these of the layer's selection (tests)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    log("argv", sys.argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    import build_pack_1017 as B
    from exllamav3.ext import exllamav3_ext as ext
    from safetensors import safe_open
    from quantize_experts_exl3 import _dequant_t
    from requant_probe import relerr

    dev = torch.device("cuda:0")
    cfg = Cfg.load(Path(a.src) / "inference" / "config.json")
    L = a.layer
    ks = [int(k) for k in a.ks.split(",")]
    grid = [float(r) for r in a.r_grid.split(",")]
    meta = json.loads((Path(a.hess) / f"layer_{L:02d}" / "meta.json").read_text())
    sel = ({int(e): v for e, v in meta["selection"].items()} if "selection" in meta
           else {int(e): {} for e in meta["experts"]})
    if a.experts:
        sel = {e: v for e, v in sel.items() if e in {int(x) for x in a.experts.split(",")}}
    evsets = ("in", "out") if a.train else ("in",)
    experts = sorted(sel)
    wp = {e: meta["experts"][str(e)]["wpsum"] for e in experts}
    src_map = json.loads((Path(a.src) / "model.safetensors.index.json").read_text())["weight_map"]
    fname = src_map[f"layers.{L}.ffn.experts.0.w1.weight"]
    # native.Harness._variant_weights needs only these; noisy weights stay fp32 for --noise-numerics fp32
    fake = SimpleNamespace(dev=dev, pack=None,
                           noisy_dtype=lambda w: torch.float32 if a.noise_numerics == "fp32" else w.dtype)

    def recon(enc, shape, k):
        o = torch.empty(shape, dtype=torch.half, device=dev)
        ext.reconstruct_had_slice(o, enc["trellis"].to(dev), enc["suh"].to(dev).half(), enc["svh"].to(dev).half(),
                                  k, True, False, 0)
        return o

    # source weights: builder orientation [in, out] fp32 (CPU) and native [out, in] bf16 (GPU)
    Wt, Wn = {}, {}
    with safe_open(str(Path(a.src) / fname), framework="pt") as fh:
        for e in experts:
            for p in ("w1", "w3", "w2"):
                s = f"layers.{L}.ffn.experts.{e}.{p}"
                Wt[(e, p)] = _dequant_t(fh.get_tensor(s + ".weight"), fh.get_tensor(s + ".scale"), "cpu")
                Wn[(e, p)] = rk.dequant_fp4_weight(fh.get_tensor(s + ".weight").to(dev), fh.get_tensor(s + ".scale").to(dev),
                                                   torch.bfloat16)
                assert torch.equal(Wn[(e, p)].float(), Wt[(e, p)].t().to(dev)), f"{s}: dequant paths differ"
    toks = {}
    for e in experts:
        with safe_open(str(Path(a.factors) / f"layer_{L:02d}" / f"expert_{e:03d}.safetensors"), framework="pt") as fh:
            x, w, row = fh.get_tensor("x").to(dev), fh.get_tensor("w").to(dev), fh.get_tensor("row").to(dev)
        if a.max_tokens and x.size(0) > a.max_tokens:
            x, w, row = x[: a.max_tokens], w[: a.max_tokens], row[: a.max_tokens]
        held = row % a.holdout_mod == 0
        toks[e] = {"in": (x, w), "out": (x[held], w[held])}
    log(f"layer {L}: {len(experts)} experts, tokens kept {[toks[e]['in'][0].size(0) for e in experts]}")

    res = {e: {"tier": sel[e].get("tier"), "rank": sel[e].get("rank"), "wpsum": wp[e],
               "n_routed": meta["experts"][str(e)]["n"], "n_in": int(toks[e]["in"][0].size(0)),
               "n_out": int(toks[e]["out"][0].size(0)), "groups": {}} for e in experts}
    w13n = {e: torch.cat([Wn[(e, "w1")], Wn[(e, "w3")]]) for e in experts}
    timing = {}
    # ---- noise and control (independent of K)
    for e in experts:
        for gname in ("GU", "D"):
            r_out = {}
            for ev in evsets:
                x, w = toks[e][ev]
                if x.size(0) == 0:
                    r_out[ev] = None
                    continue
                nn_ = a.noise_numerics
                ref_n = lambda xa, wa: expert_out(xa, wa, w13n[e], Wn[(e, "w2")], nn_, cfg)  # noqa: E731
                rows = []
                for r in grid:
                    var = Variant(name=f"reff-L{L}-{gname}", mods={L: ("noise", gname, r)}, seed=0)
                    n13, n2, _ = Harness._variant_weights(fake, L, e, var, w13n[e], Wn[(e, "w2")], ("noise", gname, r))
                    rows.append(chunked_err(lambda xa, wa: expert_out(xa, wa, n13, n2, nn_, cfg), ref_n, x, w))
                ref8 = lambda xa, wa: expert_out(xa, wa, w13n[e], Wn[(e, "w2")], "fp8", cfg)  # noqa: E731
                ctrl = chunked_err(lambda xa, wa: expert_out(xa, wa, w13n[e], Wn[(e, "w2")], "swap", cfg), ref8, x, w)
                r_out[ev] = {"noise_rho": [p[0] for p in rows], "noise_agg": [p[1] for p in rows],
                             "ctrl_rho": ctrl[0], "ctrl_agg": ctrl[1]}
            res[e]["groups"][gname] = {"noise": r_out, "K": {}}
    log("noise + control done")
    # ---- real encodes, full H (in-sample) and train-fold H (held-out)
    for k in ks:
        for gname, projs, hkey, bgroup in (("GU", ("w1", "w3"), "w13_in", "gu"), ("D", ("w2",), "w2_in", "d")):
            for hset, root in (("in", a.hess), ("out", a.train))[: len(evsets)]:
                for b0 in range(0, len(experts), a.batch):
                    chunk = experts[b0 : b0 + a.batch]
                    stems = [f"layers.{L}.ffn.experts.{e}.{p}" for e in chunk for p in projs]
                    ws = [Wt[(e, p)].clone() for e in chunk for p in projs]  # the encoder may work in place
                    hs_e = {e: B.load_hessian(Path(root), L, e, bgroup, Wt[(e, projs[0])].shape[0]) for e in chunk}
                    hs = [hs_e[e] for e in chunk for _ in projs]
                    torch.cuda.synchronize()
                    t0 = time.time()
                    encs = B.encode_hessian(ws, hs, stems, k, "cuda:0")
                    torch.cuda.synchronize()
                    dt = time.time() - t0
                    timing.setdefault(f"K{k}-{gname}-{hset}", []).append([len(stems), dt])
                    q = {}
                    for i, (stem, (enc, extra)) in enumerate(zip(stems, encs)):
                        e, p = chunk[i // len(projs)], projs[i % len(projs)]
                        wt = Wt[(e, p)].to(dev)
                        wq = recon(enc, wt.shape, k).float()
                        st = {"relerr": relerr(wq, wt),
                              "proxy": B.proxy_error(wt, wq, hs[i].to(dev)), **extra}
                        q[(e, p)] = (wq.t().contiguous(), st)
                    for e in chunk:
                        x, w = toks[e][hset]
                        g = res[e]["groups"][gname]["K"].setdefault(str(k), {})
                        stats = {p: q[(e, p)][1] for p in projs}
                        if x.size(0) == 0:
                            g[hset] = {"tensors": stats}
                            continue
                        w13q = torch.cat([q[(e, "w1")][0], q[(e, "w3")][0]]) if gname == "GU" else w13n[e]
                        w2q = q[(e, "w2")][0] if gname == "D" else Wn[(e, "w2")]
                        ref8 = lambda xa, wa: expert_out(xa, wa, w13n[e], Wn[(e, "w2")], "fp8", cfg)  # noqa: E731
                        ctl = lambda xa, wa: expert_out(xa, wa, w13n[e], Wn[(e, "w2")], "swap", cfg)  # noqa: E731
                        real = lambda xa, wa: expert_out(xa, wa, w13q, w2q, "swap", cfg)  # noqa: E731
                        r_real = chunked_err(real, ref8, x, w)
                        r_dir = chunked_err(real, ctl, x, w)
                        nz = res[e]["groups"][gname]["noise"][hset]
                        corr = r_real[0] - nz["ctrl_rho"]
                        g[hset] = {"tensors": stats, "real_rho": r_real[0], "real_agg": r_real[1],
                                   "direct_rho": r_dir[0], "direct_agg": r_dir[1], "real_corr_rho": corr,
                                   "real_corr_agg": r_real[1] - nz["ctrl_agg"],
                                   "r_eff": solve_r(grid, nz["noise_rho"], corr),
                                   "r_eff_direct": solve_r(grid, nz["noise_rho"], r_dir[0]),
                                   "r_eff_agg": solve_r(grid, nz["noise_agg"], r_real[1] - nz["ctrl_agg"])}
                    del encs, q
            log(f"K{k} {gname} done")
    # ---- pooled per (group, K, eval set), weights W_e (and per tier)
    pooled = {}
    for gname in ("GU", "D"):
        for k in ks:
            for hset in evsets:
                for tier in ("all", "hot", "mid", "cold"):
                    es = [e for e in experts if (tier == "all" or res[e]["tier"] == tier)
                          and res[e]["groups"][gname]["noise"][hset] is not None
                          and "real_rho" in res[e]["groups"][gname]["K"][str(k)][hset]]
                    if not es:
                        continue
                    wsum = sum(wp[e] for e in es)
                    pw = lambda f: sum(wp[e] * f(e) for e in es) / wsum  # noqa: E731
                    kk = lambda e: res[e]["groups"][gname]["K"][str(k)][hset]  # noqa: E731
                    nz = lambda e: res[e]["groups"][gname]["noise"][hset]  # noqa: E731
                    noise_rho = [pw(lambda e, i=i: nz(e)["noise_rho"][i]) for i in range(len(grid))]
                    noise_agg = [pw(lambda e, i=i: nz(e)["noise_agg"][i]) for i in range(len(grid))]
                    real_corr = pw(lambda e: kk(e)["real_corr_rho"])
                    rec = {"n_experts": len(es), "real_rho": pw(lambda e: kk(e)["real_rho"]),
                           "ctrl_rho": pw(lambda e: nz(e)["ctrl_rho"]), "real_corr_rho": real_corr,
                           "direct_rho": pw(lambda e: kk(e)["direct_rho"]),
                           "noise_rho": noise_rho, "r_eff": solve_r(grid, noise_rho, real_corr),
                           "r_eff_direct": solve_r(grid, noise_rho, pw(lambda e: kk(e)["direct_rho"])),
                           "r_eff_agg": solve_r(grid, noise_agg, pw(lambda e: kk(e)["real_corr_agg"])),
                           "relerr_mean": sum(sum(kk(e)["tensors"][p]["relerr"] for p in kk(e)["tensors"]) / len(kk(e)["tensors"]) for e in es) / len(es),
                           "proxy_mean": sum(sum(kk(e)["tensors"][p]["proxy"] for p in kk(e)["tensors"]) / len(kk(e)["tensors"]) for e in es) / len(es),
                           "r_eff_unweighted_median": sorted(kk(e)["r_eff"] for e in es)[len(es) // 2]}
                    pooled[f"{gname}/K{k}/{hset}/{tier}"] = rec
    sec = {key: sum(d for _, d in v) / sum(n for n, _ in v) for key, v in timing.items()}
    out = {"layer": L, "grid": grid, "ks": ks, "noise_numerics": a.noise_numerics, "holdout_mod": a.holdout_mod,
           "batch": a.batch, "pooled": pooled, "sec_per_tensor": sec, "experts": {str(e): res[e] for e in experts},
           "hess_meta": {k: meta[k] for k in ("m0", "mean_wp", "tokens", "wpow", "m0_mult", "code", "calib_sha256") if k in meta}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for key in sorted(pooled):
        if key.endswith("/all"):
            p = pooled[key]
            log(f"{key:14s} r_eff {p['r_eff']:.4f} direct {p['r_eff_direct']:.4f} agg {p['r_eff_agg']:.4f} "
                f"rho_real {p['real_rho']:.3e} ctrl {p['ctrl_rho']:.3e} relerr {p['relerr_mean']:.4f} proxy {p['proxy_mean']:.5f}")
    log("sec/tensor", json.dumps(sec))


if __name__ == "__main__":
    main()
