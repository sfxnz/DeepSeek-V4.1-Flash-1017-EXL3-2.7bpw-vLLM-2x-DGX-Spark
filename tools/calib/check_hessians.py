#!/usr/bin/env python3
"""Consistency check of a Hessian capture dir against the routing stats of the same panel run.

  python3 tools/calib/check_hessians.py --hdir DIR --stats stats.npz --layers 3

Checks per layer: one file per expert (+ prior_gu), done.json, symmetry, PSD (min eigenvalue of a few
experts), sum of per-expert token counts == routed picks in the stats, sum of wsum == routed mass.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdir", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--layers", default="3")
    a = ap.parse_args()
    st = np.load(a.stats)
    ok = True
    report = {}
    for L in [int(x) for x in a.layers.split(",")]:
        d = Path(a.hdir) / f"L{L:02d}"
        done = json.loads((d / "done.json").read_text())
        files = sorted(d.glob("e*.safetensors"))
        n_tot, w_tot, asym, mineig = 0, 0.0, 0.0, []
        for i, f in enumerate(files):
            with safe_open(str(f), "pt") as fh:
                meta = {k: json.loads(v) for k, v in fh.metadata().items()}
            n_tot += meta["n"]
            w_tot += meta["wsum"]
            if i % 64 == 0 and meta["n"] > 0:
                t = load_file(str(f))
                if "x" in t:  # fmt x: rebuild H from the stored factors
                    ww = t["w"].double().pow(meta["wpow"]).unsqueeze(-1)
                    x, act = t["x"].double(), t["act"].double()
                    t = {"H_gu": (x * ww).t() @ x, "H_d": (act * ww).t() @ act}
                    assert x.shape[0] == meta["n"]
                H = t["H_gu"].double()
                asym = max(asym, ((H - H.t()).abs().max() / H.abs().max()).item())
                ev = torch.linalg.eigvalsh(t["H_d"].double())
                mineig.append(ev.min().item() / ev.max().item())
        counts = st["counts"][L].sum()
        mass = st["mass"][L].sum()
        r = {"files": len(files), "n_sum": n_tot, "stats_picks": float(counts), "wsum": w_tot, "stats_mass": float(mass),
             "max_rel_asym": asym, "min_rel_eig_Hd": min(mineig) if mineig else None, "done": {k: v for k, v in done.items() if k != "experts"}}
        good = len(files) == 384 and n_tot == int(counts) and abs(w_tot - mass) / mass < 1e-4 and asym < 1e-5
        r["ok"] = good
        ok &= good
        report[L] = r
    print(json.dumps(report, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
