#!/usr/bin/env python3
"""The image's vllm_exl3_c (Dockerfile.e15) vs the bench module, per (K_gu, K_d) (GPU, e15 image).

  python3 kernel_study/p2b_kbits/real_ext_kbits.py --pairs 2:2,3:3,4:4,3:2,2:3 --out /data/runs/real-ext.json

The parent builds inputs per pair (census routings at m = 1, 4, 8; real and one-hot routing weights),
runs bench_kbits' module (variants 0, 1, 2) and starts one child per DSV41_P2B_COOP value (unset, 1, 2)
that imports the installed vllm_exl3_c and makes the same p2b_fused_moe calls. Pass when, bitwise:
  unset: one-hot outputs == bench v0;  "1": real-weight outputs == bench v1;  "2": == bench v2
and the child's stderr has the dataflow engaged line for "2" only and never the disarm line.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LOG_ENGAGED = "dsv41: p2b coop dataflow kernel engaged (DSV41_P2B_COOP=2)"
LOG_DISARMED = "dsv41: p2b coop dataflow lever is OFF for this call"
WORK = Path(os.environ.get("KB_DATA", "/data")) / "real_ext_work"


def child(inputs: Path, out: Path) -> int:
    mode = os.environ.get("DSV41_P2B_COOP")
    import torch

    import bench_kbits as bk  # pops DSV41_P2B_COOP; restore it for the real .so

    if mode is not None:
        os.environ["DSV41_P2B_COOP"] = mode
    import vllm_exl3_c

    data = torch.load(inputs)
    kgu, kd = data["pair"]
    layer = bk.Layer(data["layer"], 0, kgu, kd, "cuda")
    res = {}
    for name, case in data["cases"].items():
        x, ids = case["x"].cuda(), case["ids"].cuda()
        for wname in ("rw", "onehot"):
            o = torch.empty_like(x)
            vllm_exl3_c.p2b_fused_moe(x, o, *layer.tables, ids, case[wname].cuda(), kgu, kgu, kd, True, bk.INTER,
                                      bk.SWIGLU_LIMIT)
            res[f"{name}/{wname}"] = o.cpu()
    torch.cuda.synchronize()
    torch.save({"out": res, "so": str(vllm_exl3_c.__file__)}, out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="2:2,3:3,4:4,3:2,2:3,3:4,4:3,4:2,2:4")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--routings", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--child", nargs=2, type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return child(*args.child)

    import torch

    import bench_kbits as bk

    WORK.mkdir(parents=True, exist_ok=True)
    ext = bk.build_ext(False)
    bk.R.CENSUS = bk.DATA / "census_npy"
    result, ok = {"pairs": {}}, True
    for pair in args.pairs.split(","):
        kgu, kd = (int(k) for k in pair.split(":"))
        layer = bk.Layer(args.layer, 0, kgu, kd, "cuda")
        routes = bk.R.Routings(args.layer, random.Random(5))
        b = bk.Bench(ext, layer, "cuda", 5)
        cases = {}
        for m in (1, 4, 8):
            for t in range(args.routings):
                rw = b.rw(m)
                onehot = torch.zeros_like(rw)
                onehot[:, t % bk.TOPK] = 1.0
                cases[f"m{m}/t{t}"] = {"x": b.x(m).cpu(), "ids": b.ids(routes.draw("census", m)).cpu(),
                                       "rw": rw.cpu(), "onehot": onehot.cpu()}
        bench = {(v, f"{n}/{w}"): b.once(v, c["x"].cuda(), c["ids"].cuda(), c[w].cuda()).cpu()
                 for v in (0, 1, 2) for n, c in cases.items() for w in ("rw", "onehot")}
        del b, layer
        torch.cuda.empty_cache()
        inputs = WORK / f"inputs_{kgu}{kd}.pt"
        torch.save({"layer": args.layer, "pair": (kgu, kd), "cases": cases}, inputs)
        entry = {}
        for mode in (None, "1", "2"):
            env = {k: v for k, v in os.environ.items() if k != "DSV41_P2B_COOP"}
            if mode is not None:
                env["DSV41_P2B_COOP"] = mode
            outp = WORK / f"child_{kgu}{kd}_{mode or 'unset'}.pt"
            p = subprocess.run([sys.executable, __file__, "--out", str(args.out), "--child", str(inputs), str(outp)],
                               env=env, capture_output=True, text=True)
            if p.returncode != 0:
                entry[mode or "unset"] = {"pass": False, "rc": p.returncode, "stderr": p.stderr[-2000:]}
                ok = False
                continue
            got = torch.load(outp)["out"]
            if mode is None:
                cmp = {k: torch.equal(got[k].view(torch.int16), bench[(0, k)].view(torch.int16)) for k in got if k.endswith("onehot")}
            else:
                v = int(mode)
                cmp = {k: torch.equal(got[k].view(torch.int16), bench[(v, k)].view(torch.int16)) for k in got if k.endswith("/rw")}
            engaged, disarmed = LOG_ENGAGED in p.stderr, LOG_DISARMED in p.stderr
            r = {"bitwise": all(cmp.values()), "calls": len(cmp), "engaged_line": engaged, "disarm_line": disarmed}
            r["pass"] = r["bitwise"] and engaged == (mode == "2") and not disarmed
            ok &= r["pass"]
            entry[mode or "unset"] = r
        result["pairs"][pair] = entry
        print(f"[real-ext] K_gu={kgu} K_d={kd}: {json.dumps(entry)}", flush=True)
    result["pass"] = ok
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"pass={ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
