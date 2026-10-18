#!/usr/bin/env python3
"""Served-model KL / NLL against the native (unquantized) 1017 panel dump. HTTP + stdlib only.

The native harness (`tools/calib/cli.py panel --dump-topk 128`) writes, per panel row, at every
scored position p (predicting token p+1): the top-128 ids/logprobs of the full fp32 softmax and the
NLL of the actual token. This tool sends the same token ids to the serve (/v1/completions,
prompt_logprobs=K, K <= the serve's --max-logprobs, vLLM default 20) and compares position by
position:

  dnll    served NLL - native NLL of the actual next token (exact, no top-k truncation)
  top1    served argmax == native argmax
  kl      KL(native || served) on the native top-K tokens plus one tail bucket. A served logprob
          missing from the served top-K is imputed as the served K-th logprob (an upper bound on
          q, so that term is underestimated); `missing` is the share of such terms. Coarse-graining
          only lowers KL, so with no missing terms the estimate is a lower bound of the full KL.

Image rows are skipped (/v1/completions takes no images; vision is checked by smoke_vision.py and
quality_eval.py vision). Rows longer than --max-len are skipped (prompt_logprobs materialises the
logits of every prompt token of a chunk; boot with MAX_NUM_BATCHED_TOKENS=2048 if headroom is tight).

  python3 tools/served_kl.py --panel /home/sfxnz/projects/data/dsv41-1017/calib/panel.jsonl \
      --native /home/sfxnz/projects/data/dsv41-1017/calib-harness/panel_native.topk.safetensors \
      --out served_kl.json
"""
from __future__ import annotations

import argparse
import array
import json
import math
import struct
import sys
import time
import urllib.error
import urllib.request

_DT = {"F32": "f", "I32": "i", "I64": "q"}


class TopK:
    """Minimal safetensors reader for the native dump (per-row pos/ids/logprobs/nll)."""

    def __init__(self, path: str):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.base = 8 + n
        self.h = json.loads(self.f.read(n))

    def rows(self) -> list[str]:
        return sorted({k.split("/")[0] for k in self.h if "/" in k})

    def get(self, key: str):
        v = self.h[key]
        a, b = v["data_offsets"]
        self.f.seek(self.base + a)
        arr = array.array(_DT[v["dtype"]])
        arr.frombytes(self.f.read(b - a))
        return arr, v["shape"]

    def row(self, rid: str) -> dict:
        pos, _ = self.get(f"{rid}/pos")
        ids, (n, k) = self.get(f"{rid}/ids")
        lps, _ = self.get(f"{rid}/logprobs")
        nll, _ = self.get(f"{rid}/nll")
        return {"pos": list(pos), "k": k, "nll": list(nll),
                "ids": [ids[i * k:(i + 1) * k] for i in range(n)],
                "lps": [lps[i * k:(i + 1) * k] for i in range(n)]}


def kl_topk(p_ids, p_lps, q: dict[int, float], k: int) -> tuple[float, int]:
    """KL(P||Q) over P's top-k tokens + one tail bucket. q: served {id: logprob} (top-K + actual)."""
    vals = sorted(q.values(), reverse=True)
    q_floor = vals[min(k, len(vals)) - 1]  # served K-th logprob (q also holds the actual token)
    kl, sp, sq, miss = 0.0, 0.0, 0.0, 0
    for t, lp in zip(p_ids[:k], p_lps[:k]):
        lq = q.get(t)
        if lq is None:
            lq, miss = q_floor, miss + 1
        p = math.exp(lp)
        kl += p * (lp - lq)
        sp += p
        sq += math.exp(lq)
    pt, qt = max(1.0 - sp, 0.0), max(1.0 - sq, 1e-12)
    if pt > 0:
        kl += pt * math.log(pt / qt)
    return max(kl, 0.0), miss


def post(url: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()[:300]}") from exc


def score_row(rec: dict, nat: dict, plp: list, k: int) -> dict:
    """Per-position comparison; plp[i] = served {id: logprob} for prompt token i."""
    toks = rec["tokens"]
    out = {"n": 0, "dnll": 0.0, "nll_native": 0.0, "nll_served": 0.0, "top1": 0, "kl": 0.0,
           "missing": 0, "terms": 0}
    for j, p in enumerate(nat["pos"]):
        q = plp[p + 1]
        lq_true = q[toks[p + 1]]
        out["n"] += 1
        out["nll_native"] += nat["nll"][j]
        out["nll_served"] += -lq_true
        out["dnll"] += -lq_true - nat["nll"][j]
        out["top1"] += int(max(q, key=q.get) == nat["ids"][j][0])
        kl, miss = kl_topk(nat["ids"][j], nat["lps"][j], q, k)
        out["kl"] += kl
        out["missing"] += miss
        out["terms"] += min(k, nat["k"])
    return out


def summarize(parts: list[dict]) -> dict:
    n = sum(p["n"] for p in parts)
    if not n:
        return {"n": 0}
    return {"n": n,
            "kl": round(sum(p["kl"] for p in parts) / n, 6),
            "dnll": round(sum(p["dnll"] for p in parts) / n, 6),
            "nll_native": round(sum(p["nll_native"] for p in parts) / n, 6),
            "nll_served": round(sum(p["nll_served"] for p in parts) / n, 6),
            "top1": round(sum(p["top1"] for p in parts) / n, 5),
            "missing": round(sum(p["missing"] for p in parts) / max(sum(p["terms"] for p in parts), 1), 5)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--native", required=True)
    ap.add_argument("--k", type=int, default=20, help="prompt_logprobs (<= serve --max-logprobs)")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    url = a.url.rstrip("/")
    with urllib.request.urlopen(url + "/v1/models", timeout=60) as r:
        model = a.model or json.loads(r.read())["data"][0]["id"]
    nat = TopK(a.native)
    have = set(nat.rows())
    recs = [json.loads(line) for line in open(a.panel) if line.strip()]
    rows, skipped = [], {"image": 0, "long": 0, "no_native": 0}
    t0 = time.time()
    for rec in recs:
        if rec["images"]:
            skipped["image"] += 1
            continue
        if rec["n_tokens"] > a.max_len:
            skipped["long"] += 1
            continue
        if rec["id"] not in have:
            skipped["no_native"] += 1
            continue
        t = time.time()
        resp = post(url + "/v1/completions", {"model": model, "prompt": rec["tokens"], "max_tokens": 1,
                                              "temperature": 0, "prompt_logprobs": a.k}, a.timeout)
        plp = [None] + [{int(t_): v["logprob"] for t_, v in e.items()}
                        for e in resp["choices"][0]["prompt_logprobs"][1:]]
        s = score_row(rec, nat.row(rec["id"]), plp, a.k)
        s.update(id=rec["id"], domain=rec["domain"], n_tokens=rec["n_tokens"], s=round(time.time() - t, 2))
        rows.append(s)
        print(f"{rec['id']} {rec['domain']:<12} n={s['n']:<5} kl={s['kl'] / max(s['n'], 1):.4f} "
              f"dnll={s['dnll'] / max(s['n'], 1):+.4f} top1={s['top1'] / max(s['n'], 1):.3f} {s['s']}s",
              flush=True)
    doms = sorted({r["domain"] for r in rows})
    res = {"model": model, "k": a.k, "max_len": a.max_len, "rows": len(rows), "skipped": skipped,
           "elapsed_s": round(time.time() - t0, 1), "all": summarize(rows),
           "by_domain": {d: summarize([r for r in rows if r["domain"] == d]) for d in doms},
           "per_row": rows}
    print("SERVED_KL " + json.dumps({"all": res["all"], "by_domain": res["by_domain"]}))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
