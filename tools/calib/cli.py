#!/usr/bin/env python3
"""Native layer-streaming harness for DeepSeek-V4.1-Flash-1017 (one GPU). Run inside dsv41-quant151 via
tools/calib/run_gpu.sh on spark2.

Commands (all write a JSON summary to --out; raw arrays next to it):
  gen     greedy chat generation (full re-forward per token; for validation only)
  nll     teacher-forced NLL on tests/quality/nll_passages.jsonl (BOS + passage, first 16 tokens unscored)
  panel   routing stats (--stats .npz), native top-k logits dump (--dump-topk K), per-token NLL,
          optional Hessian capture (--hessian-dir, --hessian-layers)
  sens    noise-injection sensitivity: variants L:GROUP:r (GROUP = GU | D | ALL), exact KL vs native
  swap    substitute EXL3-reconstructed experts from a pack (--pack) for --layers (each separately, or
          --joint for one all-layers variant), exact KL vs native

Panel JSONL rows (one of): {"tokens": [...]} -- incl. the calib/panel records of tools/calib/data (images
  with pre-expanded spans and loss_spans are honored: only assistant tokens are scored) | {"text": "...", "bos": true} |
  {"prompt": "<raw encoded prompt with <｜deepseek_image｜> placeholders>", "images": [{"url": path}...]} |
  {"messages": [...openai/harmony...], "thinking_mode": "chat"}  (+ optional "completion": text)
  Optional per row: "id", "domain", "score_start" (first scored target index; default 1, or the
  completion start when a completion is given).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native import D, GU, Harness, Head, Row, Variant  # noqa: E402

SNAP = Path.home() / ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
PASSAGES = Path(__file__).resolve().parents[2] / "tests/quality/nll_passages.jsonl"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def parse_layers(s: str, n: int = 40) -> list[int]:
    if s in ("all", ""):
        return list(range(n))
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def make_harness(a) -> Harness:
    over = json.loads(a.cfg_override) if a.cfg_override else None
    return Harness(a.ckpt, index=a.index, engram=not a.no_engram, cfg_overrides=over,
                   attn_chunk=a.attn_chunk, log=log, smooth=getattr(a, "smooth", False),
                   act_dtype=torch.float32 if getattr(a, "smooth_fp32", False) else torch.bfloat16)


def rows_from_jsonl(hz: Harness, path, limit=None, score_all=False) -> list[Row]:
    """score_all: ignore loss_spans/score_start and score every text target after BOS."""
    hz.ref_import()
    from image_processor import prepare_vl_inputs
    tok = hz.tokenizer
    args = hz.model_args()
    rows = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        if limit and len(rows) >= limit:
            break
        d = json.loads(line)
        rid, dom = str(d.get("id", i)), d.get("domain", "default")
        types, images, score = None, None, d.get("score_start", 1)
        score_mask = None
        if "tokens" in d:  # incl. the calib/panel sets of tools/calib/data (load.py schema)
            ids = list(d["tokens"])
            if d.get("images") or "loss_spans" in d:
                sys.path.append(str(Path(__file__).resolve().parent / "data"))  # calib-data load.py
                import image_processor
                import load as calib_load
                d.setdefault("n_tokens", len(ids))
                d["_root"] = str(Path(path).resolve().parent)
                if d.get("images"):
                    types = calib_load.token_types(d)
                    images = calib_load.ref_image_inputs(d, image_processor, args)
                if "loss_spans" in d and not score_all:
                    score_mask = torch.tensor(calib_load.loss_mask(d), dtype=torch.bool)
        elif "text" in d:
            ids = ([0] if d.get("bos", True) else []) + tok.encode(d["text"], add_special_tokens=False)
        else:
            if "messages" in d:
                from encoding import encode_case
                prompt, recs = encode_case(d, d.get("thinking_mode", "chat"))
            else:
                prompt, recs = d["prompt"], d.get("images", [])
            ids, types, images = prepare_vl_inputs(prompt, recs, tok, args)
            if d.get("completion"):
                comp = tok.encode(d["completion"], add_special_tokens=False)
                if "score_start" not in d:
                    score = len(ids)
                ids = ids + comp
                types = types + [-1] * len(comp)
            if images is None:
                types = None
        rows.append(Row(id=rid, tokens=torch.tensor(ids, dtype=torch.long), domain=dom,
                        token_types=None if types is None else torch.tensor(types, dtype=torch.long),
                        images=images, score_start=1 if score_all else int(score), score_mask=score_mask))
    return rows


def summary(hz: Harness, rows, head: Head, t0: float) -> dict:
    ntok = sum(r.n for r in rows if r.variant is None)
    tot = time.time() - t0
    out = {"tokens": ntok, "rows": sum(r.variant is None for r in rows), "wall_s": round(tot, 1),
           "tok_per_s": round(ntok / tot, 1), "timing_s": {k: round(v, 1) for k, v in hz.timing.items()},
           "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    s = sum(head.nll[r.id][0] for r in rows if r.variant is None)
    n = sum(head.nll[r.id][1] for r in rows if r.variant is None)
    out["nll"] = {"mean": s / max(n, 1), "scored": n}
    by_dom = {}
    for r in rows:
        if r.variant is None:
            a = by_dom.setdefault(r.domain, [0.0, 0])
            a[0] += head.nll[r.id][0]
            a[1] += head.nll[r.id][1]
    out["nll"]["by_domain"] = {k: {"mean": v[0] / max(v[1], 1), "n": v[1]} for k, v in by_dom.items()}
    return out


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))
    log("wrote", path)


# ------------------------------------------------------------------------------------- commands

def cmd_nll(a):
    hz = make_harness(a)
    tok = hz.tokenizer
    rows = []
    for line in Path(a.passages).read_text().splitlines()[: a.limit or None]:
        p = json.loads(line)
        ids = [0] + tok.encode(p["text"], add_special_tokens=False)
        rows.append(Row(id=p["id"], tokens=torch.tensor(ids), domain="nll_passages", score_start=1 + a.skip))
    head = Head(topk_dump=a.dump_topk)
    t0 = time.time()
    rows = hz.run(rows, head_cb=head)
    out = summary(hz, rows, head, t0)
    out["per_passage"] = {r.id: head.nll[r.id][0] / max(1, head.nll[r.id][1]) for r in rows}
    write(a.out, out)
    if a.dump_topk:
        save_dump(head, Path(a.out).with_suffix(".topk.safetensors"))
    log(json.dumps({k: out[k] for k in ("tokens", "wall_s", "tok_per_s", "nll", "peak_gpu_gib")})[:600])


def save_dump(head: Head, path):
    from safetensors.torch import save_file
    t = {}
    for rid, d in head.dump.items():
        for k, v in d.items():
            t[f"{rid}/{k}"] = v.contiguous()
    save_file(t, str(path), metadata={"doc": "per row: pos = predicting position (target pos+1); ids/logprobs top-k; "
                                             "lse = logsumexp of the full fp32 logits; nll of the actual next token"})
    log("wrote", path)


def cmd_gen(a):
    hz = make_harness(a)
    hz.ref_import()
    from encoding import encode_messages
    from image_processor import prepare_vl_inputs
    items = [{"prompt": p} for p in a.prompt or []]
    if a.prompts:
        items += [json.loads(l) for l in Path(a.prompts).read_text().splitlines() if l.strip()]
    prompts = [it["prompt"] for it in items]
    tok = hz.tokenizer
    base = []
    for it in items:
        content = it["prompt"]
        if it.get("image"):
            content = [{"type": "image_url", "image_url": {"url": it["image"]}}, {"type": "text", "text": it["prompt"]}]
        text, media = encode_messages([{"role": "user", "content": content}], thinking_mode=a.thinking_mode,
                                      return_multi_modal_data=True)
        ids, types, images = prepare_vl_inputs(text, media["images"], tok, hz.model_args())
        base.append((ids, types if images else None, images))
    gen = [[] for _ in prompts]
    done = [False] * len(prompts)
    t0 = time.time()
    for step in range(a.max_new):
        rows = []
        for i, (ids, types, images) in enumerate(base):
            tt = None if types is None else torch.tensor(types + [-1] * len(gen[i]))
            rows.append(Row(id=str(i), tokens=torch.tensor(ids + gen[i]), token_types=tt, images=images,
                            score_start=len(ids) + len(gen[i])))
        hz.timing.clear()
        head = Head(keep_last_logits=True)
        hz.log = (lambda *x: None) if step else log
        hz.run(rows, head_cb=head)
        for i in range(len(prompts)):
            if not done[i]:
                nxt = int(head.last_logits[str(i)].argmax())
                gen[i].append(nxt)
                done[i] = nxt == tok.eos_token_id
        log(f"step {step} {time.time() - t0:.0f}s", [tok.decode(g) for g in gen])
        if all(done):
            break
    out = {"prompts": prompts, "completions": [tok.decode(g) for g in gen], "token_ids": gen,
           "thinking_mode": a.thinking_mode, "wall_s": round(time.time() - t0, 1)}
    write(a.out, out)


def cmd_panel(a):
    from hooks import DiskSink, HessianCapture, RoutingStats
    hz = make_harness(a)
    rows = rows_from_jsonl(hz, a.panel, a.limit, a.score_all)
    hooks = []
    if a.stats:
        hooks.append(RoutingStats(a.stats))
    if a.hessian_dir:
        hooks.append(HessianCapture(parse_layers(a.hessian_layers), DiskSink(a.hessian_dir, a.max_pending_gb),
                                    wpow=a.hessian_wpow, prior_tokens=a.prior_tokens, fmt=a.hessian_fmt))
    head = Head(topk_dump=a.dump_topk)
    t0 = time.time()
    rows = hz.run(rows, hooks=Multi(hooks), head_cb=head)
    out = summary(hz, rows, head, t0)
    write(a.out, out)
    if a.dump_topk:
        save_dump(head, Path(a.out).with_suffix(".topk.safetensors"))
    torch.save({r.id: head.per_tok[r.id] for r in rows}, Path(a.out).with_suffix(".nll.pt"))


class Multi:
    """Fan a call out to several hooks."""

    def __init__(self, hooks):
        self.h = hooks

    def __getattr__(self, name):
        if name == "wants_expert":
            return lambda layer: any(h.wants_expert(layer) for h in self.h)
        if name == "prior_tokens":
            def pt(layer):
                for h in self.h:
                    p = h.prior_tokens(layer)
                    if p is not None:
                        return p
                return None
            return pt

        def call(*args, **kw):
            for h in self.h:
                if name in ("expert", "layer_tokens", "expert_prior") and not h.wants_expert(args[0]):
                    continue
                getattr(h, name)(*args, **kw)
        return call


def variant_report(hz: Harness, head: Head, rows, t0) -> dict:
    out = summary(hz, rows, head, t0)
    nat_nll = {}
    for r in rows:
        if r.variant is None:
            a = nat_nll.setdefault(r.domain, [0.0, 0])
            a[0] += head.nll[r.id][0]
            a[1] += head.nll[r.id][1]
    res = {}
    for name, doms in head.kl.items():
        e = getattr(hz, "out_err", {}).get(name)
        tot = [sum(v[0] for v in doms.values()), sum(v[1] for v in doms.values()), sum(v[2] for v in doms.values())]
        nn = sum(v[0] for v in nat_nll.values()) / max(1, sum(v[1] for v in nat_nll.values()))
        res[name] = {
            "kl_mean": tot[0] / max(tot[1], 1),
            "dnll": tot[2] / max(tot[1], 1) - nn,
            "by_domain": {d: {"kl_mean": v[0] / max(v[1], 1), "dnll": v[2] / max(v[1], 1) - nat_nll[d][0] / max(nat_nll[d][1], 1),
                              "n": v[1]} for d, v in doms.items()},
            # routed-experts output error at the first modified layer: aggregate ||dy||^2/||y||^2 (dominated by
            # massive-activation tokens) and the per-token mean of ||dy_t||^2/||y_t||^2
            "routed_out_rel_mse": None if e is None else e[0] / max(e[1], 1e-30),
            "routed_out_rel_mse_tokmean": None if e is None else e[2] / max(e[3], 1),
            "drafter_in_rel_mse": {str(L): v[0] / max(v[1], 1e-30) for L, v in head.drafter.get(name, {}).items()},
        }
        tk = [head.tok_kl[r.id] for r in rows if r.variant is not None and r.variant.name == name and r.id in head.tok_kl]
        if tk:
            k = torch.cat(tk).sort(descending=True).values
            res[name]["kl_top1pct_share"] = (k[: max(1, k.numel() // 100)].sum() / k.sum().clamp_min(1e-30)).item()
            res[name]["kl_by_row"] = {r.id.split("@")[0]: head.tok_kl[r.id].mean().item() for r in rows
                                      if r.variant is not None and r.variant.name == name and r.id in head.tok_kl}
    out["variants"] = res
    return out


def sens_variant(L: int, g: str, r: str, seed: int, shared_noise: bool) -> Variant:
    """Name L{L}-{g}-r{r}[-s{seed}]; shared_noise: one noise draw per (layer, group, seed) across r."""
    name = f"L{L}-{g}-r{r}" + (f"-s{seed}" if seed else "")
    return Variant(name=name, mods={L: ("noise", g, float(r))}, seed=seed,
                   noise_key=f"L{L}-{g}" if shared_noise else None)


def cmd_sens(a):
    hz = make_harness(a)
    rows = rows_from_jsonl(hz, a.panel, a.limit, a.score_all)
    variants = []
    for spec in a.variant or []:  # L:GROUP:r[:seed]
        L, g, r, *sd = spec.split(":")
        variants.append(sens_variant(int(L), g, r, int(sd[0]) if sd else a.seed, a.shared_noise))
    for L in parse_layers(a.sweep_layers) if a.sweep_layers else []:
        for g in a.groups.split(","):
            for r in a.r.split(","):
                variants.append(sens_variant(L, g, r, a.seed, a.shared_noise))
    assert len({v.name for v in variants}) == len(variants), "duplicate variant names"
    hz.stack = a.smooth and not a.no_stack
    hz.w32 = a.fp32_weights
    log(f"{len(variants)} variants x {len(rows)} rows, smooth={a.smooth}, stack={hz.stack}, shared_noise={a.shared_noise}")
    head = Head(keep_tok_kl=True)
    t0 = time.time()
    rows = hz.run(rows, variants=variants, head_cb=head)
    out = variant_report(hz, head, rows, t0)
    out["mode"] = "smooth" if a.smooth else "fp8"
    out["stack"] = hz.stack
    out["act_dtype"] = str(hz.act_dtype)
    out["fp32_weights"] = hz.w32
    out["shared_noise"] = a.shared_noise
    out["panel"] = str(a.panel)
    write(a.out, out)
    torch.save(head.tok_kl, Path(a.out).with_suffix(".tokkl.pt"))


def cmd_swap(a):
    hz = make_harness(a)
    rows = rows_from_jsonl(hz, a.panel, a.limit, a.score_all)
    layers = parse_layers(a.layers)
    if a.joint:
        variants = [Variant(name="swap-" + ("all" if len(layers) == 40 else a.layers), mods={L: ("swap",) for L in layers})]
    else:
        variants = [Variant(name=f"swap-L{L}", mods={L: ("swap",)}) for L in layers]
    head = Head()
    t0 = time.time()
    rows = hz.run(rows, variants=variants, head_cb=head, pack_dir=a.pack)
    out = variant_report(hz, head, rows, t0)
    out["pack"] = str(a.pack)
    out["pack_quantization_config"] = hz.pack.qcfg
    write(a.out, out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(SNAP), help="1017 snapshot dir")
    ap.add_argument("--index", default=None, help="model.safetensors.index.json if absent from --ckpt")
    ap.add_argument("--no-engram", action="store_true", help="skip Engram (NOT faithful; debugging only)")
    ap.add_argument("--attn-chunk", type=int, default=256)
    ap.add_argument("--cfg-override", default=None, help="JSON dict of inference/config.json overrides (tests)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("nll")
    p.add_argument("--passages", default=str(PASSAGES))
    p.add_argument("--skip", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dump-topk", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_nll)

    p = sub.add_parser("gen")
    p.add_argument("--prompt", action="append")
    p.add_argument("--prompts", help="JSONL with {'prompt': ...}")
    p.add_argument("--max-new", type=int, default=24)
    p.add_argument("--thinking-mode", default="chat", choices=["chat", "thinking"])
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_gen)

    p = sub.add_parser("panel")
    p.add_argument("--panel", required=True)
    p.add_argument("--score-all", action="store_true", help="score all text targets, not only loss_spans")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--stats", help="routing stats .npz")
    p.add_argument("--dump-topk", type=int, default=0)
    p.add_argument("--hessian-dir")
    p.add_argument("--hessian-layers", default="all")
    p.add_argument("--hessian-wpow", type=float, default=1.0, help="routing-weight exponent p in H = sum w^p x x^T")
    p.add_argument("--hessian-fmt", default="x", choices=["x", "h"], help="x: routed activations (default), h: fp32 H")
    p.add_argument("--prior-tokens", type=int, default=512)
    p.add_argument("--max-pending-gb", type=float, default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_panel)

    p = sub.add_parser("sens")
    p.add_argument("--panel", required=True)
    p.add_argument("--score-all", action="store_true", help="score all text targets, not only loss_spans")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--variant", action="append", help="L:GROUP:r[:seed], e.g. 3:GU:0.1")
    p.add_argument("--smooth", action="store_true", help="B-1 smooth mode: bf16 activations, frozen routing/indexer")
    p.add_argument("--shared-noise", action="store_true", help="one noise draw per (layer, group, seed) across r")
    p.add_argument("--no-stack", action="store_true", help="smooth mode: per-stream expert GEMMs (bit-exact r=0)")
    p.add_argument("--smooth-fp32", action="store_true", help="smooth mode with fp32 activations (split bf16 GEMMs)")
    p.add_argument("--fp32-weights", action="store_true", help="with --smooth-fp32: fp32 weights + plain fp32 GEMMs (slow)")
    p.add_argument("--sweep-layers", default="")
    p.add_argument("--groups", default="GU,D")
    p.add_argument("--r", default="0.1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_sens)

    p = sub.add_parser("swap")
    p.add_argument("--panel", required=True)
    p.add_argument("--score-all", action="store_true", help="score all text targets, not only loss_spans")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--pack", required=True)
    p.add_argument("--layers", default="all")
    p.add_argument("--joint", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_swap)

    a = ap.parse_args(argv)
    log("argv", sys.argv)
    with torch.inference_mode():
        a.fn(a)


if __name__ == "__main__":
    main()
