#!/usr/bin/env python3
"""Verify the built sets and write manifest.json + samples.txt.

    python check.py [--root /home/sfxnz/projects/data/dsv41-1017/calib]

Checks: record structure (load.validate); re-encoding every record from its stored messages with the
reference encoder reproduces its tokens exactly (images read back from disk, sha256 verified);
disjointness of calib vs panel by prompt key, token-sequence hash, image hash and assistant-text 32-gram
overlap; per-domain token counts. Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encode as E  # noqa: E402
import load  # noqa: E402

NGRAM = 32


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=load.DEFAULT_ROOT)
    args = ap.parse_args()
    root = args.root
    (root / "logs").mkdir(exist_ok=True)
    logf = open(root / "logs" / f"check-{time.strftime('%Y%m%dT%H%M%S')}.log", "w")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")

    log(f"check host={platform.node()} time={time.strftime('%Y-%m-%dT%H:%M:%S%z')} root={root}")
    enc = E.Encoder()
    failures = []
    sets, stats, ngrams, samples = {}, {}, {}, []
    for name in ("calib", "panel"):
        recs = load.load_set(name, root)
        sets[name] = recs
        st = dict(n_seq=0, n_tokens=0, n_loss_tokens=0, n_image_tokens=0, n_images=0, domains={}, subdomains={},
                  thinking_mode=collections.Counter(), lang=collections.Counter(), length_hist=collections.Counter(),
                  max_len=0)
        grams = set()
        shown = set()
        for rec in recs:
            try:
                load.validate(rec)
                imgs = {}
                for im in rec["images"]:
                    b = (root / im["path"]).read_bytes()
                    if hashlib.sha256(b).hexdigest() != im["sha256"]:
                        raise AssertionError(f"image sha mismatch {im['path']}")
                    imgs[im["path"]] = b
                r = enc.encode(rec["messages"], rec["thinking_mode"], rec["reasoning_effort"], imgs)
                if r["tokens"] != rec["tokens"]:
                    raise AssertionError("re-encode differs from stored tokens")
                mask = load.loss_mask(rec)
                if mask != r["loss_mask"]:
                    raise AssertionError("loss mask differs from re-encode")
            except Exception as e:
                failures.append(f"{rec['id']}: {e}")
                continue
            n, nl = rec["n_tokens"], sum(mask)
            ni = sum(i["n_tokens"] for i in rec["images"])
            st["n_seq"] += 1
            st["n_tokens"] += n
            st["n_loss_tokens"] += nl
            st["n_image_tokens"] += ni
            st["n_images"] += len(rec["images"])
            st["max_len"] = max(st["max_len"], n)
            d = st["domains"].setdefault(rec["domain"], dict(n_seq=0, n_tokens=0, n_loss_tokens=0, n_image_tokens=0, n_images=0))
            d["n_seq"] += 1
            d["n_tokens"] += n
            d["n_loss_tokens"] += nl
            d["n_image_tokens"] += ni
            d["n_images"] += len(rec["images"])
            sd = st["subdomains"].setdefault(rec["subdomain"], [0, 0])
            sd[0] += 1
            sd[1] += n
            st["thinking_mode"][rec["thinking_mode"]] += n
            st["lang"][rec["lang"]] += n
            st["length_hist"]["<1K" if n < 1024 else "1-2K" if n < 2048 else "2-4K" if n < 4096 else
                              "4-8K" if n < 8192 else "8-16K" if n < 16384 else "16-32K"] += 1
            toks = rec["tokens"]
            for a, b in rec["loss_spans"]:
                for k in range(a, b - NGRAM + 1, 4):
                    grams.add(hash(tuple(toks[k:k + NGRAM])))
            if (rec["domain"], rec["thinking_mode"]) not in shown:
                shown.add((rec["domain"], rec["thinking_mode"]))
                samples.append((rec, mask))
        for d in st["domains"].values():
            d["frac_tokens"] = round(d["n_tokens"] / max(1, st["n_tokens"]), 4)
        stats[name] = st
        ngrams[name] = grams
        log(f"{name}: {st['n_seq']} seqs {st['n_tokens']} tokens (loss {st['n_loss_tokens']}, image {st['n_image_tokens']} in {st['n_images']} images), max {st['max_len']}")
        for dn in load.DOMAINS:
            d = st["domains"].get(dn)
            if d:
                log(f"   {dn:13s} {d['n_seq']:4d} seqs {d['n_tokens']:7d} tok {100*d['frac_tokens']:5.1f}%  loss {d['n_loss_tokens']:7d}  img {d['n_images']}")

    # disjointness
    def keyset(recs, f):
        out = set()
        for r in recs:
            out.update(f(r))
        return out
    checks = {
        "prompt_keys": (lambda r: r["key"].split("+")),
        "tokens_sha256": (lambda r: [r["tokens_sha256"]]),
        "image_sha256": (lambda r: [i["sha256"] for i in r["images"]]),
        "first_user_text": (lambda r: [json.dumps(next((m["content"] for m in r["messages"] if m["role"] == "user"), ""), ensure_ascii=False)]),
    }
    overlap = {}
    for k, f in checks.items():
        inter = keyset(sets["calib"], f) & keyset(sets["panel"], f)
        overlap[k] = len(inter)
        if inter:
            failures.append(f"overlap in {k}: {len(inter)}")
    g_inter = ngrams["calib"] & ngrams["panel"]
    overlap[f"assistant_{NGRAM}gram_frac_of_panel"] = round(len(g_inter) / max(1, len(ngrams["panel"])), 5)
    log(f"overlap: {overlap}")
    ids = [r["id"] for s in sets.values() for r in s]
    if len(ids) != len(set(ids)):
        failures.append("duplicate ids")

    # samples
    with open(root / "samples.txt", "w", encoding="utf-8") as f:
        for rec, mask in samples:
            t = rec["tokens"]
            txt = enc.tok.decode(t, skip_special_tokens=False)
            a, b = rec["loss_spans"][0]
            f.write(f"===== {rec['id']} {rec['domain']} {rec['subdomain']} {rec['thinking_mode']} effort={rec['reasoning_effort']} "
                    f"n={rec['n_tokens']} loss={sum(mask)} images={len(rec['images'])}\n")
            f.write("--- head (image spans shown as repeated <｜deepseek_image｜>):\n" + txt[:1500] + "\n")
            f.write("--- first assistant span (prev 3 tokens | span):\n" + enc.tok.decode(t[max(0, a - 3):a], skip_special_tokens=False)
                    + " | " + enc.tok.decode(t[a:b], skip_special_tokens=False)[:1200] + "\n")
            f.write("--- tail:\n" + txt[-600:] + "\n\n")

    def plain(o):
        if isinstance(o, collections.Counter):
            return dict(o)
        if isinstance(o, dict):
            return {k: plain(v) for k, v in o.items()}
        return o
    info = json.loads((root / "build_info.json").read_text()) if (root / "build_info.json").exists() else {}
    manifest = dict(
        description="DeepSeek-V4.1-Flash-1017 calibration (calib) and held-out (panel) token sets; see README.md",
        model=dict(repo="deepseek-ai/DeepSeek-V4.1-Flash-1017", revision=E.REVISION,
                   tokenizer_json_sha256=sha_file(enc.ref_dir / "tokenizer.json"),
                   encoding_py_sha256=sha_file(enc.ref_dir / "encoding" / "encoding.py"),
                   image_processor_py_sha256=sha_file(enc.ref_dir / "inference" / "image_processor.py"),
                   image_token_id=E.IMAGE_TOKEN_ID),
        files={n: dict(path=f"{n}.jsonl", sha256=sha_file(root / f"{n}.jsonl")) for n in ("calib", "panel")},
        n_image_files=len(list((root / "images").glob("*"))),
        sets=plain(stats), disjointness=overlap, failures=failures,
        build=info, checked=dict(host=platform.node(), time=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    for fl in failures[:50]:
        log("FAIL " + fl)
    log(f"failures: {len(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
