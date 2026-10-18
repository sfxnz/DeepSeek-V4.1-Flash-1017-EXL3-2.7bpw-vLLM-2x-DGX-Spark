#!/usr/bin/env python3
"""Build the sensitivity panel (decision B-1): a stratified subset of calib.jsonl (never panel.jsonl).

Per domain, records are taken in a seeded random order until the domain's token target is met; a text-only
record that would overshoot is truncated (prefix, so it stays a valid causal input; loss_spans clipped).
Image records are never truncated. Image paths are made absolute so the panel can live anywhere.
Options for a second, disjoint panel: --exclude (skip the ids of other panels), --family-rr (text domains:
round-robin over subdomain families, e.g. code/mceval-* vs code/agentic-swe), --long-cap, --min-take,
--max-shared-prefix (skip a text record whose token prefix repeats a selected or excluded record's, e.g. the
662-token agentic-swe system prompt: a truncated copy would only duplicate those positions).
Writes OUT (jsonl) and OUT.manifest.json (ids, sources, token/image counts per domain, input sha256).

  python3 tools/calib/sens_panel.py --calib .../calib/calib.jsonl --out .../sens/sens_panel.jsonl
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

DEFAULT_TARGETS = {"chat": 2400, "code": 2400, "math": 2400, "tool": 2400, "multilingual": 2400,
                   "encyclopedic": 2400, "image": 4500, "long": 5000}
DEFAULT_CAP = 1200  # max tokens per text record (more records per domain); "long" is exempt


def truncate(rec: dict, n: int) -> dict:
    assert not rec.get("images")
    r = dict(rec)
    r["tokens"] = rec["tokens"][:n]
    r["n_tokens"] = n
    r["loss_spans"] = [[a, min(b, n)] for a, b in rec.get("loss_spans", []) if a < n]
    r["truncated_from"] = rec["n_tokens"]
    return r


def family(rec: dict) -> str:
    """Subdomain family: 'code/mceval-julia' -> 'mceval', 'tool/hermes-Use Apps' -> 'hermes'."""
    return rec["subdomain"].split("/", 1)[-1].split("-")[0]


def lcp(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def build(calib: Path, targets: dict, seed: int, min_images: int, cap: int = DEFAULT_CAP, min_take: int = 512,
          exclude: dict | None = None, family_rr: bool = False, long_cap: int | None = None,
          max_shared_prefix: int | None = None):
    """exclude: {id: tokens} of records not to take (and, with max_shared_prefix, not to repeat)."""
    exclude = exclude or {}
    recs = [json.loads(l) for l in calib.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r["id"] not in exclude]
    seen = list(exclude.values())  # token lists of excluded + selected text records
    root = calib.parent
    rng = random.Random(seed)
    out = []
    for dom, tgt in targets.items():
        pool = [r for r in recs if r["domain"] == dom]
        rng.shuffle(pool)
        if dom == "image":  # round-robin over subdomains (records <= cap), first a two-image record
            by_sub = {}
            for r in pool:
                if r["n_tokens"] <= cap:
                    by_sub.setdefault(r["subdomain"], []).append(r)
            two = [r for r in pool if len(r["images"]) > 1 and r["n_tokens"] <= cap][:1]
            rr = []
            while any(by_sub.values()):
                for sub in sorted(by_sub):
                    if by_sub[sub]:
                        rr.append(by_sub[sub].pop(0))
            pool = two + [r for r in rr if r not in two]
        elif family_rr:  # families in seeded order, records in pool (seeded) order within a family
            by_fam = {}
            for r in pool:
                by_fam.setdefault(family(r), []).append(r)
            fams = sorted(by_fam)
            rng.shuffle(fams)
            rr = []
            while any(by_fam.values()):
                for f in fams:
                    if by_fam[f]:
                        rr.append(by_fam[f].pop(0))
            pool = rr
        have, n_img = 0, 0
        for r in pool:
            if have >= tgt and (dom != "image" or n_img >= min_images):
                break
            room = tgt - have
            if r.get("images"):
                if dom == "image" and have + r["n_tokens"] > tgt * 1.25 and n_img >= min_images:
                    continue
                r = dict(r)
                r["images"] = [dict(im, path=str((root / im["path"]).resolve())) for im in r["images"]]
                n_img += 1
            else:
                n = min(r["n_tokens"], room, (long_cap or r["n_tokens"]) if dom == "long" else cap)
                if n < min(min_take, r["n_tokens"]):
                    continue
                if max_shared_prefix is not None:
                    if any(lcp(r["tokens"][:n], t) > max_shared_prefix for t in seen):
                        continue
                    seen.append(r["tokens"][:n])
                if n < r["n_tokens"]:
                    r = truncate(r, n)
            have += r["n_tokens"]
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", default="/home/sfxnz/projects/data/dsv41-1017/calib/calib.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1017)
    ap.add_argument("--min-images", type=int, default=4)
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--targets", default=None, help="JSON {domain: tokens} (default %s)" % DEFAULT_TARGETS)
    ap.add_argument("--min-take", type=int, default=512, help="skip a text record whose take would be shorter")
    ap.add_argument("--long-cap", type=int, default=None, help="max tokens per 'long' record (default: no cap)")
    ap.add_argument("--family-rr", action="store_true", help="text domains: round-robin over subdomain families")
    ap.add_argument("--exclude", action="append", default=[], help="JSONL panel whose record ids are excluded")
    ap.add_argument("--max-shared-prefix", type=int, default=None,
                    help="skip a text record sharing a longer token prefix with a selected/excluded record")
    a = ap.parse_args()
    calib = Path(a.calib)
    targets = json.loads(a.targets) if a.targets else DEFAULT_TARGETS
    exclude = {d["id"]: d["tokens"] for p in a.exclude for d in map(json.loads, filter(str.strip, Path(p).read_text().splitlines()))}
    rows = build(calib, targets, a.seed, a.min_images, a.cap, a.min_take, exclude, a.family_rr, a.long_cap,
                 a.max_shared_prefix)
    assert not {r["id"] for r in rows} & exclude.keys()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    by = {}
    for r in rows:
        d = by.setdefault(r["domain"], {"records": 0, "tokens": 0, "image_records": 0, "images": 0,
                                        "image_tokens": 0, "truncated": 0, "max_len": 0})
        d["records"] += 1
        d["tokens"] += r["n_tokens"]
        d["max_len"] = max(d["max_len"], r["n_tokens"])
        d["truncated"] += "truncated_from" in r
        if r.get("images"):
            d["image_records"] += 1
            d["images"] += len(r["images"])
            d["image_tokens"] += sum(im["n_tokens"] for im in r["images"])
    man = {
        "doc": "Sensitivity panel (B-1): stratified subset of calib.jsonl; score all text targets (--score-all).",
        "source": str(calib.resolve()), "source_sha256": hashlib.sha256(calib.read_bytes()).hexdigest(),
        "seed": a.seed, "targets": targets, "cap": a.cap, "min_images": a.min_images,
        **({"min_take": a.min_take, "long_cap": a.long_cap, "family_rr": a.family_rr,
            "exclude": [str(Path(p).resolve()) for p in a.exclude], "excluded_ids": len(exclude),
            "max_shared_prefix": a.max_shared_prefix}
           if (a.exclude or a.family_rr or a.long_cap or a.min_take != 512 or a.max_shared_prefix is not None) else {}),
        "total_tokens": sum(r["n_tokens"] for r in rows), "records": len(rows),
        "by_domain": by,
        "rows": [{"id": r["id"], "domain": r["domain"], "subdomain": r.get("subdomain"), "n_tokens": r["n_tokens"],
                  "truncated_from": r.get("truncated_from"), "n_images": len(r.get("images") or [])} for r in rows],
        "panel_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    Path(str(out) + ".manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: man[k] for k in ("total_tokens", "records", "by_domain")}, indent=1))


if __name__ == "__main__":
    main()
