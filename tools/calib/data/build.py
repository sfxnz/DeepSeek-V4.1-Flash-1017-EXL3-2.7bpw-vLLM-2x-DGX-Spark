#!/usr/bin/env python3
"""Build the disjoint CALIB (~400K tokens) and PANEL (~150K tokens) sets for DeepSeek-V4.1-Flash-1017.

    python build.py [--out /home/sfxnz/projects/data/dsv41-1017/calib] [--seed 1017]

Needs tokenizers, pyarrow, Pillow (see README in the output dir for the venv used). Network: HF Hub.
Each prompt key (and each image) is used at most once across both sets, so the sets are disjoint by
construction; check.py verifies it on the written files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encode as E  # noqa: E402
import sources as S  # noqa: E402

SET_TOKENS = {"calib": 400_000, "panel": 150_000}
DOMAIN_SHARE = {"chat": .20, "code": .20, "math": .15, "tool": .10, "multilingual": .10, "long": .10,
                "encyclopedic": .05, "image": .10}
MAX_LEN = {"chat": 8192, "code": 8192, "math": 8192, "tool": 8192, "multilingual": 8192,
           "encyclopedic": 8192, "image": 6144}
MIN_LEN = 160
# (domain, name, generator, share within domain)
PLAN = [
    ("chat", "llama-nemotron-chat", S.llama_nemotron("SFT/chat/chat.jsonl", "chat/llama-nemotron"), .60),
    ("chat", "smoltalk-magpie-ultra", S.smoltalk, .40),
    ("code", "llama-nemotron-code", S.llama_nemotron("SFT/code/code_v1.1.jsonl", "code/llama-nemotron", thinking_only=True), .35),
    ("code", "opc-mceval", S.opc_mceval, .25),
    ("code", "swe-smith", S.swe_smith, .40),
    ("math", "nemotron-math", S.nemotron("math", "math/nemotron-r1", thinking_only=True), .60),
    ("math", "nemotron-stem", S.nemotron("stem", "math/stem-nemotron-r1", thinking_only=True), .40),
    ("tool", "nemotron-tool", S.nemotron("tool", "tool/nemotron"), .60),
    ("tool", "hermes-fc", S.hermes, .40),
    ("multilingual", "zh-r1", S.zh_r1, .25),
    ("multilingual", "lightblue-r1", S.lightblue, .35),
    ("multilingual", "aya", S.aya, .40),
    ("encyclopedic", "wikipedia-en", S.wikipedia, 1.0),
    ("image", "photo-coco", S.llava("sharegpt4v(coco)", "image/photo", "GPT-4V (ShareGPT4V)"), .15),
    ("image", "photo-laion", S.llava("allava_instruct_laion4v", "image/photo", "GPT-4V (ALLaVA)"), .15),
    ("image", "ocr", S.llava("textocr(gpt4v)", "image/ocr", "GPT-4V"), .20),
    ("image", "chart", S.llava("chartqa(cauldron,llava_format)", "image/chart", "ChartQA (human)"), .20),
    ("image", "document", S.llava("infographic(gpt4v)", "image/document", "GPT-4V"), .20),
    ("image", "diagram", S.llava("ai2d(gpt4v)", "image/diagram", "GPT-4V"), .10),
]
# Long context: one sequence each; (set, kind, target tokens)
LONG_PLAN = [("calib", "book", 17_200), ("calib", "code", 17_400), ("calib", "qa", 17_000), ("panel", "qa", 16_600)]
LONG_GEN = {"book": S.pg19, "code": S.github_code, "qa": S.longbench2}


def tokens_sha(ids):
    return hashlib.sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest()


class Builder:
    def __init__(self, out: Path, seed: int):
        self.out, self.seed = out, seed
        self.enc = E.Encoder()
        self.records = {"calib": [], "panel": []}
        self.used_keys, self.used_images = set(), set()
        self.images = {}  # url -> bytes (only for kept records)
        (out / "logs").mkdir(parents=True, exist_ok=True)
        self.logf = open(out / "logs" / f"build-{time.strftime('%Y%m%dT%H%M%S')}.log", "w")
        self.ctx = {"raw": out / "raw", "log": self.log}
        self.stats = {}
        self.lang_tokens = {}

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    def ntok(self, text):
        return len(self.enc.tok.encode(text, add_special_tokens=False).ids)

    def encode_cand(self, c, max_len):
        r = self.enc.encode(c["messages"], c["thinking_mode"], c["reasoning_effort"], c["image_bytes"])
        if len(r["tokens"]) > max_len and c.get("truncatable"):
            # keep the longest prefix ending in an assistant turn that fits
            ends = [k for k, m in enumerate(c["messages"]) if m["role"] == "assistant"]
            lo, hi, best = 0, len(ends) - 1, None
            while lo <= hi:
                mid = (lo + hi) // 2
                rr = self.enc.encode(c["messages"][:ends[mid] + 1], c["thinking_mode"], c["reasoning_effort"], c["image_bytes"])
                if len(rr["tokens"]) <= max_len:
                    best, lo = (ends[mid], rr), mid + 1
                else:
                    hi = mid - 1
            if best is None:
                return None
            c["messages"] = c["messages"][:best[0] + 1]
            c["truncated_at_message"] = best[0] + 1
            r = best[1]
        return r

    def add(self, set_name, domain, c, r):
        ids = r["tokens"]
        spans, a = [], None
        for t, m in enumerate(r["loss_mask"] + [0]):
            if m and a is None:
                a = t
            elif not m and a is not None:
                spans.append([a, t])
                a = None
        rec = dict(set=set_name, domain=domain, subdomain=c["subdomain"], lang=c["lang"],
                   thinking_mode=c["thinking_mode"], reasoning_effort=c["reasoning_effort"],
                   n_tokens=len(ids), tokens=ids, loss_spans=spans,
                   images=[dict(path=im["url"], **{k: v for k, v in im.items() if k != "url"}) for im in r["images"]],
                   messages=c["messages"], source=c["source"], tokens_sha256=tokens_sha(ids), key=c["key"])
        if c.get("truncated_at_message"):
            rec["source"] = dict(rec["source"], truncated_to_messages=c["truncated_at_message"])
        self.records[set_name].append(rec)
        self.used_keys.add(c["key"])
        for im in r["images"]:
            self.used_images.add(im["sha256"])
            self.images[im["url"]] = c["image_bytes"][im["url"]]

    # ------------------------------------------------------------ long
    def build_long(self):
        done = []
        for kind in ("book", "code", "qa"):
            wants = [(s, t) for s, k, t in LONG_PLAN if k == kind]
            rng = random.Random(f"{self.seed}:long:{kind}")
            gen = LONG_GEN[kind](rng, self.ctx)
            for set_name, target in wants:
                for c in gen:
                    if c["key"] in self.used_keys:
                        continue
                    c = self.make_long(c, target, rng)
                    if c is None:
                        continue
                    r = self.enc.encode(c["messages"], "chat", None, {})
                    n = len(r["tokens"])
                    if not (16_000 <= n <= 32_768):
                        self.log(f"  long {kind}: {n} tokens out of range, skipped")
                        continue
                    self.add(set_name, "long", c, r)
                    done.append((set_name, kind, n))
                    self.log(f"long {set_name} {kind}: {c['subdomain']} {n} tokens ({c['source'].get('url') or c['source'].get('path') or c['source'].get('_id')})")
                    break
        return done

    def make_long(self, c, target, rng):
        if c["long_kind"] == "qa":
            n = self.ntok(c["messages"][0]["content"])
            if not (target - 600 <= n <= target + 3000):
                return None
            return c
        text = c["text"]
        if c["long_kind"] == "book":
            paras = [p for p in text.split("\n\n") if p.strip()]
            start = int(len(paras) * (0.1 + 0.3 * rng.random()))
            ctx_tok, gen_tok = target - 2600, 2300
            a, b, n = [], [], 0
            for p in paras[start:]:
                k = self.ntok(p) + 1
                if n + k <= ctx_tok:
                    a.append(p)
                elif n + k <= ctx_tok + gen_tok:
                    b.append(p)
                else:
                    break
                n += k
            if n < ctx_tok + gen_tok * 0.6 or not b:
                return None
            user = f"Continue the following passage from the book \"{c['title']}\".\n\n" + "\n\n".join(a)
            asst = "\n\n".join(b)
        else:
            lines = text.split("\n")
            budget, a, b, n = target - 400, [], [], 0
            cut = None
            for k, ln in enumerate(lines):
                t = self.ntok(ln) + 1
                if n + t > budget:
                    break
                n += t
                cut = k
            if cut is None or n < target * 0.7:
                return None
            keep = lines[:cut + 1]
            split = int(len(keep) * 0.8)
            lang = (c.get("code_lang") or "").lower().replace(" ", "")
            user = (f"Here is the beginning of `{c['source']['path']}` from the {c['source']['repo_name']} repository "
                    f"({c['source']['file_license']} license). Continue the file.\n\n```{lang}\n" + "\n".join(keep[:split]) + "\n```")
            asst = f"```{lang}\n" + "\n".join(keep[split:]) + "\n```"
        c = dict(c, messages=[{"role": "user", "content": user}, {"role": "assistant", "content": asst}],
                 thinking_mode="chat", reasoning_effort=None, image_bytes={})
        return c

    # ------------------------------------------------------------ main domains
    def build_domains(self, long_tokens):
        rest = {s: SET_TOKENS[s] - long_tokens.get(s, 0) for s in SET_TOKENS}
        norm = sum(v for d, v in DOMAIN_SHARE.items() if d != "long")
        for domain, name, gen_fn, share in PLAN:
            target = {s: rest[s] * DOMAIN_SHARE[domain] / norm * share for s in SET_TOKENS}
            have = {s: 0 for s in SET_TOKENS}
            rng = random.Random(f"{self.seed}:{name}")
            self.log(f"== {domain}/{name}: target calib {target['calib']:.0f} panel {target['panel']:.0f}")
            seen = skipped_len = misfit = 0
            t0 = time.time()
            try:
                for c in gen_fn(rng, self.ctx):
                    seen += 1
                    if seen > 20000:
                        self.log(f"  !! {name}: gave up after {seen} candidates")
                        break
                    if c["key"] in self.used_keys or any(h in self.used_images for h in c.get("image_sha", [])):
                        continue
                    try:
                        r = self.encode_cand(c, MAX_LEN[domain])
                    except Exception as e:  # malformed source row
                        self.log(f"  skip {name} row ({type(e).__name__}: {str(e)[:120]})")
                        continue
                    if r is None or not (MIN_LEN <= len(r["tokens"]) <= MAX_LEN[domain]) or not any(r["loss_mask"]):
                        skipped_len += 1
                        continue
                    # Fill the set that is furthest from its target, without overshooting a set by more than
                    # max(1000, 15%) tokens; a set within 3% of its target counts as full.
                    n = len(r["tokens"])
                    rem = {x: target[x] - have[x] for x in SET_TOKENS}
                    open_sets = sorted((x for x in SET_TOKENS if rem[x] > 0.03 * target[x]),
                                       key=lambda x: rem[x] / target[x], reverse=True)
                    if not open_sets:
                        break
                    fit = [x for x in open_sets if n <= rem[x] + max(1000, 0.15 * target[x])]
                    if not fit:
                        misfit += 1
                        if misfit > 60:
                            self.log(f"  {name}: stopping, 60 candidates too long for the remaining budget {rem}")
                            break
                        continue
                    pick = fit[0]
                    if c["lang"] != "en" and len(fit) > 1:  # spread each language over both sets
                        pick = min(fit, key=lambda x: self.lang_tokens.get((x, c["lang"]), 0) / SET_TOKENS[x])
                    self.add(pick, domain, c, r)
                    have[pick] += n
                    self.lang_tokens[(pick, c["lang"])] = self.lang_tokens.get((pick, c["lang"]), 0) + n
            except Exception as e:
                self.log(f"  !! {name} FAILED: {type(e).__name__}: {e}")
            self.stats[name] = dict(domain=domain, candidates=seen, skipped_len=skipped_len, have=have,
                                    target={k: round(v) for k, v in target.items()}, seconds=round(time.time() - t0))
            self.log(f"   {name}: have calib {have['calib']} panel {have['panel']} (cands {seen}, len-skipped {skipped_len}, {time.time()-t0:.0f}s)")

    # ------------------------------------------------------------ output
    def write(self):
        img_dir = self.out / "images"
        img_dir.mkdir(exist_ok=True)
        for url, b in self.images.items():
            p = self.out / url
            if not p.exists():
                p.write_bytes(b)
        for s, recs in self.records.items():
            random.Random(f"{self.seed}:order:{s}").shuffle(recs)
            with open(self.out / f"{s}.jsonl", "w", encoding="utf-8") as f:
                for k, rec in enumerate(recs):
                    rec = dict(id=f"{s}-{k:05d}", **rec)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/home/sfxnz/projects/data/dsv41-1017/calib"))
    ap.add_argument("--seed", type=int, default=1017)
    ap.add_argument("--scale", type=float, default=1.0, help="debug: scale token budgets")
    ap.add_argument("--only", default="", help="debug: comma list of PLAN names (and 'long')")
    args = ap.parse_args()
    for k in SET_TOKENS:
        SET_TOKENS[k] = int(SET_TOKENS[k] * args.scale)
    if args.only:
        keep = set(args.only.split(","))
        PLAN[:] = [p for p in PLAN if p[1] in keep]
        if "long" not in keep:
            LONG_PLAN.clear()
    b = Builder(args.out, args.seed)
    b.log(f"build start host={platform.node()} seed={args.seed} argv={sys.argv}")
    b.log(f"zh parquet rev (pinned) -> {S.REV['Congliu/Chinese-DeepSeek-R1-Distill-data-110k']}")
    long_done = b.build_long()
    long_tokens = {}
    for s, _, n in long_done:
        long_tokens[s] = long_tokens.get(s, 0) + n
    b.build_domains(long_tokens)
    b.write()
    build_info = dict(host=platform.node(), finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), seed=args.seed,
                      argv=sys.argv, bytes_fetched=b.ctx.get("bytes_fetched", 0), per_source=b.stats,
                      long=long_done, set_tokens_target=SET_TOKENS, domain_share=DOMAIN_SHARE, max_len=MAX_LEN,
                      min_len=MIN_LEN, revisions=S.REV, licenses=S.LICENSE)
    (args.out / "build_info.json").write_text(json.dumps(build_info, indent=1, ensure_ascii=False))
    b.log("build done; run check.py to write manifest.json")


if __name__ == "__main__":
    main()
