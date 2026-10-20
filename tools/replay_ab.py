#!/usr/bin/env python3
"""Greedy A/B for DSV41_DECODER_REPLAY_ROWS across two boots (HTTP only, stdlib).

  python3 tools/replay_ab.py gen   --out armA.json   # boot with DSV41_DECODER_REPLAY_ROWS=0
  python3 tools/replay_ab.py turns --out armA-turns.json
  python3 tools/replay_ab.py prefix --replay-rows 0 --out armA-prefix.json  # add --retention 4096 on kv-pack
  ... reboot with DSV41_DECODER_REPLAY_ROWS=2670, the same commands (prefix: --replay-rows 2670) -> armB*.json
  python3 tools/replay_ab.py compare armA.json armB.json
  python3 tools/replay_ab.py compare armA-turns.json armB-turns.json
  python3 tools/replay_ab.py compare armA-prefix.json armB-prefix.json  # exit 1 on a token or hit-check miss

gen: one seeded pseudo-word document per --lengths (tools/corpus.py, novel=True, so both
boots send the same text) plus a question, greedy, 64 tokens, top-5 logprobs of every
generated token. Not prompt_logprobs: those requests keep every prompt row (no replay) and
materialise full-vocab logits. Run it on a fresh boot so nothing is prefix-cached.

turns: a --turn-tokens document question (turn 1), then three follow-ups, each with the
prefix-cache hit delta from /metrics: resend (turn 1 again), continue (turn 1 + its reply
+ a new question) and diverge (turn 1 + a different assistant text + the new question,
i.e. the history no longer matches right after turn 1's prompt).

prefix: prefix-cache hits after a trimmed prompt (DSV41_PREFIX_END_FALLBACK, junction
anchors, DSV41_PREFIX_REPLY_RETAIN), each against the hit e15 should land, computed from
/tokenize token ids (pass --replay-rows / --retention / --fallback / --reply-retain as booted):
  resend_r{R} / follow_r{R} / newq_r{R}: a fresh --turn-tokens prompt padded to N = R mod 128,
    R in 0 1 63 64 65 127, sent twice, then a turn that diverges right after N, then the same
    document with another final question (boot I1's d_same_doc_new_question);
  sib1..sib3: three requests sharing only a --system-tokens system prompt: 1 and 2 miss,
    2 flags the junction J, 3 hits J (or J-128);
  reply: a long answer, then a turn that repeats it verbatim: with reply retention the hit
    lands within 192 tokens of the shared end, without it at the prompt end.
Each case prints PASS/FAIL; `compare` across a replay-off and a replay-on arm (same flags
otherwise) passes when every reply has identical greedy tokens and both arms' hit checks pass
(the arms' hits differ only where expected, e.g. kv-pack checkpoints serve the replay-off arm).

compare: per item, the first generated token that differs, the largest |logprob| gap of
the chosen tokens before it, the top-5 overlap, and the prefix hits of both arms.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "benches"))

QUESTION = "\n\nIn one sentence, which word appears most often in the text above?"
NEW_QUESTION = "\n\nIn one sentence, does the text above contain any English sentences?"
FOLLOW_UP = "Now name a second frequent word and say why it might repeat."
CHAT_KWARGS = {"thinking": False, "reasoning_effort": "low"}
RESIDUES = (0, 1, 63, 64, 65, 127)
OTHER = "~~~ (no answer) ~~~"  # an assistant turn no reply starts with: the history diverges at N
SIBLING_QUESTIONS = (
    "Which word appears most often in the text above? One sentence.",
    "Name three words that appear in the first paragraph above.",
    "Is the text above prose or a list? Answer in one sentence.",
)
LONG_ANSWER = "\n\nList forty distinct words from the text above, one per line, numbered."
BLOCK, PEEK = 128, 64  # e15's hit unit and the DSpark peek block


def chat(url: str, model: str, messages: list, max_tokens: int, timeout: int) -> dict:
    body = json.dumps({
        "model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
        "logprobs": True, "top_logprobs": 5, "chat_template_kwargs": CHAT_KWARGS,
        "return_tokens_as_token_ids": True,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode())
    choice = out["choices"][0]
    content = (choice.get("logprobs") or {}).get("content") or []
    return {
        "text": choice["message"].get("content") or "",
        "prompt_tokens": int(out["usage"]["prompt_tokens"]),
        "completion_tokens": int(out["usage"].get("completion_tokens") or len(content)),
        "tokens": [c["token"] for c in content],
        "logprobs": [c["logprob"] for c in content],
        "top": [[t["token"] for t in c.get("top_logprobs") or []] for c in content],
    }


def tokenize_chat(base: str, model: str, messages: list, timeout: int = 120) -> list[int]:
    """Prompt token ids of a chat request (generation prompt included), from vLLM's /tokenize."""
    body = json.dumps({"model": model, "messages": messages, "chat_template_kwargs": CHAT_KWARGS}).encode()
    req = urllib.request.Request(f"{base}/tokenize", data=body, headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return [int(t) for t in json.loads(resp.read().decode())["tokens"]]


def token_ids(item: dict) -> list[int]:
    """Generated ids from return_tokens_as_token_ids ("token_id:123")."""
    return [int(t.split(":", 1)[1]) for t in item["tokens"] if t.startswith("token_id:")]


def common_prefix(a: list, b: list) -> int:
    return first_divergence(a, b)


def cached_blocks(prompt_len: int, kv_len: int, *, fallback: bool, junction: int = 0, retention: int = 0,
                  replay_rows: int = 0, reply: bool = False, hit: int = 0) -> set:
    """64-token SWA blocks e15 caches for a request prefilled from ``hit`` with ``kv_len`` tokens
    of KV: the tail (2 window blocks + the peek block) of its replay boundary and junction,
    floored to 128, with the prefix-end fallback also of the boundary one unit below each;
    kv-pack's checkpoints every ``retention`` tokens; with reply retention every block at or past
    the prompt end. Decoder replay at W = ``replay_rows`` (exact, >= 2414) keeps blocks below
    its frontier N - (W - 2286) out unless they hold a junction anchor's [J-256, J) or its peek
    block (decoder_replay.CacheGuard)."""
    tails = set()
    for b in [prompt_len - 1] + ([junction] if junction else []):
        p = b // BLOCK * BLOCK
        tails |= {p} | ({p - BLOCK} if fallback and b >= 2 * BLOCK else set())
    if retention:
        tails |= set(range(retention, kv_len + 1, retention))
    blocks = {q // PEEK + k for q in tails for k in (-2, -1, 0)}
    if reply:
        blocks |= set(range(prompt_len // PEEK, kv_len // PEEK))
    blocks = {b for b in blocks if hit // PEEK <= b < kv_len // PEEK}
    if replay_rows and prompt_len - hit > replay_rows:  # this prefill trims
        front = prompt_len - (replay_rows - 18 * 127)
        anchor = junction if hit < junction < prompt_len else 0

        def kept(b: int) -> bool:
            lo = b * PEEK
            return lo >= front or (bool(anchor) and anchor - 2 * BLOCK <= lo and (lo + PEEK <= anchor or lo == anchor))

        blocks = {b for b in blocks if kept(b)}
    return blocks


def landing(blocks: set, shared: int, cap: int) -> int:
    """e15's SWA hit for a prompt sharing ``shared`` tokens (hit capped at ``cap``): the largest
    128-aligned P within the full-attention hit whose window blocks and peek block [P, P+64) are
    cached and inside the shared prefix."""
    limit = min(shared, cap)
    return max((p for p in range(BLOCK, limit // BLOCK * BLOCK + 1, BLOCK)
                if p + PEEK <= limit and {p // PEEK - 2, p // PEEK - 1, p // PEEK} <= blocks), default=0)


def doc(tok, target: int, seed: int) -> str:
    from corpus import build_doc, char_ratio, trim_to_tokens

    ratio = char_ratio(tok, novel=True)
    return trim_to_tokens(build_doc(target, ratio, seed=seed, novel=True), target, tok, ratio=ratio)


def first_divergence(a: list, b: list) -> int:
    """Index of the first differing token, or the common length when one is a prefix."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def compare_item(a: dict, b: dict) -> dict:
    n = first_divergence(a["tokens"], b["tokens"])
    gaps = [abs(x - y) for x, y in zip(a["logprobs"][:n], b["logprobs"][:n])]
    overlap = [len(set(x) & set(y)) / max(1, len(x)) for x, y in zip(a["top"][:n], b["top"][:n])]
    return {
        "prompt_tokens": (a["prompt_tokens"], b["prompt_tokens"]),
        "tokens": (len(a["tokens"]), len(b["tokens"])),
        "first_divergence": n,
        "identical": a["tokens"] == b["tokens"],
        "max_abs_dlogprob": max(gaps, default=0.0),
        "min_top5_overlap": min(overlap, default=1.0),
        "hits": (a.get("hits"), b.get("hits")),
    }


def prefix_cases(a, tok, run, check) -> None:
    """The `prefix` mode's cases (see the module docstring)."""
    base = a.url
    ids = lambda messages: tokenize_chat(base, a.model, messages)  # noqa: E731
    user2 = {"role": "user", "content": FOLLOW_UP}
    other = {"role": "assistant", "content": OTHER}

    def padded(text: str, residue: int, tries: int = 8):
        pad = 0
        for _ in range(tries):
            content = text + " x" * pad + QUESTION
            n = len(ids([{"role": "user", "content": content}]))
            if n % BLOCK == residue:
                return content, n
            pad += (residue - n) % BLOCK
        raise RuntimeError(f"could not pad a prompt to N = {residue} mod {BLOCK} (last N={n})")

    opts = dict(fallback=bool(a.fallback), retention=a.retention, replay_rows=a.replay_rows)
    for r in RESIDUES:  # resends and at-N follow-ups, every tail the prefix-end fallback covers
        content, n = padded(doc(tok, a.turn_tokens, seed=5000 + r), r)
        user1 = {"role": "user", "content": content}
        run(f"cold_r{r}", [user1])
        check(f"cold_r{r}", 0)
        # its short answer's decode blocks do not reach a 128-aligned boundary past N + 64
        blocks = cached_blocks(n, n, **opts)
        run(f"resend_r{r}", [user1])
        check(f"resend_r{r}", landing(blocks, n, n - 1))
        follow = [user1, other, user2]
        f_ids = ids(follow)
        run(f"follow_r{r}", follow)
        check(f"follow_r{r}", landing(blocks, common_prefix(ids([user1]), f_ids), len(f_ids) - 1))
        # boot I1's d_same_doc_new_question: the same document, another final question
        newq = [{"role": "user", "content": content[: -len(QUESTION)] + NEW_QUESTION}]
        q_ids = ids(newq)
        run(f"newq_r{r}", newq)
        check(f"newq_r{r}", landing(blocks, common_prefix(ids([user1]), q_ids), len(q_ids) - 1))

    system = {"role": "system", "content": doc(tok, a.system_tokens, seed=6100)}  # siblings
    sib = [[system, {"role": "user", "content": q}] for q in SIBLING_QUESTIONS]
    sib_ids = [ids(m) for m in sib]
    for k, messages in enumerate(sib):
        run(f"sib{k + 1}", messages)
    d12, d23 = common_prefix(sib_ids[0], sib_ids[1]), common_prefix(sib_ids[1], sib_ids[2])
    n1, n2 = len(sib_ids[0]), len(sib_ids[1])
    check("sib1", 0)
    first = cached_blocks(n1, n1, **opts)
    hit2 = landing(first, d12, n2 - 1)
    check("sib2", hit2)
    full = min(d12, n2 - 1) // BLOCK * BLOCK  # e15: request 2's full-attention hit
    junction = full if hit2 < full else 0  # ... where its SWA hit lags it
    blocks = first | cached_blocks(n2, n2, junction=junction, hit=hit2, **opts)
    check("sib3", landing(blocks, d23, len(sib_ids[2]) - 1))

    user1 = {"role": "user", "content": doc(tok, a.turn_tokens, seed=7100) + LONG_ANSWER}  # verbatim reply
    p_ids = ids([user1])
    run("reply_turn", [user1], max_tokens=a.reply_tokens)
    first = a.result["reply_turn"]
    cont = [user1, {"role": "assistant", "content": first["text"]}, user2]
    c_ids = ids(cont)
    run("reply", cont)
    n, kv = len(p_ids), len(p_ids) + first["completion_tokens"] - 1  # the last sampled token has no KV
    blocks = cached_blocks(n, kv, reply=bool(a.reply_retain), **opts)
    check("reply", landing(blocks, common_prefix(p_ids + token_ids(first), c_ids), len(c_ids) - 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gen", "turns", "prefix", "compare"))
    ap.add_argument("files", nargs="*", help="compare: A.json B.json")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--lengths", type=int, nargs="+", default=[8192, 32768, 102400])
    ap.add_argument("--turn-tokens", type=int, default=20000)
    ap.add_argument("--system-tokens", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--reply-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=1800)
    # prefix: the boot's settings, for the expected hits
    ap.add_argument("--replay-rows", type=int, default=2670, help="DSV41_DECODER_REPLAY_ROWS of the boot")
    ap.add_argument("--retention", type=int, default=0, help="--prefix-cache-retention-interval (kv-pack: 4096)")
    ap.add_argument("--fallback", type=int, default=1, help="DSV41_PREFIX_END_FALLBACK of the boot")
    ap.add_argument("--reply-retain", type=int, default=1, help="DSV41_PREFIX_REPLY_RETAIN of the boot")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.mode == "compare":
        left, right = (json.loads(Path(f).read_text()) for f in a.files)
        rows = {k: compare_item(left[k], right[k]) for k in left if k in right and k != "checks"}
        print(json.dumps(rows, indent=1))
        same = [k for k, v in rows.items() if v["identical"]]
        # hits are checked per arm against what that boot should land (prefix mode): kv-pack's
        # checkpoints, say, serve the replay-off arm only
        failed = [f"{side}:{k}" for side, arm in (("A", left), ("B", right))
                  for k, v in arm.get("checks", {}).items() if not v["pass"]]
        print(f"identical greedy tokens: {len(same)}/{len(rows)}; per-arm hit checks failed: {failed or 'none'}",
              flush=True)
        return 0 if len(same) == len(rows) and not failed else 1
    from corpus import make_tokenizer
    from warm_prefix import hits_delta, prefix_counters

    url, metrics = f"{a.url}/v1/chat/completions", f"{a.url}/metrics"
    tok = make_tokenizer(a.url, a.model)
    result: dict = {}
    a.result = result

    def run(key, messages, max_tokens=None):
        before = prefix_counters(metrics)
        result[key] = chat(url, a.model, messages, max_tokens or a.max_tokens, a.timeout)
        result[key]["hits"] = hits_delta(before, prefix_counters(metrics))
        print(key, result[key]["prompt_tokens"], "tokens, hits", result[key]["hits"], "|", result[key]["text"][:80],
              flush=True)

    def check(key, want):
        got = result[key]["hits"]
        ok = got is not None and int(got) == want
        result.setdefault("checks", {})[key] = {"hits": got, "expected": want, "pass": ok}
        print(f"{'PASS' if ok else 'FAIL'} {key}: hits {got}, expected {want}", flush=True)

    if a.mode == "gen":
        for n in a.lengths:
            run(f"gen{n}", [{"role": "user", "content": doc(tok, n, seed=4100 + n % 997) + QUESTION}])
    elif a.mode == "prefix":
        prefix_cases(a, tok, run, check)
        failed = [k for k, v in result["checks"].items() if not v["pass"]]
        print(f"prefix checks: {len(result['checks']) - len(failed)}/{len(result['checks'])} pass", flush=True)
    else:
        user1 = {"role": "user", "content": doc(tok, a.turn_tokens, seed=4242) + QUESTION}
        run("turn1", [user1])
        reply = {"role": "assistant", "content": result["turn1"]["text"]}
        other = {"role": "assistant", "content": "I could not tell from the text."}
        user2 = {"role": "user", "content": FOLLOW_UP}
        run("resend", [user1])
        run("continue", [user1, reply, user2])
        run("diverge", [user1, other, user2])
    if a.out:
        a.out.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
