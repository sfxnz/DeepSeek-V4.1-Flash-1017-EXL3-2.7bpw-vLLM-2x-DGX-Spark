#!/usr/bin/env python3
"""N long sessions resident at once on the live serve, answer-checked (stdlib only).

Each session is one needle document of about --tokens prompt tokens (tests/quality_eval.py's filler,
needle and scoring; its own seed and nonce, so no two share a prefix), asked for its vault code with
--max-tokens greedy tokens and ignore_eos, so a session keeps decoding at ~--tokens context while the
later ones prefill. All N are sent at once. /metrics is sampled every --poll seconds (requests running
and waiting, KV cache usage, preemptions) and the JSON summary is rewritten after every sample.
Exit 1 on an HTTP error, a missed code, any preemption, or fewer than N requests running at once.

   python3 tools/resident_soak.py OUT.json --sessions 4 --tokens 450000 --max-tokens 4096
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import quality_eval as q  # noqa: E402

METRICS = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_usage": "vllm:kv_cache_usage_perc",
    "preemptions": "vllm:num_preemptions_total",
}


def metric_values(text: str) -> dict:
    """Each metric summed over its label sets in a Prometheus text page; None when absent."""
    out = {}
    for key, name in METRICS.items():
        vals = [float(m.group(1)) for m in
                re.finditer(rf"^{re.escape(name)}(?:\{{[^}}]*\}})? ([0-9.eE+-]+)$", text, re.M)]
        out[key] = sum(vals) if vals else None
    return out


def verdict(sessions: list[dict], samples: list[dict], n: int) -> list[str]:
    problems = []
    for s in sessions:
        if s.get("error"):
            problems.append(f"session {s['i']}: {s['error']}")
        elif not s.get("found"):
            problems.append(f"session {s['i']}: code {s['code']} not in {s.get('answer')!r}")
    pre = [x["preemptions"] for x in samples if x.get("preemptions") is not None]
    if not pre:
        problems.append("no vllm:num_preemptions_total samples")
    elif pre[-1] > pre[0]:
        problems.append(f"preemptions +{pre[-1] - pre[0]:.0f}")
    running = [x["running"] for x in samples if x.get("running") is not None]
    if not running or max(running) < n:
        problems.append(f"max running {max(running) if running else None} < {n}")
    return problems


def document(i: int, tokens: int, tokens_per_paragraph: float, nonce: str) -> tuple[str, str]:
    seed = 7000 + i
    name, code = q.needle_code(seed)
    n = max(4, int((tokens - 64) / tokens_per_paragraph))
    doc = q.needle_doc(q.filler_paragraphs(seed, n), (0.1, 0.5, 0.9, 0.3)[i % 4], name, code,
                       f"Archive {nonce}-{seed}. Field notes follow.")
    return f"{doc}\n\nWhat is the vault code assigned to {name}? Reply with only the code.", code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=450000, help="prompt tokens per session (approximate)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--poll", type=float, default=5.0)
    a = ap.parse_args()
    c = q.Client(a.url, None, timeout=6 * 3600)
    sample = q.filler_paragraphs(1, 200)
    tokens_per_paragraph = c.count("\n\n".join(sample)) / len(sample)  # the filler is homogeneous
    nonce = uuid.uuid4().hex[:8]
    sessions = []
    for i in range(a.sessions):
        prompt, code = document(i, a.tokens, tokens_per_paragraph, nonce)
        sessions.append({"i": i, "code": code, "prompt": prompt})
    samples: list[dict] = []
    t0 = time.time()

    def run(s: dict) -> None:
        start = time.time()
        try:
            out = c.post("/v1/chat/completions", {
                "messages": [{"role": "user", "content": s.pop("prompt")}], "max_tokens": a.max_tokens,
                "temperature": 0, "ignore_eos": True, "chat_template_kwargs": q.CHAT_KWARGS})
            text = out["choices"][0]["message"].get("content") or ""
            s.update(found=s["code"] in text[:64].upper(), answer=text.strip()[:60],
                     prompt_tokens=out["usage"]["prompt_tokens"], completion_tokens=out["usage"]["completion_tokens"])
        except Exception as exc:  # noqa: BLE001
            s["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        s["s"] = round(time.time() - start, 1)
        s["done_at_s"] = round(time.time() - t0, 1)

    def dump(final: bool = False) -> None:
        summary = {"sessions": [{k: v for k, v in s.items() if k != "prompt"} for s in sessions],
                   "elapsed_s": round(time.time() - t0, 1), "max_running": max(
                       (x["running"] for x in samples if x.get("running") is not None), default=None),
                   "max_kv_usage": max((x["kv_usage"] for x in samples if x.get("kv_usage") is not None), default=None),
                   "samples": samples}
        if final:
            summary["problems"] = verdict(sessions, samples, a.sessions)
        Path(a.out).write_text(json.dumps(summary, indent=1))

    samples.append({"t": 0.0, **metric_values(c.get("/metrics"))})
    threads = [threading.Thread(target=run, args=(s,), daemon=True) for s in sessions]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(a.poll)
        try:
            samples.append({"t": round(time.time() - t0, 1), **metric_values(c.get("/metrics"))})
        except Exception as exc:  # noqa: BLE001
            samples.append({"t": round(time.time() - t0, 1), "error": str(exc)[:120]})
        last = samples[-1]
        print(f"[resident] {last['t']:.0f}s running={last.get('running')} waiting={last.get('waiting')} "
              f"kv={last.get('kv_usage')} preemptions={last.get('preemptions')}", flush=True)
        dump()
    samples.append({"t": round(time.time() - t0, 1), **metric_values(c.get("/metrics"))})
    dump(final=True)
    problems = verdict(sessions, samples, a.sessions)
    print("RESIDENT " + json.dumps({"problems": problems, "sessions": [
        {k: s.get(k) for k in ("i", "found", "prompt_tokens", "completion_tokens", "s")} for s in sessions]}), flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
