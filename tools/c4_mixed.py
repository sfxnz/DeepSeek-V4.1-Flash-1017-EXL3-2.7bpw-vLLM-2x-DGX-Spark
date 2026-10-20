#!/usr/bin/env python3
"""Mixed 4-session cell: 3 streams keep decoding while a 4th prefills long novel prompts.

  python3 tools/c4_mixed.py run --out A.json        # the timeline below, one arm
  python3 tools/c4_mixed.py solo --out A-solo.json  # every prompt of a run alone (c=1 identity)
  python3 tools/c4_mixed.py compare A.json B.json   # greedy identity + metric deltas

run timeline: start --streams sessions (distinct prose prompts, greedy, ignore_eos,
--decode-max-tokens; a stream whose request ends starts its next one). The first
--turn-streams of them take short chat turns instead (a fresh ~--turn-tokens novel prompt,
--turn-max-tokens out, back to back): their TTFT is a new turn's wait while another
session prefills; ITL statistics use the other (pure decode) streams only. With
MAX_NUM_SEQS=4, streams + the prefilling session must stay <= 4, or a turn waits for a
slot instead of for the scheduler. Once every stream streams: --base-s of decode only,
then each --prefill length in turn (novel pseudo-word document + one question, 64 tokens
out) with --gap-s of decode only after each, then --tail-s. Then the streams are closed
(aborted).

Every request carries a per-run cache_salt, so nothing hits the prefix cache (a repeat on
the same boot is a cold prefill again; Engram pages may be warmer). Prompts depend only on
--seed, so two arms (boots) send the same text.

Recorded: per stream, every streamed chunk's arrival time and token count (DSpark emits
1..k+1 tokens per step), greedy tokens with top-2 logprobs per request; per window
(base, prefill <N>, gap <N>, tail): inter-chunk ITL p50/p90/p99 and max stall (the open
interval at the window end counts), decode tok/s per stream, turn TTFTs; per prefill:
TTFT, prompt tokens / TTFT, prefix-cache hits (/metrics delta, expect 0); MemAvailable
minima per window on this node and on --worker-host (ssh, 1 s). Stdlib only.

compare: per request, the first differing token; at a divergence the top-2 logprob margin
in both arms (a near-tie is margin < --tie nats in either arm). Solo files (one request
alone in the scheduler, scheduled identically with the sched lever on or off) must match
token for token and logprob for logprob.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "benches"))

CHAT_KWARGS = {"thinking": False, "reasoning_effort": "low"}
QUESTION = "\n\nIn two sentences, what is the text above about, and which word repeats most?"
TOPICS = (
    "how sparse attention lets a language model read very long documents",
    "why unified memory changes how an inference server budgets its caches",
    "the history and engineering of suspension bridges",
    "how a compiler turns source code into machine instructions",
    "the ecology of coral reefs and what threatens them",
    "how speculative decoding trades extra compute for lower latency",
)
STREAM_TPL = ("Write a long, detailed essay about {topic}. Use many sections and examples, and keep "
              "writing until you are told to stop.")
REC_FIELDS = ("tokens", "logprobs", "top2", "aborted", "usage", "sent", "first", "done")


# ------------------------------------------------------------------------ pure helpers


def percentile(xs: list[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100); None for no samples."""
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(q / 100 * len(s)) - 1))]


def window_stats(timelines: dict[str, list[tuple[float, int]]], t0: float, t1: float) -> dict:
    """ITL and stall statistics of decode streams over [t0, t1].

    timelines: stream -> [(arrival time, tokens in the chunk)] in arrival order. An interval
    counts when its later chunk arrives inside the window; the gap from a stream's last
    chunk (or t0) to t1 also counts as a stall, so a window spent waiting shows up."""
    itl, stalls, per_stream = [], [], {}
    for name, tl in timelines.items():
        times = [t for t, _ in tl]
        toks = sum(n for t, n in tl if t0 < t <= t1)
        prev = max([t for t in times if t <= t0], default=None)
        for t in (t for t in times if t0 < t <= t1):
            if prev is not None:
                itl.append(t - prev)
            prev = t
        stalls.append(t1 - (prev if prev is not None else t0))
        per_stream[name] = toks / (t1 - t0) if t1 > t0 else 0.0
    return {
        "t0": t0, "t1": t1, "seconds": t1 - t0, "chunks": len(itl),
        "itl_p50_s": percentile(itl, 50), "itl_p90_s": percentile(itl, 90), "itl_p99_s": percentile(itl, 99),
        "itl_mean_s": statistics.fmean(itl) if itl else None,
        "max_stall_s": max(itl + stalls) if itl or stalls else None,
        "decode_tok_s_per_stream": per_stream,
        "decode_tok_s_per_stream_min": min(per_stream.values()) if per_stream else None,
    }


def first_divergence(a: list, b: list) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def margin(rec: dict, i: int) -> float | None:
    """top-1 minus top-2 logprob at position i of a request record (None if unknown)."""
    top = rec["top2"][i] if i < len(rec.get("top2") or []) else None
    if not top or len(top) < 2:
        return None
    return top[0] - top[1]


def compare_requests(a: dict, b: dict, tie: float) -> dict:
    """{key: verdict} for requests present in both runs; verdicts: identical, prefix,
    near_tie (diverged where either arm's top-2 margin < tie), diverged."""
    out = {}
    for key in sorted(set(a) & set(b)):
        ta, tb = a[key]["tokens"], b[key]["tokens"]
        n = first_divergence(ta, tb)
        row = {"len": (len(ta), len(tb)), "first_divergence": n}
        if n == len(ta) == len(tb):
            row["verdict"] = "identical"
            lpa, lpb = a[key].get("logprobs") or [], b[key].get("logprobs") or []
            row["max_abs_dlogprob"] = max((abs(x - y) for x, y in zip(lpa, lpb)), default=0.0)
        elif n == min(len(ta), len(tb)):
            row["verdict"] = "prefix"  # one side was aborted earlier; the common part agrees
        else:
            ma, mb = margin(a[key], n), margin(b[key], n)
            row["margin"] = (ma, mb)
            near = any(m is not None and m < tie for m in (ma, mb))
            row["verdict"] = "near_tie" if near else "diverged"
        out[key] = row
    return out


# ------------------------------------------------------------------------ HTTP


def chat_stream(url: str, model: str, prompt: str, max_tokens: int, salt: str, *, ignore_eos: bool,
                stop: threading.Event | None = None, on_chunk=None, timeout: int = 3600) -> dict:
    """One streamed greedy chat request; returns timings, tokens and top-2 logprobs."""
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
        "ignore_eos": ignore_eos, "logprobs": True, "top_logprobs": 2, "cache_salt": salt,
        "chat_template_kwargs": CHAT_KWARGS,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    rec = {"sent": time.perf_counter(), "first": None, "done": None, "tokens": [], "logprobs": [], "top2": [],
           "chunks": [], "usage": {}, "aborted": False}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            if stop is not None and stop.is_set():
                rec["aborted"] = True
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                rec["usage"] = ev["usage"]
            choices = ev.get("choices") or []
            content = ((choices[0].get("logprobs") or {}).get("content") or []) if choices else []
            if not content:
                continue
            now = time.perf_counter()
            if rec["first"] is None:
                rec["first"] = now
            rec["chunks"].append((now, len(content)))
            for c in content:
                rec["tokens"].append(c["token"])
                rec["logprobs"].append(c["logprob"])
                rec["top2"].append([t["logprob"] for t in c.get("top_logprobs") or []][:2])
            if on_chunk is not None:
                on_chunk(now, len(content))
    rec["done"] = time.perf_counter()
    return rec


class MemSampler:
    """MemAvailable (MiB) at 1 s on this node and, over one ssh, on the worker node."""

    def __init__(self, worker_host: str | None) -> None:
        self.series: dict[str, list[tuple[float, int]]] = {"local": []}
        self.stop = threading.Event()
        self.threads = [threading.Thread(target=self._local, daemon=True)]
        self.proc = None
        if worker_host:
            self.series["worker"] = []
            cmd = "while :; do awk '/^MemAvailable:/{print $2}' /proc/meminfo; sleep 1; done"
            self.proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", worker_host, cmd],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self.threads.append(threading.Thread(target=self._remote, daemon=True))

    def _local(self) -> None:
        while not self.stop.is_set():
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        self.series["local"].append((time.perf_counter(), int(line.split()[1]) // 1024))
                        break
            self.stop.wait(1.0)

    def _remote(self) -> None:
        for line in self.proc.stdout:
            if self.stop.is_set():
                break
            if line.strip().isdigit():
                self.series["worker"].append((time.perf_counter(), int(line) // 1024))

    def __enter__(self):
        for t in self.threads:
            t.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        if self.proc is not None:
            self.proc.terminate()

    def minima(self, t0: float, t1: float) -> dict:
        return {k: min((v for t, v in s if t0 <= t <= t1), default=None) for k, s in self.series.items()}


# ------------------------------------------------------------------------ prompts


def build_prompts(a, tok, n_turns: int) -> dict:
    """Prompts that depend only on --seed (and the repo's corpus code): both arms send the same text."""
    from corpus import char_ratio
    from micro import fresh_doc

    ratio = char_ratio(tok, novel=True)
    streams = [STREAM_TPL.format(topic=TOPICS[i % len(TOPICS)]) + (f" (Part {i // len(TOPICS) + 1}.)" if i >= len(TOPICS) else "")
               for i in range(a.streams)]
    prefills = {}
    for n in a.prefill:
        doc, ntok = fresh_doc(n - 64, ratio, tok, seed=a.seed + n % 9973, novel=True)
        prefills[f"p{n}"] = {"prompt": doc + QUESTION, "doc_tokens": ntok}
    turns = []
    for j in range(n_turns):
        doc, _ = fresh_doc(a.turn_tokens, ratio, tok, seed=a.seed * 7 + 101 + j, novel=True)
        turns.append(doc + "\n\nIn one sentence: what is this text about?")
    return {"streams": streams, "prefills": prefills, "turns": turns}


# ------------------------------------------------------------------------ modes


def run_mode(a) -> dict:
    from corpus import make_tokenizer
    from warm_prefix import hits_delta, prefix_counters

    base = a.url.rsplit("/v1", 1)[0]
    metrics = f"{base}/metrics"
    tok = make_tokenizer(base, a.model)
    prompts = build_prompts(a, tok, a.turns_max if a.turn_streams else 0)
    salt = f"c4-{a.salt or time.time_ns()}"
    stop = threading.Event()
    timelines: dict[str, list[tuple[float, int]]] = {}  # pure decode streams: the ITL samples
    turn_lines: dict[str, list[tuple[float, int]]] = {}  # turn-taking streams
    requests: dict[str, dict] = {}
    turn_ttft: list[tuple[float, float]] = []  # (sent, ttft)
    errors: list[str] = []
    n_turn = min(a.turn_streams, a.streams)

    def decode_stream(i: int) -> None:
        tl = (turn_lines if i < n_turn else timelines).setdefault(f"s{i}", [])
        j = 0
        while not stop.is_set():
            if i < n_turn:
                if j >= len(prompts["turns"]):
                    return
                prompt, max_tokens, key = prompts["turns"][j], a.turn_max_tokens, f"t{i}.{j}"
            else:
                prompt, max_tokens, key = prompts["streams"][i] + ("" if j == 0 else f" (Continue, part {j + 1}.)"), \
                    a.decode_max_tokens, f"s{i}.{j}"
            try:
                rec = chat_stream(a.url, a.model, prompt, max_tokens, f"{salt}-{key}", ignore_eos=True, stop=stop,
                                  on_chunk=lambda t, n: tl.append((t, n)))
            except Exception as exc:  # noqa: BLE001 - recorded, the cell goes on
                errors.append(f"{key}: {exc!r}")
                return
            requests[key] = rec
            if i < n_turn and rec["first"] is not None:
                turn_ttft.append((rec["sent"], rec["first"] - rec["sent"]))
            j += 1

    events: dict[str, float] = {}
    prefills: dict[str, dict] = {}
    windows: dict[str, dict] = {}
    with MemSampler(a.worker_host) as mem:
        threads = [threading.Thread(target=decode_stream, args=(i,), daemon=True) for i in range(a.streams)]
        for t in threads:
            t.start()
        deadline = time.perf_counter() + a.ready_timeout
        while time.perf_counter() < deadline and not all(
                (turn_lines if i < n_turn else timelines).get(f"s{i}") for i in range(a.streams)):
            time.sleep(0.2)
        events["ready"] = time.perf_counter()
        time.sleep(a.base_s)
        events["base_end"] = time.perf_counter()
        windows["base"] = (events["ready"], events["base_end"])
        for key, p in prompts["prefills"].items():
            before = prefix_counters(metrics)
            rec = chat_stream(a.url, a.model, p["prompt"], 64, f"{salt}-{key}", ignore_eos=False)
            hits = hits_delta(before, prefix_counters(metrics))
            requests[key] = rec
            ttft = (rec["first"] or rec["done"]) - rec["sent"]
            ptok = int(rec["usage"].get("prompt_tokens") or 0)
            prefills[key] = {"prompt_tokens": ptok, "ttft_s": ttft, "prefill_tok_s": ptok / ttft if ttft else None,
                             "total_s": rec["done"] - rec["sent"], "prefix_hits": hits}
            windows[f"prefill {key}"] = (rec["sent"], rec["first"] or rec["done"])
            g0 = time.perf_counter()
            time.sleep(a.gap_s)
            windows[f"gap {key}"] = (g0, time.perf_counter())
            print(f"{key}: {ptok} tokens, TTFT {ttft:.1f} s, {prefills[key]['prefill_tok_s']:.0f} tok/s, hits {hits}",
                  flush=True)
        t0 = time.perf_counter()
        time.sleep(a.tail_s)
        windows["tail"] = (t0, time.perf_counter())
        stop.set()
        for t in threads:
            t.join(timeout=a.ready_timeout)
        mem_out = {name: mem.minima(t0w, t1w) for name, (t0w, t1w) in windows.items()}
        mem_out["all"] = mem.minima(events["ready"], time.perf_counter())
    stats = {}
    for name, (t0w, t1w) in windows.items():
        st = window_stats(timelines, t0w, t1w)
        st["turn_ttft_s"] = [round(x, 3) for s, x in turn_ttft if t0w <= s <= t1w]
        stats[name] = st
    return {
        "mode": "run", "args": vars(a), "salt": salt, "errors": errors, "prefills": prefills, "windows": stats,
        "mem_min_mib": mem_out, "timelines": timelines, "turn_timelines": turn_lines,
        "requests": {k: {f: v[f] for f in REC_FIELDS} for k, v in requests.items()},
    }


def solo_mode(a) -> dict:
    """Each run prompt alone (stream prompts' first request, turn prompts, prefills), greedy."""
    from corpus import make_tokenizer

    base = a.url.rsplit("/v1", 1)[0]
    prompts = build_prompts(a, make_tokenizer(base, a.model), a.solo_turns)
    salt = f"c4solo-{a.salt or time.time_ns()}"
    items = [(f"s{i}.0", p, a.solo_tokens) for i, p in enumerate(prompts["streams"])]
    items += [(f"t0.{j}", p, a.turn_max_tokens) for j, p in enumerate(prompts["turns"])]
    items += [(k, p["prompt"], 64) for k, p in prompts["prefills"].items()]
    out = {}
    for key, prompt, n in items:
        rec = chat_stream(a.url, a.model, prompt, n, f"{salt}-{key}", ignore_eos=key[0] in "st")
        out[key] = {f: rec[f] for f in REC_FIELDS}
        out[key]["ttft_s"] = (rec["first"] or rec["done"]) - rec["sent"]
        print(f"solo {key}: {len(rec['tokens'])} tokens, TTFT {out[key]['ttft_s']:.2f} s", flush=True)
    return {"mode": "solo", "args": vars(a), "requests": out}


def compare_mode(a) -> dict:
    left, right = (json.loads(Path(f).read_text()) for f in a.files)
    rows = compare_requests(left["requests"], right["requests"], a.tie)
    verdicts = {}
    for r in rows.values():
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    out = {"verdicts": verdicts, "requests": rows}
    if left.get("mode") == right.get("mode") == "solo":
        out["solo_exact"] = all(r["verdict"] == "identical" and r["max_abs_dlogprob"] == 0.0 for r in rows.values())
    if left.get("mode") == right.get("mode") == "run":
        out["windows"] = {
            w: {k: (left["windows"][w].get(k), right["windows"][w].get(k))
                for k in ("itl_p50_s", "itl_p90_s", "itl_p99_s", "max_stall_s", "decode_tok_s_per_stream_min",
                          "turn_ttft_s")}
            for w in left["windows"] if w in right["windows"]}
        out["prefills"] = {k: (left["prefills"][k], right["prefills"].get(k)) for k in left["prefills"]}
        out["mem_min_mib"] = (left["mem_min_mib"].get("all"), right["mem_min_mib"].get("all"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("run", "solo", "compare"))
    ap.add_argument("files", nargs="*", help="compare: A.json B.json")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--streams", type=int, default=3)
    ap.add_argument("--turn-streams", type=int, default=1, help="of --streams, how many take short turns")
    ap.add_argument("--turn-tokens", type=int, default=1024)
    ap.add_argument("--turn-max-tokens", type=int, default=96)
    ap.add_argument("--turns-max", type=int, default=120)
    ap.add_argument("--decode-max-tokens", type=int, default=8000)
    ap.add_argument("--prefill", type=int, nargs="+", default=[65536, 131072])
    ap.add_argument("--base-s", type=float, default=20.0)
    ap.add_argument("--gap-s", type=float, default=10.0)
    ap.add_argument("--tail-s", type=float, default=10.0)
    ap.add_argument("--ready-timeout", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=4300)
    ap.add_argument("--salt", default="", help="cache_salt suffix (default: time); same salt = prefix hits")
    ap.add_argument("--worker-host", default="spark2", help="'' = this node only")
    ap.add_argument("--solo-tokens", type=int, default=192)
    ap.add_argument("--solo-turns", type=int, default=3)
    ap.add_argument("--tie", type=float, default=0.25, help="compare: near-tie margin (nats)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.mode == "compare":
        result = compare_mode(a)
        print(json.dumps({k: v for k, v in result.items() if k != "requests"}, indent=1))
    else:
        result = run_mode(a) if a.mode == "run" else solo_mode(a)
        if a.mode == "run":
            for name, st in result["windows"].items():
                print(f"{name:>16}: ITL p50 {st['itl_p50_s']} p99 {st['itl_p99_s']} stall {st['max_stall_s']} "
                      f"min tok/s/stream {st['decode_tok_s_per_stream_min']} turns {st['turn_ttft_s']}", flush=True)
            print("MemAvailable min MiB", result["mem_min_mib"]["all"], "errors", result["errors"], flush=True)
    if a.out:
        a.out.write_text(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
