"""Source adapters: pinned HF datasets -> candidate conversations in OpenAI message format.

Every adapter is a generator `fn(rng, ctx)` yielding candidate dicts:
    key        str   identity of the prompt (sha256 of source + first user text); a key is used at most once
    subdomain  str
    lang       str
    messages   list  OpenAI-style messages (image blocks use url = "images/<sha256>.<ext>")
    thinking_mode, reasoning_effort
    image_bytes dict url -> bytes
    source     dict  dataset, revision, file, row_group, row, license, generator
Row groups are drawn from files chosen by the seeded rng and read over HTTP range requests (hub.py).
"""

from __future__ import annotations

import ast
import hashlib
import json
import re

import hub

# Pinned revisions (resolved 2026-10-17 via the HF API).
REV = {
    "nvidia/Nemotron-Post-Training-Dataset-v1": "74e23eb6f830fef4a9e96a92f6f6262214cbb9a8",
    "nvidia/Llama-Nemotron-Post-Training-Dataset": "ab2a40d258a6a4d9d4c277d702aeea445081766c",
    "HuggingFaceTB/smoltalk": "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa",
    "OpenCoder-LLM/opc-sft-stage2": "7d28f40d579edd7c24402d17d0c7639f991e6f8d",
    "SWE-bench/SWE-smith-trajectories": "08e109b4a59eaeebf80e4675cd125d42e7ac99a4",
    "NousResearch/hermes-function-calling-v1": "dae3e1d28cfbcf4b915c04ea1e072030529b4bda",
    # parquet-converter commit of refs/convert/parquet, resolved 2026-10-17 (pinned so rebuilds cannot drift)
    "Congliu/Chinese-DeepSeek-R1-Distill-data-110k": "ebbde07b432b436f441b810fd302297867800bab",
    "lightblue/reasoning-multilingual-R1-Llama-70B-train": "242ea5e1d94e47357b3bd756175e277d16996202",
    "CohereLabs/aya_dataset": "f9ea04583f02a8f86404ff6c58bf75fe637df8a2",
    "emozilla/pg19": "c021754c8e01c5b1cc83a1f549c1f97fbbb756b8",
    "codeparrot/github-code-clean": "c48d40f9e70f0196f8236901ee35807f7d6c44c0",
    "zai-org/LongBench-v2": "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
    "wikimedia/wikipedia": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    "lmms-lab/LLaVA-OneVision-Data": "7ca5e5bf8b2006d5dfa0549756198474f0897f63",
}
LICENSE = {
    "nvidia/Nemotron-Post-Training-Dataset-v1": "cc-by-4.0",
    "nvidia/Llama-Nemotron-Post-Training-Dataset": "cc-by-4.0",
    "HuggingFaceTB/smoltalk": "apache-2.0 (smol-magpie-ultra)",
    "OpenCoder-LLM/opc-sft-stage2": "mit",
    "SWE-bench/SWE-smith-trajectories": "mit",
    "NousResearch/hermes-function-calling-v1": "apache-2.0",
    "Congliu/Chinese-DeepSeek-R1-Distill-data-110k": "apache-2.0",
    "lightblue/reasoning-multilingual-R1-Llama-70B-train": "apache-2.0",
    "CohereLabs/aya_dataset": "apache-2.0",
    "emozilla/pg19": "apache-2.0 (dataset); texts are Project Gutenberg public domain",
    "codeparrot/github-code-clean": "dataset apache-2.0; each file keeps its own license (only mit/apache-2.0/bsd/isc kept)",
    "zai-org/LongBench-v2": "apache-2.0",
    "wikimedia/wikipedia": "cc-by-sa-3.0 / gfdl",
    "lmms-lab/LLaVA-OneVision-Data": "apache-2.0 (annotations); images carry their source datasets' terms",
}
DEFAULT_SYSTEM = "You are a helpful assistant."


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def effort(rng):
    """Reasoning effort mix: high 55%, max 20%, low 13%, any 1-100 12%."""
    r = rng.random()
    return 75 if r < .55 else 100 if r < .75 else 50 if r < .88 else rng.randint(1, 100)


def split_think(text):
    """'<think>\\nR\\n</think>\\n\\nA' -> (R, A); (None, text) when there is no think block."""
    m = re.match(r"\s*<think>(.*?)</think>(.*)\Z", text, re.S)
    if not m:
        return None, text
    return m.group(1).strip(), m.group(2).lstrip()


def _src(repo, file, rg, row, generator=None, **kw):
    d = dict(dataset=repo, revision=REV[repo], file=file, row_group=rg, row=row, license=LICENSE[repo])
    if generator:
        d["generator"] = generator
    d.update(kw)
    return d


def parquet_rows(repo, files, rng, columns, max_groups=60, ctx=None, per_group=8):
    """Yield (file, rg, row_idx, row) from random row groups of random files, rows shuffled; at most
    `per_group` rows per row group (adjacent rows are often topically clustered)."""
    rev = REV[repo]
    seen = set()
    for _ in range(max_groups):
        f = rng.choice(files)
        pf, hf = hub.parquet(repo, rev, f)
        rg = rng.randrange(pf.num_row_groups)
        if (f, rg) in seen:
            continue
        seen.add((f, rg))
        rows = pf.read_row_group(rg, columns=columns).to_pylist()
        if ctx is not None:
            ctx["bytes_fetched"] = ctx.get("bytes_fetched", 0) + hf.fetched
            ctx["log"](f"  fetched {repo}:{f} rg={rg}/{pf.num_row_groups} rows={len(rows)} {hf.fetched/1e6:.1f} MB")
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        for i in idx[:per_group]:
            yield f, rg, i, rows[i]


def jsonl_rows(repo, path, rng, ctx, chunk=1_500_000, max_chunks=400, per_window=8):
    """Yield (path, byte_offset, line_idx, row) from random byte windows of a large remote jsonl."""
    f = hub.HttpFile(hub.resolve_url(repo, REV[repo], path))
    for _ in range(max_chunks):
        off = rng.randrange(max(1, f.size - chunk))
        f.seek(off)
        before = f.fetched
        lines = f.read(chunk).split(b"\n")[1:-1]  # drop the partial first/last lines
        ctx["bytes_fetched"] = ctx.get("bytes_fetched", 0) + f.fetched - before
        ctx["log"](f"  fetched {repo}:{path} @{off} lines={len(lines)}")
        idx = list(range(len(lines)))
        rng.shuffle(idx)
        for j in idx[:per_window]:
            yield path, off, j, json.loads(lines[j])


def _files(repo, prefix, ctx, pattern=None):
    fs = hub.list_files(repo, REV[repo], prefix)
    fs = [f for f in fs if f.endswith(".parquet") and (pattern is None or re.search(pattern, f))]
    if not fs:
        raise RuntimeError(f"no parquet files for {repo}/{prefix} {pattern}")
    return fs


# ------------------------------------------------------------------ Nemotron v1 (R1-0528 / Qwen3-235B)
def nemotron(split, subdomain, thinking_only=False):
    repo = "nvidia/Nemotron-Post-Training-Dataset-v1"

    def gen(rng, ctx):
        files = _files(repo, "data", ctx, rf"/{split}-\d{{5}}-of-")
        for f, rg, i, r in parquet_rows(repo, files, rng, ["uuid", "generator", "reasoning", "messages", "metadata"], ctx=ctx):
            msgs = r["messages"]
            if not msgs or any(m["role"] == "user" and len((m["content"] or "").strip()) < 16 for m in msgs):
                continue  # externally sourced prompts are blank or "-" in this release
            think = r["reasoning"] == "on"
            if thinking_only and not think:
                continue
            out = []
            meta = json.loads(r["metadata"] or "{}") if r["metadata"] else {}
            tools = meta.get("tools") if isinstance(meta, dict) else None
            if tools:
                out.append({"role": "system", "content": DEFAULT_SYSTEM, "tools": tools})
            ok = True
            for m in msgs:
                if m["role"] == "assistant":
                    rc, content = split_think(m["content"] or "")
                    if think and rc is None:
                        ok = False
                        break
                    a = {"role": "assistant", "content": content}
                    if think:
                        a["reasoning_content"] = rc
                    if m.get("tool_calls"):
                        a["tool_calls"] = [{"id": tc.get("id") or "", "type": "function",
                                            "function": {"name": tc["function"]["name"],
                                                         "arguments": tc["function"]["arguments"]}}
                                           for tc in m["tool_calls"]]
                    out.append(a)
                elif m["role"] in ("user", "system", "tool"):
                    out.append({"role": m["role"], "content": m["content"] or ""})
                else:
                    ok = False
            if not ok or out[-1]["role"] != "assistant":
                continue
            first_user = next(m["content"] for m in out if m["role"] == "user")
            yield dict(key=sha(repo + "\0" + first_user), subdomain=subdomain, lang="en", messages=out,
                       thinking_mode="thinking" if think else "chat",
                       reasoning_effort=effort(rng) if think else None, image_bytes={},
                       source=_src(repo, f, rg, i, r["generator"], uuid=r["uuid"]))
    return gen


# ------------------------------------------------------------------ Llama-Nemotron SFT (DeepSeek-R1 "on" / Llama-Nemotron "off")
def llama_nemotron(path, sub, thinking_only=False):
    return lambda rng, ctx: _llama_nemotron(rng, ctx, path, sub, thinking_only)


def _llama_nemotron(rng, ctx, path, sub, thinking_only):
    repo = "nvidia/Llama-Nemotron-Post-Training-Dataset"
    for f, off, j, r in jsonl_rows(repo, path, rng, ctx):
        inp = r.get("input")
        if not isinstance(inp, list) or not inp or inp[-1].get("role") != "user":
            continue
        if any(not (m.get("content") or "").strip() for m in inp) or any(m.get("role") not in ("user", "assistant") for m in inp):
            continue
        think = r.get("reasoning") == "on"
        if thinking_only and not think:
            continue
        if len(inp[-1]["content"].strip()) < 16:
            continue
        rc, content = split_think(r["output"] or "")
        if think and rc is None:
            continue
        a = {"role": "assistant", "content": content.strip()}
        if think:
            a["reasoning_content"] = rc
        msgs = [{"role": m["role"], "content": m["content"]} for m in inp] + [a]
        yield dict(key=sha(repo + "\0" + inp[0]["content"]), subdomain=sub + ("-r1" if think else "-off"),
                   lang="en", messages=msgs, thinking_mode="thinking" if think else "chat",
                   reasoning_effort=effort(rng) if think else None, image_bytes={},
                   source=_src(repo, f, None, None, r.get("generator"), byte_offset=off, line_in_window=j))


# ------------------------------------------------------------------ smoltalk magpie-ultra (Llama-3.1-405B)
def smoltalk(rng, ctx):
    repo = "HuggingFaceTB/smoltalk"
    files = _files(repo, "data/smol-magpie-ultra", ctx, r"/train-")
    for f, rg, i, r in parquet_rows(repo, files, rng, ["messages", "category", "quality"], ctx=ctx):
        if r["quality"] not in ("good", "excellent"):
            continue
        msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
        if not msgs or msgs[-1]["role"] != "assistant":
            continue
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        yield dict(key=sha(repo + "\0" + first_user), subdomain=f"chat/magpie-{r['category']}", lang="en",
                   messages=msgs, thinking_mode="chat", reasoning_effort=None, image_bytes={},
                   source=_src(repo, f, rg, i, "Llama-3.1-405B-Instruct"))


# ------------------------------------------------------------------ OpenCoder mceval (multi-language code)
def opc_mceval(rng, ctx):
    repo = "OpenCoder-LLM/opc-sft-stage2"
    files = _files(repo, "mceval_instruct", ctx)
    for f, rg, i, r in parquet_rows(repo, files, rng, ["instruction", "output"], ctx=ctx):
        m = re.search(r"```(\w[\w+#-]*)", r["output"] or "")
        lang = m.group(1).lower() if m else "unknown"
        msgs = [{"role": "user", "content": r["instruction"].strip()},
                {"role": "assistant", "content": r["output"].strip()}]
        yield dict(key=sha(repo + "\0" + msgs[0]["content"]), subdomain=f"code/mceval-{lang}", lang="en",
                   messages=msgs, thinking_mode="chat", reasoning_effort=None, image_bytes={},
                   source=_src(repo, f, rg, i, "opc-sft-stage2 (mceval_instruct)"))


# ------------------------------------------------------------------ SWE-smith agentic coding (tool split)
SWE_TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a bash command in the repository's environment and return its output.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The bash command to run."}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "str_replace_editor",
        "description": "Custom editing tool for viewing, creating and editing files. `view` shows a file with line "
                       "numbers or lists a directory; `create` writes a new file; `str_replace` replaces exactly one "
                       "occurrence of old_str with new_str; `insert` inserts new_str after insert_line; "
                       "`undo_edit` reverts the last edit to path.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "enum": ["view", "create", "str_replace", "insert", "undo_edit"]},
            "path": {"type": "string", "description": "Absolute path to file or directory."},
            "file_text": {"type": "string", "description": "Content of the file for `create`."},
            "old_str": {"type": "string", "description": "String to replace for `str_replace`."},
            "new_str": {"type": "string", "description": "Replacement string for `str_replace` / `insert`."},
            "insert_line": {"type": "integer", "description": "Line after which to insert for `insert`."},
            "view_range": {"type": "array", "items": {"type": "integer"},
                           "description": "Optional [start, end] line range for `view`."}},
            "required": ["command", "path"]}}},
    {"type": "function", "function": {
        "name": "submit", "description": "Submit the current changes as the final solution.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]


def _pyval(s):
    if not isinstance(s, str):
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def _text(c):
    c = _pyval(c)
    if isinstance(c, list):
        return "\n\n".join(b.get("text", "") for b in c if isinstance(b, dict))
    return c or ""


def swe_smith(rng, ctx):
    repo = "SWE-bench/SWE-smith-trajectories"
    files = _files(repo, "data", ctx, r"/tool-")
    for f, rg, i, r in parquet_rows(repo, files, rng, ["messages", "instance_id", "resolved", "model", "traj_id"], ctx=ctx):
        if not r["resolved"]:
            continue
        raw = json.loads(r["messages"])
        msgs = []
        for m in raw:
            role = m["role"]
            if role == "system":
                msgs.append({"role": "system", "content": _text(m["content"]), "tools": SWE_TOOLS})
            elif role == "user":
                msgs.append({"role": "user", "content": _text(m["content"])})
            elif role == "assistant":
                a = {"role": "assistant", "content": _text(m["content"])}
                tcs = _pyval(m.get("tool_calls")) or []
                if tcs:
                    a["tool_calls"] = [{"id": tc.get("id", ""), "type": "function",
                                        "function": {"name": tc["function"]["name"],
                                                     "arguments": tc["function"]["arguments"]}} for tc in tcs]
                msgs.append(a)
            elif role == "tool":
                ids = _pyval(m.get("tool_call_ids")) or [""]
                msgs.append({"role": "tool", "content": _text(m["content"]), "tool_call_id": ids[0]})
        if len(msgs) < 4 or msgs[0]["role"] != "system" or "BEGIN FUNCTION" in msgs[0]["content"]:
            continue  # keep only native function-calling trajectories (no text-format <function=...> calls)
        if any(m["role"] == "assistant" and ("<function=" in m["content"] or not m.get("tool_calls")) for m in msgs[:-1]):
            continue
        yield dict(key=sha(repo + "\0" + r["instance_id"]), subdomain="code/agentic-swe", lang="en", messages=msgs,
                   thinking_mode="chat", reasoning_effort=None, image_bytes={}, truncatable=True,
                   source=_src(repo, f, rg, i, r["model"], traj_id=r["traj_id"]))


# ------------------------------------------------------------------ Hermes function calling (multi-turn, tool results)
def hermes(rng, ctx):
    repo = "NousResearch/hermes-function-calling-v1"
    path = ctx["raw"] / "hermes" / "func-calling.json"
    hub.download(repo, REV[repo], "func-calling.json", path)
    rows = json.loads(path.read_text())
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    for i in idx:
        r = rows[i]
        conv = r["conversations"]
        try:
            tools = json.loads(r["tools"]) if r.get("tools") else None
        except Exception:
            tools = None
        if not tools:
            continue
        msgs, ok, n_call = [{"role": "system", "content": DEFAULT_SYSTEM, "tools": tools}], True, 0
        for c in conv:
            frm, v = c["from"], c["value"]
            if frm == "system":
                continue
            if frm == "human":
                msgs.append({"role": "user", "content": v})
            elif frm == "gpt":
                calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", v, re.S)
                text = re.sub(r"<tool_call>.*?</tool_call>", "", v, flags=re.S).strip()
                a = {"role": "assistant", "content": text}
                if calls:
                    tcs = []
                    for j, cj in enumerate(calls):
                        try:
                            d = json.loads(cj)
                        except Exception:
                            try:
                                d = ast.literal_eval(cj)
                            except Exception:
                                ok = False
                                break
                        n_call += 1
                        tcs.append({"id": f"call_{n_call}", "type": "function",
                                    "function": {"name": d["name"], "arguments": json.dumps(d.get("arguments", {}), ensure_ascii=False)}})
                    a["tool_calls"] = tcs
                msgs.append(a)
            elif frm == "tool":
                for k, resp in enumerate(re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>", v, re.S) or [v]):
                    msgs.append({"role": "tool", "content": resp})
            else:
                ok = False
        if not ok or msgs[-1]["role"] != "assistant" or not any(m["role"] == "tool" for m in msgs):
            continue
        first_user = next(m["content"] for m in msgs if m["role"] == "user")
        yield dict(key=sha(repo + "\0" + first_user), subdomain=f"tool/hermes-{r.get('category', '')}", lang="en",
                   messages=msgs, thinking_mode="chat", reasoning_effort=None, image_bytes={},
                   source=_src(repo, "func-calling.json", None, i, "hermes-function-calling-v1", id=r.get("id")))


# ------------------------------------------------------------------ multilingual
def zh_r1(rng, ctx):
    repo = "Congliu/Chinese-DeepSeek-R1-Distill-data-110k"
    files = _files(repo, "default/train", ctx)
    for f, rg, i, r in parquet_rows(repo, files, rng, ["input", "content", "reasoning_content", "repo_name", "score"], ctx=ctx):
        if not r["input"] or not r["content"] or not r["reasoning_content"] or (r["score"] or 0) < 8:
            continue
        msgs = [{"role": "user", "content": r["input"].strip()},
                {"role": "assistant", "content": r["content"].strip(), "reasoning_content": r["reasoning_content"].strip()}]
        yield dict(key=sha(repo + "\0" + msgs[0]["content"]), subdomain="multilingual/zh", lang="zh", messages=msgs,
                   thinking_mode="thinking", reasoning_effort=effort(rng), image_bytes={},
                   source=_src(repo, f, rg, i, "DeepSeek-R1", repo_name=r["repo_name"]))


LB_LANG = {"Japanese": "ja", "Russian": "ru", "Arabic": "ar", "Korean": "ko", "German": "de", "French": "fr",
           "Spanish": "es", "Chinese": "zh", "Italian": "it", "Portuguese": "pt", "Vietnamese": "vi", "Thai": "th",
           "Indonesian": "id", "Turkish": "tr", "Hindi": "hi", "Polish": "pl", "Dutch": "nl", "Bengali": "bn",
           "Swahili": "sw", "Ukrainian": "uk", "Persian": "fa", "Hebrew": "he", "Greek": "el", "Czech": "cs",
           "Swedish": "sv", "Hungarian": "hu", "Finnish": "fi", "Romanian": "ro", "Malay": "ms", "Amharic": "am"}


CORE_LANGS = ["es", "de", "fr", "ja", "ru", "ar", "ko", "pt", "it", "tr", "vi", "hi", "id", "pl", "nl", "uk", "fa", "he", "th"]


def _round_robin(buckets, rng, order):
    """Yield items alternating over languages (in `order`, then any others), each bucket shuffled."""
    langs = [l for l in order if buckets.get(l)] + sorted(l for l in buckets if l not in order and buckets[l])
    for l in langs:
        rng.shuffle(buckets[l])
    while any(buckets.get(l) for l in langs):
        for l in langs:
            if buckets.get(l):
                yield l, buckets[l].pop()


def _read_parquet(repo, path, ctx, columns):
    import pyarrow.parquet as pq
    local = hub.download(repo, REV[repo], path, ctx["raw"] / repo.replace("/", "__") / path)
    return pq.read_table(local, columns=columns).to_pylist()


def lightblue(rng, ctx):
    repo = "lightblue/reasoning-multilingual-R1-Llama-70B-train"
    path = "data/train-00000-of-00001.parquet"
    cols = ["language", "translated_prompt", "response", "is_valid_think", "is_valid_res",
            "prompt_think_lang_correct", "answer_correctness", "row_id"]
    buckets = {}
    for i, r in enumerate(_read_parquet(repo, path, ctx, cols)):
        lang = LB_LANG.get(r["language"])
        if not lang or lang == "zh" or not (r["is_valid_think"] and r["is_valid_res"] and r["prompt_think_lang_correct"]):
            continue  # zh comes from the native-Chinese R1 set
        if (r["answer_correctness"] or 0) < 4:
            continue
        rc, content = split_think(r["response"] or "")
        if rc is None or not content.strip():
            continue
        buckets.setdefault(lang, []).append((i, r, rc, content))
    for lang, (i, r, rc, content) in _round_robin(buckets, rng, CORE_LANGS):
        msgs = [{"role": "user", "content": r["translated_prompt"].strip()},
                {"role": "assistant", "content": content.strip(), "reasoning_content": rc}]
        yield dict(key=sha(repo + "\0" + msgs[0]["content"]), subdomain=f"multilingual/{lang}", lang=lang,
                   messages=msgs, thinking_mode="thinking", reasoning_effort=effort(rng), image_bytes={},
                   source=_src(repo, path, None, i, "DeepSeek-R1-Distill-Llama-70B", row_id=r["row_id"]))


AYA_LANGS = {"arb": "ar", "rus": "ru", "zho": "zh", "spa": "es", "deu": "de", "fra": "fr", "jpn": "ja", "kor": "ko",
             "por": "pt", "ita": "it", "hin": "hi", "tur": "tr", "vie": "vi", "ind": "id", "pes": "fa", "ukr": "uk",
             "pol": "pl", "nld": "nl", "ben": "bn", "tha": "th", "swh": "sw", "heb": "he", "ell": "el"}


def aya(rng, ctx):
    """Human-written prompt/answer pairs packed 4-10 per conversation, same language (multi-turn chat),
    languages taken round-robin."""
    repo = "CohereLabs/aya_dataset"
    path = "data/train-00000-of-00001.parquet"
    buckets = {}
    for i, r in enumerate(_read_parquet(repo, path, ctx, ["inputs", "targets", "language_code", "annotation_type"])):
        lang = AYA_LANGS.get(r["language_code"])
        if not lang or lang == "zh" or r["annotation_type"] != "original-annotations":
            continue
        if not r["inputs"].strip() or len(r["targets"].strip()) < 40:
            continue
        buckets.setdefault(lang, []).append((i, r))
    for l in buckets:
        rng.shuffle(buckets[l])
    convs = {}
    for l, items in buckets.items():
        out = []
        while len(items) >= 4:
            k = rng.randint(4, 10)
            out.append(items[:k])
            items = items[k:]
        convs[l] = out
    for lang, items in _round_robin(convs, rng, CORE_LANGS):
        msgs = []
        for _, x in items:
            msgs += [{"role": "user", "content": x["inputs"].strip()}, {"role": "assistant", "content": x["targets"].strip()}]
        yield dict(key=sha(repo + "\0" + "\0".join(x["inputs"] for _, x in items)), subdomain=f"multilingual/{lang}",
                   lang=lang, messages=msgs, thinking_mode="chat", reasoning_effort=None, image_bytes={},
                   source=_src(repo, path, None, [i for i, _ in items], "human (Aya annotators)"))


# ------------------------------------------------------------------ encyclopedic
def wikipedia(rng, ctx):
    repo = "wikimedia/wikipedia"
    files = _files(repo, "20231101.en", ctx)
    for f, rg, i, r in parquet_rows(repo, files, rng, ["id", "title", "text", "url"], ctx=ctx, per_group=40):
        text = r["text"].strip()
        if len(text) < 4000 or len(text) > 24000:
            continue
        msgs = [{"role": "user", "content": f"Write an encyclopedia article about \"{r['title']}\"."},
                {"role": "assistant", "content": text}]
        yield dict(key=sha(repo + "\0" + r["id"]), subdomain="encyclopedic/wikipedia-en", lang="en", messages=msgs,
                   thinking_mode="chat", reasoning_effort=None, image_bytes={},
                   source=_src(repo, f, rg, i, "human (Wikipedia editors)", page_id=r["id"], url=r["url"]))


# ------------------------------------------------------------------ long context (raw text; build.py cuts to length)
def pg19(rng, ctx):
    repo = "emozilla/pg19"
    files = _files(repo, "data", ctx, r"/(validation|test)-")  # 150 books, small files
    for f, rg, i, r in parquet_rows(repo, files, rng, ["short_book_title", "publication_date", "url", "text"], ctx=ctx, per_group=10**9):
        if len(r["text"]) < 200_000:
            continue
        yield dict(key=sha(repo + "\0" + r["url"]), subdomain="long/book", lang="en", long_kind="book",
                   title=r["short_book_title"], text=r["text"],
                   source=_src(repo, f, rg, i, "human (public-domain book)", url=r["url"]))


CODE_LANGS = {"Python", "Java", "C++", "C", "C#", "Go", "Rust", "TypeScript", "Ruby", "Scala"}
PERMISSIVE = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc"}


def github_code(rng, ctx):
    repo = "codeparrot/github-code-clean"
    files = [f"data/train-{k:05d}-of-00880.parquet" for k in range(880)]
    for f, rg, i, r in parquet_rows(repo, files, rng, ["code", "repo_name", "path", "language", "license", "size"], ctx=ctx, per_group=10**9):
        if r["license"] not in PERMISSIVE or not (60_000 <= r["size"] <= 140_000) or r["language"] not in CODE_LANGS:
            continue
        lines = [ln.strip() for ln in r["code"].split("\n") if ln.strip()]
        if not lines or len(set(lines)) / len(lines) < 0.6 or sum(ln.startswith("#define") for ln in lines) / len(lines) > 0.2:
            continue  # generated tables / register maps
        if max(len(ln) for ln in lines) > 400 or re.search(r"(?i)(\.min\.|log|dump|generated|vendor|third_party|fixture|test_?data|/data/)", r["path"]):
            continue
        kw = len(re.findall(r"\b(def|class|func|fn|function|public|private|struct|impl|return|static|const|let|var)\b", r["code"]))
        if kw / len(lines) < 0.08:
            continue  # mostly data, not code
        yield dict(key=sha(repo + "\0" + r["repo_name"] + "/" + r["path"]), subdomain=f"long/code-{r['language'].lower()}",
                   lang="en", long_kind="code", title=f"{r['repo_name']}/{r['path']}", text=r["code"],
                   code_lang=r["language"], source=_src(repo, f, rg, i, "human (GitHub)", file_license=r["license"],
                                                        repo_name=r["repo_name"], path=r["path"]))


def longbench2(rng, ctx):
    repo = "zai-org/LongBench-v2"
    path = ctx["raw"] / "longbench2" / "data.json"
    hub.download(repo, REV[repo], "data.json", path)
    rows = json.loads(path.read_text())
    idx = [k for k, r in enumerate(rows) if r.get("length") == "short"]
    rng.shuffle(idx)
    for k in idx:
        r = rows[k]
        q = ("Please read the following text and answer the question below.\n\n<text>\n" + r["context"].strip() +
             "\n</text>\n\nWhat is the correct answer to this question: " + r["question"].strip() + "\nChoices:\n" +
             "".join(f"({c}) {r['choice_' + c].strip()}\n" for c in "ABCD") +
             "\nFormat your response as follows: \"The correct answer is (insert answer here)\".")
        msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": f"The correct answer is ({r['answer']})."}]
        yield dict(key=sha(repo + "\0" + r["_id"]), subdomain=f"long/retrieval-{r['domain']}", lang="en",
                   long_kind="qa", messages=msgs, thinking_mode="chat", reasoning_effort=None, image_bytes={},
                   source=_src(repo, "data.json", None, k, "human (LongBench-v2 annotators)", _id=r["_id"]))


# ------------------------------------------------------------------ images (LLaVA-OneVision-Data, GPT-4V/4o answers)
def _ext(b):
    if b[:3] == b"\xff\xd8\xff":
        return "jpg"
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "webp"
    if b[:3] == b"GIF":
        return "gif"
    return "bin"


def _llava_turns(conv, url):
    if not any(c["from"] == "human" and "<image>" in c["value"] for c in conv):
        conv = [dict(c) for c in conv]  # cauldron-style rows: the image is implicit before the first question
        first = next(k for k, c in enumerate(conv) if c["from"] == "human")
        conv[first]["value"] = "<image>\n" + conv[first]["value"]
    msgs = []
    for c in conv:
        v = c["value"]
        if c["from"] == "human":
            if "<image>" in v:
                pre, _, post = v.partition("<image>")
                blocks = []
                if pre.strip():
                    blocks.append({"type": "text", "text": pre.strip()})
                blocks.append({"type": "image_url", "image_url": {"url": url}})
                if post.replace("<image>", "").strip():
                    blocks.append({"type": "text", "text": post.replace("<image>", "").strip()})
                msgs.append({"role": "user", "content": blocks})
            else:
                msgs.append({"role": "user", "content": v.strip()})
        elif c["from"] == "gpt":
            msgs.append({"role": "assistant", "content": v.strip()})
        else:
            return None
    return msgs


def llava(subset, subdomain, generator):
    repo = "lmms-lab/LLaVA-OneVision-Data"

    def gen(rng, ctx):
        files = _files(repo, subset, ctx)
        pending = None
        for f, rg, i, r in parquet_rows(repo, files, rng, ["id", "image", "conversations"], ctx=ctx, per_group=4):
            img = r["image"] or {}
            b = img.get("bytes")
            if not b:
                continue
            h = hashlib.sha256(b).hexdigest()
            url = f"images/{h}.{_ext(b)}"
            msgs = _llava_turns(r["conversations"], url)
            if not msgs or msgs[-1]["role"] != "assistant" or not isinstance(msgs[0]["content"], list):
                continue
            item = dict(key=sha(repo + "\0" + subset + "\0" + str(r["id"])), msgs=msgs, url=url, b=b, h=h,
                        src=_src(repo, f, rg, i, generator, id=r["id"]))
            if pending is None and rng.random() < 0.25:  # pair with the next sample: a two-image conversation
                pending = item
                continue
            items = [pending, item] if pending else [item]
            pending = None
            msgs = [m for it in items for m in it["msgs"]]
            yield dict(key="+".join(it["key"] for it in items), subdomain=subdomain, lang="en", messages=msgs,
                       thinking_mode="chat", reasoning_effort=None, image_bytes={it["url"]: it["b"] for it in items},
                       image_sha=[it["h"] for it in items],
                       source=dict(items[0]["src"], extra=[it["src"] for it in items[1:]]) if len(items) > 1 else items[0]["src"])
    return gen
