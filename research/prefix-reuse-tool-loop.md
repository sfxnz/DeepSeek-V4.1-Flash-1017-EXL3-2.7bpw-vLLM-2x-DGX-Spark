# How much cached prefix does a thinking-on tool loop reuse per turn at long context?

Research for the wayfinder ticket of that name, under the map "plan the production-regime arm campaign on DGX OS 7.6.0". Static reading only, 2026-10-30. No boot, no GPU, no docker.

**Regime.** Four concurrent sessions with thinking on and tool calling, contexts up to 500K tokens. Requests go through the API gateway (port 8000, deepseek-recipe 0.2.0) to vLLM on 127.0.0.1:8001, with the recipe's prefix-cache defaults.

**Sources and abbreviations.** Repo paths are relative to the recipe at `8087818`.
- **DR**: deepseek-recipe source at `/home/sfxnz/projects/data/dsv41-1017/api-parity/upstream/deepseek-recipe`, `935817c`. That is one docs commit after "Release 0.2.0" `93b0fd2`, and the workspace version is 0.2.0 (`Cargo.toml:15`). The render and reasoning files are unchanged since that release (`git log`). The source's `_native.pyi` is byte-identical to the pinned wheel's `/home/sfxnz/.cache/dsv41-gateway/deepseek_recipe-0.2.0/deepseek_recipe/_native.pyi`. The wheel itself is native code, so this source stands in for it. That substitution is inferred, from the version, the commit and the identical stub.
- **IMG**: the image's extracted vLLM, `/home/sfxnz/projects/data/dsv41-1017/api-parity/image-src/vllm`.
- **HF**: the 1017 reference encoder, `/home/sfxnz/projects/data/dsv41-1017/upstream-2026-10-29/src/hf-1017/encoding/encoding.py`. Read only.
- **Docs**: https://api-docs.deepseek.com/guides/thinking_mode, fetched 2026-10-30.

**Notation.** All lengths are in tokens.
- N = the previous turn's prompt length.
- G = its generated tokens, including EOS.
- E = N + G.
- S = the new turn's appended tokens: tool results or the user message, plus the `<｜Assistant｜><think>` header.
- P0 = ⌊(N−1)/128⌋·128.

---

## Answer

**Gist.** The previous turn's prompt and reply are reused up to 65-192 tokens from their end. The condition is that the client passes `reasoning_content` back verbatim, as the DeepSeek docs require with tools. A client that drops it gets a hit 64-191 tokens before the previous prompt's end, and that turn's reasoning is lost. A full re-prefill at 500K needs early history to change, or the blocks to be evicted.

| Scenario | Client's `reasoning_content` | First divergence from prompt + reply | Hit lands at | Recomputed |
|---|---|---|---|---|
| Tool-result follow-up | Passed back verbatim (the docs' rule) | E, if the text round trip is exact. Otherwise the first lossy spot inside the reply (§1.3) | The largest 128-multiple P with P + 64 ≤ min(D, last cached row); reply blocks are cached | 65-192 old tokens + S. With an inexact round trip, add the reply tail after the divergence |
| Tool-result follow-up | Dropped or never sent | N (the reply's first reasoning token becomes `</think>`) | P0 when N − P0 ≥ 64, else P0 − 128 | 64-191 old tokens + the re-rendered `</think>`, content, DSML and EOS + S. The G reasoning tokens are lost |
| New user message after a loop | Passed back for all turns (the docs' rule) | Same as the first row (E) | Same as the first row | 65-192 + S (`<｜User｜>` + message + header) |
| New user message after a loop | Never sent | N | P0 or P0 − 128 | 64-191 + the re-rendered last answer + S |
| New user message after a loop | Passed back during the loop, stripped once a new user message arrives | The end of the prompt of the first assistant turn whose reasoning is now empty, often early in the session | The latest SWA-retained run at or below it that has survived eviction; else 0 | Everything after that point: up to the whole context, about 500K |
| Plain resend, original completed | n/a | None (the hit is capped at N − 1) | P0 when N − P0 ≥ 65, else P0 − 128 | 65-192 |
| Plain resend, original aborted mid-prefill | n/a | None | The original's own hit (inferred, §3.3) | Everything the original had computed, again |

Every row assumes the first system message, tools, effort and thinking switch are unchanged, and that the session stays on one render path. It also assumes the blocks were not evicted while the client ran its tool (§2.6, §4).

**500K re-prefill in normal use?** Not from the render. In a tool loop with a consistent client, the next prompt always diverges at or after the previous prompt's end. It can happen in two ways:
- **Render side.** A client edits early history, such as tools, effort, the system prompt, compaction, or stripping earlier reasoning (§4).
- **Cache side.** A waiting session's blocks are evicted under pool pressure at 4 × ~500K. This is inferred (§2.6), and only a boot can say whether it happens within real tool gaps.

**Only a boot can settle:**
1. Whether the thinking-on, T=1.0 round trip through the gateway is token-exact, i.e. whether the hit lands within 192 of E or earlier in the reply.
2. Whether a session's blocks survive a tool gap at c=4 near 500K.

---

## 1. What the 1017 render keeps between turns

### 1.1 Prompt layout through the gateway

The gateway converts each request with `ConversionOptions(default_thinking_mode=False)`, renders it with `DeepseekV41Encoding`, and encodes it with the pack tokenizer (`gateway/dsv41_gateway.py:323-327,394-396`). It sends the token IDs to vLLM's token-in route (`:367`). vLLM never re-renders an API-path request.

The render, in DR `deepseek-recipe-encoding/src/v4/dsv41.rs`:
- **Prefix.** BOS, then at index 0 with thinking on `<｜System｜>Reasoning Effort: {50|75|100} …\n\n` (`:21-30,144-156`). The tool prompt, and any response format, go at the end of the first system message (`:258-284`).
- **Assistant turn.** `<｜Assistant｜>` + `<think>` + reasoning + `</think>` + content + DSML block + `<｜end▁of▁sentence｜>` (`:206-231`). The DSML block is `\n\n<｜DSML｜ calls>\n…</｜DSML｜ calls>` (`:91-121`), and non-string argument values are written Python-style (`:102-105`).
- **Tool result after an assistant turn.** `<｜User｜><tool_result>…</tool_result>`, with `\n\n` between consecutive results or a following user message (`:157-166,190-193`). Results are sorted by call order (`:321-419`).
- **Generation header.** `<｜Assistant｜><think>` in thinking mode (`:298-306`).

So a turn's prompt ends in `<think>`, and the model's reply starts with its reasoning. If the reply is rendered back exactly, the next prompt is the previous prompt + reply + EOS + S.

### 1.2 Reasoning: kept with tools, for every earlier turn

- **deepseek-recipe.** `drop_thinking = !renders_tool_prompt` (`dsv41.rs:285-286`), and reasoning renders when `thinking_mode && index > 0 && !drop_thinking` (`:206`). With tools declared, every earlier assistant turn keeps its reasoning, across tool results and across new user messages.
- **HF reference.** The same rule: any message with tools sets `effective_drop_thinking = False` (`encoding.py:785-788`), so `thinking_part` is kept for every assistant turn (`:691-695`). Only without tools are reasoning and the turns before the last user message dropped (`:749-767`).
- **vLLM path.** `docker/patch/chat_encoder_1017.py` brings the image's encoder to that 1017 logic (`:1-8`), and the render report confirms the same keep and drop behaviour (`/home/sfxnz/projects/data/dsv41-1017/api-parity/findings/render/REPORT.md:197`). This path serves only fallback requests and direct 8001 calls.
- **Missing reasoning.** In thinking mode a missing `reasoning_content` is filled with `""` (DR `deepseek-recipe/src/protocol/reasoning_content.rs:51-78`). `FillEmpty` is the default (`deepseek-recipe/src/request/options.rs:31-34`), and the gateway does not override it (`dsv41_gateway.py:323-325`). An empty reasoning renders `<think></think>` (`dsv41.rs:208-226`).
- **What clients should send.**
  - The docs (thinking_mode, "Tool Calls"): `reasoning_content` "must be fully passed back to the API in all subsequent requests", "even for turns where the model did not perform a tool call". "If your code does not correctly pass back `reasoning_content`, the API will return a 400 error."
  - The docs ("Input and Output Parameters"): with tools, "the `reasoning_content` of all previous turns should be passed back … and will be concatenated into the context". Without tools it "does not need to be passed back".
  - The library's opt-in `Reject` enforces only the current loop: messages after the last user message (`reasoning_content.rs:60-77`).
  - Our gateway accepts all three client behaviours without error.

### 1.3 What the round trip can lose (the reply as text, not tokens)

The client sees text and sends text back, and the gateway re-encodes it. The parser (DR `deepseek-recipe/src/stream/output_parser.rs`) is not lossless:
- **Leading newlines of reasoning** are dropped (`ReasoningStart`, `:12-15,110-114,573-580`). A model that writes `\n` after `<think>` makes the next prompt diverge at the reply's first token.
- **Newlines before `</think>`** are dropped (`NewlinesAndMarker`, `:582-588`), and so are **leading newlines of content** (`:584-586,441-449`). Each of these moves the divergence to the end of the reasoning.
- **Newlines before `<｜DSML｜`** match any count (`:326-334`). Bytes between DSML tags are skipped (`:480-493`). The render always writes `\n\n<｜DSML｜ calls>\n` and `\n` between invokes. A model that spaced the block differently diverges inside the DSML block.
- **Non-string argument values** keep the model's raw bytes in the returned arguments (`:163-213`). The render then re-serializes them Python-style (`dsv41.rs:102-105`). Compact JSON such as `[1,2]` becomes `[1, 2]`.
- **Tokenization.** The encoder encodes each plain-text span between special tokens as one region (render report §1, "Token encoding"). A T=1.0 sample that took a non-canonical token split re-encodes differently from that token on.
- **Reference.** The HF parser cuts at `</think>`, the DSML start and EOS, and trims nothing (`encoding.py:946-990`). Its intended round trip is exact. The trims are deepseek-recipe's.

Every one of these spots lies inside the reply, where all rows ran every layer (§2.4) and every block is cached (§2.3). They move the hit back from E by at most the reply tail after the spot. Which of them the 1017 model hits at T=1.0 cannot be read from code.

### 1.4 What changes the start of the prompt (a full re-prefill)

All of these change BOS, the effort line, the first system message or its tool block, so the divergence lands within the first system message:
- the effort or the thinking switch (`dsv41.rs:144-156`);
- the tool list, a description, or the response format (`:258-284`);
- `tool_choice: "none"` or a request without tools. This removes the tool block and also drops every earlier turn's reasoning (`:285-286`; render report R7);
- system-prompt edits.

Two more cases change history further in:
- **Client-side compaction** or edits to old tool results.
- **Switching render paths.** A request with a vLLM-only field falls back to vLLM's own render (`dsv41_gateway.py:96-137,375-384`). That render differs from the API's, for example in the tool JSON (render report R13: `"description": null` vs `""`). A session that alternates paths re-prefills from its tool block.

The one varying system line the API itself strips is the `x-anthropic-billing-header` (DR `deepseek-recipe/src/protocol/openai/chat_completion/request/convert.rs:373-394`).

---

## 2. What the cache does with a divergence point D

### 2.1 Hit unit

- Hits land on 128-token boundaries (scheduler block; `flags.md:39`).
- SWA groups use 64-token blocks, and 43 of the 50 KV groups are SWA (FINAL `results/2026-10-19-1017-perf2/FINAL/boot-grep-boot.txt:92`; `AGENTS.md` KV layout).
- An SWA group serves a hit at P only from its cached window blocks [P−128, P) plus the DSpark peek block [P, P+64) (`docker/patch/prefix_end_fallback.py:3-5`). In vLLM terms, `need` = ⌈127/64⌉ + 1 contiguous blocks, and the eagle block is dropped (IMG `v1/core/single_type_kv_cache_manager.py:912-923,1049-1058`).
- So a hit at P needs **P + 64 ≤ D** and P ≤ N_new − 1.
- Compressor rings are `prefix_cacheable=False` and take no part in the lookup (IMG `v1/core/kv_cache_coordinator.py:685-691`). The combined hit is the fixed point of the full-attention prefix and the SWA search (`:778-911`).

### 2.2 Which SWA blocks get a hash (only those can serve a hit)

- **e15 sparse retention.** `--prefix-cache-retention-interval 4096` keeps a tail at every 4096 boundary, plus tails at the replay boundary N−1 and at a shared-prefix junction (IMG `single_type_kv_cache_manager.py:469-494,1062-1091`; `flags.md:39`).
- **`DSV41_PREFIX_END_FALLBACK=1`** also keeps b−128 for each boundary b (`prefix_end_fallback.py:14-18,53-57`). A resend, or a follow-up that diverges at N, then hits P0 or P0−128, recomputing at most 192 tokens (`flags.md:219`).
- **`DSV41_PREFIX_REPLY_RETAIN=1`** caches every SWA block at or past the prompt end as it fills, the block straddling N included (`docker/patch/prefix_reply_retain.py:12-18,46-52`). Without it the reply's SWA blocks are never cached, and a follow-up that repeats the reply re-prefills the whole reply (`:3-10`).

### 2.3 Decoder replay (`DSV41_DECODER_REPLAY_ROWS=2670`)

- **When it trims.** Only when the turn's new rows exceed W: `prompt_len − hit > 2670`. Otherwise the frontier is 0 and nothing is masked (`docker/patch/decoder_replay.py:186-194`).
- **What a trimmed prefill loses.** Layers 21-39 skip rows below N − 2670. SWA blocks below the frontier N − 384 are masked out of the cache, and that includes the 4096 checkpoints there (`:72-94`). The last 384 prompt rows stay exact in every layer, which covers P0−128's window (`:39-55`; `flags.md:218`).
- **Decode rows.** These always run every layer (`flags.md:218`), so every reply block is exact and cacheable (`prefix_reply_retain.py:15-17`).
- **Consequence for a tool loop.** Every hit near the previous prompt end or inside the reply is valid. A divergence more than about 128 tokens before a trimmed prompt's end, inside that prompt's own new rows, cannot hit there. It recomputes from the latest earlier retained run (`decoder_replay.py:85-94`).

### 2.4 Where the search lands

- **The search.** The SWA lookup scans right to left from the full-attention hit and stops at the first aligned run of `need` cached blocks (IMG `single_type_kv_cache_manager.py:964-1021`).
- **Candidate runs in a session's history.** Each earlier turn's prompt-end tails (P0, P0−128); every reply block; the 4096 checkpoints in untrimmed rows; and junction anchors.
- **Junctions.** When the full-attention groups match further than the SWA groups, e15 records the longest single-group hit as a junction J (IMG `v1/core/kv_cache_manager.py:285-291`). A trimmed prefill then keeps [J−2542, J) exact, so the *next* request that diverges there hits J (`decoder_replay.py:57-70`). The current one does not.

### 2.5 Measured so far

Boot I4 ran `tools/replay_ab.py prefix` and passed 28/28 (`results/2026-10-19-1017-perf2/I4/prefix-B.txt:64-66`). It covered resends and follow-ups at every N mod 128 in {0, 1, 63, 64, 65, 127}. Its reply case was a follow-up that repeats the reply verbatim: 20,258 prompt tokens with a hit of 20,096.

That case ran thinking off, greedy, content only, on vLLM's chat endpoint (`tools/replay_ab.py:55,229-238`). No boot has measured a thinking-on, T=1.0, tool-calling round trip through the gateway.

### 2.6 Eviction (inferred from code; not measured)

- **Between turns the blocks are evictable.** While a session waits for its tool, it has no request, so all its blocks have refcount 0.
- **Free order.** A finished request is freed group by group (IMG `kv_cache_coordinator.py:328-336`), each group tail-first (`single_type_kv_cache_manager.py:546-547`). Blocks with a hash are appended to the LRU tail, and blocks without one are prepended and reused first (IMG `v1/core/block_pool.py:725-749`). Allocation pops from the head and evicts any hash it finds there (`:649-668`).
- **What rotates the LRU.** Under reply retention, every SWA block a decoding session releases has a hash. So each decoded token of every other session sends about 43/64 ≈ 0.67 hashed blocks to the LRU tail and takes as many from the head. Without reply retention those blocks have no hash, and they are reused head-first (LIFO) without evicting anything.
- **Headroom.**
  - The pool is 4,160,749,568 B / 46,080 B ≈ 90,300 blocks (`flags.md:31,149`).
  - Four sessions growing to 500K need 3.82 of 3.875 GiB (`flags.md:28`), about 1.4% headroom, or ~1,300 blocks.
  - FINAL's 4 × ~447K soak peaked at KV usage 0.876 (`results/2026-10-19-1017-perf2/FINAL/soak.txt:122`), about 11,000 free blocks.
- **Survival horizon.** A waiting session's tail survives about (free blocks queued ahead of it) / 0.67 tokens decoded by the other sessions. That is order 2K tokens at 4 × 500K and order 16K at 4 × 447K, before prefill steps, which also allocate. These are rough estimates.
- **Effect.** Once the tail goes, the hit falls back to an older surviving SWA run. Those runs were freed earlier and so go first. Otherwise the hit is 0: a full re-prefill.

---

## 3. Per scenario

### 3.1 Tool-result follow-up inside a loop

- **Render.** Unchanged history, plus the reply, EOS, `<｜User｜><tool_result>…</tool_result>` and the header (§1.1). Every earlier turn's reasoning renders the same way again (§1.2).
- **Client passes `reasoning_content` back (the docs' rule).**
  - D = E when the round trip is exact.
  - The cached rows end at E − 1, because the last sampled token has no KV (`tools/replay_ab.py:236`).
  - With every reply block cached (§2.2), the hit is the largest 128-multiple P with P + 64 ≤ 64·⌊(E−1)/64⌋. That gives **E − P ∈ [65, 192]**; this range is derived from the rules above.
  - Recompute = (65..192) + S. Example: N = 300,000, G = 2,000 and a 3,000-token tool result recompute about 3,070-3,200 and reuse 301,808-301,935.
  - With a lossy spot at D inside the reply (§1.3), recompute = (E − D) + (64..191) + S. The worst lossy spot is a leading newline in the reasoning, which costs the whole reply.
- **Client drops `reasoning_content`.**
  - D = N exactly: the earlier messages render identically, and `<think>` is followed by `</think>` instead of the first reasoning token.
  - The hit is P0 or P0 − 128, i.e. **N − hit ∈ [64, 191]** (`prefix_end_fallback.py:9-17`; `flags.md:219`).
  - Recompute = that + the re-rendered `</think>` + content + DSML + EOS + S.
  - The reasoning is never reused. The model loses it, and the official API would answer 400 (§1.2).
- **Replay.** If S plus the tail exceeds 2,670, the prefill trims. This affects only later divergences inside S (§2.3).

### 3.2 New user message after a loop

- **Reasoning passed back for all turns.** Same as §3.1: the user message follows `<eos>` as `<｜User｜>{msg}` (`dsv41.rs:157-166`), so the recompute is 65-192 + S.
- **Never sent.** Same as §3.1's drop case.
- **Passed back during the loop, then stripped at the new user message.** This matches the docs' no-tools advice applied with tools, or the HF no-tools rule.
  - Every earlier assistant turn changes from `<think>{r}</think>` to `<think></think>`.
  - D = the prompt end of the earliest such turn, often the session's first prompt.
  - The full-attention groups match to D, and the SWA search looks for that turn's prompt-end tails or anything older (§2.4). Those blocks were freed long ago and are first in line for eviction (§2.6).
  - So the recompute is everything after D: up to the whole context. It may start from 0.
  - J = D becomes a junction for later requests only (§2.4).

### 3.3 Plain resend

- **Original completed.** D = N, capped at N − 1. The hit is P0 when N − P0 ≥ 65, else P0 − 128: 65-192 recomputed for every N (`prefix_end_fallback.py:9-17`; I4's resend cases).
- **Original aborted mid-prefill** (inferred, e.g. a client timeout on a long prompt).
  - If that prefill trimmed, its computed rows below N − 384 never got SWA hashes (§2.3), and the tail at N − 1 was never reached.
  - The resend hits at the original's own hit, if those SWA blocks survived, and recomputes everything the original had done.
  - The full-attention match past that becomes a junction, so a further resend can hit there (§2.4).

---

## 4. Worst case at 500K

- **Normal tool loop.** No full re-prefill comes from the render. With a client that passes reasoning back verbatim, or never sends it, keeping tools, effort and system prompt fixed, each turn recomputes at most 192 old tokens plus its own new text (plus a reply tail if the round trip is lossy).
- **Re-prefill from the start (or from an old retained run).**
  - Any change in §1.4.
  - Stripping earlier reasoning at a new user message (§3.2).
  - Eviction of a waiting session's blocks under pool pressure (§2.6, inferred). This one needs no client mistake at all: it depends on occupancy, tool-gap length and the other sessions' decode rate.
- **Cost.** At I4's 64K prefill rate of 2,867 tok/s (`flags.md:218`), 500K is more than 170 s alone. The prefill cap gives a long prefill next to decoders about 0.76-0.80 of solo speed (`flags.md:221`). Both are lower-bound estimates, since attention cost grows with length.
- **First turns.** The first prompt of each session is a cold prefill. Sessions sharing a system and tool block share it from the third session on (the second anchors the junction; `decoder_replay.py:57-65`).

## 5. What only a boot can settle

1. **Round-trip exactness, thinking on, T=1.0, through the gateway (§1.3).** This depends on how often the 1017 model writes newlines around reasoning or content, spaces DSML differently, emits compact JSON values, or samples non-canonical splits. It decides whether a tool-loop hit lands 65-192 tokens before E or a reply's length earlier.
   - **Probe.** Run one thinking-on tool session through port 8000 at T=1.0 for 10+ turns, passing reasoning back verbatim. Per turn, compare `usage.prompt_cache_hit_tokens` (the gateway writes vLLM's `cached_tokens` into it, `gateway/dsv41_gateway.py:484-485,493-510`) with the previous turn's `prompt_tokens + completion_tokens`. A shortfall above 192 is a lossy round trip.
   - It fits the production bench cell (T1-CELL in the upstream audit, `/home/sfxnz/projects/data/dsv41-1017/upstream-2026-10-29/REPORT.md:153`).
2. **Eviction at c=4 near 500K (§2.6).** Whether a waiting session's blocks survive realistic tool gaps while three others decode and prefill. This also measures whether reply retention's hashed SWA churn shortens that survival.
   - **Probe.** Run four sessions near 450-500K. One pauses for a timed tool gap; record its next turn's hit against the expected E − 192.
3. Static reading fully determines the drop-reasoning follow-up, the never-sent case and the completed resend. I4 already measured their hit landings for a thinking-off chat (§2.5).
