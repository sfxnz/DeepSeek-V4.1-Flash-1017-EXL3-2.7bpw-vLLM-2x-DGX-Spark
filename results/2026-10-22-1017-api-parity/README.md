# api-parity evidence (2026-10-22): the DeepSeek API's own front end in front of vLLM

Pack `sfxnz/DeepSeek-V4.1-Flash-1017-EXL3` @ `2.7bpw-mcg-hq`, TP=2 on spark1 (head) and spark2 (worker), image `e16-rows`. DeepSeek publishes [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) (MIT, 0.2.0 of 2026-10-20), the library its API front end is built on: it converts Chat Completions, Anthropic Messages and Responses requests into one conversation, renders the V4.1 prompt, encodes it with only the template's own special tokens as control tokens, resolves the API's generation settings, and parses the output back into each protocol's response. The 1017 model card names it as the production form of the prompt format. This campaign compared our serve with it request by request, then put it in front of vLLM. The api-parity stack:

| PR | Change |
|---|---|
| 1/5 | this folder |
| 2/5 | serve flags: no default effort (thinking without one gets the API's 75), `tool_choice: none` renders no tools, xgrammar pinned, 16 stop strings, cached-token usage |
| 3/5 | 16-bit grayscale images scaled like the API (they reached the model almost white) |
| 4/5 | `gateway/`: deepseek-recipe on port 8000 in front of vLLM's token-in route on `127.0.0.1:8001` (`API_GATEWAY=1`; `0` = the serve before this stack) |
| 5/5 | `quality_eval.py --api`: the live API-parity gate, baseline recorded on boot APIG |

`flags.md` and the PR descriptions cite this folder:

- `research/<track>.md`: one report per research track: what deepseek-recipe does (file:line in its source), what our serve did (file:line in the image's vLLM), every divergence with a repro and its impact, what already matched, open questions. Finding ids (R1, T1, S11, V1, D3, P15, ...) are the reports' own. Paths under `/home/sfxnz/projects/data/dsv41-1017/api-parity/` are the campaign's working directory; the files the PRs cite are copied under `harness/` here.
- `research/data/`: the parity summaries the reports quote.
- `harness/`: the scripts and outputs `flags.md` cites (`.log` saved as `.txt`), the gateway review probes (`review/`, `review/pass2/`) and the gateway's in-image check.
- `flags-checks/`: PR 2/5's flags against the image's own `vllm serve` parser on both ranks' argv.
- `APIG/`: the validation boot of the stack tip.

## Method

- **Reference.** The deepseek-recipe 0.2.0 wheel (manylinux aarch64) ran as the oracle: its conversion, rendering, encoding, sampling resolution, stream processor and image preprocessor. The HF 1017 `encoding/encoding.py` @ `1bae0e1` is the model author's prompt-format reference; where the two disagree, the reports say so.
- **Our side.** The `e16-rows` image's own vLLM code, CPU only and without weights: its request models, `OnlineRenderer`, the deepseek_v41 tokenizer, parsers and detokenizer, the chat, Messages and Responses handlers over a scripted engine, the structured-output manager, the multimodal processor.
- **Tracks.** render (165 Chat requests, three ways), tokenizer (27 special-token spellings, 64 requests, BOS/EOS/stop, 110 decodes), parsing (59 cases x 9 delta patterns, 240 fuzzed and 180 stop-string streams), sampling (a 1140-request matrix), vision (48 images, a 31,684-size sweep), protocols (87 requests over the three APIs, 24 output scenarios, HTTP probes), structured (46 requests, grammar probes, DSpark replay), conventions (how this repo lands a change, what the quality gates cover).

## What differed, and what fixes it

| Area | DeepSeek API | Serve before the stack | Fixed by |
|---|---|---|---|
| `thinking: {type}` (Chat, Messages) | decides thinking | ignored: thinking stayed off (render R1, protocols P01/P08) | 4/5 |
| Effort `minimal` / `medium` / `xhigh` / `ultra` | 50 / 75 / 75 / 100 | HTTP 400 (render R2, sampling S3) | 4/5 |
| Thinking on, no effort | 75 | 50, from the default kwarg `reasoning_effort: low` (render R6) | 2/5 |
| `response_format` | the schema is written into the system prompt | grammar only: the model never saw the schema (structured S1) | 4/5 |
| `response_format` with tools | the tool constraint wins | a JSON grammar over the whole answer: no tool call possible (structured S2) | 4/5 |
| Special-token spellings in request text | plain text | control tokens: role/turn injection, EOS mid-prompt (tokenizer T1, 26 of 27 spellings) | 4/5 |
| `tool_choice: none` | no tools in the prompt | tool block rendered (render R7) | 2/5 and 4/5 |
| `prefix: true` continuation | continues the message | closed with EOS (render R4) | 4/5 |
| Stop strings | answer content only | also reasoning and tool markup: answers lost, tool calls cut (parsing D1, sampling S7) | 4/5 |
| `max_tokens` hit inside a tool call | `finish_reason: length` | `tool_calls` with `{}` or invalid JSON (parsing D2) | 4/5 |
| Reasoning field | `reasoning_content` | `reasoning` (parsing D3) | 4/5 |
| Unset `max_tokens` | 8192 / 65536 / 131072 (max effort) | the whole context left, ~500K tokens (sampling S1) | 4/5 |
| Thinking-mode sampling | T=1.0, top_p >= 0.95; chat top_p 1.0 | the client's values (sampling S5/S6; user decision) | 4/5 |
| Responses `input_text` parts, effort `medium`, the `apply_patch` custom tool, namespace tools | accepted; `apply_patch` as a function tool, `ns::f` with the namespace description | 400, 400, dropped, `ns__f` (protocols P15, P02, P17) | 4/5 |
| Messages system blocks, mid-conversation system, tool_result images | the API's layout | joined with "", hoisted, moved out (protocols P09-P11, vision V3) | 4/5 |
| Strict tool arguments; zero-argument calls | normalized schema; the trained `">\n\n</...invoke>"` form | unconstrained; forbidden (structured S3/S4) | 4/5 |
| Tool-result images out of call order | bound in prompt order | bound in request order: the wrong image (vision V2, render R12) | 4/5 |
| Structured backend `auto` | n/a | a first json_schema routed to guidance broke every later required/named `tool_choice` until restart (structured S11) | 2/5 |
| 5 to 16 stop strings | accepted | 400 (sampling S8) | 2/5 |
| 16-bit grayscale PNG | scaled by 1/257 | clamped: almost white, PSNR 4.86 dB against the API (vision V1) | 3/5 |

Requests the API would reject, and requests with vLLM-only fields, go to vLLM unchanged (user decision), so they keep vLLM's own rendering and parsing.

## Already the same (no change)

The pack tokenizer is the HF 1017 tokenizer byte for byte (deepseek-recipe's bundled V4.1 copy differs only in the image token's spelling, same id 129264); plain text encodes to the same ids (28/28); BOS once; EOS id 1 is the only stop id; decoding matches (110/110). The served encoder reproduces all 5 HF golden vectors; the effort budgets are 50/75/100 on both sides. The quality harness's 46 chat prompts already rendered identically. xgrammar applies grammars after `</think>` and stays exact under DSpark verification; DSpark is lossless for temperature, top_p, top_k, penalties and seeds. Images: the same token grid and count for every common camera, screen and phone size.

## Not changed

- Vision: the API resizes with OpenCV and re-encodes WebP q90 where vLLM resizes with PIL (the HF reference's path): the effect on quality is unmeasured (V6). vLLM's image budget and one-pass sizing give a different token count for 3.0% of sizes (V5). `detail: low` is ignored by vLLM (V7). The recipe's SM120 attention keeps image spans causal like the public reference model, where stock vLLM ports bidirectional attention inside images (V16). All need a real vision eval first; the quality gate's image is a 64x64 red square.
- The API's own whitespace trimming, logprob split and usage fields apply through the gateway; requests that fall back keep vLLM's.
- The gateway does not stop vLLM early when the model writes text after a tool block (the API discards that text); cached tokens show only in the final usage.

## User decisions (2026-10-22)

1. Architecture: deepseek-recipe as a gateway in front of vLLM (`API_GATEWAY=0` keeps the previous serve).
2. Sampling follows the API (thinking T=1.0, top_p >= 0.95; chat top_p 1.0; the API's `max_tokens` defaults).
3. Thinking requested without an effort gets the API's 75. Thinking stays off by default.
4. Requests the API would reject fall back to vLLM unchanged: anything that worked keeps working.

## The gateway on CPU

- Prompt token ids, sampling and `max_tokens` equal deepseek-recipe's for 72 requests; SSE and bodies equal the API's own chunk generators for 36 scripted streams (1-6 token deltas, stop-string abort, `length` inside a tool call, logprobs, cache hits); 27 fallbacks reach vLLM byte for byte; the in-image HTTP check against vLLM's real token route passes 106 checks (`harness/gateway/image_check.txt`).
- vLLM's engine starts token-in grammars after `</think>` exactly as on the chat path (`harness/gateway-constraints/token_in_gate.txt`, DSpark windows in `dspark_window_token_in.txt`).
- An independent review in two passes (`harness/review/`, `harness/review/pass2/`) found 11 issues, among them a client that left after the stream grace window and never aborted vLLM, and a global image lock that stalled text requests; all were fixed and re-probed.
- Startup 0.3 s; ~120 MiB idle; a ~74K-token request prepares in ~20 ms (vLLM's own chat path took ~110 ms).

## Boot APIG

| | |
|---|---|
| Code | the stack tip @ `4432b71` (api/05-api-eval worktree, clean). The PRs as opened differ from it only in this folder, evidence paths in `flags.md`, the gateway's `MALLOC_MMAP_THRESHOLD_` (live-tested after the restart below) and the test-env fix that restart traced |
| Image | `e16-rows` sha256:df157a2d8afe, both nodes |
| Command | `SKIP_DOWNLOAD=1 AUDIT=strict ./run.sh` (`boot-APIG.sh`, `command.txt`), no other env |
| Boot | vLLM ready on 127.0.0.1:8001, the gateway on 8000; strict audit ok on head, worker and gateway (`boot.txt`, `audit-*.txt`) |

| Cell | Result |
|---|---|
| Smoke through the gateway | `smoke_chat` 323, `smoke_vision` red; `x-dsv41-gateway: api` on a plain request, `fallback` on `echo: true` (`headers.txt`) |
| `quality_eval --api` (the baseline) and an A/A rerun | PASS: render 52/52 token-identical, toolflow 6/6, reasoning 4/4, structured 3/3, stream 3/3, messages 3/3, responses 3/3, errors exact, 0 invariant violations; the rerun passes every gate against it (`api.json`, `api-aa.json`) |
| `quality_eval --quick`, vLLM directly (8001) | PASS against `quality-baseline/quick.json`: NLL 0.1453 (limit 0.1531), golden hazard 0.0165 (0.0213), tools 1.0/1.0/1.0, needle 6/6, vision red |
| `quality_eval --quick`, through the gateway (8000) | PASS: NLL 0.1448, golden hazard 0.0185, tools 1.0/1.0/1.0, needle 6/6, vision red |
| Decode c=1, gateway vs direct (median of 9) | prose 53.94 vs 54.11 tok/s, TTFT 0.178 vs 0.180 s; structured 108.50 vs 108.55 tok/s; the same DSpark acceptance |
| Decode c=4, distinct prompts, gateway (median of 5) | prose 104.9 tok/s, structured 212.2 tok/s aggregate (perf2 FINAL, direct: 96.4 / 201.1) |
| `quality_eval --full` through the gateway | PASS against `quality-baseline/full.json` (run after the gateway restart below): NLL 0.1435 (limit 0.1545), golden hazard 0.0149 (0.0285), tools 22/22, 22/22, 8/8, needle 9/9 to 128K, GSM8K 98/100 (limit 0.97), MMLU 0.917 (0.912), vision red, c=2. GSM8K with thinking: 38/40, sampled at the API's T=1.0 with a mean of 365 completion tokens, where the baseline ran greedy (39/40, 217 tokens); the Wilson gate passes (`quality-full-gateway.txt`, `full-gateway.json`) |
| Gateway memory | 124 MiB after warmup, 432 MiB after the API eval; with `MALLOC_MMAP_THRESHOLD_=131072` (restart below) 96 MiB, then 223 MiB after the same eval (`gw-stats-*.txt`) |

**Gateway restart.** At 14:51:17Z, during the first `--full` run's 128K needle, a unit-test run in the api/04-gateway worktree ran the real `stop.sh` (`test_stop_sh_orchestrate_zero_is_local_only` sets a made-up `CONTAINER_NAME` but not the gateway's name), and `stop.sh` removed the live gateway (docker events: kill, die 137, destroy; the vLLM ranks stayed up). The interrupted run is `quality-full-gateway-interrupted.txt`. PR 4/5's tests now give every real `stop.sh` run a made-up gateway name (a test pins it), and a suite run beside the live serve left the gateway untouched. The gateway was restarted with run.sh's command plus `MALLOC_MMAP_THRESHOLD_=131072`, the final PR 4/5 configuration (`gateway-restart.sh`); smoke, the API eval (against `api.json`) and `--full` then ran on it (`*-2.txt`, `api-2.json`).

## Reproduce

```bash
./run.sh                    # API_GATEWAY=1: the gateway on 8000, vLLM on 127.0.0.1:8001
python3 smoke_chat.py && python3 smoke_vision.py
python3 tests/quality_eval.py --api --baseline results/2026-10-22-1017-api-parity/APIG/api.json
python3 tests/quality_eval.py --quick --url http://127.0.0.1:8001 --baseline results/2026-10-18-1017/quality-baseline/quick.json
```
