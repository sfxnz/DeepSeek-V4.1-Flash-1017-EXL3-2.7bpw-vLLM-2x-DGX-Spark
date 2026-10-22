# Sampling track: how generation settings are resolved, DeepSeek API (deepseek-recipe 0.2.0) vs our serve

## Short summary

**Scope.** I checked how a request becomes thinking mode, effort, temperature, top_p, max_tokens and 400 errors on two sides:
- the API, as encoded by deepseek-recipe 0.2.0;
- our e16-rows vLLM with the run.sh flags, including DSpark's handling of sampling parameters.

**Method.** One matrix of 1140 Chat Completions requests: 19 thinking controls × 4 sampling settings × 5 max_tokens settings × tools off/auto/required. On top of that, 28 single-parameter probes, 60 synthetic context-limit cases, and 14 Messages/Responses cases.
- API side: run through deepseek_recipe in the host venv.
- Our side: run through vLLM's own request, renderer and serving functions in the image, CPU-only.
- DSpark: the image's MRV2 rejection kernels, run under the Triton interpreter.

**Headline.**
- Defaults that already match:
  - With no sampling fields, both sides sample at T=1.0 and top_p=1.0. The pack ships no generation_config.json, so `default_sampling_params={}`.
  - Effort budgets are 50/75/100 on both sides.
- DSpark verification is lossless for temperature, top_p, top_k, penalties and seeds, so there is no silent fallback. min_p and logit_bias are refused with a 400.
- What diverges:
  - (1) No max_tokens default: we give the whole context left, up to about 500K tokens.
  - (2) DeepSeek's own thinking controls are not understood.
  - (3) The API's thinking-mode sampling policy is not applied.
  - (4) Stop strings fire inside the reasoning.
  - (5) Limits and validation differ.
- Only 204 of 1140 matrix requests currently match field for field.

| id | behaviour | API | ours | impact | fix |
|---|---|---|---|---|---|
| S1 | max_tokens unset | 8192 chat / 65536 thinking / 131072 thinking at effort max, capped at context left | context left (499,960 for a 40-token prompt) | **High**: a degenerate generation can run for hours (500K tok at ~50 tok/s ≈ 2.8 h) and hold 1 of 4 slots; truncation points differ | wrapper patch, part B |
| S2 | top-level `thinking:{type}` | decides the mode and overrides effort; enabled with no effort → 75 | ignored (left in model_extra). `{enabled}` → chat mode; `{disabled}`+`effort:high` → thinking | **High**: silently wrong mode | wrapper, part A |
| S3 | effort aliases | minimal→50, medium→75, xhigh→75, ultra→100 | 400 for medium/minimal/xhigh (encoder) and ultra (request schema): 3 of the 7 values in the OpenAI SDK's own list | **High** for clients (SDKs send "medium") | wrapper A, plus a one-line rewrite to allow "ultra" |
| S4 | thinking on via chat_template_kwargs, no effort | (default) 75 | 50, because run.sh's default effort is "low" | Med: shallower reasoning | run.sh `--default-chat-template-kwargs '{"thinking":false}'` (your decision) |
| S5 | thinking-mode temperature / top_p | T forced to 1.0; top_p = max(requested, 0.95) | client values; T=0 runs greedy | Policy. Effect on quality not measured | wrapper part C behind a switch (your decision) |
| S6 | chat-mode top_p (V4.1) | forced to 1.0 | client value | Policy, Low | wrapper part C |
| S7 | stop strings in thinking mode | matched in content only | matched on the whole output: stops inside the reasoning, content empty | Med-High for stop+thinking clients | gate the stop check on `</think>` (overlaps the parsing track) |
| S8 | 5–16 stop strings | accepted (max 16) | 400 (VLLM_MAX_STOP_STRINGS=4) | Med for clients | run.sh `VLLM_MAX_STOP_STRINGS=16` |
| S9 | max_completion_tokens | ignored (unknown field) | honoured | Low | keep ours |
| S10 | max_tokens > 393216 | 400 | accepted up to the context | Low | optional, in the wrapper |
| S11 | context-limit 400 text and error JSON | "…you requested T tokens (P in the messages, M in the completion)…" | "…you requested M output tokens and your prompt contains P input tokens, for a total of T…" | Low (clients that parse the numbers) | 2-string rewrite in renderers/params.py |
| S12 | tool_choice required/named + thinking | 400 | accepted | Low-Med | for the structured track |
| S13 | logit_bias (any); min_p (vLLM extension) | ignored | 400 under spec decode | Low | leave |
| S14 | n≠1; top_logprobs=0 without logprobs; seed<0 | 400 | accepted | Low | leave |
| S15 | presence/frequency penalty | validated, then dropped | applied (DSpark-correct) | unknown | open question |
| S16 | parallel_tool_calls=false | ignored | keeps the first call only | Low | leave |
| S17 | Messages / Responses | Messages honours `thinking`, max_tokens optional (over the cap: clamped to 393216); Responses gets the same max_tokens defaults | Messages ignores `thinking`; max_tokens required; over the context: 400. Responses max = context left | Med | protocols track; extend the wrapper to Responses |
| S18 | default thinking when absent | ON (library default) | OFF (recipe decision) | — | not proposed |

**Recommendations**

1. **Add `docker/patch/api_sampling.py`.** It installs wrappers at import time, the same pattern as `sched_prefill_cap.py`. The prototype is `harness/sampling/proto_api_sampling.py`.
   - Part A, always on: honour top-level `thinking`, and map the effort aliases.
   - Part B, `DSV41_API_MAX_TOKENS=1`: the API's max_tokens defaults.
   - Part C, `DSV41_API_SAMPLING`: thinking mode gets T=1.0 and top_p≥0.95; chat mode gets top_p=1.0. **This is policy; you choose the default.**
   - If C is on, the quality gate's gsm8k_think component (thinking, effort high, T=0) becomes sampled. It needs a per-request opt-out (for example `vllm_xargs:{"dsv41_api_sampling":0}`, which API clients never send) or a re-baseline. smoke, bench, LAIL and the other quality components are unaffected: they run thinking off with explicit max_tokens.
   - Re-running the matrix with the prototype plus C plus the S4 default leaves only these differences:
     - S18, recipe decisions and vLLM-only kwargs: default thinking off; `chat_template_kwargs` honoured (the API ignores it) and numeric effort through kwargs.
     - S9: max_completion_tokens, kept on purpose.
     - S10: the 393,216 bound.
     - S12: tool_choice+thinking.
     - "ultra".
2. **run.sh.**
   - Set `VLLM_MAX_STOP_STRINGS=16` in the head's env.
   - Consider dropping `reasoning_effort` from the default kwargs. Chat prompts stay byte-identical, because effort is only rendered in thinking mode.
3. **Stop sequences** should match only after `</think>` (token 128822, not a special token) when the prompt ended in `<think>`. Coordinate with the parsing track.
4. **Error text.** Rewrite the context-limit message to the API's wording.
5. **Do not use `--override-generation-config`.**
   - `max_new_tokens` turns into a hard cap on every request: explicit 20000 → 8192, thinking 60000 → 8192.
   - It cannot vary by mode.
   - T and top_p already default to 1.0, so setting them there changes nothing.

**Where things are**

I could not write REPORT.md here, so the full report is the rest of this message. Everything else is under `/home/sfxnz/projects/data/dsv41-1017/api-parity/harness/sampling/`:
- `run_all.sh` regenerates everything in about 2.3 min; no GPU, and the image runs as your user.
- Results: `api_results.json` and `ours_results.json`.
- Tables quoted below: `compare_out.txt`. Every matrix row on both sides: `matrix_full.tsv`.
- Prototype and its re-runs: `proto_api_sampling.py`, `proto_check.py`, `proto_results*.json`.
- DSpark check: `dspark_lossless.py` and `dspark_lossless_n8000.log`.
- Stop-string demos: `api_stop_in_reasoning.py` and `ours_stop_in_reasoning.py`.
- Messages/Responses: `protocols_*.py` and `protocols_*.json`.

---

# Full report

## 1. What upstream does

All paths are under `upstream/deepseek-recipe/`.

**Thinking mode**
- The rule is in `deepseek-recipe/src/protocol/reasoning.rs:58-78`, `resolve_thinking_mode`:
  - An explicit `thinking.type` wins. `adaptive` is an alias of `enabled` (`:6-14`).
  - Otherwise effort `none` turns thinking off, and any other effort turns it on.
  - Otherwise the `ConversionOptions.default_thinking_mode` decides; it defaults to **true** (`request/options.rs:150-170`).
- With thinking on and no effort, the effort is High.
- Effort mapping (`reasoning.rs:24-49`): none / minimal→Low / low→Low / medium→High / high→High / xhigh→Xhigh / ultra→Max / max→Max. A numeric effort fails deserialization: "invalid type: integer `80`, expected string or map".
- V4.1 renders Low=50, None/High/Xhigh=75, Max=100 (`deepseek-recipe-encoding/src/v4/dsv41.rs:21-28`). The HF `encoding/encoding.py:440-474` accepts low/high/max or an integer 1–100, and defaults to high.
- Per protocol:
  - Chat: `thinking` > `reasoning_effort` > default (`chat_completion/request/convert.rs:49-53`).
  - Messages: `thinking` > `output_config.effort` > default (`anthropic/messages/request/convert.rs:55-71`); `thinking.budget_tokens` is ignored.
  - Responses: `reasoning.effort` > default only (`openai/responses/request/convert.rs:741-748`).

**Temperature and top_p**
- Validation order (`protocol/validation.rs:44-111`): n, presence_penalty, frequency_penalty, temperature [0,2], top_p (0,1], top_logprobs with logprobs [0,20], max_tokens [1, bound], seed [0, 2^63).
- Temperature is 1.0 in thinking mode, otherwise the requested value or 1.0 (`:112-118`).
- `ModelVersion::resolve_top_p` (`request/options.rs:85-92`):
  - V4.1 thinking: max(requested or 1.0, 0.95).
  - V4.1 chat: 1.0.
  - V4: 0.95 with tools / 1.0 in thinking mode; the requested value otherwise.
  - For V4.1, tools do not change anything.
- Chat passes `parse_tool_calls` as has_tools (`convert.rs:150-152`).
- `supports_image` is V4.1 only (`options.rs:70-72`).

**max_tokens**
- Upper bound: `DEFAULT_MAX_TOKENS_UPPER_BOUND = 393216` (`options.rs:96`), selected per mode at `:234-240`.
  - Chat and Responses reject a larger value: "Invalid max_tokens value, the valid range of max_tokens is [1, 393216]".
  - Messages clamps it (`clamp_max_tokens: true`, `anthropic convert.rs:99-107`).
- `ConversationRequest::resolve_max_tokens` (`request/mod.rs:207-235`):
  - If the requested value is unset, the default is 8192 (chat), 65536 (thinking) or 131072 (thinking with Max effort) (`:155-166`).
  - The result is lowered to context minus prompt, and raised to at least 1.
  - If prompt + requested > context: 400 "This model's maximum context length is {ctx} tokens. However, you requested {total} tokens ({prompt} in the messages, {requested} in the completion). Please reduce the length of the messages or completion." When max_tokens is unset, `{requested}` is 0.
  - The caller supplies the context limit. The model card says 1M; I used our 500000.

**Accepted, dropped or rejected**
- The Chat schema is at `chat_completion/request/schema.rs:17-60`. Unknown fields are ignored, so these are silently dropped: `max_completion_tokens`, `logit_bias`, `user`, `chat_template_kwargs`, `top_k`, `min_p`, `repetition_penalty`.
- `presence_penalty` and `frequency_penalty` are validated and then dropped (`mod.rs:1-13`: "Extract before conversion if the backend supports it").
- `n` other than 1: 400.
- `parallel_tool_calls` must be a bool and is then ignored.
- `seed` and `logprobs`/`top_logprobs` are passed through.
- `stop`: at most 16 entries, none empty (`validation.rs:155-185`).
- `stream_options` without `stream`: 400 (`convert.rs:569-587`).
- Required or named tool_choice in thinking mode: 400 "Thinking mode does not support this tool_choice" (`client_tools.rs:94-98`).
- Stop sequences are parse-side only. They are matched only in the Content and JSON stages, never in Reasoning or ToolCalls (`stream/output_parser.rs:411-466` and `:567-589`); empty ones are dropped (`:104-108`).
- Error JSON: `{"error":{"message","type":"invalid_request_error","param":null,"code":"invalid_request_error"}}` (`error_response.rs:33-65`).

The README's "Generation settings" paragraph says the same things (`README.md:28-33`). The HF model card recommends T=1.0, top_p=1.0, a 1M context and max_tokens ≥ 256K (`README.md:77-84`); that is higher than the API's own defaults.

## 2. What our serve does

Paths are under `image-src/vllm` unless a recipe path is given.

Note that image-src already contains the chat_encoder_1017 rewrite. The image's baked sitecustomize reports "already applied", and all my image runs used the real image.

**Thinking resolution**
- run.sh:664 sets `--default-chat-template-kwargs '{"thinking":false,"reasoning_effort":"low"}'`.
- `ChatCompletionRequest.build_chat_params` (`entrypoints/openai/chat_completion/protocol.py:570-603`) adds `reasoning_effort` and sets `enable_thinking = effort != "none"` (`:582-588`).
- Server defaults then fill in missing keys only (`renderers/params.py:40-52`, `:111-140`; `renderers/online_renderer.py:395-413`).
- The V4.1 tokenizer decides (`tokenizers/deepseek_v41.py:72-88`):
  - `thinking = bool(thinking or enable_thinking)`, which is OR semantics, so an effort turns thinking on even when `chat_template_kwargs.thinking` is false;
  - True if neither key is present; effort "none" turns it off;
  - no effort → "high";
  - only low/high/max or an integer 1–100 are allowed, otherwise ValueError, which the server returns as 400 BadRequestError (`serve/exception_handling/register.py:27-39`).
- Budgets are 50/75/100 (`tokenizers/deepseek_v41_encoding.py:210-241`).
- The request schema's `reasoning_effort` Literal is none/minimal/low/medium/high/xhigh/max (`protocol.py:244-256`), so "ultra" fails pydantic validation.
- Top-level `thinking` is not a field. OpenAIBaseModel uses extra="allow" and only logs it at debug level (`serve/engine/protocol.py:22-51`).
- Messages (`entrypoints/anthropic/protocol.py:121-140`) has no `thinking` field and requires max_tokens; `output_config.effort` → `reasoning_effort` (`anthropic/serving.py:495-513`).
- Responses: `reasoning.effort` → effort and enable_thinking (`responses/protocol.py:314-345`).

**Sampling**
- `default_sampling_params = model_config.get_diff_sampling_param()` (`chat_completion/serving.py:172`), with `generation_config="auto"` (`config/model.py:330`).
- With no generation_config.json, transformers builds the config from config.json and it has no sampling keys, so the result is `{}`. My harness ran vLLM's own code on the pack and got exactly that, and the boot log has no "overridden" warning.
- `to_sampling_params` (`protocol.py:661-750`): the request value, else the default, else `_DEFAULT_SAMPLING_PARAMS` (T 1.0, top_p 1.0, top_k 0, min_p 0, rep 1.0; `:628-634`). Penalties, n, seed, stop, logit_bias, top_k, min_p and repetition_penalty all pass through.
- `SamplingParams` checks the same ranges as the API, with different wording (`sampling_params.py:546-615`). Temperature 0 makes it greedy and resets top_p to 1; seed −1 means random (`:494-495`).

**max_tokens**
- `get_max_tokens(max_model_len, max_completion_tokens or max_tokens, prompt_len, {}, override=None)` (`serving.py:310-318`, `serve/utils/api_utils.py:169-203`) returns min(context left, requested). Unset therefore means context left.
- Errors:
  - requested > max_model_len: "max_tokens=… cannot be greater than max_model_len…" (`renderers/params.py:237-250`);
  - prompt + requested > context: vLLM's context-limit message (`:488-513`);
  - prompt = context with nothing requested: "max_tokens must be at least 1, got 0".
- `--override-generation-config {"max_new_tokens":N}` sets both the default and `override_max_tokens` (`serving.py:173-177`), which `get_max_tokens` applies as a minimum over every request.
- MAX_MODEL_LEN is 500000 (run.sh:18).
- Stop lists are limited to `VLLM_MAX_STOP_STRINGS=4` (`entrypoints/generate/base/protocol.py:28-30`, `envs.py:113`).
- Stop strings are checked on the whole detokenized output (`v1/engine/detokenizer.py:96-140`, `:310-345`).
- `parallel_tool_calls=false` keeps the first call (`serve/utils/tool_calls_utils.py:19-37`).

**DSpark (MRV2, "Using V2 Model Runner")**
- Evidence that MRV2 is in use: boot log `.run-state/audit-head-20261020-115123.log:306`. SPEC_CONFIG is greedy dspark (run.sh:316-323).
- The rejection sampler applies the full `Sampler.apply_sampling_params` to the target logits of every verify row: logit bias, penalties, bad words, thinking budget, temperature, min_p, top-k/top-p (`v1/worker/gpu/spec_decode/rejection_sampler.py:144-181`, `v1/worker/gpu/sample/sampler.py:209-273`).
- With greedy drafts, q is one-hot (`rejection_sampler_utils.py:183-187`): accept with probability p(x), otherwise resample from p with x removed.
- A −1 draft (an adaptive-verify masked row) samples from the full target (`:760-770`).
- Penalties count the earlier drafts (`sample/penalties.py:148-158`).
- Seeds are honoured through rand(seed, pos).
- `_validate_spec_decode` refuses min_p > 0 and any logit_bias with a 400 (`sampling_params.py:1012-1025`).
- The biased `DSV41_DSPARK_SOFTMAX_VERIFY` lever is off (run.sh:188, rejected in flags.md); adaptive verify is on (`recipe.yaml:83`).

**Recipe tooling**
- `smoke_chat.py:59-65`, `smoke_vision.py:66-68`, `bench_decode.py:129-140`, `benches/{e2e,micro}.py`, `tools/c4_mixed.py`: thinking off, effort low, T=0, explicit max_tokens.
- `tools/measure_lail_prose.py:84-96`: T=0.2, thinking off.
- `tests/quality_eval.py:69-70` and `:404-408`: everything T=0 with explicit max_tokens. gsm8k_think uses `{thinking:true, effort:high}`, T=0, max_tokens 4096 (`:597`).

No recipe patch changes request or sampling resolution apart from chat_encoder_1017's effort map.

## 3. Divergences with evidence

Run `harness/sampling/run_all.sh`; the excerpts below come from `compare_out.txt`.

**S1. max_tokens default** (table M)
- `eff_none`, unset: API 8192 vs ours 499986.
- `eff_high`, unset: 65536 vs 499960.
- `eff_max`, unset: 131072 vs 499960.
- Synthetic prompt of 450000 tokens: API 8192 (chat) / 50000 (thinking) vs ours 50000.
- Both sides lower the value to the context left.
- Fix: wrapper part B. In the prototype, `max_tokens_arg` is already min(context left, …), so the code is `min(arg, default)`.
- Risk: replies with no max_tokens that run past 8192 tokens (chat) or 64K (thinking) now end with finish_reason=length. That is the API's behaviour, but below the model card's ≥256K. Set the defaults from env if you want different numbers.

**S2. Top-level thinking** (table T)
- `th_enabled`: API think/75/65536 vs ours chat/499986.
- `th_disabled+eff_high`: API chat/8192 vs ours think/75.
- `th_enabled+eff_none`: API think/75 vs ours chat.
- Fix: part A wraps `build_chat_params`. It reads `model_extra["thinking"]` and sets both `thinking` and `enable_thinking`, so the OR in the tokenizer cannot override it. Enabled with no effort sets "high". The reasoning parser receives the same kwargs.

**S3. Effort aliases** (table T)
- `eff_medium`, `eff_minimal`, `eff_xhigh`: ours 400 "DeepSeek V4.1 reasoning_effort must be low, high, max, or an integer within [1, 100]…"; the API gives 75/50/75.
- `eff_ultra`: ours 400 "Input should be 'none', … or 'max'"; the API gives 100 / 131072.
- The OpenAI SDK in the image (3.11) types the field as none/minimal/low/medium/high/xhigh/max.
- Note that chat_encoder_1017 rejects xhigh on purpose, which matches HF encoding.py. The API library accepts xhigh and renders it as 75, so mapping it at the request layer keeps the encoder strict.

**S4. Effort default**
- `ctk_thinking`: ours think/50 vs API 75.
- The prototype re-run with defaults `{"thinking":false}` gives 75.

**S5/S6. Thinking-mode policy** (table S)
- `eff_high+t0`: API T=1 vs ours T=0.
- `eff_high+t0.6_p0.9`: API T=1, top_p 0.95 vs ours 0.6 / 0.9.
- `eff_none+t0.6_p0.9`: API 0.6 / 1.0 vs ours 0.6 / 0.9.
- `ctk_think_high(quality)+t0`: API T=1 vs ours T=0. This is the gsm8k_think gate setting.
- On this pack, greedy thinking did not degenerate:
  - gsm8k_think 39/40, mean 217 tokens (`quality-baseline/full.json`);
  - perf2 FINAL 38/40, mean 243 tokens, maximum miss 1759 of 4096.
- So the quality case for C is unproven. It is API policy, and the model card's recommendation.

**S7. Stop strings in thinking mode**
- Same output fed to both sides, `stop=["Observation:"]`, with the stop string inside the reasoning.
- API: `reasoning_content` keeps "…Observation: I should not stop here…", content is 'Thought: call the tool.\n', finish_reason stop.
- Ours: "matched while decoding piece 0 (</think> seen: False)"; the returned text is 'The user wants a ReAct step. '.
- Proposed fix: a per-request gate in IncrementalDetokenizer. When the prompt's last token is `<think>` (128821), set the stop-check offset past the first `</think>` (128822; it is not a special token, so it appears in the output text).
- Risk: medium (hot path; stop strings that straddle `</think>`).

**S8. Stop list length** (table P)
- 5 and 16 strings: ours 400 "List should have at most 4 items…"; API accepts.
- 17 strings: both 400, with different text.
- Fix: `VLLM_MAX_STOP_STRINGS=16` in the head env, either in FORWARD_ENVS with a default or in the fixed env_args at run.sh:520-534. The API server reads it at import time.

**S9.** `mct_100`: ours 100 vs API the default (65536 or 8192). Keep ours: honouring a client's limit is strictly safer.

**S10.** `mt_400k`: API 400 vs ours 400000. Optional to match.

**S11.** Table L shows both texts. `chat|prompt=500000|unset`: API 1 vs ours 400 "max_tokens must be at least 1, got 0". The error JSON also differs: vLLM sends `type:"BadRequestError"` and `code:400`, while the API sends `type/code:"invalid_request_error"`. That belongs to the protocols/parsing tracks.

**S12.** `tool_choice=required/named +thinking`: API 400 vs ours accepted. This affects all 300 residual tools_required rows.

**S13–S16.** Table P has each probe. For logit_bias, ours fails loudly (400) while the API silently ignores it. Leave it.

**S17.** Table Proto:
- Messages `thinking:{enabled}`: API think/75 vs ours chat.
- Messages without max_tokens: API 65536 vs ours 400 "Field required".
- Messages max_tokens=500000: API clamps to 393216 vs ours 400 (context).
- Responses `input only`: API think/75/65536 vs ours chat/499989.
- Responses `effort=medium`: ours 400.

**DSpark is lossless** (`dspark_lossless.py`, image kernels under the Triton interpreter)
- Setup: V=64, T=0.7, top_p=0.9 (11 tokens kept), N=8000 requests per scenario.
- Reference multinomial: TV 0.0139, chi2 p 0.156.
- Greedy draft = 2nd most likely token: TV 0.0137, p 0.318; 2247/8000 drafts accepted, against an expected 2266.
- Draft = argmax: p 0.941.
- Draft outside the top-p set, or −1 (masked): p 0.406.
- Zero tokens outside the top-p support.
- Greedy T=0: always the argmax.
- Not covered: `apply_sampling_params` itself (it uses CUDA UVA buffers); I read that code instead. DSpark acceptance at T=1.0, the API's thinking setting, is unmeasured. Every bench runs at T≤0.2, so throughput under part C is unknown.

**The prototype in numbers**

| configuration | matrix requests that match (of 1140) |
|---|---|
| ours as shipped | 204 |
| + parts A and B | 190 (top_p and temperature differences are now visible once mode and max agree) |
| + part C | 424 |
| + S4 default `{"thinking":false}` | 448 |

The 692 rows that still differ in the last configuration are all accounted for:
- 300 rows: API-only 400 for tool_choice in thinking mode.
- 144 rows: API-only 400 for the 393216 bound.
- 32 rows: "ultra".
- 160 rows: max_completion_tokens, plus default thinking.
- default-thinking and vLLM-kwargs rows: S18.

`--override-generation-config '{"max_new_tokens":8192,…}'`: unset → 8192, max_tokens=20000 → 8192, thinking with 60000 → 8192. That makes it a hard cap, so I rejected it.

## 4. Already matches (no need to re-check)

- With no sampling fields, both sides use T=1.0 and top_p=1.0 in both modes. The pack has no generation_config.json, and `get_diff_sampling_param` returns `{}`.
- Effort low→50, high→75, max→100; "none" gives chat mode; low, high and max turn thinking on. A numeric top-level effort is a 400 on both sides.
- These rules are the same on both sides; only the error text differs:
  - ranges: temperature [0,2], top_p (0,1], penalties [-2,2], top_logprobs ≤ 20;
  - the empty stop string;
  - stream_options without stream;
  - max_tokens=0;
  - prompt + max > context.
- An explicit max_tokens ≤ 393216 that fits the context is used as given. Seeds ≥ 0 and logprobs pass through on both sides.
- Tools on or off do not change V4.1 sampling on either side.
- DSpark and adaptive verify preserve the target distribution and greedy output. Penalties count the drafts. SOFTMAX_VERIFY, which is biased, is off.

## 5. Open questions / not verified

- Whether DeepSeek's production backend applies presence/frequency penalties. The library drops them but leaves the choice to the caller.
- Whether the live API really ignores `max_completion_tokens`.
- What context limit the live API passes for deepseek-flash (1M per the model card).
- Whether a numeric effort of 100 via chat_template_kwargs should get the 131072 default. The prototype gives it to "max" only.
- Whether vLLM's prompt length for max_tokens and the context check counts the expanded image tokens, as the API does. For the vision track.
- Raw versus processed logprobs (MRV2 default `raw_logprobs`) against the API. For the parsing track.
- DSpark acceptance and throughput at T=1.0 / top_p 0.95. Needs a boot; I ran no GPU.
- `n>1` under DSpark: I only checked request validation.
- Responses and Messages wrappers are not prototyped. For Messages `thinking`, the protocols track needs a schema rewrite.
- The prototype's serving glue (`_effective_chat_template_kwargs` stash) was emulated in the harness, not run through a live `OpenAIServingChat`. Phase 3 should test it on a boot.
- Install point: sitecustomize runs in every process, and importing the entrypoints there costs time. For the conventions track: a wrapper versus a source rewrite of `protocol.py` / `serving.py`.
