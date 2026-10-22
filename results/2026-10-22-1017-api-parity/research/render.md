# render track: does a Chat Completions request become the same prompt on our serve as on the DeepSeek API?

## SHORT SUMMARY

**Method.** I ran 165 request cases. Each went through three paths:
- **(a) API:** deepseek_recipe 0.2.0, `ChatCompletionRequest(body).convert(ConversionOptions())`, then `DeepseekV41Encoding` render and encode with the bundled v41 tokenizer.
- **(b) HF:** the 1017 `encoding/encoding.py`.
- **(c) Served:** the image's real vLLM front end on CPU. That is `ChatCompletionRequest.model_validate`, then `OnlineRenderer(run.sh flags).render_chat`, then the engine `prompt_token_ids`.

The 165 cases are 55 structural cases run in a thinking and a non-thinking variant, 21 thinking-control cases, the 5 HF goldens sent as requests, and validation edge cases. The two variants use top-level `reasoning_effort` "none" and "high", which both sides read the same way.

**Headline.**
- 133 cases are accepted by both the API and our serve. Of those, 79 give identical token IDs, 3 differ only in tokenization (identical text), and 51 differ in the prompt text.
- The served encoder itself matches HF: both reproduce all 5 golden vectors.
- Almost every divergence comes from request handling: fields that never reach the encoder, or fields the encoder ignores. The rest are places where the API and HF use different formats.

| id | Behaviour | API (deepseek-recipe) | Ours | Impact | Smallest fix |
|---|---|---|---|---|---|
| R1 | Top-level `thinking:{type}` | enabled/adaptive → on (75); disabled overrides effort | Field ignored (pydantic extra), thinking stays off; `disabled`+`high` turns thinking on | HIGH | Rewrite `build_chat_params` to forward `thinking`; tokenizer applies "explicit thinking beats effort" |
| R2 | `reasoning_effort` values minimal/medium/xhigh/ultra | minimal→50, medium→75, xhigh→75, ultra→100 (thinking on) | 400 from the tokenizer; `ultra` 400 from the pydantic Literal | HIGH (`medium` is common in SDKs) | Alias map in `deepseek_v41.py` (chat_encoder_1017 extension); Literal rewrite for `ultra` |
| R3 | `response_format` json_object / json_schema | "## Response Format: … schema …" appended to the first system message | Never rendered: `response_format` is never passed to the tokenizer | HIGH for structured output | Renderer passes `params.response_format`; tokenizer attaches it to the first system message (the encoder already renders it) |
| R4 | `prefix:true` (and vLLM `continue_final_message`) | Prompt continues the prefix; reasoning-only prefix stays open (`<think>r`) | Ignored: prompt ends `…prefix<｜end▁of▁sentence｜>` with no assistant header | HIGH for prefix users | `prefix` → `wo_eos` (already honored, verified); drop `</think>` on a reasoning-only prefix |
| R5 | `developer` role | 400 in Chat Completions (Responses maps developer → system) | Chat mode: a separate `<｜User｜>` turn. Thinking without tools: silently dropped | MED-HIGH | Map developer → system in `_normalize_messages` |
| R6 | Thinking on with no effort given | 75 | 50, from the run.sh default kwarg `reasoning_effort:"low"` | MED | run.sh: `--default-chat-template-kwargs '{"thinking":false}'` (what-if verified) |
| R7 | `tool_choice:"none"` with tools | No tool prompt; history reasoning dropped | Full tool block rendered (379 vs 40 ids) | MED | run.sh: `--exclude-tools-when-tool-choice-none` (what-if 2/2 match) |
| R8 | `tool_choice` required / named | Prompt ends `\n\n<｜DSML｜ calls>\n`; named lists only that tool; 400 in thinking mode | No prefill; all tools listed; grammar only; accepted in thinking mode | MED | Filter tools for named now; add the prefill only together with tool-parser and grammar changes (coordinate with the structured and parsing tracks) |
| R9 | Special-token spellings typed in content | Plain text, character tokens (40 ids) | Real control tokens such as `<｜User｜>`, `</think>`, EOS (19 ids) | MED (injection) | Tokenizer track; needs special-range-aware encoding (the swap below gives it) |
| R10 | `latest_reminder` role | Rendered | 400 "Invalid role" (the encoder supports it) | MED-LOW | Allow the role; match the API's placement when the reminder follows the user message |
| R11 | Malformed, non-object, empty or double-encoded tool-call `arguments` in history | `arguments` string parameter carrying the raw text | `parse_chat_messages` rewrites them in place to `{}`, giving an empty invoke | LOW-MED | Deep-copy the messages before `parse_chat_messages` in `renderers/deepseek_v4.py` |
| R12 | Parallel tool results with images, sent out of call order | Images follow prompt order | Text is sorted by call order but `multi_modal_data` is not, so images bind to the wrong results | MED (silent, rare) | Sort tool messages before `parse_chat_messages` (the swap prototype fixes it) |
| R13 | Tool JSON in the tool prompt | `{"name","description":"","parameters":{}}`; strict schemas normalized; no `strict` key | `"description": null, "parameters": null`; `"strict": true` appended | LOW-MED | Emit the upstream shape in the tokenizer |
| R14 | Tools when message 0 is user and a later system message exists | New empty system message at index 0 | Tool block put inside the later system message | LOW | Insert at index 0 unless message 0 is system |
| R15 | Trailing assistant message without `prefix` | New `<｜Assistant｜>` header appended after the EOS | Prompt ends at EOS with no header | LOW-MED | Append the header when `add_generation_prompt` is set |
| R16, R17 | Empty system message; `x-anthropic-billing-header` line | Dropped; stripped | Rendered; kept (the varying header also breaks the prefix cache) | LOW | Drop / strip in `_normalize_messages` |
| R18 | `<｜deepseek_image｜>` typed as text | Plain text | 400 (1017 patch, HF rule) | LOW | Leave as is, or relax |
| R19 | 17 inputs the API rejects (tool-name pattern, duplicate tools, orphan tool result, missing "json", prefix not last, …) | 400 | Accepted | LOW | None needed |
| R20 | Thinking mode, no tools declared, tool calls already in the current turn | Current-turn reasoning dropped | Kept, same as HF | API and HF disagree | Keep ours (HF); open question |

**Recommendations**
1. **Two run.sh flags now, no code change:** `--exclude-tools-when-tool-choice-none` (R7) and drop `reasoning_effort:"low"` from `--default-chat-template-kwargs` (R6). The thinking default stays off. Our own harnesses all pass an explicit effort, so they are unaffected (`tests/quality_eval.py:69-70`, the smoke tests and the benches). README lines 11 and 228 and `recipe.yaml:158` need the matching edit.
2. **Patch route.** Extend `chat_encoder_1017.py` for R2, R4, R5, R10, R13, R14, R16 and R17. Add one new source rewrite of `renderers/deepseek_v4.py` that deep-copies the messages and sorts tool results before parsing, and passes `response_format` and `tool_choice` through (R3, R8 named-tool filtering, R11, R12). Add one rewrite of `ChatCompletionRequest.build_chat_params` to forward `thinking` (R1). R9 is not fixable this way.
3. **Swap route.** Replace the `DeepseekV4Renderer` prompt path with deepseek_recipe convert → render → encode, and fall back to the current path for requests deepseek_recipe rejects. My in-process prototype (nothing changed in the image or the repo) ran through the real `OnlineRenderer`: 136/136 comparable cases produce engine token IDs equal to the API's, images included, with the R12 ordering fixed. It fixes R1–R5, the prompt side of R8, and R9–R17 in one place. Before adopting it:
   - Decide the R8 prefill against the tool parser and grammar.
   - `ultra` is still blocked by vLLM's pydantic Literal.
   - Responses and `/tokenize` also use this renderer.
   - Choose a validation policy: the fallback, or strict API 400s.

**Infra notes for CONTEXT.md**
- Run containers with `--user 1000:1000 -e HOME=/tmp`; otherwise root-owned files appear under `/ap`.
- `VllmConfig` needs `DeviceConfig(device="cpu")` in a container without a GPU.
- `dr.ChatCompletionRequest(body)` raises `ValueError` for schema errors such as an unknown role. `convert()` raises `ConversionError(status_code, body)`.

**Files** (all under `/home/sfxnz/projects/data/dsv41-1017/api-parity/harness/render/`). No REPORT.md was written; the full report is below.
- `run_render.py` (3-way harness) and `corpus.py`
- `run_in_image.sh`
- `show.py <case>`, `summarize.py`
- `golden_direct.py`
- `swap_proto.py`, `swap_renderer_proto.py`
- `out/` holds `results.json`, `summary.txt`, `golden_direct.txt`, `whatif_*.json`, `swap_*.log` and `swap_*.json`.

---

## FULL REPORT

### 1. What upstream does: V4.1 format rules (deepseek-recipe @935817c; HF = 1017 `encoding/encoding.py`)

**Request conversion** (`deepseek-recipe/src/protocol/openai/chat_completion/request/`)

*Roles and fields*
- Accepted roles are system, user, assistant, tool and latest_reminder (`schema.rs:63-90`). `developer` is an unknown variant and returns 400. Unknown fields are ignored.
- Top-level controls are `thinking:{type}` and `reasoning_effort` (`schema.rs:53-57`).
- Thinking resolution (`reasoning.rs:58-78`):
  - An explicit `thinking` setting decides on/off; `adaptive` means enabled.
  - Otherwise effort `none` means off, any other effort means on, and no effort uses the default, which is on (`options.rs:444-461`).
  - With thinking on, effort maps minimal/low → Low, medium/high → High, xhigh → Xhigh, ultra/max → Max (`reasoning.rs:37-49`). The default is High. With thinking off, effort is None.
  - Numbers are a deserialization error.

*Messages* (`transform_messages`, `convert.rs:415-566`)
- A system message whose content is empty after stripping the billing header is dropped (`convert.rs:378-394`, `428-440`).
- Content-part lists are joined with `"\n\n"`. Images become the special `<｜image｜>` (`convert.rs:291-357`).
- An assistant message must have content or tool_calls. Images in assistant messages are rejected.
- An assistant message with tool_calls must be followed immediately by exactly one tool message per call. Tool calls with no matching result, and duplicates, are rejected.
- Missing `reasoning_content` is filled with "" in thinking mode and cleared outside it (`reasoning_content.rs:172-223`, `FillEmpty` default).

*Request-level checks*
- A json_object request needs the word "json" somewhere (`convert.rs:83-91`).
- `prefix` must be on the last message and cannot be combined with tools or json_object (`convert.rs:93-102`, `234-247`).
- Tools (`convert.rs:254-283`, `client_tools.rs:63-105`):
  - `tool_choice` none removes the tools.
  - Named keeps only the named tool and becomes Required.
  - Required or named in thinking mode is a 400.
  - Tool names must match `^[a-zA-Z0-9_-]+$` and be unique (`validation.rs:129-153`).
  - Strict tool schemas are normalized: every property required, `additionalProperties:false`, unsupported keywords dropped (`validation.rs:209-240`, `json_schema.rs`).

**Rendering** (`deepseek-recipe-encoding/src/v4/dsv41.rs`, helpers in `mod.rs`)

*BOS and system token*
- BOS is emitted once (`dsv41.rs:288`).
- At index 0, `<｜System｜>` is emitted if thinking is on or message 0 is a system message (`dsv41.rs:144-156`).
- With thinking on, the effort line follows: `Reasoning Effort: {50|75|100} (range 1-100, the higher the value, the more thorough the reasoning)\n\n`, where Low → 50, None/High/Xhigh → 75, Max → 100 (`dsv41.rs:21-30`).

*Messages*
- **System:** a system message after index 0 gets its own `<｜System｜>` (mid-conversation system). `SystemMessage.tools` renders `### Updated Tool Schemas`; that path is used by Responses, not Chat Completions (`dsv41.rs:168-181`).
- **User and tool:** `<｜User｜>` starts the turn, unless the previous message is user or tool, in which case `"\n\n"` separates them (`dsv41.rs:157-166`). Tool results render as `<tool_result>{c}</tool_result>`.
- **Latest reminder:** `<｜latest_reminder｜>{c}` is written in place, with no header. A reminder directly after another reminder is dropped (`dsv41.rs:186-189`, `338`).
- **Assistant** (`dsv41.rs:194-232`):
  - Rendered as `<｜Assistant｜>` + marker + reasoning + content + DSML + `<｜end▁of▁sentence｜>`.
  - Reasoning is rendered only when thinking is on, the index is above 0, and either tools are declared or the message is a prefix. Otherwise the marker is `</think>`.
  - A prefix message gets no EOS. A prefix with empty content and no tool calls keeps `<think>{r}` open.
  - Only the grounding tokens (`<｜ref｜>` etc.) inside assistant content or reasoning count as special (`mod.rs:24-33`, `197-208`).
- **DSML tool calls** (`dsv41.rs:91-121`, `mod.rs:38-43`):
  - Layout: `\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="n">\n<｜DSML｜ parameter name="k" string="true|false">v</…>\n…</｜DSML｜ calls>`.
  - Arguments are parsed once. If they are not a JSON object, they become `{"arguments": raw}`.
  - Non-string values are written in Python `json.dumps` style (`core/json.rs:417-424`).

*Whole conversation*
- **Tool prompt and response format** (`dsv41.rs:250-284`). Both are appended to the first system message; an empty system message is inserted at index 0 if message 0 is not one.
  - Tool prompt text: `mod.rs:56-90` + `dsv41.rs:70-88`. Each schema is `{"name","description":d or "","parameters"}`.
  - Response format text: `\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{json}`. The JSON is `{"type":"json_object"}` or the json_schema `schema`. Regex is not rendered.
- **Dropping past reasoning:** reasoning is dropped exactly when no tool prompt is rendered (`dsv41.rs:286`).
- **Generation header:** `<｜Assistant｜>` + `<think>` or `</think>` is appended unless the last message is a prefix. For Required it is followed by `\n\n<｜DSML｜ calls>\n` (`dsv41.rs:298-310`).
- **Normalization:** consecutive user messages are merged with `"\n\n"`. Tool results are sorted by the order of the preceding calls (`dsv41.rs:321-419`). Image sources are collected in that final order (`mod.rs:109-121`).

**Token encoding** (`encoding/src/tokenizer.rs:90-117`)
- Special ranges encode to their added-vocab IDs. Each plain-text region is encoded without the added vocabulary, so typed special spellings stay plain text.
- The bundled v41 tokenizer differs from the pack's in exactly one entry: ID 129264 is `<｜image｜>` (special=true) upstream and `<｜deepseek_image｜>` (special=false) in the pack.

**HF encoding.py differences from upstream**
- It renders the client's tool `function` dict verbatim (`encoding.py:84-98`), including `strict`, any extra keys and namespaces.
- It has no notion of request validation, `prefix`, tool_choice or a Required prefill. `wo_eos` drops the EOS (`encoding.py:697-708`) but always closes the reasoning block.
- The generation header is appended after user, developer and mid-conversation system messages, and before a following latest_reminder. A trailing assistant message gets no header (`encoding.py:712-742`).
- Past reasoning is dropped only before the last real user turn, and only when no tools are declared (`encoding.py:562-582`, `749-807`). Tool-result-only user messages do not start a turn.
- It supports numeric effort 1-100 (`encoding.py:456-474`), `task` tokens, and the `<｜deepseek_image｜>` placeholder (`encoding.py:39`). Text spelling that placeholder is rejected (`encoding.py:355-367`).

### 2. What our serve does (image-src = the image; both encoder files already carry the baked 1017 rewrite)

Request flow (the `vllm/` paths below are in image-src):
1. **Validation.** `ChatCompletionRequest` (`entrypoints/openai/chat_completion/protocol.py`):
   - Unknown top-level fields, `thinking` among them, are kept in `model_extra` and never used (`entrypoints/serve/engine/protocol.py:22-24`).
   - `reasoning_effort` is a Literal without `ultra` (`protocol.py:244-256`).
   - `reasoning_content` is renamed to `reasoning` (`protocol.py:521-547`).
   - `tool_choice` defaults to auto when tools are present (`protocol.py:915-918`).
   - Message keys outside the schema survive, for example `prefix`, `wo_eos` and `task`.
2. **Renderer.** `OnlineRenderer.render_chat` → `preprocess_chat` (`renderers/online_renderer.py:175-180`, `395-413`):
   - Tools are passed as `tool.model_dump()`, which keeps `description`/`parameters` as None and keeps `strict` (`entrypoints/generate/base/protocol.py:224-239`).
   - Tools are passed even when `tool_choice` is none, because the exclude flag is off.
   - Template kwargs are merged as server defaults < request `chat_template_kwargs` < `{add_generation_prompt, continue_final_message, documents, reasoning_effort, enable_thinking}` + `tools` + `tokenize=False` (`renderers/params.py:40-52`, `111-149`; `protocol.py:570-605`).
   - `tool_choice` and `response_format` are stored on ChatParams but never handed to the tokenizer.
3. **`DeepseekV4Renderer.render_messages`** (`renderers/deepseek_v4.py:35-87`) runs `parse_chat_messages` first, then `tokenizer.apply_chat_template(messages=request.messages, **kwargs)`. `parse_chat_messages` rewrites each tool call's `arguments` in place to a dict, or `{}` when it is not a JSON object (`entrypoints/chat_utils.py:1998`, `2043-2117`). Its `multi_modal_data` follows message order.
4. **DSV41 tokenizer** (`tokenizers/deepseek_v41.py`):
   - `_normalize_messages` (`deepseek_v41.py:21-47`) accepts only system, developer, user, assistant and tool (latest_reminder is a 400), maps `reasoning` back to `reasoning_content`, and flattens content parts with `"\n\n"`.
   - `apply_chat_template` (`deepseek_v41.py:56-103`):
     - Puts tools on the first system message found anywhere, or a new one at index 0.
     - Sets thinking from `thinking` or `enable_thinking`; effort `none` turns it off.
     - Validates effort ∈ {low, high, max, 1..100}.
     - Reads `drop_thinking`.
   - It ignores `add_generation_prompt`, `continue_final_message`, `prefix`, tool_choice and response_format.
5. **Encoder** (`tokenizers/deepseek_v41_encoding.py`):
   - It is HF's 1017 logic plus a `developer` role rendered as `<｜User｜>` (`deepseek_v41_encoding.py:402-415`).
   - That role is dropped by `_drop_thinking_messages` in thinking mode without tools (`deepseek_v41_encoding.py:548-567`).
6. **Tokenization:** the flat prompt string is encoded with the full added vocabulary and `add_special_tokens=False` (`renderers/base.py:585-600`).

The recipe sets the flags at worktree `run.sh:660-664`. The `chat_encoder_1017.py` patch (sitecustomize `:669-672`) is already baked into the e15 and e16 images: Dockerfile.e16 lines 52-53 run `py_compile`, which executes the baked sitecustomize at build time. At serve time the patch reports "already applied".

### 3. Evidence per divergence

All scripts are in harness/render. Unless noted, each excerpt is in `out/summary.txt` under the case name; `python3 show.py <case>` prints the full prompts.
- **R1** `ctl_thinking_enabled`: the API prompt ends `<｜Assistant｜><think>` at effort 75; ours ends `</think>` with kwargs `thinking=False`. `ctl_thinking_disabled_effort_high`: the API is in chat mode, ours thinks at 75.
- **R2** `ctl_effort_medium`/`minimal`/`xhigh`: ours returns "400 ValueError: DeepSeek V4.1 reasoning_effort must be low, high, max, or an integer within [1, 100]". `ctl_effort_ultra`: pydantic 400. The API thinks at 75/50/75/100.
- **R3** `rf_json_schema@think`: the API has 91 ids including "## Response Format: … {"type": "object", …}"; ours has 40 ids with no format text. API and HF are identical here.
- **R4** `prefix_content@chat`: the API ends `</think>Autumn wind` (12 ids); ours ends `Autumn wind<｜end▁of▁sentence｜>` (13 ids). `prefix_reasoning_only@think`: the API ends `<think>Let me count syllables`; ours ends `…</think><｜end▁of▁sentence｜>`. `prefix_via_wo_eos`: sending `wo_eos:true` already gives the open prefix on our serve.
- **R5** `developer_role@think`: our prompt does not contain "Be terse.". `@chat` renders `<｜User｜>Be terse.<｜User｜>Hello`. The API returns 400 "unknown variant `developer`". Upstream Responses maps developer → system (`responses/request/convert.rs:421-450`).
- **R6** `ctl_ctk_thinking`: ours renders "Reasoning Effort: 50". With `--default-ctk '{"thinking":false}'` it matches the API at 75 (`out/whatif_default_ctk_no_effort.json`).
- **R7** `tools_choice_none@think`: 40 vs 379 ids. With `--exclude-tools-when-none` both variants match (`out/whatif_exclude_tools_none.json`). The flag is `entrypoints/launchers/cli_args.py:108`.
- **R8** `tools_choice_required@chat`: the API ends `</think>\n\n<｜DSML｜ calls>\n` (359 vs 354 ids). `tools_choice_named@chat`: the API schemas list only `search` (284 vs 354 ids).
- **R9** `sp_user_text`: the strings are equal, but the API has 40 ids and ours 19 (`…128803,223,128804,223,128821,223,128822,223,128825,223,1…`). The same applies to `</think>` in assistant history and to `｜DSML｜` in tool results.
- **R10** `latest_reminder@*`: ours returns "400 Invalid role: latest_reminder"; API equals HF. `latest_reminder_last`: the API renders `<｜User｜>q<｜latest_reminder｜>r<｜Assistant｜>…`, HF renders `<｜User｜>q<｜Assistant｜></think><｜latest_reminder｜>r` with no header at the end.
- **R11** `tc_args_malformed`/`nonobject`/`empty`: API and HF render `<｜DSML｜ parameter name="arguments" string="true">{"query": "x"</…>`; ours renders `<｜DSML｜ invoke name="search">\n\n</｜DSML｜ invoke>`. `tc_args_double_encoded`: the API keeps the raw text, HF decodes twice, ours renders an empty invoke.
- **R12** `img_tool_results_reordered`: ours places the images at (463, **202**), (675, **184**). The big image sits in call 1's slot. The control case gives small 185 / big 203. The swap prototype gives (463, 184), (657, 202).
- **R13** `tools_no_description`: the API renders `{"name": "ping", "description": "", "parameters": {}}`, ours `…"description": null, "parameters": null}`, HF `{"name": "ping"}`. `tools_strict_valid`: the API's normalized schema drops `minLength` and has no `strict`; ours keeps `minLength` and appends `"strict": true`.
- **R14** `tools_mid_system_first_user`: the API starts `<BOS><｜System｜>\n\n## Tools…`; ours puts the tool block after `…<｜System｜>Be brief.`.
- **R15** `trailing_assistant@think`: the API ends `</think>Autumn wind<eos><｜Assistant｜><think>`; ours ends `<think></think>Autumn wind<eos>`.
- **R16** `empty_system@chat`: 5 vs 6 ids. **R17** `billing_header_system`: 11 vs 36 ids.
- **R18** `sp_user_deepseek_image`: ours returns 400; the API keeps it as plain text.
- **R19** The 17 "API 400 / ours ok" cases are listed in `out/summary.txt`.
- **R20** `tc_no_tools_declared@think`: the API renders `<｜Assistant｜></think>\n\n<｜DSML｜ calls>…`; HF and ours render `<think>I should call the tool.</think>…`.
- **Vision track:** `file` content parts. The API accepts the flat `{type:file,file_data}` shape; ours returns 400 "Unsupported chat content part type: 'file'". The OpenAI nested shape is rejected by both.
- **Goldens** (`out/golden_direct.txt`):
  - HF and the image encoder reproduce all 5 vectors.
  - deepseek_recipe reproduces vectors 1 and 2 plus its generation header, and vector 5 exactly.
  - Vector 3 differs: upstream renders tools on a mid-conversation system message as "### Updated Tool Schemas"; Chat Completions cannot express that layout.
  - Vector 4 uses the internal `task` token, which upstream does not support.

### 4. Already matching (no need to re-check)

- **Basics:** BOS appears once, with no double BOS even when `add_special_tokens=true`. The system-token and effort-line rules match, and so do the 50/75/100 effort values. The `<think>`/`</think>` generation header and the EOS rules match.
- **Reasoning history:** past reasoning is dropped in multi-turn without tools and kept with tools; a missing reasoning becomes `<think></think>`.
- **Tool prompt:** template text, DSML tag names with the leading space, JSON separators, `ensure_ascii=False`, floats such as `1e-05`, and key order when the client sends name, description and parameters.
- **DSML tool calls in history:** single and parallel calls, nested arguments, non-ASCII, quotes, backslashes and newlines, floats and big ints, content alongside calls, and `content:null`.
- **Tool results:** wrapping, the `"\n\n"` join, sorting by call order, content-part lists, and a result followed by a user message.
- **Messages:** consecutive users merged; content-part arrays in system, user and assistant messages; mid-conversation system messages, including one at the end; two system messages at the start; whitespace and empty content.
- **Images (before expansion):** placement, ID 129264 and the joins match in system, user, tool and two-image cases.
- **Controls:** top-level effort none/low/high/max, `thinking` enabled + max, and `chat_template_kwargs` thinking + high all match. `tool_choice` auto, explicit or absent, matches.
- **Golden 5** matches on all three sides.

### 5. Swap feasibility: deepseek_recipe inside the image

- **What to replace.** Replace `DeepseekV4Renderer.render_messages` and `render_messages_async` (`renderers/deepseek_v4.py:35-87`).
- **What it must return.** It must return `(conversation, TokensPrompt{prompt_token_ids, prompt, multi_modal_data, multi_modal_uuids})`. `parse_dec_only_prompt` accepts token lists, tokenization is skipped, and vLLM's DSV41 processor expands ID 129264 (`models/deepseek_v4_1/common/mm_preprocess.py:376-396`).
- **Steps** (prototype: `swap_renderer_proto.py`):
  1. Deep-copy the messages before `parse_chat_messages`, which is still used to load the images.
  2. Rebuild the request body: `reasoning` → `reasoning_content`; tool `arguments` back to strings; tools without None values; `tool_choice` and `response_format` taken from ChatParams.
  3. Forward the client's own `thinking`, top-level effort and `chat_template_kwargs` through `build_chat_params`, because ChatParams cannot tell client kwargs from server defaults.
  4. Convert with `ConversionOptions(default_thinking_mode=False).with_special_tokens(SpecialTokens(image="<｜deepseek_image｜>")).with_model_alias(served, ModelVersion.DeepseekV41Flash)`.
  5. Encode with the pack tokenizer.
  6. Reorder the `multi_modal_data` items to follow `rendered.image_sources[*].path` (`.messages[i]`).
  7. Map `ConversionError.body` to `VLLMValidationError` (400). Catch the `ValueError` the request constructor raises for schema errors.
- **Results.**
  - `swap_proto.py`: 134/134 cases rebuilt from vLLM-validated requests give IDs identical to the API's with the pack tokenizer.
  - `swap_renderer_proto.py` through the real `OnlineRenderer`:
    - **Strict mode:** 136 matches; 21 requests get 400 on both sides. Four get 400 from our serve but are accepted by the API: one `file` part, `ultra`, and two vLLM-only knobs.
    - **Fallback mode:** 17 requests the API would reject go through the current path instead.
    - The single remaining difference is that our serve accepts a `reasoning` key in history, because vLLM's rename makes it indistinguishable from `reasoning_content`.
- **Edge cases to settle:**
  - **Required prefill:** the output then starts inside `<｜DSML｜ calls>`, so the deepseek_v41 tool parser and grammar must accept that (structured and parsing tracks).
  - **Reasoning-only prefix:** the reasoning parser must start in the reasoning state.
  - **`ultra`:** blocked by the pydantic Literal.
  - **vLLM-only knobs:** numeric effort, `drop_thinking`, `task`.
  - **Validation policy:** strict 400s or the fallback.
  - **Other users of this renderer:** Responses and `/tokenize`. Gate the swap on Chat Completions requests or extend it.
  - **Deployment:** install the abi3 aarch64 wheel and its bundled `deepseek_recipe.libs` OpenCV `.so` files in a Dockerfile layer; pin 0.2.0.
  - **Cost:** Rust render and encode cost is negligible.

### 6. Open questions / not verified

- Which reference to follow where the API and HF disagree: R10 after-user placement, R15, R16, R20, and R13's verbatim client tool dict in HF.
- Whether the hosted API uses non-default `ConversionOptions` (for example rejecting missing `reasoning_content`); this cannot be checked offline.
- How the model actually behaves after the R4 and R15 prompts; no GPU was used.
- Image token counts per image are for the vision track (184–203 positions observed).
- Strict-schema normalization and Required-prefill interplay: structured track. `reasoning` vs `reasoning_content` in responses: parsing track.
