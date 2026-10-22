# protocols track: API surface (Chat extras, Responses, Anthropic Messages) and integration architecture

## SUMMARY (short form)

**What was compared.** deepseek-recipe 0.2.0 (the DeepSeek API front-end library) against our serve. Our side is the image's vLLM with the worktree patches applied by sitecustomize, run CPU-only. Three test sets:
- **87 request cases** (37 Chat, 27 Messages, 23 Responses), rendered to prompt token IDs on both sides. 28 match exactly.
- **24 output scenarios**: the same model output token IDs pushed through deepseek-recipe's StreamProcessor and through the image's real chat, messages and responses handlers, using a scripted mock engine.
- **HTTP probes** through the image's real FastAPI app.

**What already matches:**
- **Chat:** plain chat, effort low/high/max (budgets 50/75/100), tools with `auto`, tool round trips with `reasoning_content` passed back, a single image.
- **Messages:** basic requests, a string `system`, `output_config.effort`.
- **Responses:** string input, function round trips, reasoning items without `encrypted_content`, the streaming event order.

**What breaks:**
- **Responses is unusable for standard clients.** `input_text` parts and `reasoning.effort:"medium"` (the defaults of the OpenAI SDK and Codex) both return 400. `apply_patch` and `tool_search` are dropped. Developer messages render as `<｜User｜>`.
- **The DeepSeek and Anthropic `thinking` switches are ignored**, so requests that set them get no thinking.
- **Responses use `reasoning` instead of `reasoning_content`**, and logprobs are not split between content and reasoning.

| ID | Divergence (DeepSeek API → ours) | Impact | Smallest fix |
|---|---|---|---|
| P01 | Chat `thinking:{type:enabled}` is ignored, so there is no thinking (37 vs 11 tokens) | High | B. Or C: map it in `ChatCompletionRequest.build_chat_params` |
| P02 | reasoning effort `minimal`/`medium`/`xhigh` → 400 on ours (API maps them to 50/75/75). Same for Messages `output_config.effort` and Responses `reasoning.effort` | High | C: map the aliases in the chat_encoder_1017 rewrite of `tokenizers/deepseek_v41.py`. Or B |
| P03 | Response field is `reasoning`, not `reasoning_content`. Logprobs are not split, and `</think>` and EOS land in content logprobs | High | B. C: emit an alias field (keep `reasoning`) |
| P04 | `prefix:true` continuation, or any trailing assistant message, is closed with EOS and gets no generation header (vLLM's `continue_final_message` does the same) | Med | B. C: map `prefix` to `wo_eos` plus the parser start stage |
| P05 | `tool_choice:"none"` still renders the tools (11 vs 276 tokens), in Chat and Messages | Med | **run.sh `--exclude-tools-when-tool-choice-none`** (verified: identical IDs) |
| P06 | `required`/named tool choice: the API prefills `\n\n<｜DSML｜ calls>\n` and keeps only the named tool; ours forces the call with a grammar and keeps all tools. With thinking on, the API returns 400 and ours accepts | Med | B (overlaps the structured track) |
| P07 | `json_object`/`json_schema`: the API writes a "## Response Format" schema into the prompt; ours only uses a grammar. Response format `regex` → 400 on ours | Med | B (structured track) |
| P08 | Messages `thinking:{type:enabled\|adaptive}` is ignored (no such field in vLLM's schema) | High | B. C: add the field and map it |
| P09 | Messages `system` blocks are joined with "" (API joins with "\n\n") | Med | C: one line. Or B |
| P10 | Messages mid-conversation `system` messages are hoisted into the top system block with a "" join (the API keeps them in place, a V4.1 feature) | Med-High | C: set merge_inline_system=False for deepseek_v41. Or B |
| P11 | Messages `tool_result`: text joined with "\n", images moved outside `<tool_result>`, `tool_reference` → 400, `defer_loading`/`tool_addition` unsupported | Med | B |
| P12 | Messages `thinking` blocks joined with "" (API "\n\n") | Low-Med | C: one line. Or B |
| P13 | Messages `tool_choice:any` → ours forces a call with a grammar (API treats it as `auto`) | Med | C: one line. Or B |
| P14 | Messages: `document` block → 400 (API inserts placeholder text); `max_tokens` is required; `stop_sequences` limited to 4 (API 16); `max_tokens` above the context → 400 (API clamps to 393216) | Low-Med | B. **env `VLLM_MAX_STOP_STRINGS=16`** |
| P15 | Responses `input_text` parts and `function_call_output` parts → 400 "supports text and image content only" | **High** | C: treat `input_text`/`output_text` as text in `_normalize_messages`. Or B |
| P16 | Responses developer message renders as `<｜User｜>` (API: `<｜System｜>`) | Med | C (render track). Or B |
| P17 | Responses: `apply_patch` custom tool dropped, client `tool_search` ignored, namespaces named `ns__f` with no description (API: `ns::f` plus the namespace description) | High (Codex) | B |
| P18 | Responses reasoning item carrying `encrypted_content` → 400 (the API ignores the field) | Med | C: remove the raise. Or B |
| P19 | Responses: only the first part of a multi-part assistant message kept; parallel calls not sorted by call_id; `input_image` requires `detail`; `top_logprobs` only works with `include` | Low-Med | B |
| P20 | Responses and Messages stream/body details: `reasoning_part` events, `phase` null, `encrypted_content` null, no `ping`, missing cache usage fields | Low | B |
| P21 | Usage: the API's final chunk always carries usage (and with `include_usage`, every other chunk has `usage:null`); ours sends usage only with `include_usage`, as a separate chunk, counts the EOS token, and has no cache-hit keys | Low-Med | B. Partial: `--enable-prompt-tokens-details` |
| P22 | The API rejects these and ours accepts them: `n>1`, `top_logprobs>20`, `max_tokens>393216`, empty `messages`, `json_object` without the word "json", developer role in Chat | Low | B |
| P23 | Error bodies: ours uses an integer `code`, type `BadRequestError`, and pydantic dumps; `/v1/messages` mixes two error shapes; an unknown model name gets 404 (API 400) | Low | B. Optional extra `--served-model-name` aliases |
| P24 | Chat: the `x-anthropic-billing-header` line stays in `system` (API strips it); `latest_reminder` role → 400 | Low | B. C: small |

**Prototype (B and A2 share the code; only the transport differs).**
- **How it works:** deepseek-recipe converts the request and encodes it (pack tokenizer, image spelling `<｜deepseek_image｜>`). It sends a GenerateRequest (token IDs, image content parts, sampling settings) to the image's own `/inference/v1/generate` handler, which streams token IDs back. StreamProcessor turns those into protocol events.
- **Exact parity:** 22/22 scenarios across all three protocols reproduce the API's prompt token IDs and response bodies/SSE exactly (after blanking IDs and timestamps). That covers thinking, tools, logprobs and `length` finishes.
- **Images work:** one placeholder per image plus `content_parts` is accepted, and vLLM expands the spans. Image token counts equal the API's on 8/8 sizes, whether vLLM gets the original image or the API-preprocessed one.
- **Cost:** for a 74K-token agentic request, convert is 1.2 ms and encode 17–20 ms. Our current path takes 108–124 ms. The HTTP hop adds about 10 ms, StreamProcessor about 0.45 µs per token, and the process uses about 110 MB RSS.

**Recommendation: architecture B (a head-only gateway sidecar), staged.**
1. **Stage 0, run.sh only:** add `--exclude-tools-when-tool-choice-none`, `VLLM_MAX_STOP_STRINGS=16` and `--enable-prompt-tokens-details`. Optionally add the two cheap C fixes for hard 400s (P15 `input_text`, P02 effort aliases) to the existing chat_encoder_1017 rewrite. Set `VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1`, which Stage 1 needs.
2. **Stage 1, opt-in front end:** a deepseek-recipe gateway on :8000, with vLLM moved to 127.0.0.1:8001.
   - It runs in its own container from the same image; no rebuild is needed.
   - The 0.2.0 wheel is pinned by sha256 and mounted read-only on the head only.
   - The gateway handles `/v1/chat/completions`, `/v1/messages` and `/v1/responses`, and proxies every other path to vLLM.
3. **Stage 2:** re-record the quality baselines through the gateway, check streaming and cancellation on a live boot, then make it the default.
4. **Alternatives:** keep A2 (the same code as an in-process endpoint plugin) as a fallback if spark1 memory can't take another ~110–150 MB process. C as the primary path is not worth it: 25+ anchored rewrites with ongoing drift, and parity would still be approximate.

**Files.** There is no REPORT.md (subagents cannot write one); this message is the report. Everything is under `/home/sfxnz/projects/data/dsv41-1017/api-parity/harness/protocols/`:
- **Runner:** `run_in_image.sh` mounts the worktree patches like run.sh, plus the wheel and `deepseek_recipe.libs`, and chowns outputs back.
- **API side:** `upstream_lib.py`. **Our side:** `ours_lib.py` (render path) and `ours_serve.py` (full handlers with a mock engine).
- **Cases and scenarios:** `cases.py`, `scenarios.py`.
- **Comparisons:** `compare_render.py` → `out/render_compare.md` and `out/render_compare_extra.md`; `compare_respond.py` → `out/respond_compare.md`.
- **HTTP probes:** `http_probe.py` → `out/http_probe_scaleout{0,1}.json`.
- **Prototype:** `gateway_proto.py`, `compare_gateway.py` → `out/gateway_compare.md`.
- **Costs:** `proto_bench.py` → `out/proto_bench.json`; `proto_images.py` → `out/proto_images.json`; `mem_probe.py`.
- **Flag test:** `flag_probe.py` → `out/ours_render_flag_exclude_tools.json`.

**Notes for CONTEXT.md:** image-src is not pristine (it already carries the chat_encoder_1017 rewrite), and importing the wheel in the image needs the `deepseek_recipe.libs` mount. Both are already in the harness.

---

## FULL REPORT

### 1. What upstream does (deepseek-recipe @935817c, paths under `deepseek-recipe/src/` unless noted)

**Shared conversion** (`request/options.rs`, `request/mod.rs`, `protocol/*.rs`)
- **Defaults:** thinking on (`options.rs:155`), Messages `web_search` rejected (`:158`), `max_tokens` upper bound 393216 (`:96`).
- **Model names:** a model not in the table → 400 "The supported API model names are …, but you passed X." (`:212-228`). Our name has to be registered with `with_model_alias`.
- **Thinking switch** (`protocol/reasoning.rs:58-78`): an explicit `thinking.type` decides. `adaptive` is an alias of `enabled` (`:11`). Otherwise effort `none` turns thinking off and any other effort turns it on; with no effort the default applies.
- **Effort mapping** (`reasoning.rs:43-46`): minimal and low → Low (50); medium and high → High (75); xhigh → Xhigh, rendered 75 in V4.1; ultra and max → Max (100). Thinking with no effort means High.
- **Sampling** (`protocol/validation.rs:44-125`):
  - Validated in a fixed order: n=1 only, penalties in [-2,2], temperature in [0,2], top_p in (0,1], top_logprobs ≤ 20 and only with `logprobs`, max_tokens in [1, bound], seed < 2^63.
  - Temperature is forced to 1.0 in thinking mode.
  - top_p for V4.1 is max(top_p, 0.95) in thinking mode and 1.0 otherwise (`options.rs:85-92`).
- **Tools:** at most 16 stop sequences (`validation.rs:156`). Tool names must match `^[a-zA-Z0-9_-]+$` and be at most 128 characters. A required or named tool choice in thinking mode → 400 "Thinking mode does not support this tool_choice" (`protocol/client_tools.rs:94-98`). Tool choice `none` drops the tools (`chat_completion/request/convert.rs:259-261`, messages `convert.rs:235`, responses `tools.rs:307`).
- **max_tokens resolution** (`request/mod.rs:207`, `docs/tokenizer.md:194-203`): a missing value becomes 8192 non-thinking, 65536 thinking, or 131072 at max effort, lowered to fit the context. Prompt plus requested max_tokens above the limit → 400 with the DeepSeek text ("This model's maximum context length is …").
- **Tool-call IDs** (`protocol/tool_call_id.rs:226-236`): `call_{NN}_[ET_]{20 alnum}{4-digit checksum}`. ET marks a thinking turn with empty reasoning; only `ReasoningContentBehavior::Reject` reads it.
- **Output constraint** (`protocol/output_constraint.rs`, `request/mod.rs:79-93`): computed but never enforced. Enforcement is left to the caller.

**Chat Completions** (`protocol/openai/chat_completion/**`)
- **Request fields** (`request/schema.rs:75-118`):
  - `thinking:{type, budget_tokens}`, `reasoning_effort`, `logprobs`/`top_logprobs`, `stream_options.include_usage`.
  - `response_format` of type text, json_object, json_schema or regex.
  - `tool_choice`, `strict` tools, `parallel_tool_calls` (validated, then ignored).
- **Roles** (`:120-148`): system, user, assistant (with `reasoning_content` and `prefix`), tool, and `latest_reminder`. There is no developer role, so `developer` is rejected.
- **Conversion** (`request/convert.rs`):
  - Empty `messages` → 400 (`:421`).
  - The `x-anthropic-billing-header` line is stripped from system content (`:378-394`).
  - `prefix` is allowed only on the final assistant message (`:234-247`) and picks the parser start stage (`:209-230`).
  - `json_object` requires the word "json" somewhere in the messages (`:87-91`).
  - `stream_options` without `stream` → 400 (`:581-585`).
- **Prompt effects** (render cases): `json_object`/`json_schema` add a "## Response Format …" system section. `required`/named tool choice ends the prompt with `\n\n<｜DSML｜ calls>\n` (parsing starts inside the tool calls). Strict tools get their schema normalized.
- **Response shape** (`response/schema.rs`, `response/chunk_generator.rs`):
  - Messages and deltas carry `reasoning_content` (`schema.rs:61`).
  - `logprobs` has separate `content` and `reasoning_content` lists (`:162`).
  - Usage carries `prompt_tokens_details.cached_tokens`, `completion_tokens_details.reasoning_tokens` (thinking only), and `prompt_cache_hit_tokens`/`prompt_cache_miss_tokens` (`:214-215`).
  - In a stream the first chunk is role plus an empty content or reasoning field. The final chunk carries `finish_reason` and always carries usage; with `include_usage`, every other chunk has `usage:null` (`chunk_generator.rs:84-90, 195-210`). The stream ends with `[DONE]`.

**Responses** (`protocol/openai/responses/**`)
- **Request** (`request/schema.rs`):
  - Fields: input (string or items), instructions, max_output_tokens, `reasoning.effort`/`summary`, `text.format` (json_schema or json_object), `top_logprobs` (any value turns logprobs on), tools, tool_choice, user.
  - Item types: message, function_call, function_call_output, reasoning (`encrypted_content` ignored, `:171`, convert `:496`), custom_tool_call(_output), tool_search_call/output, web_search_call (dropped). Unknown types are ignored.
  - Content parts: `input_text`, `output_text`, `input_image` (`detail` optional), `input_file` (becomes "[Unsupported Document]").
  - `store`, `previous_response_id` and `include` are unknown fields and ignored.
- **Tools** (`request/tools.rs`):
  - Namespace tools become `{ns}::{name}` in V4.1, with the namespace description prepended to each tool's description (`:122-132, 345-347`).
  - The `apply_patch` custom tool becomes a function tool with a long grammar description (`:369-395`). Any other custom tool → 400.
  - A client-executed `tool_search` becomes a `tool_search` function. Tools it discovers are declared with `defer_loading` in a System message placed after the output (`:157-175`, convert `tool_search_output`).
  - `web_search` is dropped.
- **Conversion** (`request/convert.rs`):
  - Developer messages become System in V4.1 (`:429-445`).
  - `instructions` becomes the first System message.
  - Multi-part content is joined with "\n\n" (`:106, :143`).
  - A tool-call group is sorted by call_id (`:294`).
  - In V4.1 an `<image_resize_notice>` developer message after a tool output is placed right after that output.
- **Output** (`response/chunk_generator.rs`):
  - Events: response.created, response.in_progress, output_item.added, content_part.added (also used for reasoning, `:143`), reasoning_text/output_text/function_call_arguments/custom_tool_call_input deltas, then the done events, then response.completed or response.incomplete.
  - Reasoning items have `content:[{type:reasoning_text}]`, `summary:[]` and `encrypted_content:"{response_id}-0"` (`:285`).
  - Message items have `phase` `final_answer`, or `commentary` when a tool call follows (`:412`).
  - Request settings are echoed (temperature, top_p, tools, text, tool_choice, max_output_tokens as requested).

**Anthropic Messages** (`protocol/anthropic/messages/**`)
- **Request** (`request/schema.rs`):
  - `max_tokens` is optional (`:24`) and clamped to 393216 (`convert.rs:104`).
  - `thinking.type` takes priority over `output_config.effort` (`:31-32`).
  - `system` may be a string or text blocks. `role:"system"` messages are allowed mid-conversation.
  - Block types: text, image (base64/url; `file_id` → 400), document, thinking, tool_use, tool_result (text/image/document/tool_reference), server_tool_use/web_search_tool_result, tool_addition.
  - Server tools → 400 "tools[i]: server tools are not supported" by default. All-deferred tools → 400 (`convert.rs:217`).
  - `tool_choice:any` is treated as `auto` (`schema.rs:240`, `convert.rs:248`). `metadata` is validated and ignored.
- **Conversion** (`request/convert.rs`):
  - System blocks joined with "\n\n"; a first block starting with the billing header (case-insensitive) is dropped (`:365-383`).
  - In V4.1, mid-conversation system messages stay in place as System; in V4 they are wrapped in `<system-reminder>` (`:496`).
  - A user turn becomes one user message per block (same rendered text).
  - `tool_result` parts joined with "\n\n", images kept inside the tool message.
  - `tool_reference` becomes the tool name as text, plus a System message declaring the referenced tools (`:617`).
  - `document` becomes "[Unsupported document type: {media_type}]" (`:292`).
  - `tool_use.input` is serialized with spaces (`:667`). Thinking blocks are joined with "\n\n" (`:692`).
- **Output** (`response/*`):
  - Stream: message_start (usage carries input, cache creation 0, cache read, `service_tier`), then in thinking mode a thinking block opened right away and a `ping` after the first block (`chunk_generator.rs:93, 130`).
  - Thinking blocks end with a `signature_delta` whose value is the response ID (`:44`).
  - message_delta carries `stop_reason` and `stop_sequence` (null when unset). Content filtering maps to `refusal` (`schema.rs:89`).

**Errors** (`error_response.rs:45-63`): every protocol uses `{"error":{"message","type":"invalid_request_error"|"internal_error","param":null,"code":"invalid_request_error"|type}}`. Status is 400 or 500.

**Stream processing** (`stream/*`, `docs/streaming.md`, `docs/tokenizer.md`, `_native.pyi`)
- `StreamProcessor(generator, parsing_options, tokenizer)` accepts `InferenceChunk.ready/text/tokens(ids, [(lp, [(id, lp)…])])/finish` (`_native.pyi:773, 1005`).
- It decodes each special token as its own piece, so `</think>` and DSML tags are matched as tokens only.
- Stop sequences are matched in answer content only, never in reasoning (`stream/output_parser.rs:53`). The processor stops after a match (`processor.rs:185, 206`).
- A Stop finish must not push its final token IDs (`stream/inference.rs:58-60`): the gateway drops the EOS ID.
- Reasoning tokens are counted with the tokenizer.

**Example servers** (`server-py/main.py:98-145, 169-208`; `server-rs/src/lib.rs`): they render prompt text and resolve images (OpenCV, WebP). The inference backend is a mock that yields text chunks; their processor has no tokenizer (README:313-324). Any real backend must encode with the tokenizer and resolve max_tokens itself.

**V4.1 specifics:**
- The bundled tokenizer spells ID 129264 `<｜image｜>`; the pack spells it `<｜deepseek_image｜>`. The IDs are the same, so encoding with the pack tokenizer needs `SpecialTokens(image="<｜deepseek_image｜>")`.
- Developer messages map to System. Namespaces use `::`. Mid-conversation System messages, `defer_loading`, and `tool_search` are supported.

### 2. What our serve does (image-src under `vllm/`, recipe @7851999)

**Routes** (`out/http_probe_scaleout0.json`):
- `/v1/chat/completions`, `/v1/chat/completions/batch`, `/v1/completions`.
- `/v1/responses` plus GET `/v1/responses/{id}` and `/cancel`.
- `/v1/messages`, `/v1/messages/count_tokens`.
- `/tokenize`, `/detokenize`, `/v1/models`, `/health`, `/metrics`, `/load`, `/ping`, `/invocations`, `/generative_scoring`, `/version`.
- With `VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1`, which run.sh does not set (`envs.py:1880`, `scale_out/factories.py:64-101`), these are added: `/inference/v1/generate` and the render/derender endpoints for chat, completions and messages (`out/http_probe_scaleout1.json`).

**Request → prompt path.** run.sh:660-666 sets `--tokenizer-mode deepseek_v41`, `--tool-call-parser deepseek_v41`, `--reasoning-parser deepseek_v41` and default kwargs `{"thinking":false,"reasoning_effort":"low"}`. No chat template is set.
- **Chat:** `OnlineRenderer.render_chat` → `DeepseekV4Renderer` → the deepseek_v41 tokenizer's `apply_chat_template` gets the **raw** request messages (`renderers/deepseek_v4.py:32-52`) → `_normalize_messages` (`tokenizers/deepseek_v41.py:21-47`).
  - It accepts roles system, developer, user, assistant and tool only (`:27-28`), and content parts of type text, image_url, input_image and image_pil only (`:39-45`).
  - It computes thinking from kwargs `thinking`/`enable_thinking` (`:72-80`) and validates effort against low/high/max (`:81-88`). That validation comes from the recipe's chat_encoder_1017 rewrite, which is already baked into the image.
  - Then `encode_messages` from `deepseek_v41_encoding.py` runs. It renders developer as `<｜User｜>` (`:402-406`).
- **Top-level request fields:**
  - The only thinking switch is `reasoning_effort`, which becomes `enable_thinking` (`openai/chat_completion/protocol.py:570-603`).
  - An unknown field such as `thinking` is accepted and dropped with a debug log (`serve/engine/protocol.py:22-51`).
  - `stop` is limited to `VLLM_MAX_STOP_STRINGS=4` (`generate/base/protocol.py:28-30`, `envs.py:1154`).
- **Messages:** `AnthropicServingMessages.to_chat_completion_request` (`anthropic/serving.py:191`).
  - System blocks are joined with "" (`:241-255`).
  - Mid-conversation system messages are merged into the top block because there is no chat template (`:150-151`).
  - Thinking is joined with "" (`:329`). Tool input goes through `json.dumps` (`:383`).
  - `tool_result` text is joined with "\n", images are moved to a separate user message, and `tool_reference` is passed through as list content, which `_normalize_messages` then rejects (`:411-462`).
  - Effort is copied to `reasoning_effort` (`:514`); `tool_choice:any` becomes `required` (`:549`).
  - The pydantic schema has no `thinking` field and requires `max_tokens` (`anthropic/protocol.py:121-139`). Its block types have no `document` (`:39-47`), and tools must have `input_schema` (`:79`).
- **Responses:** `construct_input_messages` (`openai/responses/utils.py:161-335`).
  - A reasoning item with `encrypted_content` raises (`:275-276`).
  - An output message keeps only `content[0]` (`:298`). Plain dicts pass through unchanged, so `input_text` parts reach `_normalize_messages` and fail (`:335`).
  - Tools: only function and namespace tools are used (`tool_parsers/utils.py:204-221`), namespaces use `__` (`:172`), and custom/tool_search tools are dropped.
  - Logprobs only with `include:["message.output_text.logprobs"]` (`responses/protocol.py:460-468`). `previous_response_id` without the store env → 404.
- **Output:** the DeepSeekV41Parser (`parser/deepseek_v41.py`, thinking detection in `parser/deepseek_v4.py:221-240`) splits reasoning and DSML calls.
  - The fields are `reasoning` (`chat_completion/protocol.py:71`; `generate/base/protocol.py:329-333`).
  - Usage comes only with `include_usage`, as a separate chunk with empty `choices`.
- **Errors:** `{"error":{"message","type","param","code":<int>}}` (`serve/exception_handling/error_response.py`). `/v1/messages` translates handler errors to the Anthropic shape (`anthropic/api_router.py:39-48`), but request-validation errors stay OpenAI-shaped (probe `messages_no_max_tokens`).

**Token-in / token-out**
- **`/inference/v1/generate`** (`scale_out/token_in_token_out/`):
  - The request carries `token_ids` plus `sampling_params` (a full SamplingParams in JSON), plus either `content_parts` (image URLs; the server resolves them and runs the model's MM processor on the token prompt, `serving.py:147-170`) or `features` (pre-rendered base64 MM kwargs, `:171-198`).
  - It streams `token_ids` per delta, with logprobs as `token_id:N` strings (`:396-545`), and sends final usage when `include_usage` is set. Cached tokens are reported only with `--enable-prompt-tokens-details`.
  - A bare image placeholder sent without image data is accepted silently (`out/proto_bench.json` bare_placeholder…), so a gateway must always send `content_parts` when the prompt has image placeholders.
- **`/v1/completions`:** accepts token-ID prompts and `return_token_ids`, but no multimodal input.
- **Image spans:** vLLM expands each image to N copies of 129264 and prepends 0–1 alignment pads with ID 129265 (`models/deepseek_v4_1/common/mm_preprocess.py:16-23, 376-463`), a deliberate vLLM-side change.

**Recipe patches:** `docker/patch/chat_encoder_1017.py` is the only protocol-relevant patch. It changes the effort budgets and validation (so `xhigh` is rejected), the last-user index, and tool and image merges. No patch touches the Anthropic, Responses or OpenAI handlers.

### 3. Divergence table (full)

Evidence files are in `harness/protocols/out/`. **render** means `render_compare.md` (or `_extra.md`), with both sides' prompts in `upstream_render*.json` and `ours_render*.json`. **respond** means `respond_compare.md`. **http** means `http_probe_scaleout0.json`.

| ID | Behaviour | API (deepseek-recipe) | Ours | Impact | Evidence | Proposed fix (where) | Risk |
|---|---|---|---|---|---|---|---|
| P01 | Chat `thinking:{type}` | Decides thinking (`reasoning.rs:63-71`) | Silently ignored | High: DeepSeek-doc clients never get thinking | render `chat_thinking_field_enabled`: API 37 tok `…Reasoning Effort: 75…<｜Assistant｜><think>`, ours 11 tok `…<｜Assistant｜></think>` | B. C: patch `build_chat_params` (`chat_completion/protocol.py:570`) to map `model_extra["thinking"]["type"]` to `enable_thinking` | Low |
| P02 | Effort aliases | minimal→50, medium→75, xhigh→75 | 400 "…reasoning_effort must be low, high, max…" for Chat, Messages `output_config.effort` and Responses `reasoning.effort` | High: `medium` is the OpenAI SDK/Codex default | http `chat_effort_medium`; render_extra `msg_effort_medium`, `msg_thinking_adaptive_effort_medium`, `resp_effort_medium`, `resp_effort_xhigh` | C: alias map inside the existing chat_encoder_1017 rewrite of `tokenizers/deepseek_v41.py:75-88`. Or B | Low |
| P03 | Reasoning field and logprobs split | `reasoning_content`; logprobs `{content, reasoning_content}` without markers | `reasoning`; one `content` logprobs list that includes `</think>` and EOS | High: DeepSeek clients read and pass back `reasoning_content` | respond `chat_think_full`; `ours_respond_logprobs.json` (18 entries) vs `upstream_respond_logprobs.json` (1 + 15) | B. C: serializer alias in ChatMessage/DeltaMessage (keep `reasoning` too) plus logprob split in the parser | Med (vLLM-native clients) |
| P04 | Prefix continuation / trailing assistant message | `prefix:true` continues without EOS; a non-prefix trailing assistant gets a new `<｜Assistant｜></think>` turn | Ends with `…<｜end▁of▁sentence｜>` and no header; `continue_final_message` is ignored by the encoder | Med: the model generates after EOS | render `chat_prefix_continuation`, `chat_prefix_continuation_vllm` | B. C: map `prefix`→`wo_eos` in `_normalize_messages` (the encoder supports `wo_eos`, `deepseek_v41_encoding.py:370, 489`) and set the parser start stage | Med |
| P05 | `tool_choice:"none"` | Drops the tools from the prompt | Renders them (276 vs 11 tokens) | Med | render `chat_tools_choice_none`, `msg_tool_choice_none`; with the flag the IDs are equal (`ours_render_flag_exclude_tools.json`) | **run.sh `--exclude-tools-when-tool-choice-none`** | Low |
| P06 | `required`/named tool choice | Prompt ends `\n\n<｜DSML｜ calls>\n`, only the named tool is kept, parsing starts in tool calls; 400 in thinking mode | Grammar (structural tag) forcing; all tools rendered; thinking accepted | Med: different prompt; grammar plus DSpark | render `chat_tools_required_nothink` (281 vs 276), `chat_tools_named_nothink` (281 vs 326), `chat_tools_required_think`, `msg_tool_choice_tool_think` | B (coordinate with the structured track) | Med |
| P07 | `response_format` | json_object/json_schema render "## Response Format:\n\nYou MUST strictly adhere…" plus the schema; regex is passed through; `json_object` needs the word "json" | Grammar only, no prompt text; `regex` → 400 (schema literal) | Med | render `chat_json_object` (36/12), `chat_json_schema` (54/11), `resp_text_json_schema` (59/11), `chat_regex`, `chat_json_object_no_word` | B (structured track) | Med |
| P08 | Messages `thinking.type` | Decides thinking; `adaptive` = enabled | No such field (pydantic drops it) | High: Claude Code sends `thinking` | render `msg_thinking_enabled`, `msg_thinking_adaptive` (37 vs 11) | B. C: add the field to `anthropic/protocol.py:121` and map it in `_handle_output_config` | Low |
| P09 | Messages system blocks | Joined "\n\n" | Joined "" | Med: Claude Code sends several system blocks | render `msg_system_blocks`: `You are terse.\n\nUse metric units.` vs `You are terse.Use metric units.` | C: `"\n\n".join` at `anthropic/serving.py:255`. Or B | Low |
| P10 | Messages mid-conversation system message | `<｜System｜>` in place (also counts as a user turn for thinking-drop) | Hoisted into the top system block, "" join | Med-High | render `msg_mid_system` | C: return False from `_detect_merge_inline_system` for deepseek_v41 (`anthropic/serving.py:150`). Or B | Low |
| P11 | Messages `tool_result` and deferred tools | "\n\n" join; images inside `<tool_result>`; `tool_reference` becomes text plus deferred tool declarations | "\n" join; images moved out of the tool result; `tool_reference` → 400 | Med (Claude Code tool search) | render `msg_tool_roundtrip`, `msg_tool_result_image`, `msg_defer_loading`; http `messages_tool_reference` | B (C partial: join and image placement) | Med |
| P12 | Messages thinking history | Joined "\n\n" | Joined "" | Low-Med | render `msg_thinking_history` | C: `anthropic/serving.py:329`. Or B | Low |
| P13 | Messages `tool_choice:any` | Treated as `auto` | `required` plus grammar | Med | render `msg_tool_choice_any` (same prompt; ours `structural_tag`) | C: `anthropic/serving.py:549`. Or B | Low |
| P14 | Messages limits and blocks | `document` → text placeholder; `max_tokens` optional; ≤16 stops; `max_tokens` clamped to 393216 | `document` → 400; `max_tokens` required; ≤4 stops; 500000 → 400 | Low-Med | render `msg_document_block`, `msg_no_max_tokens`, `msg_stop_17`, `msg_max_tokens_500k` | B; `VLLM_MAX_STOP_STRINGS=16` | Low |
| P15 | Responses text parts | Accepted, joined "\n\n" | 400 "DeepSeek V4.1 supports text and image content only; got 'input_text'" (also `function_call_output` parts) | **High**: standard SDK/Codex shape | render `resp_input_text_parts`, `resp_function_output_parts`, `resp_input_image_detail`; http `responses_input_text` | C: treat `input_text`/`output_text` as text in `_normalize_messages` (`deepseek_v41.py:36-45`). Or B | Low |
| P16 | Developer role | Responses: System (V4.1); Chat: role rejected | `<｜User｜>` (`deepseek_v41_encoding.py:402`) | Med: Codex sends developer messages | render `resp_instructions_developer`, `chat_developer_role` | C (render track: the reference encoder has no developer role). Or B | Low |
| P17 | Responses tools | `apply_patch` function tool (+1080 tok description) emitted as `custom_tool_call`; client `tool_search`; namespace `ns::f` plus description | Custom tools and `tool_search` dropped; `ns__f` with no description | High (Codex) | render `resp_apply_patch` (1090/10), `resp_tool_search_client` (311/274), `resp_namespace_tool` | B | — |
| P18 | `encrypted_content` on a reasoning item | Ignored | 400 "Encrypted content is not supported." | Med | render `resp_reasoning_item_encrypted`; http | C: remove the raise at `responses/utils.py:275-276`. Or B | Low |
| P19 | Responses history details | All parts kept; calls sorted by call_id; `detail` optional; `top_logprobs` enables logprobs | First part only; order as sent; `detail` required (pydantic 400); logprobs only with `include` | Low-Med | render `resp_assistant_multi_parts`, `resp_parallel_calls_unsorted`, `resp_input_image`, `resp_top_logprobs` | B | — |
| P20 | Responses/Messages output details | `content_part.added` for reasoning; `phase`; `encrypted_content:"{id}-0"`; message `incomplete` on length; echoes `tool_choice:"auto"` and text format; Messages `ping`, `stop_sequence:null`, cache usage fields, `signature:""` at block start | `reasoning_part.added/done`; `phase:null`; `encrypted_content:null`; message `completed`; `tool_choice:"none"`, `text:null`; no `ping`, `stop_sequence` omitted, no cache fields; random signature | Low | respond `resp_think_stream`, `resp_text_full`, `resp_length_full`, `msg_*` | B | — |
| P21 | Usage | Usage always in the final chunk; `usage:null` on the others with `include_usage`; cache hit/miss keys; EOS not counted | Usage only with `include_usage`, as a separate chunk; EOS counted (+1) | Low-Med | respond `chat_text_stream`, `chat_text_stream_usage` | B. Partial: `--enable-prompt-tokens-details`, `--enable-force-include-usage` | Low |
| P22 | Validation | Rejects n>1, top_logprobs>20, max_tokens>393216, empty messages, `json_object` without "json" | Accepts them (an empty `messages` renders just BOS) | Low | render `chat_n2`, `chat_logprobs_top21`, `chat_max_tokens_400k`, `chat_empty_messages`, `chat_json_object_no_word` | B | — |
| P23 | Errors and model names | Single OpenAI-style shape with DeepSeek texts; unknown model → 400 | Integer `code`, type `BadRequestError`, pydantic dumps; `/v1/messages` mixes shapes; unknown model → 404; missing model accepted; context-overflow text differs | Low | http `chat_model_deepseek_chat`, `messages_no_max_tokens`, `chat_context_overflow` | B; optional extra `--served-model-name` aliases | Low |
| P24 | Chat billing header and `latest_reminder` | Strips the billing line; `latest_reminder` supported | Keeps it (38 vs 16 tokens); 400 "Invalid role: latest_reminder" | Low | render `chat_system_billing_header`, `chat_latest_reminder` | B (C small) | Low |
| P25 | Others (Low) | Tool IDs `call_00_…`; content before a tool call trimmed of "\n\n"; `top_k` ignored | `chatcmpl-tool-…`; trailing "\n\n" kept; `top_k` honored | Low | respond `chat_tool_full`, `msg_tool_stream`; render `msg_temperature_top_k` | B (parsing and sampling tracks) | — |

### 4. Already matching (no need to re-check)

Prompt token IDs are identical, ignoring vLLM's image-span expansion and pads, for:
- **Chat:** `chat_plain`, `thinking:disabled`, effort `none`/`low`/`high`/`max` (budgets 50/75/100), `chat_history_reasoning_content` (thinking + tools history), `chat_tools_auto`, `chat_tool_content_parts`, `chat_image`.
- **Messages:** `msg_plain`, `msg_system_string`, `msg_system_billing_block` (first block), `msg_user_multi_text_blocks` (Claude Code shape), `msg_effort_high`/`max`, `msg_tool_roundtrip_nonascii`, `msg_user_image`, `msg_temperature_top_k`.
- **Responses:** `resp_string`, `resp_typed_message_string`, `resp_default_thinking`, `resp_effort_high`, `resp_function_roundtrip`, `resp_reasoning_item`, `resp_web_search_tool` (dropped on both), `resp_top_logprobs` (prompt only).

Output side (respond): parsed reasoning, content and tool arguments; `finish_reason` stop/length/tool_calls; Messages `stop_reason` mapping and block sequence with `signature_delta`; Responses item types and streaming order, `incomplete_details.max_output_tokens`.

Image token counts: API and vLLM agree for 8/8 sizes (`out/proto_images.json`). Basic validation that both sides perform: top_logprobs without logprobs, stream_options without stream (texts differ).

### 5. Integration architectures

| | A1: in-process source rewrites of the vLLM handlers | A2: in-process endpoint plugin (same code as B) | B: gateway sidecar → `/inference/v1/generate` | C: targeted fixes, deepseek-recipe as test oracle |
|---|---|---|---|---|
| Protocols | All three, exact | All three, exact | All three, exact (22/22 prototype) | Fix by fix; output schema (`reasoning_content`, usage, events) needs vLLM model changes |
| Images | Yes (renderer with token prompt + MM data) | Yes (`serve_tokens` `content_parts`) | Yes (`content_parts`; counts equal 8/8) | Unchanged |
| Logprobs | Yes | Yes | Yes (`token_id:N` → IDs) | The split needs parser work |
| Prep latency, 74K-token request | ~20 ms | ~20 ms + 4.5 ms validation | ~20 ms + ~10 ms hop (434 KB JSON) | ~110 ms (today) |
| Per-token cost | 0.45 µs | 0.45 µs | + SSE parse per chunk (≪ 16–40 ms per token) | Today |
| DSpark / prefix-cache patches | Engine-side, unaffected | Same | Same; required tool choice uses a prompt prefill, not a grammar | Grammar-forced tools stay |
| Effort | Medium-high | Medium | Medium (≈400–600 LOC + proxy + tests) | High (25+ rewrites across 6+ files) |
| Risk | Couples to serving internals; a bug takes the serve down | Young plugin API; routes attach after core routes, so shadowing needs reordering | Extra process (~110 MB RSS measured, ~150 MB with uvicorn) on tight spark1; scale-out env | Anchors drift; parity approximate |
| Drift | High | Medium | Low (HTTP contract plus pinned wheel) | High |
| Ops | Restart reloads the 97 GB model | Same | Gateway restarts and upgrades without touching vLLM | Same as today |

Facts that shape this:
- The API server runs only on the head, so the wheel (20 MB native + 11 MB OpenCV, abi3 aarch64, no runtime deps) is needed only there. It can be mounted read-only like docker/patch; no image rebuild and no 23.7 GB transfer to spark2.
- The image already has fastapi, uvicorn and httpx for a gateway.
- The repo's own tools (`quality_eval.py`, smoke, bench) use `chat_template_kwargs` and `/v1/completions`. The gateway must proxy unknown paths and shim `chat_template_kwargs.{thinking, reasoning_effort}` to the DeepSeek fields, or those tools must call vLLM's port directly.

Gateway contract details (implemented in `gateway_proto.py`):
- `ConversionOptions(default_thinking_mode=False)` keeps the recipe's thinking-off default; register the alias and `SpecialTokens(image="<｜deepseek_image｜>")`.
- `resolve_max_tokens(len(ids) + image adjustment, 500000)`; leave a margin of one token per image for vLLM's pads.
- Do not send stop strings to vLLM (it would also stop inside reasoning). Abort by closing the stream when the processor finishes.
- Drop the trailing EOS ID on a `stop` finish.
- Send images as data URLs; prefer the API-preprocessed bytes so vLLM never fetches URLs. Install a DestinationFilter.
- Optionally send `detokenize:false`.

### 6. Prototype results

- **Parity:** `compare_gateway.py` gives prompt IDs equal to the API's own encode and normalized responses equal, 22/22: chat full/stream/usage, thinking, tools, thinking+tools, logprobs, length; Messages text/stream/thinking/tool/length; Responses text/stream/thinking/tool/length.
- **Cost** (`proto_bench.json`, 4 CPUs, 74,473-token agentic request):

  | Step | Chat | Messages | Responses |
  |---|---|---|---|
  | API convert | 1.19 ms | 1.33 ms | 1.16 ms |
  | API encode | 20.4 ms | 16.6 ms | 16.6 ms |
  | Ours (current path) | 107.6 ms | 124.5 ms | 110.3 ms |

  The B hop: JSON dumps 2.3 ms, loads 2.6 ms, pydantic GenerateRequest 4.5 ms. StreamProcessor: 0.89 ms per 2000 tokens. On the host (20 cores), memory was 67 MB after loading the tokenizer and 112 MB after the 74K encode, which took 10.9 ms (`mem_probe.py`).
- **Images:** for 8 sizes from 32×24 to 4000×3000 and 64×4000, the API token count equals vLLM's span for both the original and the API-preprocessed WebP (`proto_images.json`).

### 7. Open questions / not verified

1. Does DeepSeek's production API enforce OutputConstraint (JSON/strict/required) by constrained decoding? The library leaves it to the caller. The token-in path has no reasoning-parser kwargs, so a grammar applied in thinking mode would also constrain the reasoning.
2. Places where the API library and the reference encoder disagree (render track to decide):
   - **Developer role:** API renders System; the reference encoder has no such role; ours renders User.
   - **Trailing non-prefix assistant message:** the API adds a new `<｜Assistant｜>` turn; the reference and ours end at EOS.
   - **Chat `file` part:** the API expects flat `file_data`; OpenAI nests it under `file`; ours doesn't support the part at all.
3. vLLM's token stream reports cached tokens only in the final usage chunk, so message_start and response.created cannot carry cache hits without a small ServingTokens change.
4. The top-k logprob list from `_create_tokens_logprobs` (`serving.py:506-545`) takes the first N dictionary entries, so the sampled token may take one slot. Not checked against a real engine.
5. Not tested live: HTTP cancellation timing, backpressure, real DSpark multi-token deltas (the mock used 1–4-token chunks), `detokenize:false`, and whether spark1 memory has room for the gateway.
6. Pixel parity when vLLM is handed API-preprocessed images (token counts match; pixels are the vision track's call).
7. Harness note: `vllm serve` copies `--reasoning-parser` into `args.structured_outputs_config` when it builds the engine config, and the harness replicates that step. Without it the reasoning split disappears; that is a harness artifact, not a serve bug.
