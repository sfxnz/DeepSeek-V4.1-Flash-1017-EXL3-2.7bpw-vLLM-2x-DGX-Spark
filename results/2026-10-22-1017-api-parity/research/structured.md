# Track `structured`: summary (the full report follows the summary)

**Scope.** I compared `response_format` (json_object, json_schema, regex), `tool_choice` (auto, required, named, none), strict tools, grammar backends, the reasoning gate, and how all of this interacts with DSpark.
- Upstream reference: deepseek-recipe 0.2.0 @935817c, cross-checked against HF `encoding/encoding.py`.
- Our side: the vLLM in the e16-rows image plus the run.sh flags. No recipe patch touches these code paths: there are no `dsv41` marks in parser/, tool_parsers/, v1/structured_output/, renderers/ or entrypoints/openai/. The only baked rewrite is the 1017 chat encoder, and it does not touch `response_format` or `tool_choice`.

**Bottom line.** Grammar enforcement itself works:
- The backend is xgrammar.
- `--reasoning-parser deepseek_v41` gates the grammar so it applies only after `</think>`. In non-thinking mode the prompt already ends in `</think>`, so it applies from the first token.
- It works with DSpark (CPU replay at k=5 below, and the live D4 json_schema check passed 7/7).

Six problems change output quality or client-visible behaviour:
1. The model never sees the response format.
2. `response_format` together with auto tool calling makes tool calls impossible.
3. Strict tool arguments are not constrained.
4. Zero-argument tool calls are forced off the trained format.
5. The tool prompt differs for strict, none, named and bare tools.
6. The default "auto" backend can break tool calling until restart.

| id | behaviour | API (deepseek-recipe) | ours | impact |
|---|---|---|---|---|
| S1 | json_object / json_schema in the prompt | adds `## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}` to the first system message (dsv41.rs:253-284) | not rendered. The renderer drops ChatParams.response_format (renderers/deepseek_v4.py:48-52, params.py:144-149), although the encoder supports it (deepseek_v41_encoding.py:367,397-400) | HIGH: the model fills a grammar it cannot see (field descriptions and meaning invisible) |
| S2 | response_format + tools, tool_choice auto, non-strict tools | tool constraint wins (kind `tool_call`); the format is only rendered in the prompt (output_constraint.rs:34-51) | JSON grammar on the whole answer (jump-forward `{`), so **no tool call is possible** (abstract_parser.py:545-581 only drops response_format when it builds a structural tag) | HIGH for agent clients that send both |
| S3 | strict tools: argument schema | constraint carries the normalized schema: every property required, nothing extra (validation.rs:209-240) | image builder constrains tool names and DSML syntax only (structural_tag_registry.py:247-250 TODO) | MED: unknown, missing, mistyped or enum-violating arguments pass |
| S4 | zero-argument call under any DSML grammar | trained/rendered form is `">\n\n</｜DSML｜ invoke>` (dsv41.rs:115; HF encoding.py identical) | builder forbids the blank line (`invoke` = Star(param), registry.py:285-289) | MED: forced off-distribution token; for non-strict tools the only alternatives are closing or inventing a parameter |
| S5 | tool rendering for strict / bare tools | `{name, description or "", parameters or {}}`; strict parameters normalized (key order, all required, extra keywords dropped) | `"strict": true/false` leaks into the prompt, `"description": null`, `"parameters": null`, strict parameters not normalized | LOW-MED: prompt drift |
| S6 | named tool_choice | prompt lists only the named tool; ≥1 call of it allowed | prompt lists all tools; grammar allows exactly one call | LOW |
| S7 | tool_choice none | tools dropped from the prompt (convert.rs:259-261) | tools still rendered (`--exclude-tools-when-tool-choice-none` is off); a DSML block the model emits is stripped from the content | LOW-MED |
| S8 | required / named prefill | prompt ends `</think>\n\n<｜DSML｜ calls>\n` (dsv41.rs:307-310) | no prefill; the grammar forces the same text (jump-forward `\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="`) | LOW: same text, about 5 extra forced tokens |
| S9 | strict json_schema | constrained with the normalized schema (all properties required); partial `required` gives 400 | schema passed as written, `strict` ignored: optional properties may be omitted | LOW-MED |
| S10 | request validation | 400 on: json_object without the word "json", mixed strict/non-strict tools, response_format + strict tools, required/named in thinking mode, strict-schema errors. Accepts `{"type":"regex"}` and json_schema without `name` | the reverse in each case: we accept the first group and return 400 for regex and for a missing `name` | LOW (client-visible) |
| S11 | backend "auto" (run.sh sets none) | n/a | the engine's single backend is fixed by the first structured request (structured_output/__init__.py:125-165). A schema with `multipleOf` or pattern+maxLength routes to guidance; if that is the first request after boot, every later required / named / auto+strict request fails (`KeyError: 'triggers'`, backend_guidance.py:267) until restart | MED robustness |
| S12 | json_object grammar | constraint "any JSON" (empty schema) | `{"type":"object"}` only (backend_xgrammar.py:89-92) | LOW |

**Already matching:**
- The grammar is gated by `</think>`.
- DSML tag names and layout for calls with at least one argument: upstream-rendered blocks are accepted, including Python-style JSON values and multi-line strings.
- Tool names are constrained.
- required gives ≥1 call, and text after the calls block is rejected.
- Structured outputs work with DSpark plus the adaptive verify patch.

**Top recommendations**
1. **New `docker/patch` source rewrite (prompt side, fixes S1, S5, S6 and S7's prompt part).**
   - `renderers/deepseek_v4.py` passes `response_format` and `tool_choice` from ChatParams into `apply_chat_template`.
   - `tokenizers/deepseek_v41.py` attaches the response format and the tools to messages[0] (inserting an empty system message if needed), renders tools as deepseek-recipe does (strict ones via my normalization port), and keeps only the named tool.
   - Result: byte-identical to deepseek-recipe on all 29 comparable matrix rows (5 of them after stripping the required prefill), versus 4/29 today.
2. **run.sh flags.** Add `--exclude-tools-when-tool-choice-none` (S7) and `--structured-outputs-config '{"backend":"xgrammar"}'` (S11; `multipleOf` and pattern+maxLength schemas then get a clean 400 instead of poisoning the engine).
3. **Request-side constraint shim around `Parser.adjust_request`, which runs after rendering.**
   - The tool constraint wins over response_format (S2).
   - The upstream 400s for mixed strict tools and for response_format + strict tools.
   - strict json_schema is normalized for the grammar (S9).
   - The prototype reproduces upstream's constraint choice and 400 texts on all 10 affected cases.
4. **Replace the deepseek_v41 structural-tag builder with the schema-aware prototype (S3, S4).**
   - It accepts all 7 upstream-rendered valid samples, including the zero-argument call, `$ref` into `$defs`, and a `["string","null"]` union.
   - It rejects all 14 malformed ones.
   - It compiles in ≤24 ms.
   - Strict tools get their parameters in declared order (xgrammar's own convention).
5. **Optional client-compat shim (S10):** map `response_format {"type":"regex"}` to `structured_outputs.regex`, and default a missing json_schema `name`.
6. **After landing,** run a boot smoke with DSpark on at c=1 and c=4: tool_choice required, named and auto+strict, plus json_schema in thinking mode. Check for no `grammar rejected tokens` errors, and record acceptance and time per output token.
7. **Guard:** never set `VLLM_ENFORCE_STRICT_TOOL_CALLING=0`. With it off, required/named fall back to vLLM's JSON-array schema (`[{"name":..,"parameters":..}]`) instead of DSML.

**Paths**
- Harness: `/home/sfxnz/projects/data/dsv41-1017/api-parity/harness/structured/`. Everything re-runs with `run_all.sh`.
  - Prototypes: `dsml_grammar.py` (builder) and `strict_schema.py` (normalization port, 13/13 parity with upstream).
  - Checks: `prompt_fix_check.py`, `adjust_fix_check.py`, `grammar_probe.py`, `spec_grammar_sim.py`, `backend_pinning_check.py`, `served_matrix.py`, `upstream_matrix.py`, `upstream_dsml_samples.py`, `hf_dsml_check.py`, `upstream_strict_norm.py`, `cases.py`.
  - Each has a `*.log` beside it.
- Data: `/home/sfxnz/projects/data/dsv41-1017/api-parity/findings/structured/` contains `upstream_matrix.json`, `served_matrix.json`, `dsml_samples.json`, `grammar_probe.json`, `prompt_fix_check.json`, `spec_grammar_sim.json` and `upstream_strict_norm.txt`. As instructed, there is no REPORT.md; the full report is below.

---

# Full report

## 1. What upstream does (deepseek-recipe 0.2.0 @935817c)

**Conversion: Chat Completions** (`deepseek-recipe/src/protocol/openai/chat_completion/request/convert.rs`)
- json_object requires the word "json" somewhere in the messages, case-insensitive (`:83-91`; `validation.rs:268-283`). json_object with a prefix message gives 400 (`:93-97`).
- `convert_tools` (`:254-283`): tool_choice `none` drops every tool, so there is no tool prompt and no constraint.
- `convert_client_tools` (`client_tools.rs:63-105`):
  - Normalizes each tool's schema (`validation.rs:209-240`). Strict tools get `normalize_strict_schema`; non-strict ones are validated with `jsonschema` and must be `type: object`.
  - A named choice keeps only that tool's definition (`:84-90`).
  - Named and required both become `ToolChoice::Required` and are rejected in thinking mode with "Thinking mode does not support this tool_choice" (`:94-98`).
- Response format (`:108-128`):
  - Request types: `json_object`, `json_schema{schema, name?, strict?}`, `regex{regex}`, `text` (`schema.rs:171-189`; `name` is optional).
  - A strict json_schema is normalized for the constraint but rendered as written (`validation.rs:249-264`).
- `resolve_output_constraint` (`output_constraint.rs:28-60`):
  - With tools and tool_choice ≠ none, the constraint is `ToolCalls{tools, required}`.
  - Errors: mixing strict and non-strict tools gives 400 "Cannot use strict tool with non-strict tool in the same request". All-strict tools plus any response_format (even `text`) gives 400 "response_format should not be used with tools".
  - Otherwise: json_object gives `Json{None}`, json_schema gives `Json{schema}`, regex gives `Regex`, text gives no constraint.
- `OutputConstraint` (`request/mod.rs:62-126`):
  - `kind()` is one of `json`, `regex`, `tool_call`, `tool_call_required`.
  - `schema()` is compact JSON of the schema, the regex, or the tool list as Chat Completions function tools (with `strict`). For json_object it is an empty string ("any JSON value").
  - Whether a backend enforces it "is the caller's decision". The real dumps are in `upstream_matrix.json`. Example (strict tools): `[{"type":"function","function":{"name":"get_weather",...,"parameters":{...,"additionalProperties":false,"required":["city","unit","days"]},"strict":true}},...]`.
- Parsing options (`convert.rs:150-161`): `starts_in_tool_calls` is set for Required (the prompt opened the block). `parse_json_output` is set for json_object only: it extracts fenced or bare JSON and turns the surrounding text into spaces (`stream/output_parser.rs:36-56,105-125`).

**Strict normalization** (`json_schema.rs:16-287`)
- Uses a typed schema, so only retained keywords survive:
  - object: properties, additionalProperties, required, description, `$def`
  - string: description, pattern, format ∈ {email, hostname, ipv4, ipv6, uuid}, enum
  - number/integer: description, const, default, minimum/maximum, exclusiveMinimum/exclusiveMaximum, multipleOf
  - array: items, description
  - Key order is canonicalized.
- Every object gets `required` = all of its properties and `additionalProperties: false`.
- 400 cases: a partial `required` list, `additionalProperties: true`, a nested empty object, a node with no type, or an unknown format (e.g. `date-time`).
- **Quirk:** `$defs` and `definitions` are dropped (only `$def` is kept), leaving a dangling `$ref` (`:65-69`).
- `upstream_strict_norm.txt` shows each case. My port in `strict_schema.py` reproduces all 13 probes (`PORT parity 13/13`).

**V4.1 rendering** (`deepseek-recipe-encoding/src/v4/dsv41.rs`)
- The tool prompt (`:69-88`; `mod.rs:46-52,56-90`) lists each tool as `{"name","description" or "","parameters"}` with Python-style separators. Strict tools carry their normalized parameters; no `strict` key is rendered.
- The response format (`:253-284`): json_object renders `{"type": "json_object"}`, json_schema renders the schema as written, and regex/text render nothing. It is appended after the tool prompt to messages[0] if that is a system message; otherwise an empty system message is inserted at index 0.
- `tool_choice == Required` with a tool prompt appends `\n\n<｜DSML｜ calls>\n` after `<｜Assistant｜></think>` (`:307-310`).
- DSML tag names are `" calls"`, `" invoke"` and `" parameter"`, with a leading space (`:17-19`). A zero-argument call renders as `<｜DSML｜ invoke name="f">\n\n</｜DSML｜ invoke>` (`:113-116`). `hf_dsml_check.log` confirms HF encoding.py renders the same thing.

**V4.1-specific:** the leading-space DSML tags, mid-conversation system messages, and the numeric effort line (thinking mode adds `Reasoning Effort: NN` before the response-format block).

**Where HF encoding.py and deepseek-recipe disagree:** HF `tools_from_openai_format`/`render_tools` dump the function dict as given (encoding.py:84-99,509-520). It would therefore render a client's `"strict": true` and omit a missing description, where deepseek-recipe renders `{name, description or "", parameters}` with normalized strict parameters. The DSML and response-format templates are identical.

## 2. What our serve does (image vLLM 0.1.dev20904+g179dd0fa9; run.sh:657-667)

**Flags.** run.sh passes `--tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41`. There is no `--structured-outputs-config` and no `--exclude-tools-when-tool-choice-none`. The image environment does not set `VLLM_ENFORCE_STRICT_TOOL_CALLING`, which defaults to True (envs.py:1777-1779).

**Request types** (`entrypoints/generate/base/protocol.py:70-145`)
- `ResponseFormat.type` must be one of text, json_object or json_schema, so regex gets a 400 pydantic literal error. `JsonSchemaResponseFormat.name` is required.
- json_object becomes `StructuredOutputsParams(json_object=True)`, which compiles as `{"type":"object"}` (backend_xgrammar.py:89-92).
- json_schema becomes `json=schema`. The `strict` flag is ignored; xgrammar's default `strict_mode=True` forbids undeclared keys and enforces declared key order.

**Prompt**
- `DeepseekV4Renderer.render_messages` (renderers/deepseek_v4.py:48-52,75-79) calls `apply_chat_template` with `get_apply_chat_template_kwargs()` (renderers/params.py:144-149). That function omits `ChatParams.response_format` and `tool_choice`, even though `build_chat_params` fills them (chat_completion/protocol.py:603-604). Only the Kimi K3 renderer uses them (renderers/kimi_k3.py:168-171).
- `tokenizers/deepseek_v41.py:56-103` attaches `tools` to the first system message found anywhere, which may be a mid-conversation one (`:65-70`), and never sets `response_format`.
- Tool dicts come from `FunctionDefinition.model_dump`, which keeps `description: None` and `parameters: None` and keeps `strict` unless it is None (generate/base/protocol.py:224-236).
- With tool_choice none, `tool_dicts` is still passed (online_renderer.py:175-179; the flag defaults to False, launchers/cli_args.py:108).

**Constraint**
- `ParserManager.get_parser` builds a `DelegatingParser` (parser_manager.py:75-160). After rendering, `adjust_request` runs (online_renderer.py:442-475).
- `_apply_structural_tag` (abstract_parser.py:545-581): for auto, required or named tool_choice, it calls `get_model_structural_tag("deepseek_v41", ...)` (tool_parsers/structural_tag_registry.py:107-153). That returns None for auto unless some tool is strict (`:119-120`); otherwise it builds the vLLM builder at `:236-320` (names and syntax only).
- When a tag is built, `response_format` is set to None (`:578-580`). With no tag, the JSON fallback (`abstract_tool_parser.py:118-165`) only applies to required/named. Because `structural_tag_model` is set and the environment flag is on, `supports_required_and_named=False` (`:64-70`), so required/named output goes through the DSML auto parser.
- Backend "auto" picks xgrammar per request, falling back to guidance or outlines (sampling_params.py:1176-1221). The engine uses one backend, chosen from the first structured request (v1/structured_output/__init__.py:125-165). The guidance backend only reads legacy structural tags (backend_guidance.py:262-283).

**Reasoning gate**
- `--reasoning-parser` sets `structured_outputs_config.reasoning_parser` (engine/arg_utils.py:2521-2523), with `enable_in_reasoning=False` (config/structured_outputs.py:41).
- The serving layer passes `reasoning_ended = parser.is_reasoning_end(prompt_ids)` (chat_completion/serving.py:356-371). The engine gates the bitmask with `should_fill_bitmask`, `should_advance` and `trim_reasoning_for_advance` (structured_output/__init__.py:370-499).
- `ParserEngine.is_reasoning_end` scans backwards for `</think>` (128822) versus `<think>` (128821) (parser/engine/parser_engine.py:611-629).
- Non-thinking prompts end in `</think>`, so the grammar is on from the first token. Thinking prompts end in `<think>`, so the grammar starts after the model's `</think>`. This is `served_matrix.log` `gate_open=True/False`.

**DSpark**
- The scheduler grammar-validates drafts (`validate_tokens`) and pads the rejected ones with -1 (scheduler.py:2359-2389). `grammar_bitmask` builds k+1 rows per request, unconstrained until `</think>` falls inside the draft window (structured_output/__init__.py:280-359).
- MRv2 maps rows by (request, position) with `mask_stride=decode_query_len` (v1/worker/gpu/structured_outputs.py:13-120; model_runner.py:467-473,1440-1447). `max_rollback_tokens` is the DSpark k_max=5 (backend_xgrammar.py:129-133).
- Masked target logits give p(invalid draft)=0, so such drafts are always rejected. The recipe's `dspark_adaptive.py` (-1 drafts past the confidence cut; shapes unchanged) is compatible: row m is the grammar state after the m validated drafts.

## 3. Divergences (evidence and fixes)

**S1: response format absent from the prompt. HIGH (json_schema), MED (json_object).**
- Evidence (`prompt_fix_check.log`): `json_schema/chat served==upstream False, fixed==upstream True`.
  - Upstream: `<｜System｜>\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{"type": "object", "properties": {"name": {"type": "string", "description": "Full name"}, ...}<｜User｜>...`
  - Ours: `<｜begin▁of▁sentence｜><｜User｜>Extract the person: ...<｜Assistant｜></think>`
- Fix: a source rewrite of renderers/deepseek_v4.py (forward `response_format` and `tool_choice`) and tokenizers/deepseek_v41.py; see `fixed_apply_chat_template` in prompt_fix_check.py. Put it in a new `docker/patch/*.py`, or fold it into chat_encoder_1017.py.
- Risk: low. Only requests with a response_format, tools, or a mid-conversation system message change prompt.
- Note: my fix also moves tools from a mid-conversation system message to messages[0] (`mid_system_tools` now matches upstream). This overlaps the `render` track.

**S2: response_format + tools (auto, non-strict) blocks tool calls. HIGH.**
- Evidence (`served_matrix.log`): `rf_plus_tools/chat so=JSON:'{"type": "object", ...' jf='{'`. Upstream: `rf_plus_tools/chat kind=tool_call`.
- Fix: a shim before stock `adjust_request` (`adjust_fix_check.py`): drop response_format from the grammar when tools are active, and keep it in the prompt (S1). Raise `VLLMValidationError` for upstream's two 400s.
- Result (`adjust_fix_check.log`): `rf_plus_tools JSON -> none`, `rf_plus_strict_tools STRUCTURAL_TAG -> 400 response_format should not be used with tools`, `tools_auto_mixed -> 400 Cannot use strict tool with non-strict tool...`.
- Risk: low.

**S3: strict tool arguments unconstrained. MED.**
- Evidence (`grammar_probe.log`): `cur_required_strict` and `cur_auto_strict` accept `unknown_param`, `int_as_string`, `int_out_of_range`, `enum_violation`, `missing_param_strict`, `duplicate_param`, `wrong_order_strict` and `nested_missing_key_strict`. The `new_*` grammars reject all of them.
- Fix: `dsml_grammar.get_deepseek_v41_structural_tag`, registered as the deepseek_v41 builder through a source rewrite of structural_tag_registry.py or re-registration in sitecustomize. Strict tools are normalized by `strict_schema.normalize_strict` with `$defs` re-attached.
- Risk: medium. Declared argument order is enforced, so an eval boot is needed.

**S4: zero-argument calls. MED.**
- Evidence: `valid/noarg` is rejected by every `cur_*` grammar (`REJECT@12:' name="get_time'+'">\n\n'`) and accepted by every `new_*` grammar. `hf_dsml_check.log`: `noarg: HF == deepseek-recipe: True '...get_time">\n\n</｜DSML｜ invoke>...'`.
- Fix: the same builder (it accepts both the blank-line and compact forms).
- Risk: low.

**S5: tool rendering. LOW-MED.**
- Evidence (`tools_bare` and `tools_auto_strict`). Served: `{"name": "ping", "description": null, "parameters": null}` and `..."additionalProperties": false}, "strict": true}`. Upstream: `{"name": "ping", "description": "", "parameters": {}}` and `..."additionalProperties": false, "required": ["city", "unit", "days"]}}`.
- Fix: the S1 rewrite (`_upstream_tool` plus `normalize_strict`).

**S6: named tool_choice. LOW.** Served `tools_named` renders web_search too. The fix is the S1 rewrite (filter to the named tool). I kept the one-call grammar; upstream allows several calls of the named tool, so `stop_after_first=False` would be needed to match exactly.

**S7: tool_choice none. LOW-MED.** `tools_none` is `False` before the fix and `True` after. The fix is the run.sh flag `--exclude-tools-when-tool-choice-none` (or the S1 rewrite).

**S8: required prefill generated, not prompted. LOW.**
- Evidence (`spec_grammar_sim.json` C): the rows allow `[2,1,4,6,2,1]` tokens. The forced opener is drafted and accepted as `["\n\n","<","｜DSML｜"," calls",">\n"]`, the same tokens as the upstream prefill (271, 30, 128825, 10699, 1018).
- Recommendation: no change. Prefilling would require the parser to start inside the block.

**S9: strict json_schema normalization. LOW-MED.**
- Evidence: `json_schema_strict` stock `required=None` becomes `['name','age','email']` after the fix. In the probe, `person_no_email` is accepted by `json_strict_served` and rejected by `json_strict_norm`.
- Fix: the S2 shim.

**S10: request validation. LOW.** `served_matrix.log` shows regex as `literal_error ... 'regex'` and a missing name as `json_schema.name Field required`. The upstream 400s are in `upstream_matrix.log` (14 of 46 rows). The fix is an optional pydantic before-validator.

**S11: backend auto pinning. MED.**
- Evidence (`backend_pinning_check.log`): `[auto] multipleOf json_schema -> request backend 'guidance'` and `engine backend guidance: tool_choice=required structural tag FAILS: KeyError: 'triggers'`.
- Fix: run.sh `--structured-outputs-config '{"backend":"xgrammar"}'`, which gives `[xgrammar] multipleOf -> VLLMValidationError` (a clean 400).
- Trade-off: those schemas are accepted by upstream.

**S12: json_object top level. LOW.** `json_array / json_object: REJECT@0`. No action proposed.

## 4. Already matching (no need to re-check)
- The grammar starts after `</think>` in thinking mode and at the first token in non-thinking mode; reasoning tokens are trimmed before the grammar advances (`spec_grammar_sim` A1/A2).
- The image's DSML structural tag accepts every upstream-rendered call that has arguments (`weather_full`, `two_calls`, `multiline_string`), including Python-style JSON values.
- Unknown tool names, unclosed invokes, bad JSON values, and text after the block (required/named) are all rejected.
- Required gives ≥1 call; named is forced to the named tool; auto with strict tools still allows prose.
- Structured outputs with DSpark and adaptive verify: invalid drafts are truncated and masked, and later rows are free (`spec_grammar_sim` B: drafts `["Ada","\"",-1,-1,-1]`, the `Hello` row masked to 136 allowed tokens). Live evidence: worktree `results/2026-10-19-1017-perf2/D4/grammar.txt` shows `GRAMMAR 7/7 valid` (json_schema strict, thinking off, c=1 at k=5 and c=4 at k=3, adaptive on).
- xgrammar compile times are 1-24 ms (`grammar_probe.log`).
- Without response_format or tools, prompts are identical (`text`, `tools_auto`: served==upstream True).

## 5. Open questions / not verified
- What DeepSeek's production backend actually enforces is unknown: `tool_call` with non-strict tools (names only?), argument order for strict tools, and whether json_object allows a top-level array. deepseek-recipe leaves enforcement to the caller.
- Upstream drops `$defs` from strict schemas, leaving a dangling `$ref` in both the prompt and the constraint. Faithful parity would reproduce this upstream bug. My grammar keeps `$defs`; the prompt port reproduces upstream exactly. Decide which to follow.
- The GPU half was not run: MRv2's bitmask kernel and the rejection sampler with structural tags, thinking-mode JSON, or c=4 mixed batches. Recommendation 6 covers this, and should also record CPU bitmask cost per step at k=5.
- Read in code, not reproduced: `include_reasoning=false` in thinking mode makes the serving layer set `reasoning_ended=True` (serving.py:356-357). The grammar would then apply from the first token, even though the prompt ends in `<think>`.
- The Responses API (`text.format`) and Anthropic Messages tool_choice paths were not examined; they belong to the protocols track.
- `parallel_tool_calls=false`: vLLM filters to the first call (serving.py:778,1100); upstream ignores the field. This belongs to the parsing track.
