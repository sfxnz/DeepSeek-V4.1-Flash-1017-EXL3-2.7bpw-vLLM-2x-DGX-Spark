# Parsing track: model output to API response, DeepSeek API (deepseek-recipe 0.2.0) vs our serve

Method. The same generated token IDs went through both stacks. Special tokens are single IDs; plain text is encoded without the added vocabulary, which matches upstream's PromptTokenizer (405 of 405 IDs identical).
(a) Upstream: deepseek_recipe 0.2.0 StreamProcessor, then ChatCompletionChunkGenerator, then ChatCompletionResponse.accumulate.
(b) Ours: the image's real vLLM path. Parser adjust_request, then SamplingParams, then the v1 OutputProcessor (detokenizer and stop strings), then the real OpenAIServingChat stream and full generators on a stub instance. CPU only.
Coverage: 59 cases, each non-stream plus 9 step patterns (1 to 6 tokens, matching DSpark k≤5), plus 240 fuzzed streams and 180 stop-string streams.
No recipe patch touches this path. The only text-path rewrite is chat_encoder_1017, which is render side.

DIVERGENCES (severity = client or quality impact)
- **D1, HIGH. `stop` strings.**
  - API: stops are matched only in the answer content, never in reasoning or inside the DSML tool block.
  - Ours: stops are matched over all detokenized text, including reasoning and DSML markup.
  - Thinking-mode stops like "Observation:", "\n\n" or "</think>" that appear in reasoning end the request there: content is null, the answer is lost, finish_reason is "stop".
  - A stop string inside tool args or markup gives args "{}" (non-stream) or '{"query": "' (stream, invalid JSON) with finish "tool_calls", or no tool call at all.
  - Messages `stop_sequences` behave the same way.
  - Fix P2: answer-only stop window in vllm/v1/engine/detokenizer.py. Prototyped: all 8 stop cases then match the API.
- **D2, HIGH. max_tokens hit inside a tool call.**
  - API: finish "length" with partial args.
  - Ours: finish "tool_calls" with args "{}" (non-stream) or invalid JSON (stream). Clients will execute truncated calls.
  - Fix P3: serving rewrite so "length" wins. Prototyped, works.
- **D3, MED-HIGH. Reasoning field name.**
  - API: `reasoning_content` in both message and delta.
  - Ours: `reasoning` only, verified in stream and non-stream.
  - Input still accepts both spellings: the validator renames `reasoning_content` to `reasoning`, the tokenizer maps it back, and the text reaches the prompt (`<think>MARKER</think>`).
  - DeepSeek-API clients therefore see no reasoning. If they echo back `reasoning_content` (now missing), the tool-loop turn renders `<think></think>` and the next prompt loses the reasoning.
  - Fix P4: also serialize `reasoning_content`. Prototyped, works.
- **D4, MED. Non-stream parsing is text-only.**
  - Plain-text "</think>" or "<think>" in reasoning or content cuts reasoning short, or drops or moves content into reasoning. This happens even with thinking off: content 'Wrap it as  please.' with reasoning 'secret'.
  - Streaming is token-aware and correct, so stream and non-stream disagree.
  - Fix P1b: split at the character offset of the `</think>` token, and run the tool pass with think markers neutralized. Prototyped: fixes all such cases and keeps valid tool JSON.
- **D5, MED-LOW. Whitespace.**
  - API: drops leading newlines of reasoning and content, and the newlines right before `</think>` and before the DSML block. Keeps spaces.
  - Ours: keeps the leading "\n\n", rstrips all trailing reasoning whitespace including spaces, and keeps "\n\n" before tool calls.
  - Echoed history then renders as "…\n\n\n\n<｜DSML｜ calls>". That drifts from the generated tokens and shortens prefix-cache / REPLY_RETAIN hits.
  - Fix P5 (spec below, not prototyped).
- **D6, MED-LOW. Logprobs.**
  - API: separate `content` and `reasoning_content` lists; markup and EOS dropped.
  - Ours: one list that includes reasoning, `</think>`, DSML tags and EOS.
  - Byte-fallback tokens come back as "" with no bytes instead of `\xNN`.
- **D7, LOW-MED. Usage.**
  - completion_tokens is +1 because EOS is counted.
  - Streamed reasoning_tokens drift: 81 of 240 fuzz streams differ. It reports 0 instead of 6 when reasoning and EOS arrive in one DSpark step.
  - Stream usage is sent only with include_usage. The API always puts usage on the final chunk.
  - No prompt_cache_hit/miss_tokens.
- **D8, LOW-MED. Named tool_choice.** API finish "tool_calls", ours "stop". Fixed by P3.
- **D9, LOW. Tool-arg text.**
  - Ours re-serializes `string="false"` values (21.50→21.5, `{"a":1}`→`{"a": 1}`, 1e2→100.0).
  - Ours coerces by schema: "5"→5, 123→"123", "null"→null for nullable strings, invalid JSON→string.
  - API copies the raw text, so invalid JSON is possible there. Results are equal as JSON apart from the coercions.
- **D10, LOW.** Text after the tool block: the API drops it. Ours drops it in non-stream but streams it as content.
- **D11, LOW.** Duplicate parameter: the streamed args stick at invalid '{"query": "a'.
- **D12, LOW. Shapes.** null vs "" for content and empty reasoning; first and final chunk shapes; tool IDs `chatcmpl-tool-*` vs `call_NN_[ET_]…`.
- **D13, LOW. DSML edge cases.**
  - DSML spelled with plain tokens is treated as markup.
  - With tool_choice none or no tools, emitted DSML is stripped by us but returned verbatim by the API.
  - V4 unspaced tags are parsed only by the API.
  - A re-emitted `<think>` is dropped by us.
- **D14, LOW.** A final 6-token step that holds the `</think>` token, a plain "</think>" and EOS mis-binds when streamed.
- JSON mode: fenced output and preambles differ only if the json_object grammar is off. Bare JSON is identical.

MATCHES
- `</think>` split with thinking on and off; EOS or length inside reasoning.
- Spaced DSML grammar: all parameter types, escapes, parallel calls and indices, empty invoke `{}`, a tool block without `</think>`, required choice, unknown names passed through.
- Multi-byte splits with no U+FFFD leaks; grounding tokens kept verbatim.
- Stop in content: 180 of 180 streams match.
- Stream equals non-stream text and tools in 240 of 240 fuzz streams.
- The parser's thinking flag equals the renderer's.

RECOMMENDATIONS (Option A: targeted vLLM source rewrites behind levers)
All four were prototyped as harness monkeypatches and validated on the 59 cases and both fuzzers with no regressions.
1. P2: answer-only stop matching in detokenizer.py. Fixes D1 for chat and Messages.
2. P3: serving.py finish_reason, with "length" winning and named choice reporting "tool_calls". Fixes D2 and D8.
3. P4: add `reasoning_content` next to `reasoning` in ChatMessage and DeltaMessage. Fixes D3.
4. P1b: token-aware non-stream split in DelegatingParser.parse. Fixes D4.
5. Later: P5 whitespace (D5) and a logprobs split (D6). For D7 the cheap levers are `--enable-force-include-usage` and `--enable-prompt-tokens-details`; the protocols track should decide.

Option B (wrap deepseek_recipe's StreamProcessor as the vLLM Parser) was prototyped through the real serving code. Content, reasoning and tool calls match the API in all 59 cases. It is not recommended now:
- Stop handling has to move into the processor, and there is no early abort. Generation ran to EOS: 16 tokens vs 12.
- It needs the wheel plus its bundled OpenCV libs in the image, and a ParserManager hook.
- P2–P4 are still needed for serving-level behaviour.

PATHS
- Harness: /home/sfxnz/projects/data/dsv41-1017/api-parity/harness/parsing/ — run_in_image.sh, common.py, cases.py, run_diff.py, fuzz_stream.py, fuzz_stop.py, reasoning_field_io.py, usage_no_include.py, debug_stop_rnd3.py, proto_patches.py, run_patches.py, fuzz_with_patches.py, proto_recipe_parser.py, run_proto.py.
- Outputs: /home/sfxnz/projects/data/dsv41-1017/api-parity/harness/parsing/out/
  - diff_summary.txt and diff_results.json: the 59-case baseline.
  - patches_p1bp2p3p4/: Option A results. proto/ and proto_stops/: Option B results.
  - fuzz_stream*.json, fuzz_stop*.json, reasoning_field_io.txt.
- run_in_image.sh mounts the whole venv site-packages (including deepseek_recipe.libs) and runs as the host uid with HOME=/tmp.

==================================== FULL REPORT ====================================

## 0. Coordinator items
- **Item 1.** I checked whether any recipe patch touches the output path:
  - No `docker/patch/*.py` or sitecustomize rewrite targets `vllm/parser/*`, `tool_parsers/*`, `reasoning/*`, `entrypoints/openai/chat_completion/*`, `v1/engine/detokenizer.py` or `output_processor.py`.
  - None of those image-src files carries a dsv41 marker.
  - Every harness run used the image itself, with its baked sitecustomize active, so the results show the code clients actually hit.
- **Item 2.** run_in_image.sh mounts the whole `venv/lib/python3.12/site-packages`, which includes `deepseek_recipe.libs`.
- **Item 3.** The upstream side always sends model "DeepSeek-V4.1-Flash".
- **Item 4.** This message is the report.
- **Item 5.** The field name is confirmed (D3). Script: harness/parsing/reasoning_field_io.py. Output: out/reasoning_field_io.txt.
  - Ours, non-stream message keys: ['content','reasoning','role']. Stream delta keys: ['content','reasoning','role'].
  - Upstream, non-stream: {"role","content","reasoning_content"}. Stream delta keys: ['content','reasoning_content','role'].
  - Input: an assistant history message with `reasoning_content` and one with `reasoning` both render `<｜Assistant｜><think>SECRET-REASONING-MARKER</think>\n\n<｜DSML｜ calls>…`. With neither field it renders `<think></think>`.
  - Code path: validator chat_completion/protocol.py:528-546 renames reasoning_content to reasoning; chat_utils.py:1999-2004 keeps both; tokenizers/deepseek_v41.py:29-30 maps reasoning back to reasoning_content for the encoder.

## 1. Upstream (deepseek-recipe @935817c), output to Chat Completions

### Input
- The backend sends token IDs per step as `InferenceChunk::Tokens`.
- The caller must drop the final IDs (EOS) unless the finish reason is Length (stream/inference.rs:58-62).
- Decoder (stream/decoder.rs:160-234):
  - Buffers until the UTF-8 is complete (holds on U+FFFD).
  - Decodes every added-vocabulary token as its own piece and flags it as special. `is_special_token` means any added token (decoder.rs:82-86), so `<think>`, `</think>` and `｜DSML｜` qualify even though tokenizer.json marks them special=false.

### ParsingOptions for Chat (request/convert.rs:150-160)
- `parse_tool_calls` = tools non-empty AND tool_choice != none.
- `starts_in_tool_calls` = tool_choice is required or named. That requires thinking off (client_tools.rs:96), and the prompt then ends with `\n\n<｜DSML｜ calls>\n` (deepseek-recipe-encoding/src/v4/dsv41.rs:307-310).
- `parse_json_output` = json_object only (not json_schema).
- `thinking_initial_stage` = ReasoningStart, or a continuation stage for a `prefix` message (convert.rs:209-230).
- `stop_sequences` = request.stop, at most 16 and none empty (validation.rs:156-183).

### Stage machine (stream/output_parser.rs:390-601)
Markers are matched byte by byte (KMP). A marker's special segments (`</think>`, `｜DSML｜`) match only a special-token piece; its plain segments match only plain pieces (82-94, 365-373). So a spelled-out "</think>" is just text.

- **Reasoning (567-590):**
  - Leading newlines are dropped.
  - `[\n*]<｜DSML｜` goes to ToolCalls: a tool block ends reasoning even without `</think>`.
  - `[\n*]</think>` (the token) goes to Content.
  - Everything else is reasoning, including `<think>` and a plain "</think>".
- **Content (418-464):**
  - Stop sequences have priority.
  - `[\n*]<｜DSML｜` goes to ToolCalls.
  - Leading newlines are dropped; another `</think>` token is skipped.
  - JSON mode: a ```` ```json\n ```` or ```` ```\n ```` fence, or a bare `{`/`[`, enters Json. Text outside the JSON becomes one space per source chunk. In JSON mode, stops match only inside Json (465-479).
- **ToolCalls (480-494):** `invoke name="` starts a call. `</｜DSML｜tool_calls>` or `</｜DSML｜ calls>` goes to Finished. Everything else is skipped, so spaced (V4.1) and unspaced (V4) tags both parse.
- **ToolName** runs up to `"` (495-504).
- **Arguments (505-525):** `parameter name=` emits `{` or `, `. `</｜DSML｜` closes with `}`, or `{}` for an empty invoke.
- **Argument name (526-535):** raw bytes up to the first space, quotes included, followed by `: `.
- **`string=` attribute (536-552):** `true">` opens a `"`; the value is JSON-escaped with serde_json (processor.rs:349-356, 435-440). `false">` copies the value raw, with no validation.
- **Value (553-566):** `</｜DSML｜` closes the string.
- **Finished (591-593):** anything after the tool block or after a stop is dropped.

Result: `{"k": "v", "n": <raw>}`, keys in document order, `, ` and `: ` separators, duplicate keys kept, partial JSON on truncation (processor.rs test at 940).

### Processor (stream/processor.rs:75-232)
- completion_tokens = the tokens of every processed piece, including markup and the whole piece that holds a stop (179).
- A stop ends processing (185-192).
- finish_reason (205-217; schema.rs:143-154):
  - StopSequence → "stop".
  - Stop with a tool call already named at finish → "tool_calls".
  - Length → "length", even mid-call.
  - ContentFilter → "content_filter"; EndOfStream → "stop".
- reasoning_tokens = the tokenizer re-encoding the emitted reasoning text, thinking mode only (218-222; decoder.rs:76-80).

### Chat generator (protocol/openai/chat_completion/response/chunk_generator.rs)
- Start chunk (113-139): `{role, content:null, reasoning_content:""}` in thinking mode, else `{role, content:""}`.
- Content and reasoning deltas carry a null for the other field (135-160).
- Tool start delta (161-185): `{index, id, type, function:{name, arguments:""}}`.
  - id = `call_{idx:02}_[ET_]{20 alnum}{xxh3%10000:04}` (tool_call_id.rs:16-43).
  - `ET_` is added when thinking mode emitted no reasoning.
- Finish chunk (195-215): `{content:"", reasoning_content:null}` plus finish_reason and usage.
  - Usage is on the final chunk even without include_usage.
  - With include_usage, every chunk also carries `usage:null` (86-95).
- Usage fields (schema.rs:206-246): prompt_tokens, completion_tokens, total_tokens, prompt_tokens_details.cached_tokens, completion_tokens_details.reasoning_tokens (thinking only), prompt_cache_hit_tokens, prompt_cache_miss_tokens.
- The non-stream body is the same chunks accumulated (schema.rs:274-345): content is never null ("" at minimum), and empty thinking gives reasoning_content "".

### Logprobs
- Attached per decoded piece and split into `logprobs.content` and `logprobs.reasoning_content`.
- Markup and tool pieces drop their logprobs.
- Token strings come from the byte-level vocabulary, with `\xNN` for partial UTF-8 plus raw bytes (processor.rs:295-367; logprobs.rs:69-110; schema.rs:156-163).

### HF encoding.py (1017) vs deepseek-recipe
HF `parse_message_from_completion_text` (encoding.py:888-990):
- Keeps reasoning and content whitespace verbatim.
- Asserts `</think>` in thinking mode.
- Requires the exact `\n\n<｜DSML｜ calls` start and asserts no text after the block.
- Rejects unspaced V4 tags and duplicate params.
- Splits `ns::name` into a namespace.

deepseek-recipe is lenient and strips newlines (not spaces). The two agree on the HF golden output ("  reason  ", "summary"), and their argument format is identical.

## 2. Our serve (image-src vllm 0.1.dev20904; no recipe patch on this path)

### Configuration
- Flags (run.sh:660-664): `--tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41 --default-chat-template-kwargs '{"thinking":false,"reasoning_effort":"low"}'`.
- ParserManager builds `DelegatingParser(reasoning=DeepSeekV41ParserReasoningAdapter, tool=DeepSeekV41EngineToolParser)` (parser/parser_manager.py:75-138; parser/engine/registered_adapters.py:48-51).
- Config `deepseek_v41_config` (parser/deepseek_v41.py:17-58; parser/deepseek_v4.py:133-218) is literal-terminal based.
  - Only `<think>`/`</think>`, plus BOS/EOS as DROP, are token-ID terminals (streaming_parser_engine.py:39-86, 121-140).
  - DSML tags are matched as text.

### Thinking flag and request adjustment
- Thinking uses the renderer's rule: thinking or enable_thinking, default true, and reasoning_effort "none" turns it off (deepseek_v4.py:230-236; tokenizers/deepseek_v41.py:72-78).
- It reads the merged chat_template_kwargs (serving.py:189-199, 266-273).
- The prompt-end check is `is_reasoning_end` (parser_engine.py:611-629).
- `adjust_request` sets skip_special_tokens=False (parser_engine.py:220).
- A structural tag is applied only for required/named or strict tools (structural_tag_registry.py:119, 237-320).
- VLLM_ENFORCE_STRICT_TOOL_CALLING=1 makes `supports_required_and_named=False`, so required and named choices are parsed by the same DSML parser (abstract_tool_parser.py:64-70).

### Stop strings (v1/engine/detokenizer.py:96-143, 310-362)
- Checked over the whole output text, which includes reasoning and markup because skip_special_tokens=False.
- The last `max(len(stop))-1` characters are held back while streaming.
- EOS is appended to token_ids but never decoded.

### Streaming
- serving.py:656 calls `parse_delta` (abstract_parser.py:824).
- Token-aware for `<think>`/`</think>` via strict mode (streaming_parser_engine.py:347); text-based for DSML.
- Trailing reasoning whitespace is deferred (parser_engine.py:474-503).
- Arguments are streamed as diffs of re-converted JSON prefixes (947-1012).

### Non-stream
- serving.py:970 calls `parse()` (abstract_parser.py:808-822).
- `extract_reasoning(text)` is text-only (parser_engine.py:506-531).
- Tools are then extracted from the content text (569-592, 1026-1077).
- Arguments are `json.dumps` of `{name: value}`, with `string="false"` values passed through `json.loads` (deepseek_v4.py:67-98), then schema coercion (parser_engine.py:240-403).

### Serving
- finish_reason is "tool_calls" whenever tools were streamed or auto-parsed, even on "length"; a named choice gives "stop" (serving.py:759, 1071-1075).
- completion_tokens = len(token_ids), EOS included (1121-1128).
- reasoning_tokens is always reported (159).
- Stream usage is a separate `choices: []` chunk, only with include_usage (818-871).
- Field name is `reasoning` (chat_completion/protocol.py:71; generate/base/protocol.py:332).
- Tool IDs are `chatcmpl-tool-{16hex}` (chat_utils.py:2223).
- Logprobs are one list over all generated tokens, EOS included (serving.py:1241-1305).

## 3. Divergence table
Evidence lives under harness/parsing/out. diff_summary.txt (59-case baseline) is referenced by case name. "ns" = non-stream; "st[N]" = stream with N-token steps.

### D1, HIGH. Stop strings over reasoning and markup
- API: answer content only (output_parser.rs:53-56, 411-417, 465-466). Ours: detokenizer.py:110-143 over the full text.
- Impact: thinking-mode answers are lost, tool calls are destroyed or truncated, and finish_reason is wrong. This also affects Anthropic Messages, whose stop_sequences become `stop` (anthropic/serving.py:484, 633-640).
- Evidence:
  - stop_in_reasoning: UP content='Final answer: 7.'; OURS ns content=None, reasoning='Step 1.\n\nStep 2.'. Stream st[rnd3] also lost a space: reasoning='Step1.\n\nStep 2. ' (debug_stop_rnd3.py: deferred whitespace dropped on the finish flush).
  - stop_blankline_reasoning ("\n\n"): UP content='Answer line one.'; OURS reasoning='Para one.', content=None.
  - stop_in_tool_args: UP '{"query": "Paris weather"}' tool_calls; OURS ns '{}' tool_calls, st '{"query": "' tool_calls.
  - stop_matching_markup: UP tool call; OURS content '\n\n', no call, finish stop.
  - stop_think_tag: OURS content None.
- Fix P2: new docker/patch rewrite of vllm/v1/engine/detokenizer.py, behind a lever such as DSV41_STOP_ANSWER_ONLY=1 (add it to FORWARD_ENVS).
  - In `BaseIncrementalDetokenizer.__init__`, note whether the prompt ends with the `<think>` ID.
  - In `update()`, track the answer window. It starts after the `</think>` token (or at 0 when thinking is off) and ends at the first `｜DSML｜` token.
  - Run `check_stop_strings` on `output_text[start:end]` and map the truncation index back.
  - Look up token IDs from the tokenizer rather than hard-coding them.
  - Prototype: proto_patches.apply_p2. It cleared all 8 stop cases (out/patches_p1bp2p3p4) and both fuzzers stayed clean.
- Risk: low-medium. It changes stop behaviour for every request that sets stops; the engine still aborts early.

### D2, HIGH. finish_reason when truncated mid tool call
- API: "length" (processor.rs:205-217, test at 940). Ours: "tool_calls" (serving.py:759, 1071).
- Impact: the client runs a truncated call. Args are '{}' (ns) or '{"query": "' (st, invalid JSON); tool_truncated_after_param silently drops the partial param.
- Evidence: tool_truncated_mid_value: UP '{"query": "weather in Par' finish length; OURS ns '{}' tool_calls, st '{"query": "' tool_calls.
- Fix P3: serving.py rewrite with two anchors. Streaming: `if tools_streamed[i] and output.finish_reason != "length"`. Non-stream: tool_calls only when finish_reason != "length".
- Prototype: apply_p3, validated.
- Risk: low.

### D3, MED-HIGH. `reasoning` vs `reasoning_content`
- Details and evidence: section 0 item 5.
- Fix P4: rewrite `ChatMessage._serialize` (chat_completion/protocol.py:73-78) and `DeltaMessage._serialize` (generate/base/protocol.py:335-340) to add `reasoning_content = reasoning` when present.
  - Responses and Messages read the attribute, not the serialized field, so they are unaffected.
- Prototype: apply_p4, validated.
- Risk: low. Streamed reasoning bytes roughly double.

### D4, MED. Non-stream parse is text-only (abstract_parser.py:816; parser_engine.py:512, 569-592)
- Evidence:
  - think_literal_end_tag_in_reasoning: UP reasoning='The tag </think> closes thinking.' content='Done.'; OURS ns reasoning='The tag', content=' closes thinking.Done.'; st correct.
  - think_literal_tag_pair_in_content: OURS ns content='Wrap it as  please.' reasoning='rsecret'.
  - nothink_literal_tag_pair_in_content: OURS ns reasoning='secret' even with thinking off.
  - think_literal_end_tag_in_content: OURS ns content loses "</think>".
- Fix P1b: rewrite `DelegatingParser.parse`.
  - Drop the trailing EOS. In thinking mode, split the stop-truncated text at the decoded offset of the first `</think>` token.
  - Run tool extraction with `skip_reasoning_parsing=True` on the tool engine.
  - Fall back to the current path when there is no `</think>` token.
  - Prototype: apply_p1b. It fixed every literal-tag case and kept vLLM's whole-text tool JSON.
  - Trade-off: a genuine second `</think>` token in content is kept as text (the API drops it).
  - Note: routing non-stream through parse_delta (prototype P1) also fixes the literal tags, but it inherits streaming's invalid-JSON args on duplicate or truncated params. Rejected.
- Risk: low-medium.

### D5, MED-LOW. Whitespace (deepseek_v4.py:216; parser_engine.py:474-503, 528)
- Evidence:
  - think_newlines_around: UP reasoning 'First line.\n\nSecond line.', content 'Answer text.\n'; OURS '\n\nFirst…', '\n\nAnswer text.\n'.
  - tool_content_before: OURS content '…for you.\n\n'.
  - think_trailing_spaces: UP '  reason  ' (also the HF golden); OURS '  reason'.
- Impact:
  - UI cosmetics.
  - History drift: the HF golden output always has "summary\n\n<｜DSML｜ calls>", so every tool call with content echoes back as "summary\n\n\n\n<DSML…".
  - Shorter prefix-cache / REPLY_RETAIN hits.
- Fix P5, not prototyped:
  - Reasoning: drop leading "\n"s and only the final newline run before `</think>`.
  - Content: drop leading "\n"s, and drop trailing "\n"s when a tool block follows.
  - Streaming: hold back trailing content newlines until the next non-newline text or the end.
- Risk: medium (streaming deferral).

### D6, MED-LOW. Logprobs (serving.py:1241-1305)
- Evidence: logprobs_think_tool, where OURS includes 'Plan', '.', '</think>', '｜DSML｜', ' calls', … and EOS; logprobs_multibyte, where OURS gives "" with bytes [] for a partial byte token.
- Impact: answer-confidence scoring in thinking mode mixes in reasoning and markup.
- Fix: serving patch that splits by token class (reasoning before the `</think>` ID; drop `</think>`, the DSML block from `<`+`｜DSML｜`, and EOS), plus a `reasoning_content` list.
- Risk: low; moderate effort.

### D7, LOW-MED. Usage
- completion_tokens +1 for EOS in every case (e.g. think_basic: 21 vs 22).
- Streamed reasoning_tokens drift (fuzz_stream.json: 81 of 240). think_no_end_eos st[6] rt=0 vs 6, from `_rebuild_from_anchors` token-count attribution (token_id_scanner.py:312-400).
- Stream usage is absent without include_usage (usage_no_include.py: upstream's final chunk has usage; ours has none).
- No prompt_cache_hit_tokens or prompt_cache_miss_tokens.
- Fix: flags `--enable-force-include-usage` and `--enable-prompt-tokens-details`; a usage-field patch is for the protocols track to decide. Optionally subtract EOS.
- Risk: low.

### D8, LOW-MED. Named tool_choice finish_reason "stop" vs "tool_calls"
- Evidence: tool_named_choice_nothink.
- Fix: part of P3. Prototype validated.

### D9, LOW. Tool-arg normalization and coercion (deepseek_v4.py:67-98; parser_engine.py:240-403)
- Evidence:
  - tool_types_thinking: '21.50' vs '21.5', `{"unit":"c"}` vs `{"unit": "c"}`; equal as JSON.
  - tool_int_as_string_attr: "5" vs 5.
  - tool_str_as_number_attr: 123 vs "123".
  - tool_nullable_string_null_text: "null" vs null.
  - tool_false_invalid_json: invalid JSON (API) vs a string (ours).
- Fix: keep. Optionally skip null coercion for `string="true"` values.

### D10, LOW. Text after the tool block
- Evidence: tool_content_after. Streams emit 'Done, waiting…' or '\nDone…' depending on chunking; the API drops it.
- Fix: drop TEXT_CHUNK after TOOL_END in the engine.

### D11, LOW. Duplicate parameter
- Evidence: tool_duplicate_param, where the stream stays at '{"query": "a' (parser_engine.py:1007).
- Fix: keep the first occurrence in the converter.

### D12, LOW. Shapes
- content null (ours) vs "" (API).
- Empty-thinking reasoning null vs "".
- Reasoning and content in one delta.
- Tool start delta without `arguments:""`.
- Tool IDs `chatcmpl-tool-…` vs `call_00_[ET_]…`. `ET_` matters only to the API's own request validation.
- Fix: optional.

### D13, LOW. DSML and think-marker edge cases
- tool_plain_spelled_block_in_content: OURS 'Format: ', the rest dropped.
- tool_choice_none_but_emitted and tool_no_tools_declared_but_emitted: API returns the markup verbatim as content; ours strips it.
- tool_v4_unspaced_tags: the 1017 model emits spaced tags.
- think_start_token_echo: `<think>` dropped by ours, kept as reasoning by the API.

### D14, LOW. Final-step anchor mis-binding
- Evidence: think_end_token_then_plain_end_tag_same_step st[6] gives reasoning 'r</think>', content ' ok'.
- Fix: exclude trailing DROP anchors in `_recover_holdback_text`.

### JSON mode
- json_fenced_with_preamble differs, but only without the grammar; bare JSON is identical.
- The API's output there is itself chunking-dependent.

## 4. What already matches
- `</think>`-token split with thinking on and off; empty reasoning (the text, apart from null vs ""); EOS or length inside reasoning.
- A second `</think>` token in content is dropped.
- Spaced DSML parse: string, number, bool, null, array and object params; quote, backslash, newline and tab escapes; JSON-looking strings stay strings; two parallel calls with correct indices; empty invoke `{}`; a tool block without `</think>` ends reasoning; required choice; unknown tool names passed through by both.
- finish_reason for completed calls, stop and length.
- Multi-byte characters split across tokens under every chunking.
- Grounding tokens `<｜ref｜>`/`<｜box｜>` kept verbatim (vLLM `all_special_tokens` = BOS/EOS only).
- Stop in content: 180 of 180 streams, content identical to the API.
- No stream/non-stream text or tool mismatch in 240 fuzzed streams with non-matching stops.
- The parser's thinking flag equals the renderer's.

## 5. Open questions and what I could not verify
- No GPU runs. Real-world frequency of leading newlines, plain-spelled tags, text after the block and duplicate params is unknown. The "summary\n\n<DSML" pattern is in the HF golden output, so D5 likely hits every tool call that has content.
- Grammar, structural tags and xgrammar were not executed; that belongs to the structured track. With thinking, required or named tool_choice, the API returns 400 while we render and constrain after `</think>`.
- With tool_choice none, the API omits tools from the prompt. We render them unless `--exclude-tools-when-tool-choice-none` is set (render track).
- Prefix continuation (`prefix: true` / ContentContinuation) was not tested.
- Responses and Messages share the parser and detokenizer, so D1 (Messages stop_sequences), D4 (Responses non-stream also calls `parse()`, responses/serving.py:1070) and D5 apply there. The experimental ParsableContext builds its parser without chat_template_kwargs (responses/context.py:315), so thinking defaults on; it is off by default.
- Option B packaging: I did not locate the .whl file (only the venv install), so building it into the image is unverified.
- Completion-token counts after a stop depend on step size on both sides.
