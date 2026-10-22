| case | status | API (deepseek-recipe) | ours (vLLM e16-rows + patches) | first prompt diff (API -> ours) |
|---|---|---|---|---|
| chat_plain | match | 11 tok | 11 tok |  |
| chat_thinking_field_enabled | TOKENS DIFFER | 37 tok think/high | 11 tok | delete: '<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n' -> ''; insert: '' -> '/' |
| chat_thinking_field_disabled | match | 11 tok | 11 tok |  |
| chat_effort_none | match | 11 tok | 11 tok |  |
| chat_effort_minimal | API ok / ours ERROR | ok (37 tok) | DeepSeek V4.1 reasoning_effort must be low, high, max, or an integer within [1, 100] in chat_template_kwargs |  |
| chat_effort_low | match | 37 tok think/low | 37 tok |  |
| chat_effort_medium | API ok / ours ERROR | ok (37 tok) | DeepSeek V4.1 reasoning_effort must be low, high, max, or an integer within [1, 100] in chat_template_kwargs |  |
| chat_effort_high | match | 37 tok think/high | 37 tok |  |
| chat_effort_xhigh | API ok / ours ERROR | ok (37 tok) | DeepSeek V4.1 reasoning_effort must be low, high, max, or an integer within [1, 100] in chat_template_kwargs |  |
| chat_effort_max | match | 37 tok think/max | 37 tok |  |
| chat_ctk_thinking_true | TOKENS DIFFER | 11 tok | 37 tok | insert: '' -> '<｜System｜>Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough the reasoning)\n\n'; delete: '/' -> '' |
| chat_prefix_continuation | TOKENS DIFFER | 16 tok | 17 tok | insert: '' -> '<｜end▁of▁sentence｜>' |
| chat_prefix_continuation_vllm | TOKENS DIFFER | 19 tok | 17 tok | delete: '<｜Assistant｜></think>' -> '' |
| chat_latest_reminder | API ok / ours ERROR | ok (17 tok) | Invalid role: latest_reminder |  |
| chat_history_reasoning_content | match | 350 tok think/high | 350 tok |  |
| chat_history_reasoning_field | TOKENS DIFFER | 347 tok think/high | 350 tok | insert: '' -> 'Need weather.' |
| chat_tools_auto | match | 276 tok | 276 tok |  |
| chat_tools_choice_none | TOKENS DIFFER | 11 tok | 276 tok | insert: '' -> '<｜System｜>\n\n## Tools\n\nYou have access to a set of tools to help answer the user\'s question. You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="$TOOL_NAME">\n<｜DSML｜ parameter name="$PARAMETER_NAME" string="true\|false">$PARAMETER_VALUE</｜DSML｜ parameter>\n...\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name="$TOOL_NAME2">\ |
| chat_tools_required_nothink | TOKENS DIFFER | 281 tok | 276 tok | delete: '\n\n<｜DSML｜ calls>\n' -> '' |
| chat_tools_required_think | API ERROR / ours ok | Thinking mode does not support this tool_choice | ok (300 tok) |  |
| chat_tools_named_nothink | TOKENS DIFFER | 281 tok | 326 tok | insert: '' -> '{"name": "other", "description": "Get the weather", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}\n'; delete: '\n\n<｜DSML｜ calls>\n' -> '' |
| chat_tools_strict | TOKENS DIFFER | 282 tok | 281 tok | delete: 'additionalProperties": false, "' -> ''; insert: '' -> ', "strict": true' |
| chat_json_object | TOKENS DIFFER | 36 tok | 12 tok | delete: '<｜System｜>\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{"type": "json_object"}' -> '' |
| chat_json_object_no_word | API ERROR / ours ok | Prompt must contain the word 'json' in some form to use 'response_format' of type 'json_object'. | ok (11 tok) |  |
| chat_json_schema | TOKENS DIFFER | 54 tok | 11 tok | delete: '<｜System｜>\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}' -> '' |
| chat_regex | API ok / ours ERROR | ok (11 tok) | 6 validation errors for ChatCompletionRequest response_format.function-wrap[__log_extra_fields__()].type   Input should be 'text', 'json_object' or 'json_schema' [type=literal_error, input_value='rege |  |
| chat_n2 | API ERROR / ours ok | Invalid n value (currently only n = 1 is supported) | ok (11 tok) |  |
| chat_logprobs_top21 | API ERROR / ours ok | Invalid top_logprobs value, the valid range of top_logprobs is [0, 20]. | ok (11 tok) |  |
| chat_top_logprobs_without_logprobs | both error | Invalid top_logprobs and logprobs value, logprobs must be set to true if top_logprobs is used. | when using `top_logprobs`, `logprobs` must be set to true. (parameter=top_logprobs) |  |
| chat_stream_options_no_stream | both error | stream_options should be set along with stream = true | Stream options can only be defined when `stream=True`. (parameter=stream_options) |  |
| chat_max_tokens_400k | API ERROR / ours ok | Invalid max_tokens value, the valid range of max_tokens is [1, 393216] | ok (11 tok) |  |
| chat_system_billing_header | TOKENS DIFFER | 16 tok | 38 tok | insert: '' -> 'x-anthropic-billing-header: cc_version=1.2.3; cch=abcd\n' |
| chat_image | match | 8 tok | 8 tok img spans [194] |  |
| chat_file_image | both error | .messages[0]: file must have a file_id or file_data | Unsupported chat content part type: 'file'. Supported types: audio_embeds, audio_url, image_embeds, image_pil, image_url, input_audio, input_image, input_text, output_text, prompt_embeds, refusal, tex |  |
| chat_tool_content_parts | match | 325 tok | 325 tok |  |
| msg_plain | match | 11 tok | 11 tok |  |
| msg_no_max_tokens | API ok / ours ERROR | ok (11 tok) | 1 validation error for AnthropicMessagesRequest max_tokens   Field required [type=missing, input_value={'model': 'deepseek-ai/De...': {'type': 'disabled'}}, input_type=dict]     For further informatio |  |
| msg_thinking_enabled | TOKENS DIFFER | 37 tok think/high | 11 tok | delete: '<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n' -> ''; insert: '' -> '/' |
| msg_thinking_adaptive | TOKENS DIFFER | 37 tok think/high | 11 tok | delete: '<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n' -> ''; insert: '' -> '/' |
| msg_effort_high | match | 37 tok think/high | 37 tok |  |
| msg_effort_max | match | 37 tok think/max | 37 tok |  |
| msg_system_string | match | 17 tok | 17 tok |  |
| msg_system_blocks | TOKENS DIFFER | 21 tok | 20 tok | delete: '\n\n' -> '' |
| msg_system_billing_block | match | 17 tok | 17 tok |  |
| msg_user_multi_text_blocks | match | 24 tok | 24 tok |  |
| msg_mid_system | TOKENS DIFFER | 28 tok | 27 tok | insert: '' -> '.Mid-conversation instruction'; delete: 'System｜>Mid-conversation instruction.<｜' -> '' |
| msg_tool_roundtrip | TOKENS DIFFER | 329 tok | 329 tok | delete: '\n' -> '' |
| msg_tool_roundtrip_nonascii | match | 332 tok | 332 tok |  |
| msg_thinking_history | TOKENS DIFFER | 356 tok think/high | 356 tok | delete: '\n\n' -> '' |
| msg_tool_result_image | TOKENS DIFFER | 326 tok | 325 tok img spans [194] | delete: '\n\n<｜image｜>' -> ''; insert: '' -> '>\n\n<｜image｜' |
| msg_user_image | match | 9 tok | 9 tok img spans [194] |  |
| msg_tool_choice_any | match | 276 tok | 276 tok |  |
| msg_tool_choice_tool_think | API ERROR / ours ok | Thinking mode does not support this tool_choice | ok (300 tok) |  |
| msg_tool_choice_none | TOKENS DIFFER | 11 tok | 276 tok | insert: '' -> '<｜System｜>\n\n## Tools\n\nYou have access to a set of tools to help answer the user\'s question. You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="$TOOL_NAME">\n<｜DSML｜ parameter name="$PARAMETER_NAME" string="true\|false">$PARAMETER_VALUE</｜DSML｜ parameter>\n...\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name="$TOOL_NAME2">\ |
| msg_document_block | API ok / ours ERROR | ok (16 tok) | 2 validation errors for AnthropicMessagesRequest messages.0.content.str   Input should be a valid string [type=string_type, input_value=[{'type': 'document', 'so..., 'text': 'Summarize.'}], input_type |  |
| msg_server_tool_web_search | both error | tools[1]: server tools are not supported | 1 validation error for AnthropicMessagesRequest tools.1.input_schema   Field required [type=missing, input_value={'type': 'web_search_2025..._search', 'max_uses': 3}, input_type=dict]     For further  |  |
| msg_defer_loading | API ok / ours ERROR | ok (399 tok) | DeepSeek V4.1 supports text and image content only; got 'tool_reference' |  |
| msg_stop_17 | both error | Stop string array too long: 17 | 1 validation error for AnthropicMessagesRequest stop_sequences   List should have at most 4 items after validation, not 17 [type=too_long, input_value=['s0', 's1', 's2', 's3', ...3', 's14', 's15', 's1 |  |
| msg_max_tokens_500k | API ok / ours ERROR | ok (11 tok) | This model's maximum context length is 500000 tokens. However, you requested 500000 output tokens and your prompt contains 64 characters (more than 0 characters, which is the upper bound for 0 input t |  |
| msg_temperature_top_k | match | 11 tok | 11 tok |  |
| resp_string | match | 11 tok | 11 tok |  |
| resp_input_text_parts | API ok / ours ERROR | ok (11 tok) | DeepSeek V4.1 supports text and image content only; got 'input_text' |  |
| resp_typed_message_string | match | 11 tok | 11 tok |  |
| resp_instructions_developer | TOKENS DIFFER | 22 tok | 22 tok | replace: 'Sy' -> 'U'; delete: 't' -> ''; replace: 'm' -> 'r' |
| resp_default_thinking | match | 11 tok | 11 tok |  |
| resp_effort_high | match | 37 tok think/high | 37 tok |  |
| resp_function_roundtrip | match | 325 tok | 325 tok |  |
| resp_function_output_parts | API ok / ours ERROR | ok (328 tok) | DeepSeek V4.1 supports text and image content only; got 'input_text' |  |
| resp_reasoning_item | match | 353 tok think/high | 353 tok |  |
| resp_reasoning_item_encrypted | API ok / ours ERROR | ok (353 tok) | Encrypted content is not supported. |  |
| resp_parallel_calls_unsorted | TOKENS DIFFER | 366 tok | 366 tok | replace: 'Paris' -> 'Rome'; replace: 'Rome' -> 'Paris'; replace: 'Sun' -> 'Rai' |
| resp_namespace_tool | TOKENS DIFFER | 279 tok | 275 tok | replace: '::' -> '__'; delete: 'Weather tools.\\n' -> '' |
| resp_apply_patch | TOKENS DIFFER | 1090 tok | 10 tok | delete: '<｜System｜>\n\n## Tools\n\nYou have access to a set of tools to help answer the user\'s question. You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="$TOOL_NAME">\n<｜DSML｜ parameter name="$PARAMETER_NAME" string="true\|false">$PARAMETER_VALUE</｜DSML｜ parameter>\n...\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name="$TOOL_NAME2">\n...\n |
| resp_tool_search_client | TOKENS DIFFER | 311 tok | 274 tok | delete: '{"name": "tool_search", "description": "Search tools", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}\n' -> '' |
| resp_web_search_tool | match | 272 tok | 272 tok |  |
| resp_text_json_schema | TOKENS DIFFER | 59 tok | 11 tok | delete: '<｜System｜>\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"], "additionalProperties": false}' -> '' |
| resp_input_image | API ok / ours ERROR | ok (8 tok) | 248 validation errors for ResponsesRequest input.str   Input should be a valid string [type=string_type, input_value=[{'role': 'user', 'conten...CDMAAAAAElFTkSuQmCC'}]}], input_type=list]     For furt |  |
| resp_assistant_multi_parts | TOKENS DIFFER | 17 tok | 14 tok | delete: '\n\nPart two.' -> '' |
| resp_top_logprobs | match | 5 tok | 5 tok |  |
| resp_max_output_tokens_400k | API ERROR / ours ok | Invalid max_tokens value, the valid range of max_tokens is [1, 393216] | ok (5 tok) |  |
