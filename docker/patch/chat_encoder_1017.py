"""DeepSeek-V4.1-Flash-1017 chat-encoder parity for the image's vLLM V4.1 tokenizer.

The 1017 checkpoint's encoding/encoding.py changed against the September one the
image's vllm/tokenizers/deepseek_v41_encoding.py follows:

1. find_last_user_index skips user messages that hold only tool results, so a
   tool round-trip does not open a new turn (thinking mode with drop_thinking:
   the current turn's reasoning is kept across tool results, and the prompt
   after a tool result still ends in <think>).
2. merge_tool_messages passes structured content through: a tool message's
   content_blocks (when its content is not a string) and a user message's list
   content become the content_blocks.
3. encode_messages validates every message for literal image placeholder tokens
   before merging, then merges context and messages and sorts tool results over
   both (images are collected after that, in final prompt order).

Plus the reference's reasoning-effort budgets: low 50, high 75, max 100 (the
image maps low 25, high 50, xhigh 75, max 100; the 1017 golden test_output_1
renders the default "high" as 75). "xhigh" is not a 1017 effort and is now
rejected like any other unknown value.

The image flattens image parts to IMAGE_PLACEHOLDER in deepseek_v41.py
_normalize_messages before the encoder sees them, so the placeholder check runs
there, on the request's own text, before flattening.

Source rewrites of the two vLLM files, applied at boot from sitecustomize
(required: an anchor miss is image drift and fails the boot under
DSV41_PATCH_STRICT=1). Idempotent. Checked against the 1017 encoding/tests
vectors and behaviour tests by tests/chat_encoder_1017_image_check.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

MARK = "dsv41-chat-encoder-1017"
LOG_ENGAGED = "dsv41: chat encoder 1017 parity engaged"
LOG_DISARMED = "dsv41: chat encoder 1017 parity FAILED"
ENCODING_REL = "tokenizers/deepseek_v41_encoding.py"
TOKENIZER_REL = "tokenizers/deepseek_v41.py"

ENC_TOOL_OLD = """            tool_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
"""
ENC_TOOL_NEW = """            content = msg.get("content", "")  # dsv41-chat-encoder-1017
            if msg.get("content_blocks") and not isinstance(msg.get("content"), str):
                content = msg["content_blocks"]
            tool_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": content,
            }
"""

ENC_USER_OLD = """            if content_blocks is None:
                content_blocks = [{"type": "text", "text": msg.get("content", "")}]
"""
ENC_USER_NEW = """            if content_blocks is None:
                content = msg.get("content", "")  # dsv41-chat-encoder-1017
                content_blocks = (
                    content
                    if isinstance(content, list)
                    else [{"type": "text", "text": content}]
                )
"""

ENC_LAST_USER_OLD = """        if role in ["user", "developer"] or (role == "system" and idx > 0):
            last_user_index = idx
            break
    return last_user_index
"""
ENC_LAST_USER_NEW = """        if role in ["user", "developer"] or (role == "system" and idx > 0):
            # dsv41-chat-encoder-1017: a user message of tool results only is not a new turn
            content_blocks = messages[idx].get("content_blocks")
            if content_blocks and not any(
                block.get("type") in ("text", "image", "image_url")
                for block in content_blocks
            ):
                continue
            last_user_index = idx
            break
    return last_user_index
"""

ENC_ENCODE_OLD = """    context = context if context else []

    # Preprocess: merge tool messages and sort tool results
    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context) :]
    if context:
        context = merge_tool_messages(context)
        context = sort_tool_results_by_call_order(context)

    full_messages = context + messages
"""
ENC_ENCODE_NEW = """    # dsv41-chat-encoder-1017: merge context and messages, then sort tool results over both.
    # (The 1017 placeholder check runs on the request text in deepseek_v41._normalize_messages,
    # before image parts become IMAGE_PLACEHOLDER.)
    context = merge_tool_messages(context or [])
    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context) :]

    full_messages = context + messages
"""

ENC_EFFORT_OLD = """REASONING_EFFORT_MAPPINGS: Dict[str, int] = {
    "low": 25,
    "high": 50,
    "xhigh": 75,
    "max": 100,
}
"""
ENC_EFFORT_NEW = """REASONING_EFFORT_MAPPINGS: Dict[str, int] = {  # dsv41-chat-encoder-1017: 1017 reference budgets
    "low": 50,
    "high": 75,
    "max": 100,
}
"""

ENC_VALIDATE_ANCHOR = "def merge_tool_messages("
ENC_VALIDATE_NEW = '''def _validate_no_image_sp_tokens(msg: Dict[str, Any]) -> None:
    """dsv41-chat-encoder-1017: reject literal image placeholder tokens in request text."""
    content = msg.get("content")
    texts = [content] if isinstance(content, str) else []
    if isinstance(content, list):
        texts += [
            block.get("text") or ""
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
    if any(IMAGE_PLACEHOLDER in text for text in texts):
        raise ValueError(
            f"Message content contains image special token '{IMAGE_PLACEHOLDER}'. "
            "Images should be provided as image content blocks."
        )
    reasoning_content = msg.get("reasoning_content")
    if isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:
        raise ValueError(
            f"reasoning_content contains image special token '{IMAGE_PLACEHOLDER}'"
        )


'''

TOK_IMPORT_OLD = """from .deepseek_v41_encoding import (
    IMAGE_PLACEHOLDER,
    REASONING_EFFORT_MAPPINGS,
    encode_messages,
)
"""
TOK_IMPORT_NEW = """from .deepseek_v41_encoding import (
    IMAGE_PLACEHOLDER,
    REASONING_EFFORT_MAPPINGS,
    _validate_no_image_sp_tokens,  # dsv41-chat-encoder-1017
    encode_messages,
)
"""

TOK_NORMALIZE_OLD = """        if "reasoning" in message:
            message["reasoning_content"] = message["reasoning"]
        content = message.get("content")
"""
TOK_NORMALIZE_NEW = """        if "reasoning" in message:
            message["reasoning_content"] = message["reasoning"]
        _validate_no_image_sp_tokens(message)  # dsv41-chat-encoder-1017
        content = message.get("content")
"""

TOK_EFFORT_MSG_OLD = '"DeepSeek V4.1 reasoning_effort must be low, high, xhigh, max, "'
TOK_EFFORT_MSG_NEW = '"DeepSeek V4.1 reasoning_effort must be low, high, max, "  # dsv41-chat-encoder-1017'


def _sub(src: str, old: str, new: str, label: str) -> str:
    if src.count(old) != 1:
        raise ValueError(f"{MARK}: {label}: expected 1 match, found {src.count(old)}")
    return src.replace(old, new, 1)


def patch_encoding_source(src: str) -> str:
    """vllm/tokenizers/deepseek_v41_encoding.py -> 1017 parity. Idempotent."""
    if MARK in src:
        return src
    src = _sub(src, ENC_TOOL_OLD, ENC_TOOL_NEW, "merge_tool_messages tool content")
    src = _sub(src, ENC_USER_OLD, ENC_USER_NEW, "merge_tool_messages user content")
    src = _sub(src, ENC_LAST_USER_OLD, ENC_LAST_USER_NEW, "find_last_user_index")
    src = _sub(src, ENC_ENCODE_OLD, ENC_ENCODE_NEW, "encode_messages preprocessing")
    src = _sub(src, ENC_EFFORT_OLD, ENC_EFFORT_NEW, "reasoning effort budgets")
    src = _sub(src, ENC_VALIDATE_ANCHOR, ENC_VALIDATE_NEW + ENC_VALIDATE_ANCHOR, "placeholder validator")
    return src


def patch_tokenizer_source(src: str) -> str:
    """vllm/tokenizers/deepseek_v41.py -> validate request text before flattening. Idempotent."""
    if MARK in src:
        return src
    src = _sub(src, TOK_IMPORT_OLD, TOK_IMPORT_NEW, "encoder import")
    src = _sub(src, TOK_NORMALIZE_OLD, TOK_NORMALIZE_NEW, "_normalize_messages validation")
    src = _sub(src, TOK_EFFORT_MSG_OLD, TOK_EFFORT_MSG_NEW, "effort error message")
    return src


def install(vllm_root, rewrite) -> str:
    """Apply both rewrites with sitecustomize's `rewrite(path, fn)`; print the engagement line."""
    root = Path(vllm_root)
    try:
        done = [rewrite(root / ENCODING_REL, patch_encoding_source),
                rewrite(root / TOKENIZER_REL, patch_tokenizer_source)]
    except BaseException as exc:
        print(f"dsv41: chat encoder 1017 parity FAILED: {exc!r}", file=sys.stderr, flush=True)
        raise
    print(f"dsv41: chat encoder 1017 parity engaged (encoding {done[0]}, tokenizer {done[1]})",
          file=sys.stderr, flush=True)
    return f"encoding {done[0]}, tokenizer {done[1]}"
