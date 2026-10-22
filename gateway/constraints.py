"""Structured outputs for the dsv41 API gateway: deepseek-recipe's OutputConstraint -> the
`structured_outputs` field of the sampling params sent to vLLM's /inference/v1/generate.

structured_outputs_for(converted, *, thinking) reads converted.inference_options.output_constraint
(deepseek-recipe 0.2.0, request/mod.rs:62-126; `kind` and `schema` are properties) and returns:

  json, schema ""      {"json": "{}"}: any JSON value, the API's json_object constraint (vLLM's
                       {"json_object": true} would compile {"type": "object"})
  json, schema         {"json": schema}; a strict schema arrives normalized with `$defs` and
                       `definitions` dropped (json_schema.rs:65-69), so they are put back from the
                       schema as written (conversation.response_schema)
  regex                {"regex": regex}
  tool_call_required   {"structural_tag": DSML}: required and named both end the prompt with the
                       `\\n\\n<｜DSML｜ calls>\\n` prefill (dsv41.rs:307-310), so the tag starts inside
                       the block: one or more invokes, `</｜DSML｜ calls>`, then only EOS
  tool_call (auto)     the triggered DSML tag when the tools are strict (upstream rejects mixing),
                       else None, as vLLM does for auto today

Upstream already lets the tool constraint win over response_format (output_constraint.rs:28-60).
Invokes follow deepseek-recipe's rendering: strict tools take their declared parameters in order,
each once, typed (string="true" raw text, enum and pattern enforced; string="false" a JSON value
checked against the property schema); non-strict tools take any parameter names and values; an
invoke without parameters takes the trained blank line `">\\n\\n</｜DSML｜ invoke>`. A `$ref` that
names nothing (upstream drops `$defs` from strict tools) accepts any value: xgrammar rejects a
dangling ref.

Gating: requests on /inference/v1/generate carry no reasoning_ended, and the engine computes it on
the first bitmask from the prompt IDs with the --reasoning-parser deepseek_v41 parser
(v1/structured_output/__init__.py:380-391, parser/engine/parser_engine.py:611-629). A prompt that
ends in `<think>` keeps the grammar off until the model's `</think>`; one that ends in `</think>`
(or the required prefill after it) applies it from the first token, as on the chat path. So
thinking mode needs no fallback. FALLBACK is returned only for a required tool call in thinking
mode (upstream rejects it, but the prefill would follow `<think>` and the gate would never open).
A Chat prefix continuation with answer text already in the prompt gets no grammar: the grammar
cannot see that text and would force a second document after it.
"""
from __future__ import annotations

import json


class _Fallback:
    def __repr__(self) -> str:
        return "FALLBACK"


FALLBACK = _Fallback()

CALLS_START = "<｜DSML｜ calls>"
CALLS_END = "</｜DSML｜ calls>"
INVOKE_END = "</｜DSML｜ invoke>\n"
PARAM_HEAD = '<｜DSML｜ parameter name="'
PARAM_END = "</｜DSML｜ parameter>\n"
TEXT_EXCLUDES = ["</｜DSML｜ parameter>", "</｜DSML｜ invoke>", CALLS_END]
THINK_EXCLUDES = ["<think>", "</think>"]
DEF_KEYS = ("$defs", "definitions", "$def")


def _const(value):
    return {"type": "const_string", "value": value}


def _seq(*elements):
    return {"type": "sequence", "elements": list(elements)}


def _or(*elements):
    return {"type": "or", "elements": list(elements)}


def _tag(begin, content, end):
    return {"type": "tag", "begin": begin, "content": content, "end": end}


def _any_text():
    return {"type": "any_text", "excludes": TEXT_EXCLUDES}


def _resolves(node, ref: str) -> bool:
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return False
    return True


def _relax_refs(node, root):
    """Replace each `#/...` $ref that names nothing in `root` with {} (any value)."""
    if isinstance(node, list):
        return [_relax_refs(v, root) for v in node]
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/") and not _resolves(root, ref):
        return {}
    return {k: _relax_refs(v, root) for k, v in node.items()}


def _types(schema: dict) -> set | None:
    """JSON types a normalized strict property schema admits, or None when unknown ($ref)."""
    if isinstance(schema.get("type"), str):
        return {schema["type"]}
    if isinstance(schema.get("type"), list):
        return set(schema["type"])
    if isinstance(schema.get("anyOf"), list):
        alts = [_types(alt) for alt in schema["anyOf"]]
        return None if None in alts else set().union(*alts)
    return None


def _string_content(schema: dict):
    enum = schema.get("enum")
    if isinstance(enum, list) and enum and all(isinstance(v, str) and v for v in enum):
        return _or(*(_const(v) for v in enum))
    if schema.get("pattern"):
        return {"type": "regex", "pattern": schema["pattern"]}
    return _any_text()


def _strict_param(key: str, schema: dict, params: dict):
    """One parameter of a strict tool. Its JSON value is checked as a standalone schema, so the
    root's definitions come along for its refs."""
    head = f'{PARAM_HEAD}{key}" string="'
    types = _types(schema)
    if types == {"string"}:
        return _tag(head + 'true">', _string_content(schema), PARAM_END)
    schema = dict(schema, **{k: params[k] for k in DEF_KEYS if k in params and k not in schema})
    value = {"type": "json_schema", "json_schema": _relax_refs(schema, schema)}
    if types is not None and "string" not in types:
        return _tag(head + 'false">', value, PARAM_END)
    return _tag(head, _or(_seq(_const('true">'), _any_text()), _seq(_const('false">'), value)), PARAM_END)


def _loose_param():
    """Any parameter name, raw text or any JSON value (the image's deepseek_v41 builder)."""
    value = _or(_seq(_const('true">'), _any_text()),
                _seq(_const('false">'), {"type": "json_schema", "json_schema": True}))
    return _tag(PARAM_HEAD, _seq({"type": "regex", "pattern": '[^"]+'}, _const('" string="'), value), PARAM_END)


def _invoke(fn: dict):
    params = fn.get("parameters")
    props = params.get("properties") if isinstance(params, dict) else None
    if fn.get("strict") is not True:
        content = _or(_const("\n"), {"type": "plus", "content": _loose_param()})
    elif props:
        content = _seq(*(_strict_param(k, s, params) for k, s in props.items()))
    else:
        content = _const("\n")
    return _tag(f'<｜DSML｜ invoke name="{fn["name"]}">\n', content, INVOKE_END)


def _structural_tag(fmt) -> dict:
    return {"structural_tag": json.dumps({"type": "structural_tag", "format": fmt})}


def structured_outputs_for(converted, *, thinking: bool) -> dict | _Fallback | None:
    """The `structured_outputs` value for the converted request, None, or FALLBACK."""
    constraint = converted.inference_options.output_constraint
    if constraint is None:
        return None
    kind, spec = constraint.kind, constraint.schema
    if kind in ("json", "regex"):
        conversation = converted.conversation
        last = conversation.messages[-1]
        if last.prefix and last.content:
            return None
        if kind == "regex":
            return {"regex": spec}
        if not spec:
            return {"json": "{}"}
        schema = json.loads(spec)
        written = conversation.response_schema
        if isinstance(schema, dict) and isinstance(written, dict):
            schema.update({k: written[k] for k in DEF_KEYS if k in written and k not in schema})
        return {"json": json.dumps(_relax_refs(schema, schema))}
    if kind not in ("tool_call", "tool_call_required"):
        raise ValueError(f"unknown output constraint kind {kind!r}")
    fns = [tool["function"] for tool in json.loads(spec)]
    if kind == "tool_call" and not any(fn.get("strict") is True for fn in fns):
        return None
    if kind == "tool_call_required" and thinking:
        return FALLBACK
    calls = {"type": "tags_with_separator", "tags": [_invoke(fn) for fn in fns], "separator": "",
             "at_least_one": True, "stop_after_first": False}
    if kind == "tool_call":
        return _structural_tag({"type": "triggered_tags", "triggers": [CALLS_START],
                                "tags": [_tag(CALLS_START + "\n", calls, CALLS_END)], "excludes": THINK_EXCLUDES})
    return _structural_tag(_seq(calls, _const(CALLS_END)))
