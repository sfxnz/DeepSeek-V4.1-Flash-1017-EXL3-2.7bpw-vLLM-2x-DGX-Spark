#!/usr/bin/env python3
"""1017 chat-encoder parity checks for the vLLM V4.1 tokenizer (docker/patch/chat_encoder_1017.py).

Two entry points share the same checks:
  * check_encoder(enc, tests_dir): the 1017 encoding/tests golden vectors (exact match) and the
    1017 test_encoding.py behaviours through a deepseek_v41_encoding module. The host unit test
    runs it on the patched e14 fixture (tests/fixtures/dsv41_tokenizer_encoding_e14.pin.py).
  * main(): inside the serve image (no GPU), after sitecustomize applied the patch at interpreter
    start the way run.sh mounts it: the same checks on the image's own module, plus the serving
    path (deepseek_v41 apply_chat_template) for the vectors and behaviours the OpenAI API can carry.

    docker run --rm --network none --memory 4g --name dsv41-1017-chatenc \\
      -v $PWD/docker/patch:/opt/dsv41-patch:ro \\
      -v $PWD/docker/patch/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro \\
      -v $PWD/tests:/checks:ro -v <1017 snapshot>/encoding/tests:/vectors:ro \\
      --entrypoint python3 <image> /checks/chat_encoder_1017_image_check.py /vectors

Prints one line per check and exits 1 on any failure.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


def load_case(path: Path) -> dict:
    """encoding.load_cases of the 1017 reference, first case."""
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = [data]
    elif data and isinstance(data[0], dict) and "role" in data[0]:
        data = [{"messages": data}]
    case = data[0]
    messages = copy.deepcopy(case["messages"])
    if "tools" in case:
        messages[0]["tools"] = case["tools"]
    return {"messages": messages, "tools": case.get("tools"), "thinking_mode": case.get("thinking_mode") or "chat",
            "reasoning_effort": case.get("reasoning_effort")}


def flatten_images(messages: list, placeholder: str) -> list:
    """The image's _normalize_messages content rule (text parts and image placeholders joined by blank lines)."""
    out = copy.deepcopy(messages)
    for m in out:
        if isinstance(m.get("content"), list):
            m["content"] = "\n\n".join(b.get("text", "") if b.get("type") == "text" else placeholder
                                       for b in m["content"])
    return out


def vector_ids(tests_dir: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[-1]) for p in Path(tests_dir).glob("test_input_*.json"))


def tool_history():
    calls = [{"id": n, "type": "function", "function": {"name": n, "arguments": "{}"}} for n in ("a", "b")]
    return [{"role": "user", "content": "look"}, {"role": "assistant", "content": "", "tool_calls": calls}]


def drop_thinking_messages():
    return [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "reasoning_content": "r2",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "value"},
    ]


def check_encoder(enc, tests_dir: Path) -> list[tuple[str, bool, str]]:
    """Golden vectors and 1017 behaviours on a deepseek_v41_encoding module."""
    out = []
    ph = enc.IMAGE_PLACEHOLDER
    for i in vector_ids(tests_dir):
        case = load_case(Path(tests_dir) / f"test_input_{i}.json")
        want = (Path(tests_dir) / f"test_output_{i}.txt").read_text()
        try:
            got = enc.encode_messages(flatten_images(case["messages"], ph), thinking_mode=case["thinking_mode"],
                                      reasoning_effort=case["reasoning_effort"])
            out.append((f"encoder vector {i}", got == want, "" if got == want else _first_diff(got, want)))
        except Exception as exc:  # noqa: BLE001
            out.append((f"encoder vector {i}", False, repr(exc)))
    msgs = drop_thinking_messages()
    prompt = enc.encode_messages(copy.deepcopy(msgs), thinking_mode="thinking", drop_thinking=True)
    ok = ("<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>" in prompt and "r1" not in prompt
          and "<think>r2</think>" in prompt
          and prompt.endswith("<tool_result>value</tool_result><｜Assistant｜><think>"))
    out.append(("encoder drop_thinking keeps the current turn across tool results", ok, prompt[-160:]))
    nxt = enc.encode_messages(copy.deepcopy(msgs) + [{"role": "user", "content": "q3"}], thinking_mode="thinking")
    out.append(("encoder next user turn drops r2", "r2" not in nxt and nxt.endswith("q3<｜Assistant｜><think>"),
                nxt[-80:]))
    blocks = [{"type": "text", "text": "result b"}]
    m = enc.merge_tool_messages([{"role": "tool", "tool_call_id": "b", "content": None, "content_blocks": blocks}])
    out.append(("merge passes tool content_blocks through", m[0]["content_blocks"][0]["content"] == blocks, repr(m)))
    m = enc.merge_tool_messages([{"role": "user", "content": [{"type": "text", "text": "x"}]}])
    out.append(("merge passes user list content through", m[0]["content_blocks"] == [{"type": "text", "text": "x"}],
                repr(m)))
    res = [{"type": "tool_result", "tool_use_id": n, "content": f"result {n}"} for n in ("b", "a")]
    p = enc.encode_messages([{"role": "user", "content_blocks": res}], thinking_mode="chat", context=tool_history())
    out.append(("tool results sorted by call order across context",
                p.endswith("<tool_result>result a</tool_result>\n\n<tool_result>result b</tool_result>"
                           "<｜Assistant｜></think>"), p[-120:]))
    eff = getattr(enc, "REASONING_EFFORT_MAPPINGS", {})
    out.append(("reasoning effort budgets low 50 / high 75 / max 100",
                eff == {"low": 50, "high": 75, "max": 100}, repr(eff)))
    for role in ("user", "tool"):
        try:
            enc._validate_no_image_sp_tokens({"role": role, "content": f"hi {ph}"})
            out.append((f"placeholder in {role} text rejected", False, "no error"))
        except ValueError:
            out.append((f"placeholder in {role} text rejected", True, ""))
        except AttributeError as exc:
            out.append((f"placeholder in {role} text rejected", False, repr(exc)))
    return out


def check_serving(tok, tests_dir: Path, enc) -> list[tuple[str, bool, str]]:
    """deepseek_v41 apply_chat_template (tokenize=False): what the OpenAI API path renders."""
    class _Base:
        def get_added_vocab(self):
            return {}

    wrapped = tok.get_deepseek_v41_tokenizer(_Base())
    out = []
    api_roles = {"system", "user", "assistant", "tool", "developer"}
    for i in vector_ids(tests_dir):
        case = load_case(Path(tests_dir) / f"test_input_{i}.json")
        msgs = case["messages"]
        if any(m.get("role") not in api_roles or m.get("task") for m in msgs) or (
                case["tools"] and msgs[0].get("role") != "system"):
            out.append((f"serving vector {i}", True, "skipped: not expressible through the OpenAI API"))
            continue
        msgs = copy.deepcopy(msgs)
        msgs[0].pop("tools", None)
        kw = {"thinking": case["thinking_mode"] == "thinking", "tokenize": False}
        if case["reasoning_effort"] is not None:
            kw["reasoning_effort"] = case["reasoning_effort"]
        want = (Path(tests_dir) / f"test_output_{i}.txt").read_text()
        try:
            got = wrapped.apply_chat_template(msgs, tools=case["tools"], **kw)
            out.append((f"serving vector {i}", got == want, "" if got == want else _first_diff(got, want)))
        except Exception as exc:  # noqa: BLE001
            out.append((f"serving vector {i}", False, repr(exc)))
    prompt = wrapped.apply_chat_template(drop_thinking_messages(), thinking=True, tokenize=False)
    out.append(("serving drop_thinking keeps r2 across the tool result",
                "<think>r2</think>" in prompt and prompt.endswith("<tool_result>value</tool_result><｜Assistant｜><think>"),
                prompt[-160:]))
    ph = enc.IMAGE_PLACEHOLDER
    for role, extra in (("user", {}), ("tool", {"tool_call_id": "a"})):
        msgs = tool_history() + [{"role": role, "content": f"hi {ph}", **extra}]
        try:
            wrapped.apply_chat_template(msgs, thinking=False, tokenize=False)
            out.append((f"serving rejects placeholder text in a {role} message", False, "no error"))
        except ValueError:
            out.append((f"serving rejects placeholder text in a {role} message", True, ""))
    msgs = tool_history() + [{"role": "tool", "tool_call_id": n, "content": [
        {"type": "text", "text": f"result {n}"}, {"type": "image_url", "image_url": {"url": f"{n}.png"}}]}
        for n in ("b", "a")]
    p = wrapped.apply_chat_template(msgs, thinking=False, tokenize=False)
    want = (f"<｜User｜><tool_result>result a\n\n{ph}</tool_result>\n\n"
            f"<tool_result>result b\n\n{ph}</tool_result><｜Assistant｜></think>")
    out.append(("serving tool-result images follow call order", p.endswith(want), p[-200:]))
    for effort, budget in (("low", 50), ("high", 75), ("max", 100)):
        p = wrapped.apply_chat_template([{"role": "user", "content": "x"}], thinking=True, reasoning_effort=effort,
                                        tokenize=False)
        out.append((f"serving effort {effort} -> {budget}", f"Reasoning Effort: {budget} " in p, p[:120]))
    return out


def _first_diff(got: str, want: str) -> str:
    n = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
    return f"first diff at {n}: got {got[max(0, n - 40):n + 60]!r} want {want[max(0, n - 40):n + 60]!r}"


def report(results) -> int:
    bad = 0
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" -- {detail}" if detail and not ok else
                                                      (f" ({detail})" if detail.startswith("skipped") else "")))
        bad += not ok
    print(f"{len(results) - bad}/{len(results)} passed")
    return 1 if bad else 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    tests_dir = Path(argv[0])
    import vllm.tokenizers.deepseek_v41 as tok
    import vllm.tokenizers.deepseek_v41_encoding as enc

    patched = "dsv41-chat-encoder-1017" in Path(enc.__file__).read_text()
    results = [("image encoder carries the 1017 patch marker", patched, enc.__file__)]
    results += check_encoder(enc, tests_dir)
    results += check_serving(tok, tests_dir, enc)
    return report(results)


if __name__ == "__main__":
    sys.exit(main())
