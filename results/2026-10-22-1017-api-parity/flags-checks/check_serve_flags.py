"""CPU-only check, inside the serve image, of the api-parity (2/5) serve flags.

Input: /sp/argv.json = run.sh head/worker `vllm serve` argv + container env from tests/run_sh_harness.py.
1. Parse both argv with the image's own `vllm serve` parser (ServeSubcommand, as vllm/entrypoints/cli/main.py
   builds it), run its validate(), build AsyncEngineArgs. The image cannot infer a device without a GPU, so the
   CPU platform is swapped in first, as vllm's CLI itself does for `vllm bench`.
2. VLLM_MAX_STOP_STRINGS: request validation with the image's protocol models, env unset vs the forwarded value.
3. Default chat-template kwargs: old {"thinking":false,"reasoning_effort":"low"} vs the new default, merged
   with request kwargs the way OnlineRenderer.preprocess_chat does, rendered by the pack's V4.1 tokenizer.
4. --enable-prompt-tokens-details: the usage shape it adds.
"""
import json
import os
import re
import subprocess
import sys

ARGV = json.load(open("/sp/argv.json"))
PACK = "/hub/models--sfxnz--DeepSeek-V4.1-Flash-1017-EXL3/snapshots/2.7bpw-mcg-hq"
OLD_DEFAULT = {"thinking": False, "reasoning_effort": "low"}
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def flag_value(args, flag):
    return args[args.index(flag) + 1]


def part1_parse():
    from vllm import platforms
    from vllm.platforms.cpu import CpuPlatform

    if platforms.current_platform.is_unspecified():
        platforms.current_platform = CpuPlatform()
    import vllm
    from vllm.config.structured_outputs import StructuredOutputsConfig
    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    print("vllm", vllm.__version__)
    parser = FlexibleArgumentParser(description="vLLM CLI")
    sub = parser.add_subparsers(required=False, dest="subparser")
    cmd = ServeSubcommand()
    serve_parser = cmd.subparser_init(sub)
    want = {"--exclude-tools-when-tool-choice-none", "--structured-outputs-config",
            "--enable-prompt-tokens-details", "--default-chat-template-kwargs"}
    for action in serve_parser._actions:
        if want & set(action.option_strings):
            print("option", action.option_strings, type(action).__name__, "default", repr(action.default))
    parsed = {}
    for role in ("head", "worker"):
        argv = ARGV[role]["args"]
        args = parser.parse_args(argv)
        cmd.validate(args)  # validate_parsed_serve_args, as `vllm serve` runs before dispatch
        args.model = args.model_tag  # ServeSubcommand.cmd does this first (entrypoints/cli/serve.py:51-53)
        ea = vllm.AsyncEngineArgs.from_cli_args(args)
        so = ea.structured_outputs_config
        print(f"{role}: subparser={args.subparser} headless={args.headless} "
              f"exclude_tools_when_tool_choice_none={args.exclude_tools_when_tool_choice_none} "
              f"enable_prompt_tokens_details={args.enable_prompt_tokens_details} "
              f"default_chat_template_kwargs={args.default_chat_template_kwargs!r} "
              f"structured_outputs_config={type(so).__name__}(backend={so.backend!r}) "
              f"reasoning_parser={ea.reasoning_parser!r} tool_call_parser={args.tool_call_parser!r} "
              f"enable_auto_tool_choice={args.enable_auto_tool_choice}")
        check(f"{role} parses --exclude-tools-when-tool-choice-none", args.exclude_tools_when_tool_choice_none is True)
        check(f"{role} parses --enable-prompt-tokens-details", args.enable_prompt_tokens_details is True)
        check(f"{role} parses --default-chat-template-kwargs", args.default_chat_template_kwargs == {"thinking": False})
        check(f"{role} parses --structured-outputs-config",
              isinstance(so, StructuredOutputsConfig) and so.backend == "xgrammar")
        # create_engine_config copies --reasoning-parser into this config (engine/arg_utils.py:2521-2523), so the
        # grammar stays gated on </think>: the pinned config must not lose it.
        check(f"{role} keeps --reasoning-parser for the structured config", ea.reasoning_parser == "deepseek_v41")
        check(f"{role} argv value", flag_value(argv, "--structured-outputs-config") == '{"backend":"xgrammar"}')
        parsed[role] = args
    # One flag at a time against the stock value: each flag on its own changes exactly its own field.
    base = [a for a in ARGV["head"]["args"]]
    for flag, field, stock in (("--exclude-tools-when-tool-choice-none", "exclude_tools_when_tool_choice_none", False),
                               ("--enable-prompt-tokens-details", "enable_prompt_tokens_details", False)):
        argv = [a for a in base if a != flag]
        check(f"{flag} absent -> stock {stock}", getattr(parser.parse_args(argv), field) is stock)
    i = base.index("--structured-outputs-config")
    argv = base[:i] + base[i + 2:]
    args = parser.parse_args(argv)
    args.model = args.model_tag
    so = vllm.AsyncEngineArgs.from_cli_args(args).structured_outputs_config
    check("--structured-outputs-config absent -> stock backend 'auto'", so.backend == "auto", repr(so.backend))


STOP_CODE = r'''
import json
from pydantic import ValidationError
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
out = {}
for n in (4, 5, 16, 17):
    stops = [f"s{i}" for i in range(n)]
    row = {}
    for name, cls, body in (
        ("chat", ChatCompletionRequest, {"model": "m", "messages": [{"role": "user", "content": "x"}], "stop": stops}),
        ("completions", CompletionRequest, {"model": "m", "prompt": "x", "stop": stops}),
        ("messages", AnthropicMessagesRequest, {"model": "m", "max_tokens": 8,
                                                "messages": [{"role": "user", "content": "x"}], "stop_sequences": stops}),
    ):
        try:
            cls.model_validate(body)
            row[name] = "ok"
        except ValidationError as e:
            row[name] = "rejected: " + e.errors()[0]["msg"]
    out[n] = row
print("STOPJSON " + json.dumps(out))
'''


def part2_stop_strings():
    forwarded = ARGV["head"]["env"].get("VLLM_MAX_STOP_STRINGS")
    check("head env forwards VLLM_MAX_STOP_STRINGS=16", forwarded == "16", repr(forwarded))
    check("worker env forwards VLLM_MAX_STOP_STRINGS=16", ARGV["worker"]["env"].get("VLLM_MAX_STOP_STRINGS") == "16")
    results = {}
    for label, value in (("stock (unset)", None), ("forwarded", forwarded)):
        env = {k: v for k, v in os.environ.items() if k != "VLLM_MAX_STOP_STRINGS"}
        if value is not None:
            env["VLLM_MAX_STOP_STRINGS"] = value
        proc = subprocess.run([sys.executable, "-c", STOP_CODE], env=env, capture_output=True, text=True)
        line = next((l for l in proc.stdout.splitlines() if l.startswith("STOPJSON ")), None)
        if line is None:
            check(f"stop-string probe {label} ran", False, proc.stderr[-800:])
            return
        res = {int(k): v for k, v in json.loads(line[len("STOPJSON "):]).items()}
        results[label] = res
        for n, row in res.items():
            print(f"stop strings {label:<14} n={n:>2}: {row}")
    stock, fwd = results["stock (unset)"], results["forwarded"]
    for api in ("chat", "completions", "messages"):
        check(f"{api}: stock accepts 4, rejects 5", stock[4][api] == "ok" and stock[5][api] != "ok")
        check(f"{api}: forwarded accepts 5 and 16, rejects 17",
              fwd[5][api] == "ok" and fwd[16][api] == "ok" and fwd[17][api] != "ok")


def part3_default_kwargs():
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.renderers.params import merge_kwargs
    from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer

    new_default = json.loads(flag_value(ARGV["head"]["args"], "--default-chat-template-kwargs"))
    tok = DeepseekV41Tokenizer.from_pretrained(PACK)
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 17*19? Return only the integer."}]
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather",
                                               "parameters": {"type": "object", "properties": {
                                                   "city": {"type": "string"}}, "required": ["city"]}}}]
    cases = [
        ("no chat_template_kwargs (client default)", {}, False),
        ("{thinking:false}", {"chat_template_kwargs": {"thinking": False}}, False),
        ("{thinking:false, effort:low} (smoke/bench/quality CHAT_KWARGS)",
         {"chat_template_kwargs": {"thinking": False, "reasoning_effort": "low"}}, False),
        ("{thinking:true, effort:high} (quality_eval THINK_KWARGS)",
         {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}}, False),
        ("{thinking:true, effort:max}", {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"}}, False),
        ("{thinking:true, effort:low}", {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "low"}}, False),
        ("top-level reasoning_effort:high", {"reasoning_effort": "high"}, False),
        ("top-level reasoning_effort:none", {"reasoning_effort": "none"}, False),
        ("tools, no kwargs", {"tools": tools}, False),
        ("{thinking:true}, no effort", {"chat_template_kwargs": {"thinking": True}}, True),
        ("{enable_thinking:true}, no effort", {"chat_template_kwargs": {"enable_thinking": True}}, True),
        ("tools + {thinking:true}, no effort", {"tools": tools, "chat_template_kwargs": {"thinking": True}}, True),
    ]

    def render(req, body, default):
        tool_dicts = [t.model_dump() for t in req.tools] if req.tools else None
        defaults = merge_kwargs(default, dict(tools=tool_dicts, tokenize=False))  # OnlineRenderer.preprocess_chat
        params = req.build_chat_params(None, "auto").with_defaults(defaults)
        return tok.apply_chat_template(messages=body["messages"], **params.get_apply_chat_template_kwargs())

    def describe(p):
        m = re.search(r"Reasoning Effort: (\d+)", p)
        mode = "think" if p.endswith("<think>") else ("chat" if p.endswith("</think>") else "?")
        return f"{mode}/effort {m.group(1) if m else '-'}"

    for name, extra, expect_change in cases:
        body = {"model": "m", "messages": msgs, **extra}
        req = ChatCompletionRequest.model_validate(body)
        old, new = render(req, body, OLD_DEFAULT), render(req, body, new_default)
        same = old == new
        detail = f"old {describe(old)} ({len(tok.encode(old, add_special_tokens=False))} ids) -> new {describe(new)} ({len(tok.encode(new, add_special_tokens=False))} ids)"
        if expect_change:
            check(f"changes: {name}", (not same) and "Reasoning Effort: 50" in old and "Reasoning Effort: 75" in new, detail)
        else:
            check(f"identical: {name}", same, detail)


def part4_usage():
    from vllm.entrypoints.openai.chat_completion.serving import _make_prompt_tokens_details
    from vllm.entrypoints.serve.engine.protocol import UsageInfo

    for flag in (False, True):
        u = UsageInfo(prompt_tokens=1200, completion_tokens=8, total_tokens=1208,
                      prompt_tokens_details=_make_prompt_tokens_details(flag, 1024, None, None))
        print(f"usage enable_prompt_tokens_details={flag}: {json.dumps(u.model_dump(exclude_none=True))}")
    u = UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                  prompt_tokens_details=_make_prompt_tokens_details(True, 1024, None, None))
    check("usage carries prompt_tokens_details.cached_tokens", u.prompt_tokens_details.cached_tokens == 1024)


part1_parse()
part2_stop_strings()
part3_default_kwargs()
part4_usage()
print(f"RESULT {'FAIL ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
sys.exit(1 if FAILS else 0)
