"""Stock restore through run.sh's EXTRA_ARGS (appended last): parse with the image's vllm serve parser."""
import json
from vllm import platforms
from vllm.platforms.cpu import CpuPlatform
if platforms.current_platform.is_unspecified():
    platforms.current_platform = CpuPlatform()
import vllm
from vllm.entrypoints.cli.serve import ServeSubcommand
from vllm.utils.argparse_utils import FlexibleArgumentParser
parser = FlexibleArgumentParser(description="vLLM CLI")
cmd = ServeSubcommand()
cmd.subparser_init(parser.add_subparsers(required=False, dest="subparser"))
ok = True
for role, spec in json.load(open("/sp/argv_override.json")).items():
    args = parser.parse_args(spec["args"])
    cmd.validate(args)
    args.model = args.model_tag
    so = vllm.AsyncEngineArgs.from_cli_args(args).structured_outputs_config
    got = (args.exclude_tools_when_tool_choice_none, args.enable_prompt_tokens_details, so.backend, args.default_chat_template_kwargs)
    want = (False, False, "auto", {"thinking": False, "reasoning_effort": "low"})
    print(role, "effective:", got, "stock" if got == want else "NOT STOCK")
    ok &= got == want
print("RESULT", "EXTRA_ARGS restores every stock value" if ok else "FAIL")
