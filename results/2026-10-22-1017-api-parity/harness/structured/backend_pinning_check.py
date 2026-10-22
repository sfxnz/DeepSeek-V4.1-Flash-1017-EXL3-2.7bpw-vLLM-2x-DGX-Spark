"""Backend selection with run.sh's default structured-outputs config (backend "auto").

vLLM picks the engine's one backend from the FIRST structured request it sees
(v1/structured_output/__init__.py:125-165, "We only support a single backend"), while each
request's `_backend` is chosen per request in SamplingParams._validate_structured_outputs
(sampling_params.py:1176-1221: xgrammar, falling back to guidance/outlines). This shows:
  1. some json_schemas (e.g. `multipleOf`, pattern+maxLength) validate to "guidance" under "auto",
  2. the guidance backend cannot compile the deepseek_v41 structural tag (tool_choice required /
     named / auto+strict): it only reads the legacy {"triggers","structures"} shape,
  3. with backend pinned to "xgrammar" the same schema is a clean 400 at validation instead.

Run (CPU only, no network):
  docker run --rm --network none --memory 8g --cpus 4 \
    -v /home/sfxnz/projects/data/dsv41-1017/api-parity:/ap \
    -v ~/.cache/huggingface/hub:/hub:ro \
    --entrypoint python3 dsv41-flash-exl3-sm121:e16-rows /ap/harness/structured/backend_pinning_check.py
"""

import json
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cases  # noqa: E402

PACK = "/hub/models--sfxnz--DeepSeek-V4.1-Flash-1017-EXL3/snapshots/2.7bpw-mcg-hq"


def main():
    from vllm.config.structured_outputs import StructuredOutputsConfig
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams
    from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer
    from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag
    from vllm.v1.structured_output.backend_guidance import GuidanceBackend
    from vllm.v1.structured_output.backend_types import StructuredOutputOptions as O
    from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

    tok = DeepseekV41Tokenizer.from_pretrained(PACK)
    candidates = {
        "uniqueItems": {"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}}, "required": ["tags"]},
        "multipleOf": {"type": "object", "properties": {"n": {"type": "integer", "multipleOf": 5}}, "required": ["n"]},
        "pattern+maxLength": {"type": "object", "properties": {"code": {"type": "string", "pattern": "^[A-Z]+$", "maxLength": 8}}, "required": ["code"]},
        "format:idn-email": {"type": "object", "properties": {"e": {"type": "string", "format": "idn-email"}}, "required": ["e"]},
        "contains": {"type": "object", "properties": {"l": {"type": "array", "items": {"type": "integer"}, "contains": {"const": 1}}}, "required": ["l"]},
        "propertyNames": {"type": "object", "properties": {"m": {"type": "object", "propertyNames": {"pattern": "^[a-z]+$"}}}, "required": ["m"]},
    }
    mc = SimpleNamespace(is_diffusion=False)
    schema = None
    for cname, cand in candidates.items():
        for backend in ("auto", "xgrammar"):
            sp = SamplingParams(structured_outputs=StructuredOutputsParams(json=json.dumps(cand)))
            try:
                sp._validate_structured_outputs(mc, StructuredOutputsConfig(backend=backend), tok)
                chosen = sp.structured_outputs._backend
                print(f"[{backend}] {cname} json_schema -> request backend {chosen!r}")
                if backend == "auto" and chosen == "guidance" and schema is None:
                    schema = cand
            except Exception as e:
                print(f"[{backend}] {cname} json_schema -> {type(e).__name__}: {str(e)[:140]}")
    schema = schema or candidates["multipleOf"]

    tools = [ChatCompletionToolsParam.model_validate(t) for t in (cases.T_WEATHER, cases.T_SEARCH)]
    spec = json.dumps(get_model_structural_tag("deepseek_v41", tools, "required", reasoning=False).model_dump())
    vcfg = SimpleNamespace(
        structured_outputs_config=StructuredOutputsConfig(),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )
    for name, cls in (("xgrammar", XgrammarBackend), ("guidance", GuidanceBackend)):
        try:
            be = cls(vllm_config=vcfg, tokenizer=tok, vocab_size=129280)
            be.compile_grammar(O.STRUCTURAL_TAG, spec, stop_token_ids={tok.eos_token_id})
            print(f"engine backend {name}: tool_choice=required structural tag compiles")
        except Exception as e:
            print(f"engine backend {name}: tool_choice=required structural tag FAILS: {type(e).__name__}: {str(e)[:160]}")
        try:
            be = cls(vllm_config=vcfg, tokenizer=tok, vocab_size=129280)
            be.compile_grammar(O.JSON, json.dumps(schema), stop_token_ids={tok.eos_token_id})
            print(f"engine backend {name}: the guidance-routed json_schema compiles")
        except Exception as e:
            print(f"engine backend {name}: the guidance-routed json_schema FAILS: {type(e).__name__}: {str(e)[:160]}")


if __name__ == "__main__":
    main()
