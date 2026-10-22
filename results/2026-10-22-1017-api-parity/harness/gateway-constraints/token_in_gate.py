"""Part 1 evidence: does the e16-rows vLLM gate a /inference/v1/generate grammar on the reasoning
parser, as the chat path does? CPU only, the image's own objects, the run.sh serve flags.

Per request (deepseek-recipe 0.2.0 convert + encode with the pack tokenizer, exactly the gateway's
path; structured_outputs from gateway/constraints.py):
  A. POST body -> GenerateRequest (the FastAPI body model of /inference/v1/generate) ->
     ServingTokens.serve_tokens with a capturing engine client: the kwargs that reach
     engine_client.generate (token-in passes no reasoning_ended / reasoning_parser_kwargs).
  B. InputProcessor.process_inputs on the captured engine input + params (AsyncLLM.add_request's
     path): EngineCoreRequest.reasoning_ended, the structured-output backend; then
     Request.from_engine_core_request.
  C. StructuredOutputManager(vllm_config) with structured_outputs_config.reasoning_parser =
     deepseek_v41 (engine/arg_utils.py:2521-2523 copies --reasoning-parser there), driven in the
     scheduler's order (grammar_bitmask before sampling; should_advance / trim_reasoning_for_advance /
     accept_tokens after, scheduler.py:1985-2010) over a scripted output. Per step: gate state,
     whether the row is masked, whether the scripted token is allowed, what advanced the grammar.
     The same script is replayed with the chat path's explicit reasoning_ended =
     parser.is_reasoning_end(prompt_ids) (chat_completion/serving.py:356-385) for comparison.
Also: payload shape checks (valid and invalid structured_outputs bodies) against GenerateRequest +
SamplingParams._validate_structured_outputs.

Run: harness/gateway-constraints/run_in_image.sh token_in_gate.py
Writes harness/gateway-constraints/token_in_gate.json.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback

sys.path.insert(0, "/ap/harness/protocols")
sys.path.insert(0, "/gateway")

import constraints  # noqa: E402
import ours_lib  # noqa: E402
import ours_serve  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SERVED = ours_lib.SERVED
VOCAB = 129280


class CapturingEngine(ours_serve.MockEngine):
    def generate(self, engine_input, sampling_params, request_id, **kw):
        self.captured.append({"engine_input": engine_input, "sampling_params": sampling_params,
                              "request_id": request_id, "kwargs": kw})

        async def _nothing():
            return
            yield  # noqa: unreachable

        return _nothing()


def build_state():
    st = ours_lib.state()

    async def build():
        from vllm.entrypoints.generate.api_router import init_generate_state
        from vllm.entrypoints.scale_out.factories import init_scale_out_state

        eng = CapturingEngine(st.vllm_config, st.online_renderer.renderer)
        st.engine_client = eng
        st.args.structured_outputs_config.reasoning_parser = st.args.reasoning_parser
        await init_generate_state(eng, st, st.args, None, ("generate",))
        init_scale_out_state(st, st.args, eng, None)
        return eng

    eng = asyncio.get_event_loop().run_until_complete(build())
    # what `vllm serve` does when it builds the engine config (engine/arg_utils.py:2521-2523)
    st.vllm_config.structured_outputs_config.reasoning_parser = "deepseek_v41"
    return st, eng


def gateway_request(dr, enc, body):
    opts = dr.ConversionOptions(default_thinking_mode=False).with_model_alias(SERVED, dr.ModelVersion.DeepseekV41Flash)
    conv = dr.ChatCompletionRequest(json.dumps(dict(body, model=SERVED))).convert(opts)
    ids = enc.encode(conv.conversation)
    so = constraints.structured_outputs_for(conv, thinking=conv.conversation.thinking_mode)
    return conv, ids, so


def popcount_row(row):
    return int(sum(bin(int(x) & 0xFFFFFFFF).count("1") for x in row[: (VOCAB + 31) // 32]))


def bit(row, t):
    return bool((int(row[t // 32]) >> (t % 32)) & 1)


def replay(mgr, req, script, tok):
    """Scheduler order per decode step; returns one record per scripted token."""
    rid = req.request_id
    sor = req.structured_output_request
    t0 = time.time()
    while not sor.is_grammar_ready:
        if time.time() - t0 > 60:
            raise TimeoutError("grammar compile")
        time.sleep(0.01)
    grammar = sor.grammar
    req.num_computed_tokens = req.num_prompt_tokens  # final prefill chunk scheduled
    rows = []
    for t in script:
        gate_before = sor.reasoning_ended
        bm = mgr.grammar_bitmask({rid: req}, [rid], {})  # Scheduler.get_grammar_bitmask
        row = bm[0]
        n = popcount_row(row)
        req.append_output_token_ids(t)
        req.num_computed_tokens += 1
        adv = None
        if mgr.should_advance(req, new_token_ids=[t]):  # Scheduler.update_from_output
            toks = mgr.trim_reasoning_for_advance(req, [t])
            adv = (tok.decode(toks), grammar.accept_tokens(rid, toks) if toks else True)
        rows.append({"tok": tok.decode([t]), "id": t, "gate_before": gate_before, "gate_after": sor.reasoning_ended,
                     "masked": n < VOCAB, "n_allowed": n, "allowed": bit(row, t), "advanced": adv,
                     "terminated": grammar.is_terminated()})
    return rows


def main():
    import deepseek_recipe as dr

    from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
    from vllm.reasoning import ReasoningParserManager
    from vllm.v1.engine.input_processor import InputProcessor
    from vllm.v1.request import Request
    from vllm.v1.structured_output import StructuredOutputManager

    st, eng = build_state()
    tok = st.online_renderer.renderer.get_tokenizer()
    enc = dr.DeepseekV41Encoding().with_tokenizer(dr.Tokenizer.from_file(os.path.join(ours_lib.MODEL, "tokenizer.json")))
    ip = InputProcessor(st.vllm_config, st.online_renderer.renderer)
    mgr = StructuredOutputManager(st.vllm_config)
    reasoner_cls = ReasoningParserManager.get_reasoning_parser("deepseek_v41")
    out = {"config": {
        "structured_outputs_config": repr(st.vllm_config.structured_outputs_config),
        "manager_reasoner_cls": getattr(mgr.reasoner_cls, "__name__", None),
        "enable_in_reasoning": mgr.enable_in_reasoning,
    }}

    def ids(s):
        return tok.encode(s, add_special_tokens=False)

    weather = {"type": "function", "function": {"name": "get_weather", "description": "Weather.", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
    weather_strict = {"type": "function", "function": {"name": "get_weather", "description": "Weather.", "strict": True,
                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}
    block = ('<｜DSML｜ invoke name="get_weather">\n<｜DSML｜ parameter name="city" string="true">Paris'
             '</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')
    answer = '{"capital": "Paris"}'
    cases = {
        "chat_json_object": ({"messages": [{"role": "user", "content": "Reply in json: the capital of France?"}],
                              "response_format": {"type": "json_object"}},
                             ids(answer) + [1]),
        "chat_json_object_first_token_prose": ({"messages": [{"role": "user", "content": "Reply in json: the capital of France?"}],
                                                "response_format": {"type": "json_object"}},
                                               ids("Paris") + [1]),
        "thinking_json_object": ({"messages": [{"role": "user", "content": "Reply in json: the capital of France?"}],
                                  "thinking": {"type": "enabled"}, "response_format": {"type": "json_object"}},
                                 ids("The user wants JSON. Hello") + [128822] + ids(answer) + [1]),
        "chat_required_prefill": ({"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [weather],
                                   "tool_choice": "required"},
                                  ids(block) + [1]),
        "thinking_auto_strict": ({"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [weather_strict],
                                  "thinking": {"type": "enabled"}},
                                 ids("Need the tool.") + [128822] + ids("Checking.\n\n<｜DSML｜ calls>\n" + block) + [1]),
    }
    out["cases"] = {}
    for name, (body, script) in cases.items():
        rec = {}
        try:
            conv, prompt_ids, so = gateway_request(dr, enc, body)
            thinking = conv.conversation.thinking_mode
            rec["thinking"] = thinking
            rec["prompt_tail"] = tok.decode(prompt_ids[-8:])
            rec["structured_outputs"] = so if not isinstance(so, dict) or "structural_tag" not in so else {"structural_tag": "<%d chars>" % len(so["structural_tag"])}
            payload = {"model": SERVED, "token_ids": prompt_ids,
                       "sampling_params": {"temperature": 1.0, "max_tokens": 64, "structured_outputs": so}, "stream": True}
            greq = GenerateRequest.model_validate(json.loads(json.dumps(payload)))  # the HTTP body
            eng.captured.clear()
            res = asyncio.get_event_loop().run_until_complete(st.serving_tokens.serve_tokens(greq, None))
            cap = eng.captured[-1]
            rec["A_serve_tokens_result"] = type(res).__name__
            rec["A_generate_kwargs"] = sorted(cap["kwargs"])
            rec["A_reasoning_kwargs_passed"] = {k: cap["kwargs"][k] for k in ("reasoning_ended", "reasoning_parser_kwargs")
                                                if k in cap["kwargs"]}
            sp = cap["sampling_params"]
            ecr = ip.process_inputs(cap["request_id"], cap["engine_input"], sp, supported_tasks=("generate",))
            rec["B_engine_core_request"] = {"reasoning_ended": ecr.reasoning_ended,
                                            "reasoning_parser_kwargs": ecr.reasoning_parser_kwargs,
                                            "backend": ecr.sampling_params.structured_outputs._backend,
                                            "prompt_len": len(ecr.prompt_token_ids)}
            req = Request.from_engine_core_request(ecr, block_hasher=None)
            mgr.grammar_init(req)
            rec["C_token_in"] = replay(mgr, req, script, tok)
            # chat path: explicit reasoning_ended from the request's parser (serving.py:363-364)
            ctk = {"chat_template_kwargs": {"thinking": thinking}}
            chat_gate = reasoner_cls(tok, **ctk).is_reasoning_end(prompt_ids)
            ecr2 = ip.process_inputs(cap["request_id"] + "-chat", cap["engine_input"], sp.clone(), supported_tasks=("generate",))
            ecr2.reasoning_ended, ecr2.reasoning_parser_kwargs = chat_gate, ctk
            req2 = Request.from_engine_core_request(ecr2, block_hasher=None)
            mgr.grammar_init(req2)
            rec["C_chat_reasoning_ended"] = chat_gate
            chat_rows = replay(mgr, req2, script, tok)
            rec["C_same_as_chat_path"] = [(r["masked"], r["n_allowed"], r["allowed"], r["advanced"]) for r in rec["C_token_in"]] == \
                [(r["masked"], r["n_allowed"], r["allowed"], r["advanced"]) for r in chat_rows]
            # reasoner without kwargs (token-in) vs with thinking kwargs: same verdict on the prompt
            rec["is_reasoning_end(prompt)"] = {
                "no_kwargs (token-in)": reasoner_cls(tok).is_reasoning_end(prompt_ids),
                "thinking=True": reasoner_cls(tok, chat_template_kwargs={"thinking": True}).is_reasoning_end(prompt_ids),
                "thinking=False": reasoner_cls(tok, chat_template_kwargs={"thinking": False}).is_reasoning_end(prompt_ids),
            }
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["trace"] = traceback.format_exc()[-2500:]
        out["cases"][name] = rec

    # payload shapes accepted / refused by the endpoint's body model + structured-output validation
    from vllm.config.structured_outputs import StructuredOutputsConfig

    shapes = {
        "json_object": {"json_object": True},
        "json (string)": {"json": json.dumps({"type": "object", "properties": {"a": {"type": "integer"}}})},
        "json (object)": {"json": {"type": "object", "properties": {"a": {"type": "integer"}}}},
        "regex": {"regex": r"\d{4}-\d{2}-\d{2}"},
        "structural_tag (JSON string)": {"structural_tag": json.dumps({"type": "structural_tag", "format": {"type": "const_string", "value": "x"}})},
        "structural_tag (object, refused)": {"structural_tag": {"type": "structural_tag", "format": {"type": "const_string", "value": "x"}}},
        "two constraints (refused)": {"json_object": True, "regex": "a"},
        "json_object false (refused)": {"json_object": False},
        "dangling $ref (refused)": {"json": json.dumps({"type": "object", "properties": {"g": {"$ref": "#/$defs/G"}}})},
    }
    out["payload_shapes"] = {}
    for name, so in shapes.items():
        try:
            greq = GenerateRequest.model_validate({"token_ids": [1, 2], "sampling_params": {"max_tokens": 4, "structured_outputs": so}})
            sp = greq.sampling_params
            sp._validate_structured_outputs(st.vllm_config.model_config,
                                            StructuredOutputsConfig(backend="xgrammar"), tok)
            out["payload_shapes"][name] = f"accepted ({type(sp.structured_outputs).__name__}, backend {sp.structured_outputs._backend})"
        except Exception as e:  # noqa: BLE001
            out["payload_shapes"][name] = f"refused: {type(e).__name__}: {str(e).splitlines()[0][:160]}"

    with open(os.path.join(HERE, "token_in_gate.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=repr)

    print(json.dumps(out["config"], indent=1))
    for name, rec in out["cases"].items():
        print(f"== {name}")
        if "error" in rec:
            print("  ERROR", rec["error"])
            print(rec["trace"])
            continue
        print(f"  thinking={rec['thinking']} prompt tail={rec['prompt_tail']!r} structured_outputs={rec['structured_outputs']}")
        print(f"  A generate() kwargs={rec['A_generate_kwargs']} reasoning kwargs passed={rec['A_reasoning_kwargs_passed']}")
        print(f"  B {rec['B_engine_core_request']}")
        print(f"  is_reasoning_end(prompt) {rec['is_reasoning_end(prompt)']}; chat path reasoning_ended={rec['C_chat_reasoning_ended']}")
        for i, r in enumerate(rec["C_token_in"]):
            print(f"  step {i:2d} {r['tok']!r:24s} gate {str(r['gate_before']):5s}->{str(r['gate_after']):5s} "
                  f"masked={str(r['masked']):5s} n_allowed={r['n_allowed']:6d} allowed={str(r['allowed']):5s} "
                  f"advanced={r['advanced']} term={r['terminated']}")
        print(f"  token-in rows == chat-path rows: {rec['C_same_as_chat_path']}")
    print("== payload shapes")
    for k, v in out["payload_shapes"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
