#!/usr/bin/env python3
"""gateway/constraints.py inside the serve image (no GPU).

For every request in tests/fixtures/api_gateway/constraints_cases.json (converted by deepseek-recipe
0.2.0; see tests/test_api_gateway_constraints.py), structured_outputs_for runs on a stub of the
converted request. Each emitted value goes through vLLM's own /inference/v1/generate path: the
GenerateRequest body model, SamplingParams._validate_structured_outputs (backend auto must pick
xgrammar; backend xgrammar must accept it), StructuredOutputRequest's key and XgrammarBackend.compile_grammar
with the pack tokenizer. The fixture's upstream-rendered outputs must be accepted (EOS included) and
the malformed ones refused.

A gate replay then drives the engine's StructuredOutputManager the way the scheduler does for a
token-in request (no reasoning_ended, no parser kwargs) with --reasoning-parser deepseek_v41: a prompt
ending in `</think>` is masked from the first token, one ending in `<think>` only after `</think>`.

    docker run --rm --network none --memory 4g --name dsv41-gw-constraints \\
      -v $PWD:/repo:ro \\
      -v $PWD/docker/patch:/opt/dsv41-patch:ro \\
      -v $PWD/docker/patch/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro \\
      -v <dir with the pack's tokenizer.json, tokenizer_config.json, config.json>:/tok:ro \\
      --entrypoint python3 <image> /repo/tests/api_gateway_constraints_image_check.py /tok

Prints one line per check and exits 1 on any failure.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/api_gateway/constraints_cases.json"
THINK_END = "</think>"


def load_cases() -> list[dict]:
    return json.loads(FIXTURE.read_text())["cases"]


def stub(case: dict):
    """The parts of a deepseek_recipe ConversationRequest that constraints.py reads."""
    c = case["constraint"]
    return SimpleNamespace(
        inference_options=SimpleNamespace(output_constraint=None if c is None else SimpleNamespace(**c)),
        conversation=SimpleNamespace(response_schema=case["response_schema"], thinking_mode=case["thinking"],
                                     messages=[SimpleNamespace(**case["last_message"])]))


def result_kind(so) -> str | None:
    """None, "FALLBACK" or the structured_outputs key."""
    if so is None:
        return None
    if not isinstance(so, dict):
        return repr(so)
    return next(iter(so)) if len(so) == 1 else repr(sorted(so))


def feed(grammar, ids: list[int]) -> str:
    grammar.reset()
    for i, t in enumerate(ids):
        if not grammar.matcher.accept_token(t):
            return f"REJECT@{i}"
    return "ACCEPT" if grammar.matcher.is_terminated() else "INCOMPLETE"


def check_cases(constraints, tok, vocab_size: int) -> list[tuple[str, bool, str]]:
    from vllm.config.structured_outputs import StructuredOutputsConfig
    from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
    from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend
    from vllm.v1.structured_output.request import StructuredOutputRequest

    eos = tok.eos_token_id
    model_config = SimpleNamespace(is_diffusion=False)
    backend = XgrammarBackend(vllm_config=SimpleNamespace(structured_outputs_config=StructuredOutputsConfig(),
                                                          speculative_config=SimpleNamespace(num_speculative_tokens=5)),
                              tokenizer=tok, vocab_size=vocab_size)
    out = []
    for case in load_cases():
        name = case["name"]
        so = constraints.structured_outputs_for(stub(case), thinking=case["thinking"])
        out.append((f"{name}: maps to {case['expect']}", result_kind(so) == case["expect"], repr(so)[:300]))
        if not isinstance(so, dict):
            continue
        try:
            for backend_name in ("auto", "xgrammar"):
                body = json.loads(json.dumps({"token_ids": [0], "sampling_params": {"max_tokens": 8, "structured_outputs": so}}))
                sp = GenerateRequest.model_validate(body).sampling_params
                sp._validate_structured_outputs(model_config, StructuredOutputsConfig(backend=backend_name), tok)
                chosen = sp.structured_outputs._backend
                out.append((f"{name}: /inference/v1/generate accepts it, backend {backend_name} -> {chosen}",
                            chosen == "xgrammar", ""))
            kind, spec = StructuredOutputRequest.from_sampling_params(sp).structured_output_key
            t0 = time.perf_counter()
            grammar = backend.compile_grammar(kind, spec, stop_token_ids={eos})
            out.append((f"{name}: compiles ({kind.name}, {(time.perf_counter() - t0) * 1e3:.0f} ms)", True, ""))
        except Exception as exc:  # noqa: BLE001
            out.append((f"{name}: validates and compiles", False, f"{type(exc).__name__}: {exc}"[:400]))
            continue
        for i, text in enumerate(case["accept"]):
            got = feed(grammar, tok.encode(text, add_special_tokens=False) + [eos])
            out.append((f"{name}: accepts sample {i}", got == "ACCEPT", f"{got} {text[:120]!r}"))
        for i, text in enumerate(case["reject"]):
            got = feed(grammar, tok.encode(text, add_special_tokens=False) + [eos])
            out.append((f"{name}: refuses malformed {i}", got != "ACCEPT", f"{got} {text[:120]!r}"))
    return out


def check_gate(tok, vocab_size: int) -> list[tuple[str, bool, str]]:
    """The scheduler's order per step: grammar_bitmask before sampling, then should_advance /
    trim_reasoning_for_advance / accept_tokens (v1/core/sched/scheduler.py:1787-1809, 1985-2010)."""
    import vllm.v1.structured_output as so_mod
    from vllm.config.structured_outputs import StructuredOutputsConfig
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams
    from vllm.v1.request import Request

    so_mod.cached_tokenizer_from_config = lambda model_config: tok  # the pack tokenizer, no model load
    cfg = SimpleNamespace(
        structured_outputs_config=StructuredOutputsConfig(reasoning_parser="deepseek_v41"),  # run.sh --reasoning-parser
        parallel_config=SimpleNamespace(distributed_executor_backend="mp"),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        model_config=SimpleNamespace(skip_tokenizer_init=False, is_diffusion=False, get_vocab_size=lambda: vocab_size),
        speculative_config=None, num_speculative_tokens=0)
    mgr = so_mod.StructuredOutputManager(cfg)
    eos = tok.eos_token_id
    think_end = tok.convert_tokens_to_ids(THINK_END)

    def ids(s):
        return tok.encode(s, add_special_tokens=False)

    def replay(prompt: str, script: list[int]):
        sp = SamplingParams(max_tokens=64, structured_outputs=StructuredOutputsParams(json_object=True))
        sp._validate_structured_outputs(SimpleNamespace(is_diffusion=False), cfg.structured_outputs_config, tok)
        req = Request("gate", ids(prompt), sp, None)  # token-in: reasoning_ended and parser kwargs stay None
        sor = req.structured_output_request
        start = sor.reasoning_ended
        mgr.grammar_init(req)
        t0 = time.time()
        while not sor.is_grammar_ready and time.time() - t0 < 60:
            time.sleep(0.01)
        req.num_computed_tokens = req.num_prompt_tokens
        masked = []
        for t in script:
            row = mgr.grammar_bitmask({"gate": req}, ["gate"], {})[0]
            masked.append(not all(int(x) & 0xFFFFFFFF == 0xFFFFFFFF for x in row[: vocab_size // 32]))
            if masked[-1] and not (int(row[t // 32]) >> (t % 32)) & 1:
                return start, masked, f"masked out {tok.decode([t])!r}"
            req.append_output_token_ids(t)
            req.num_computed_tokens += 1
            if mgr.should_advance(req, new_token_ids=[t]):
                adv = mgr.trim_reasoning_for_advance(req, [t])
                if adv and not sor.grammar.accept_tokens("gate", adv):
                    return start, masked, f"grammar refused {tok.decode(adv)!r}"
        return start, masked, "terminated" if sor.grammar.is_terminated() else "not terminated"

    answer = ids('{"capital": "Paris"}') + [eos]
    reasoning = ids("The user wants JSON. Hello")
    head = "<｜begin▁of▁sentence｜><｜User｜>Reply in JSON: the capital of France?<｜Assistant｜>"
    out = []
    start, masked, end = replay(head + THINK_END, answer)
    out.append(("gate: token-in request arrives without reasoning_ended", start is None, repr(start)))
    out.append(("gate: prompt ending in </think> is masked from the first token",
                all(masked) and end == "terminated", f"{masked} {end}"))
    start, masked, end = replay(head + "<think>", reasoning + [think_end] + answer)
    n = len(reasoning) + 1
    out.append(("gate: prompt ending in <think> is free until </think>, masked after it",
                not any(masked[:n]) and all(masked[n:]) and end == "terminated", f"{masked} {end}"))
    return out


def report(results) -> int:
    bad = 0
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" -- {detail}" if detail and not ok else ""))
        bad += not ok
    print(f"{len(results) - bad}/{len(results)} passed")
    return 1 if bad else 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    tok_dir = Path(argv[0])
    sys.path.insert(0, str(ROOT / "gateway"))
    import constraints
    from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer

    tok = DeepseekV41Tokenizer.from_pretrained(str(tok_dir))
    vocab_size = json.loads((tok_dir / "config.json").read_text())["text_config"]["vocab_size"]  # 129280
    return report(check_cases(constraints, tok, vocab_size) + check_gate(tok, vocab_size))


if __name__ == "__main__":
    sys.exit(main())
