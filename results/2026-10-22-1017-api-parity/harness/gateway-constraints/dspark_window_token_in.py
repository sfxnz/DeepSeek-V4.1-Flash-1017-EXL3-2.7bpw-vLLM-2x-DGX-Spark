"""DSpark-sized windows (k=5) on the token-in path: StructuredOutputManager (real __init__, pack
tokenizer, --reasoning-parser deepseek_v41, num_speculative_tokens=5) driven like the scheduler
does under spec decode: update_draft_token_ids_in_output (should_advance + validate_tokens, -1 pad),
grammar_bitmask with the drafts, then update_from_output (should_advance(new) / trim / accept).
Thinking prompt + json_object; reasoning ends mid-window. Token-in request (reasoning_ended None,
no parser kwargs) vs the chat path's explicit reasoning_ended=False + chat_template_kwargs.
Run: docker ... /h/dspark_window_token_in.py  (mounts as in the repo image check, /tok = pack tokenizer)
"""
import json
import sys
import time
from types import SimpleNamespace

import vllm.v1.structured_output as so_mod
from vllm.config.structured_outputs import StructuredOutputsConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer
from vllm.v1.request import Request

K, VOCAB = 5, 129280
tok = DeepseekV41Tokenizer.from_pretrained("/tok")
so_mod.cached_tokenizer_from_config = lambda model_config: tok
cfg = SimpleNamespace(
    structured_outputs_config=StructuredOutputsConfig(reasoning_parser="deepseek_v41"),
    parallel_config=SimpleNamespace(distributed_executor_backend="mp"),
    scheduler_config=SimpleNamespace(max_num_seqs=4),
    model_config=SimpleNamespace(skip_tokenizer_init=False, is_diffusion=False, get_vocab_size=lambda: VOCAB),
    speculative_config=SimpleNamespace(num_speculative_tokens=K), num_speculative_tokens=K)
mgr = so_mod.StructuredOutputManager(cfg)
eos, think_end = tok.eos_token_id, tok.encode("</think>", add_special_tokens=False)[0]


def ids(s):
    return tok.encode(s, add_special_tokens=False)


def n_allowed(row):
    return int(sum(bin(int(x) & 0xFFFFFFFF).count("1") for x in row[: VOCAB // 32]))


def run(rid, reasoning_ended, kwargs):
    sp = SamplingParams(max_tokens=64, structured_outputs=StructuredOutputsParams(json_object=True))
    sp._validate_structured_outputs(SimpleNamespace(is_diffusion=False), cfg.structured_outputs_config, tok)
    prompt = ids("<｜begin▁of▁sentence｜><｜User｜>Reply in JSON: capital of France?<｜Assistant｜><think>")
    req = Request(rid, prompt, sp, None, reasoning_ended=reasoning_ended, reasoning_parser_kwargs=kwargs)
    sor = req.structured_output_request
    mgr.grammar_init(req)
    while not sor.is_grammar_ready:
        time.sleep(0.01)
    g = sor.grammar
    # final prefill step: bitmask (no drafts), sample the first reasoning token
    first = ids("The")[0]
    req.num_computed_tokens = req.num_prompt_tokens
    rows0 = mgr.grammar_bitmask({rid: req}, [rid], {})
    req.append_output_token_ids(first)
    req.num_computed_tokens += 1
    if mgr.should_advance(req, new_token_ids=[first]):
        g.accept_tokens(rid, mgr.trim_reasoning_for_advance(req, [first]))
    log = [{"step": "prefill", "rows_n_allowed": [n_allowed(r) for r in rows0]}]
    windows = [
        (ids(" user wants JSON for")[:K], ids(" Paris")[0]),
        (ids(" ok")[:1] + [think_end] + ids('{"capital":')[: K - 2], ids(' "')[0]),
        (ids('Paris"}')[:2] + ids(" Hello")[:1] + ids("xx")[:2], eos),
    ]
    for drafts, bonus in windows:
        drafts = list(drafts)
        if mgr.should_advance(req):  # update_draft_token_ids_in_output
            valid = g.validate_tokens(drafts)
            drafts = valid + [-1] * (len(drafts) - len(valid))
        rows = mgr.grammar_bitmask({rid: req}, [rid], {rid: drafts})
        accepted = []
        for t in drafts:
            if t == -1:
                break
            accepted.append(t)
        new = accepted + [bonus]
        req.append_output_token_ids(new)
        req.num_computed_tokens += len(new)
        adv = None
        if mgr.should_advance(req, new_token_ids=new):
            a = mgr.trim_reasoning_for_advance(req, new)
            adv = (tok.decode(a), g.accept_tokens(rid, a) if a else True)
        log.append({"drafts": [t if t == -1 else tok.decode([t]) for t in drafts], "bonus": tok.decode([bonus]),
                    "rows_n_allowed": [n_allowed(r) for r in rows], "advanced": adv, "gate": sor.reasoning_ended,
                    "terminated": g.is_terminated()})
    return log


tin = run("token-in", None, None)
chat = run("chat", False, {"chat_template_kwargs": {"thinking": True}})
print(json.dumps({"token_in": tin}, ensure_ascii=False, indent=1))
print("token-in == chat path:", tin == chat)
