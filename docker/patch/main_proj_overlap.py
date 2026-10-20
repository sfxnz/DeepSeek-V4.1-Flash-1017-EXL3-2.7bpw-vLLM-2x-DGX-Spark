"""DSpark main_proj beside the target's logits all-gather and sampler (DSV41_MAIN_PROJ_OVERLAP=1, default off).

After the target graph, the eager section runs lm_head, the logits all-gather, the sampler / rejection
chain, then propose(): combine_hidden_states (cat of the 3 target aux hidden states, main_proj MXFP8
[5120, 15360], main_norm) and the rest of the draft prep. main_proj only reads the target graph's aux
hidden states, not the sampled tokens, yet it waits for the all-gather and the sampler. 1017 B0
traces (lm_head end -> main_proj start; main_proj), rank 0 / rank 1: c=1 149 / 215 us; 346 / 355 us.
c=4 320 / 382 us; 406 / 411 us. The all-gather and the sampler are latency-bound, main_proj is
bandwidth-bound, so they can overlap.

What it does. sample_tokens registers the step's aux hidden states. The first logits gather of the
step (after the lm_head GEMM is enqueued, before its all-gather; never during a CUDA-graph capture)
records an event and runs the unwrapped combine_hidden_states(torch.cat(aux, -1)) on a side stream that
waits for that event. propose()'s combine_hidden_states call then waits for the side stream and returns
that tensor instead of computing it again. Nothing is registered when there is no DSpark-style
speculator (no combine_hidden_states), no aux hidden states, or on a non-last PP rank; a step whose
gather never runs (sharded sampler) or whose combine input does not match computes as stock.

Numerics: the same ops on the same inputs in the same order on another stream: bit-exact. The first
DSV41_MAIN_PROJ_OVERLAP_VERIFY combines (default 8, minimum 1) also run the stock op on the main stream
and compare raw bits (this also checks the stream ordering); a mismatch prints one LOG_DISARMED line
and every later step is stock.

Top-level imports are stdlib only; decode_levers.install() calls install() when the env is on.
"""

from __future__ import annotations

import os

# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dsv41: main_proj overlap self-check bit-exact"
LOG_DISARMED = "dsv41: main_proj overlap DISABLED ->"

ENV = "DSV41_MAIN_PROJ_OVERLAP"
VERIFY_ENV = "DSV41_MAIN_PROJ_OVERLAP_VERIFY"
DEFAULT_VERIFY = 8

_STATE = {"armed": True, "verify_left": DEFAULT_VERIFY, "pending": None, "ready": None, "side": None,
          "engaged": False, "combine": None, "cls": None}


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return (env.get(ENV, "0") or "0") == "1"


def verify_calls(env=None) -> int:
    env = os.environ if env is None else env
    return max(1, int(env.get(VERIFY_ENV, "") or DEFAULT_VERIFY))


def _disarm(exc) -> None:
    if _STATE["armed"]:
        _STATE["armed"] = False
        print(f"dsv41: main_proj overlap DISABLED -> stock order: {exc!r}", flush=True)


def step_aux(runner, cls):
    """(DSpark model, aux list) to precompute this step, or None."""
    st = getattr(runner, "execute_model_state", None)
    spec = getattr(runner, "speculator", None)
    if st is None or spec is None or cls is None or not getattr(runner, "is_last_pp_rank", True):
        return None
    aux = getattr(st, "aux_hidden_states", None)
    model = getattr(spec, "model", None)
    if not aux or not isinstance(model, cls):
        return None
    return model, list(aux)


def wrap_sample_tokens(orig):
    def sample_tokens(self, *args, **kwargs):
        _STATE["pending"] = step_aux(self, _STATE["cls"]) if _STATE["armed"] else None
        _STATE["ready"] = None
        try:
            return orig(self, *args, **kwargs)
        finally:
            _STATE["pending"] = None
            _STATE["ready"] = None

    sample_tokens._dsv41_main_proj_overlap = True
    return sample_tokens


def fork(torch, model, aux) -> None:
    """Enqueue the stock combine on the side stream after everything enqueued so far on this stream."""
    main = torch.cuda.current_stream()
    if _STATE["side"] is None:
        _STATE["side"] = torch.cuda.Stream()
    side = _STATE["side"]
    start = torch.cuda.Event()
    start.record(main)
    with torch.cuda.stream(side):
        side.wait_event(start)
        res = _STATE["combine"](model, torch.cat(aux, dim=-1))
        done = torch.cuda.Event()
        done.record(side)
    for t in aux:
        t.record_stream(side)
    _STATE["ready"] = (res, done, int(aux[0].shape[0]), sum(int(t.shape[-1]) for t in aux))


def wrap_gather(torch, orig):
    def _gather_logits(self, logits):
        p = _STATE["pending"]
        if p is not None and _STATE["armed"] and not torch.cuda.is_current_stream_capturing():
            _STATE["pending"] = None
            try:
                fork(torch, *p)
            except Exception as exc:  # noqa: BLE001 - the stock combine runs later instead
                _STATE["ready"] = None
                _disarm(exc)
        return orig(self, logits)

    _gather_logits._dsv41_main_proj_overlap = True
    return _gather_logits


def wrap_combine(torch, orig):
    def combine_hidden_states(self, aux_hidden_states):
        r = _STATE["ready"]
        _STATE["ready"] = None
        if r is None or not _STATE["armed"]:
            return orig(self, aux_hidden_states)
        res, done, rows, width = r
        if tuple(aux_hidden_states.shape) != (rows, width):
            return orig(self, aux_hidden_states)  # not this step's input
        main = torch.cuda.current_stream()
        main.wait_event(done)
        res.record_stream(main)
        if _STATE["verify_left"] > 0:
            ref = orig(self, aux_hidden_states)
            if not torch.equal(res.view(torch.int16), ref.view(torch.int16)):
                _disarm(RuntimeError(f"side-stream combine != stock (rows={rows})"))
                return ref
            _STATE["verify_left"] -= 1
            if not _STATE["engaged"]:
                _STATE["engaged"] = True
                print(f"dsv41: main_proj overlap self-check bit-exact (rows={rows}, width={width})", flush=True)
        return res

    combine_hidden_states._dsv41_main_proj_overlap = True
    return combine_hidden_states


def install() -> str:
    import torch
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4ForCausalLM
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    if getattr(GPUModelRunner.sample_tokens, "_dsv41_main_proj_overlap", False):
        return "already installed"
    _STATE["verify_left"] = verify_calls()
    _STATE["cls"] = DSparkDeepseekV4ForCausalLM
    _STATE["combine"] = DSparkDeepseekV4ForCausalLM.combine_hidden_states
    DSparkDeepseekV4ForCausalLM.combine_hidden_states = wrap_combine(torch, DSparkDeepseekV4ForCausalLM.combine_hidden_states)
    LogitsProcessor._gather_logits = wrap_gather(torch, LogitsProcessor._gather_logits)
    GPUModelRunner.sample_tokens = wrap_sample_tokens(GPUModelRunner.sample_tokens)
    return f"GPUModelRunner.sample_tokens / LogitsProcessor._gather_logits / DSpark combine wrapped (verify={_STATE['verify_left']})"
