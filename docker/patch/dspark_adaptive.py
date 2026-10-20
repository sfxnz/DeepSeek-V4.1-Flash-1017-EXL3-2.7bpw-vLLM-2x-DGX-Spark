"""Adaptive DSpark verify depth at fixed shapes (DSV41_DSPARK_ADAPTIVE=1, default off).

DSpark drafts k = NUM_SPECULATIVE_TOKENS tokens per request in one parallel
block (one draft forward over k rows, then k sequential Markov steps), and the
target verifies all k every step: k + 1 rows per request. With the lever each
request verifies only its confident prefix:

  m = the number of leading drafts whose survival (the running product of the
      DSpark confidence head's per-draft acceptance probabilities) is at least
      DSV41_DSPARK_TAU (default 0.15), clamped to [1, DSV41_DSPARK_VERIFY_MAX]
      (default k). TAU=0 turns the confidence cut off (VERIFY_MAX alone).
      A comma list gives one TAU per batch size (1, 2, ...; the last value
      holds for larger batches), e.g. 0.15,0.15,0.3,0.3: a larger batch pays
      more per verified row, so it can cut harder.

Shapes never change: every request still runs k + 1 verify rows, so the
captured CUDA graphs, the capture sizes and every kernel row contract stay as
they are. Rows past m (local position j > m; j = 0 is the anchor, j = i is
draft i) are masked:

  routed MoE         a masked row takes its anchor row's top-k expert ids with
                     weight 0, so it adds no distinct expert: no expert bytes.
  rejection sampler  a masked row's draft id becomes -1, which the stock
                     kernels always reject (greedy stores the argmax of row m;
                     sampled draws from row m's target distribution). The step
                     emits exactly what k' = m speculation with the same
                     drafts emits, so greedy stays greedy and sampling keeps
                     the target distribution (Leviathan et al.; m is fixed by
                     the drafter before the target runs).

The sampler reads logits rows 0..m only, and those depend on input rows 0..m
only (attention is causal, everything else is per token). What masked rows
write (target KV, the compressor ring, the draft context KV) is what rejected
rows write under fixed k, and the next step starts at or before the first
masked position and rewrites it before anything reads it.

The confidence signal: DSparkSpeculator computes sigmoid(confidence_head)
into draft_token_confidence_probs inside the captured draft graph when its
instance flag enable_adaptive_verification is set. vLLM's own
AdaptiveVerificationManager (which trims query lengths: the SM120 indexer
does not take that) is created from the same flag inside initialize_kv_cache,
so the lever sets the flag only after initialize_kv_cache returns. The
manager stays off, the config flag stays off, and the stock confidence code is
captured with the draft graphs.

This is not the rejected DSV41_DSPARK_CONF_GATE: that one scaled the Markov
bias by the confidence (it changed the drafts) and still verified all k rows,
so it could only lose acceptance (Sept: 2.40 -> 2.14, REVERT). This lever
never changes a draft; it stops paying expert reads for rows the drafter does
not believe in, and NUM_SPECULATIVE_TOKENS=5 lets confident steps go deeper.

decode_levers.install() wraps DSparkSpeculator.__init__; the first speculator
built in a process (a worker) binds the class-level hooks, inert until armed:
  GPUModelRunner.initialize_kv_cache  arm: checks, bind target routers, flag
  GPUModelRunner.prepare_inputs       mark this step's masked input rows
  FusedMoERouter.select_experts       remap masked rows (target routers only)
  rejection_sampler.rejection_sample  -1 for masked drafts
  DSparkSpeculator.propose            verify widths for the drafts just made
No hook syncs the host. Top-level imports are stdlib only.
"""

from __future__ import annotations

import math
import os

# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dsv41: dspark adaptive verify armed"
LOG_DISARMED = "dsv41: dspark adaptive verify DISABLED ->"

ENV_FLAG = "DSV41_DSPARK_ADAPTIVE"
ENV_TAU = "DSV41_DSPARK_TAU"
ENV_VERIFY_MAX = "DSV41_DSPARK_VERIFY_MAX"
DEFAULT_TAU = 0.15  # TensorFold C5 default (cut where the confidence product < 0.15)
# Levers that change the drafts or their confidences behind this one's back.
CONFLICTS = ("DSV41_DSPARK_CONF_GATE", "DSV41_DSPARK_TAIL_NGRAM")

_STATE = None  # the armed AdaptiveVerify, else None


def enabled(env) -> bool:
    return (env.get(ENV_FLAG, "0") or "0") == "1"


def settings(env, k: int) -> tuple[tuple[float, ...], int]:
    """(tau per batch size, verify cap) for draft block size k. Raises ValueError on bad values."""
    raw = (env.get(ENV_TAU, "") or "").strip()
    try:
        taus = (DEFAULT_TAU,) if not raw else tuple(float(t) for t in raw.split(","))
    except ValueError:
        taus = ()
    if not taus or any(math.isnan(t) or not 0.0 <= t <= 1.0 for t in taus):
        raise ValueError(f"{ENV_TAU}={raw!r}: want a float in [0, 1], or a comma list of them per batch size")
    raw = (env.get(ENV_VERIFY_MAX, "") or "").strip()
    cap = k if not raw else int(raw)
    if not 1 <= cap <= k:
        raise ValueError(f"{ENV_VERIFY_MAX}={raw!r}: want an int in [1, {k}] (k = NUM_SPECULATIVE_TOKENS)")
    return taus, cap


def conflicts(env) -> list[str]:
    return [name for name in CONFLICTS if (env.get(name, "0") or "0").strip() not in ("", "0")]


def verify_width(confidence, tau: float, cap: int):
    """[R, k] per-draft acceptance probabilities -> [R] int32 verify width in [1, cap].

    Survival is non-increasing, so counting the positions at or above tau counts
    the leading ones. NaN survival counts as below tau.
    """
    import torch

    survival = torch.cumprod(confidence.float(), dim=1)
    return (survival >= tau).sum(dim=1).clamp_(1, cap).to(torch.int32)


class AdaptiveVerify:
    """Per-rank state. Buffers are persistent, so captured graphs read the current step's values."""

    def __init__(self, tau, cap: int, k: int, max_num_reqs: int, max_num_tokens: int, device):
        import torch

        # tau: one float, or one per batch size (the last holds for larger batches).
        self.taus = (float(tau),) if isinstance(tau, (int, float)) else tuple(float(t) for t in tau)
        self.cap, self.k = cap, k
        # Verify width per request state; k (nothing masked) until the drafter records one.
        self.verify_len = torch.full((max_num_reqs,), k, dtype=torch.int32, device=device)
        # This step's widths, snapshotted by prepare(): the MoE remap and the rejection
        # mask read the same values even if a record() lands between them.
        self.step_len = self.verify_len.clone()
        self.masked = torch.zeros(max_num_tokens, dtype=torch.bool, device=device)
        self.iota = torch.arange(max_num_tokens, dtype=torch.int64, device=device)
        # Routing source row per input row: itself, or its anchor when masked (src[t] <= t).
        self.src = self.iota.clone()

    def prepare(self, input_batch) -> None:
        """Before the target forward: mark the masked input rows of this step."""
        import torch

        n = int(input_batch.num_tokens_after_padding)
        self.masked[:n].zero_()
        self.src[:n].copy_(self.iota[:n])
        if not input_batch.num_draft_tokens:
            return
        self.step_len.copy_(self.verify_len)
        rows = input_batch.logits_indices
        local = input_batch.expanded_local_pos
        cut = local > self.step_len[input_batch.expanded_idx_mapping]
        self.masked[rows] = cut
        self.src[rows] = torch.where(cut, rows - local, rows)

    def remap(self, topk_weights, topk_ids):
        """Router output, inside the graphs: masked rows take the anchor's experts, weight 0."""
        n = topk_ids.shape[0]
        if n > self.src.numel():
            return topk_weights, topk_ids
        return (
            topk_weights.masked_fill(self.masked[:n].unsqueeze(1), 0.0),
            topk_ids.index_select(0, self.src[:n]),
        )

    def mask_drafts(self, draft_sampled, expanded_idx_mapping, expanded_local_pos):
        """Rejection-sampler input: a masked row's draft id becomes -1 (prepare's predicate and widths)."""
        cut = expanded_local_pos > self.step_len[expanded_idx_mapping]
        return draft_sampled.masked_fill(cut, -1)

    def tau_for(self, num_reqs: int) -> float:
        return self.taus[min(max(num_reqs, 1), len(self.taus)) - 1]

    def record(self, confidence, idx_mapping) -> None:
        """After the draft: widths for the drafts just proposed, by request state.

        The TAU is this batch's: the scheduler sizes the next step from this step's batch too.
        """
        num_reqs = idx_mapping.shape[0]
        tau = self.tau_for(num_reqs)
        self.verify_len[idx_mapping] = verify_width(confidence[:num_reqs], tau, self.cap)


def _refusal(runner) -> str | None:
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    spec = getattr(runner, "speculator", None)
    if not isinstance(spec, DSparkSpeculator):
        return "needs the DSpark speculator (SPEC=dspark)"
    if int(runner.num_speculative_steps) < 2:
        return "NUM_SPECULATIVE_TOKENS < 2 leaves no draft to cut"
    if type(getattr(runner, "rejection_sampler", None)) is not RejectionSampler:
        return "needs the stock V2 RejectionSampler"
    if getattr(runner, "adaptive_verification", None) is not None:
        return "vLLM's own adaptive verification is on"
    if int(runner.model_state.num_new_sampled_tokens_per_step) != 1:
        return "needs one bonus token per step (logit row 0 = anchor)"
    par = runner.parallel_config
    if (
        par.pipeline_parallel_size != 1
        or par.data_parallel_size != 1
        or par.enable_expert_parallel
        or getattr(par, "prefill_context_parallel_size", 1) != 1
        or getattr(par, "decode_context_parallel_size", 1) != 1
        or getattr(par, "use_ubatching", False)
    ):
        return "router rows must be the batch rows (no PP, DP, EP, PCP, DCP or DBO)"
    if spec.model.model.confidence_head is None:
        return "the DSpark checkpoint has no confidence head"
    return None


def _target_routers(model) -> list:
    from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter

    seen, routers = set(), []
    for module in model.modules():
        router = getattr(module, "router", None)
        if isinstance(router, FusedMoERouter) and id(router) not in seen:
            seen.add(id(router))
            routers.append(router)
    return routers


def arm(runner, env) -> str:
    """After initialize_kv_cache: bind every hook to one state, or none."""
    global _STATE
    if _STATE is not None:
        return "already armed"
    try:
        reason = _refusal(runner)
        if reason:
            raise RuntimeError(reason)
        k = int(runner.num_speculative_steps)
        taus, cap = settings(env, k)
        routers = _target_routers(runner.model)
        if not routers:
            raise RuntimeError("the target model has no MoE router")
        state = AdaptiveVerify(taus, cap, k, runner.max_num_reqs, runner.max_num_tokens, runner.device)
    except Exception as exc:  # noqa: BLE001 - a refusal leaves the stock serve
        print(f"dsv41: dspark adaptive verify DISABLED -> stock verify: {exc!r}", flush=True)
        return "disarmed"
    for router in routers:
        router._dsv41_av_remap = state.remap
    runner.speculator.enable_adaptive_verification = True  # confidences in the draft graph
    _STATE = state
    mode = "A/A, nothing masked" if not any(taus) and cap == k else "masking"
    print(
        f"dsv41: dspark adaptive verify armed (tau={','.join(f'{t:g}' for t in taus)} verify_max={cap} k={k}, {mode}; "
        f"masked rows reuse their anchor's experts in {len(routers)} MoE layers)",
        flush=True,
    )
    return "armed"


def install(env=None) -> str:
    """Hook DSparkSpeculator.__init__; the runner hooks bind there (worker processes only)."""
    env = os.environ if env is None else env
    if not enabled(env):
        return "off"
    found = conflicts(env)
    if found:
        raise ValueError(f"{ENV_FLAG}=1 cannot combine with {found}")
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    if getattr(DSparkSpeculator.__init__, "_dsv41_av", False):
        return "already installed"
    stock_init = DSparkSpeculator.__init__

    def __init__(self, *args, **kwargs):
        stock_init(self, *args, **kwargs)
        try:
            _install_runner_hooks(env)
        except Exception as exc:  # noqa: BLE001 - unarmed hooks are inert; the serve stays stock
            print(f"dsv41: dspark adaptive verify DISABLED -> stock verify: hook install {exc!r}", flush=True)

    __init__._dsv41_av = True
    DSparkSpeculator.__init__ = __init__
    return "hooks bind at DSpark speculator init; arms after initialize_kv_cache"


def _install_runner_hooks(env) -> None:
    """Class-level hooks, once per process. Nothing acts until arm() succeeds."""
    from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
    from vllm.v1.worker.gpu import model_runner as mr
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rs
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    runner_cls = mr.GPUModelRunner
    if getattr(runner_cls.initialize_kv_cache, "_dsv41_av", False):
        return

    stock_select = FusedMoERouter.select_experts

    def select_experts(self, *args, **kwargs):
        topk_weights, topk_ids = stock_select(self, *args, **kwargs)
        remap = getattr(self, "_dsv41_av_remap", None)
        if remap is None:
            return topk_weights, topk_ids
        return remap(topk_weights, topk_ids)

    stock_rejection_sample = rs.rejection_sample

    def rejection_sample(
        target_logits, draft_logits, draft_sampled, cu_num_logits, pos, idx_mapping,
        expanded_idx_mapping, expanded_local_pos, *args, **kwargs,
    ):
        state = _STATE
        if state is not None:
            draft_sampled = state.mask_drafts(draft_sampled, expanded_idx_mapping, expanded_local_pos)
        return stock_rejection_sample(
            target_logits, draft_logits, draft_sampled, cu_num_logits, pos, idx_mapping,
            expanded_idx_mapping, expanded_local_pos, *args, **kwargs,
        )

    stock_propose = DSparkSpeculator.propose

    def propose(self, input_batch, *args, **kwargs):
        draft_tokens = stock_propose(self, input_batch, *args, **kwargs)
        state = _STATE
        if state is not None:
            state.record(self.draft_token_confidence_probs, input_batch.idx_mapping)
        return draft_tokens

    stock_prepare = runner_cls.prepare_inputs

    def prepare_inputs(self, *args, **kwargs):
        input_batch = stock_prepare(self, *args, **kwargs)
        state = _STATE
        if state is not None:
            state.prepare(input_batch)
        return input_batch

    stock_init_kv = runner_cls.initialize_kv_cache

    def initialize_kv_cache(self, *args, **kwargs):
        out = stock_init_kv(self, *args, **kwargs)
        arm(self, env)
        return out

    initialize_kv_cache._dsv41_av = True
    FusedMoERouter.select_experts = select_experts
    rs.rejection_sample = rejection_sample
    DSparkSpeculator.propose = propose
    runner_cls.prepare_inputs = prepare_inputs
    runner_cls.initialize_kv_cache = initialize_kv_cache  # last: nothing arms before every hook is in
