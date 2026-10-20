"""Prefill sharing for concurrent sessions: a per-step prefill cap and decode-only steps.

e15's V1 scheduler (v1/core/sched/scheduler.py of the 0909 image) fills each step with up
to --max-num-batched-tokens B tokens: the running requests first, in admission order
(:560-737), then waiting requests (:786-1215). A decoding request takes 1 + k tokens
(4 at DSpark k=3) and k - 1 = 2 more of the input budget (DSpark's drafting slots,
:531-536, :736); a prefilling request takes what is left (:600-602). While one session
prefills a long prompt, every step carries an ~8,170-token chunk (~3 s per chunk on the
perf2 I1 stack, more late in a long prompt) and the other sessions' streams get one step
per chunk. A new turn that arrives meanwhile is not admitted before the prefill's last
chunk: the prefill leaves input_budget at 0 and the loops stop at `input_budget <=
draft_slots` (:562, :787).

DSV41_SCHED_PREFILL_CAP=X (tokens; unset or 0 = off, else >= MIN_CAP; recipe 2048): while the scheduler
holds more than one request (running + waiting, Scheduler.get_request_counts), every
prefilling request schedules at most X tokens per step. The cap is the stock per-request
one: schedule() runs with scheduler_config.long_prefill_token_threshold set to X (or to a
smaller configured --long-prefill-token-threshold) and gets the configured value back
after the call. A prefill alone in the scheduler keeps the whole budget. These are the
semantics of vLLM 0.32's long_prefill_token_threshold (0.32rc1 scheduler.py:640-655: the
cap is not applied when only one request is eligible), so on 0.32 the knob maps to
--long-prefill-token-threshold X. With p prefilling requests a step carries up to p * X
prefill tokens.

Decoder replay tail (decoder_replay.py, end-anchored W >= its exact minimum, armed on this
scheduler): rows before a prompt's last W run layers 0..cut only (21 of 40 on 1017), its
last W rows run every layer. While a prefilling or waiting request's next chunk reaches
its last W rows (remaining < W + X), the cap for that step is X * (cut + 1) / layers, so
a tail chunk costs about what an X-token chunk before the tail costs.

DSV41_SCHED_DECODE_STEPS=n (unset or 0 = off, max MAX_DECODE_STEPS; recipe 1): after a step that
carried prefill tokens while a request was decoding, the next n steps defer prefill work
through the stock data-parallel prefill-balancing path, schedule(throttle_prefills=True)
(:552-556 defer_prefills, :587-591 running chunks skipped, :957-960 new prefills not
admitted). Those steps run the decoding requests alone, at a CUDA-graph size. The stock
path does not defer while requests wait that could not be admitted (prefill_capacity_bound,
:1223-1224); such a step carries prefill and restarts the count.

A step "carried prefill" when a request in it was scheduled more than 1 + k tokens (a
decode step schedules at most 1 + k); a prompt's last chunk of <= 1 + k tokens does not
count.

Output tokens: this is scheduling only. A request alone in the scheduler is scheduled
exactly as stock. With other requests present, chunk boundaries and batch composition
change; greedy outputs can then differ only through the serve's non-batch-invariance.

Install refuses (one LOG_DISARMED line, the scheduler left stock) unless the image's
Scheduler.schedule reads the threshold at its running and waiting sites, takes
throttle_prefills and has the DP deferral (when n > 0), Scheduler.get_request_counts
exists, and AsyncScheduler inherits schedule(). Top-level imports are stdlib only.
"""

from __future__ import annotations

import functools
import os

# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dsv41: sched prefill cap armed"
LOG_DISARMED = "dsv41: sched prefill cap DISABLED ->"

ENV_CAP = "DSV41_SCHED_PREFILL_CAP"
ENV_STEPS = "DSV41_SCHED_DECODE_STEPS"
# Below ~256 rows a prefill step still reads every routed expert (~11 ms per layer at 256
# rows as at 1024), so prefill throughput collapses; smaller values are a typo.
MIN_CAP = 256
MAX_DECODE_STEPS = 8

# Lines of the image's Scheduler.schedule the wrapper relies on (verbatim, stripped).
CONTRACT = (
    "if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:",  # running
    "threshold = self.scheduler_config.long_prefill_token_threshold",  # waiting
)
CONTRACT_STEPS = (
    "throttle_prefills and not self.prefill_capacity_bound",
    "if defer_prefills and request.is_prefill_chunk:",
    "elif defer_prefills and num_computed_tokens < request.num_tokens - 1:",
)


def knobs(env=None) -> tuple[int, int]:
    """(cap tokens, decode-only steps); (0, 0) is stock scheduling."""
    env = os.environ if env is None else env
    cap = int(env.get(ENV_CAP, "") or 0)
    steps = int(env.get(ENV_STEPS, "") or 0)
    if cap != 0 and cap < MIN_CAP:
        raise ValueError(f"{ENV_CAP}={cap} must be 0 (off) or >= {MIN_CAP}")
    if not 0 <= steps <= MAX_DECODE_STEPS:
        raise ValueError(f"{ENV_STEPS}={steps} must be in 0..{MAX_DECODE_STEPS}")
    return cap, steps


def threshold(base: int, cap: int, num_requests: int) -> int:
    """long_prefill_token_threshold for one schedule() call (0 = no per-request cap)."""
    if cap <= 0 or num_requests <= 1:
        return base
    return cap if base <= 0 else min(base, cap)


def carried_prefill(num_scheduled_tokens, num_spec_tokens: int) -> bool:
    """Whether a step scheduled some request more tokens than a decode step can (1 + k)."""
    return any(n > 1 + num_spec_tokens for n in num_scheduled_tokens.values())


def decoding(running) -> bool:
    """A running request past its prefill: the condition of the stock DP deferral (:556)."""
    return any(not r.is_prefill_chunk for r in running)


def replay_geometry(sched) -> tuple[int, int, int]:
    """(tail rows W, layers every prompt row runs, layers) when decoder replay is armed
    end-anchored on this scheduler (decoder_replay.py sets its prefix-cache guard on the KV
    cache manager in Scheduler.__init__), else (0, 1, 1)."""
    guard = getattr(getattr(sched, "kv_cache_manager", None), "_dsv41_replay", None)
    if guard is None:
        return 0, 1, 1
    import decoder_replay as dr

    mc = sched.vllm_config.model_config
    cfg = getattr(mc, "hf_text_config", None) or mc.hf_config
    cut, after, window = dr.geometry(cfg.kv_source_layer_ids, cfg.num_hidden_layers, cfg.sliding_window)
    if dr.mode_for(int(guard.rows), window, after) != "exact":
        return 0, 1, 1
    return int(guard.rows), cut + 1, cut + 1 + after


def step_cap(cap: int, geometry, prefills) -> int:
    """The cap of one step: X * (cut + 1) / layers while a prefilling or waiting request's
    next X-token chunk reaches its last W prompt rows (those run every layer), else X."""
    rows, enc, layers = geometry
    if cap and rows and any(r.num_tokens - r.num_computed_tokens < rows + cap for r in prefills):
        return max(1, cap * enc // layers)
    return cap


def _geometry(sched) -> tuple[int, int, int]:
    """replay_geometry(), once per scheduler; a failure turns tail weighting off, loudly."""
    geo = getattr(sched, "_dsv41_replay_geometry", None)
    if geo is None:
        try:
            geo = replay_geometry(sched)
        except Exception as exc:  # noqa: BLE001 - the cap itself still applies
            print(f"dsv41: sched prefill cap: no decoder replay tail weighting ({exc!r})", flush=True)
            geo = (0, 1, 1)
        if geo[0]:
            print(f"dsv41: sched prefill cap: decoder replay tail of {geo[0]} rows, cap x {geo[1]}/{geo[2]} "
                  "while a chunk reaches it", flush=True)
        sched._dsv41_replay_geometry = geo
    return geo


def contract_problem(schedule_src: str, schedule_params, async_overrides: bool, has_counts: bool,
                     steps: int) -> str | None:
    """Why the wrapper must not install on this scheduler, or None."""
    lines = {line.strip() for line in schedule_src.splitlines()}
    missing = [s for s in CONTRACT + (CONTRACT_STEPS if steps else ()) if s not in lines]
    if missing:
        return f"Scheduler.schedule lacks {missing}"
    if "throttle_prefills" not in schedule_params:
        return "Scheduler.schedule takes no throttle_prefills"
    if async_overrides:
        return "AsyncScheduler overrides schedule()"
    if not has_counts:
        return "Scheduler.get_request_counts is missing"
    return None


def wrap(orig, cap: int, steps: int):
    """Scheduler.schedule with the cap and the decode-only steps (state on the instance)."""

    @functools.wraps(orig)
    def schedule(self, throttle_prefills: bool = False):
        owe = getattr(self, "_dsv41_decode_steps_owed", 0)
        defer = steps > 0 and owe > 0 and decoding(self.running)
        cfg = self.scheduler_config
        base = cfg.long_prefill_token_threshold
        n = sum(self.get_request_counts())
        if cap and n > 1:
            prefills = [r for r in self.running if r.is_prefill_chunk] + list(self.waiting) + list(self.skipped_waiting)
            thr = threshold(base, step_cap(cap, _geometry(self), prefills), n)
        else:
            thr = base
        if thr != base:
            cfg.long_prefill_token_threshold = thr
        try:
            out = orig(self, throttle_prefills or defer)
        finally:
            if thr != base:
                cfg.long_prefill_token_threshold = base
        if steps:
            if carried_prefill(out.num_scheduled_tokens, self.num_spec_tokens):
                self._dsv41_decode_steps_owed = steps if decoding(self.running) else 0
            elif defer:
                self._dsv41_decode_steps_owed = owe - 1
        return out

    schedule._dsv41_prefill_cap = (cap, steps)
    return schedule


def describe(cap: int, steps: int) -> str:
    parts = []
    if cap:
        parts.append(f"cap {cap} tokens per prefilling request while the scheduler holds > 1 request")
    if steps:
        parts.append(f"{steps} decode-only step(s) after each step that carried prefill while a request decodes")
    return "; ".join(parts)


def install(env=None) -> str:
    """'off' (both knobs 0), 'armed', 'already installed' or 'refused' (LOG_DISARMED printed)."""
    import inspect

    cap, steps = knobs(env)
    if not cap and not steps:
        return "off"
    from vllm.v1.core.sched import async_scheduler as async_mod
    from vllm.v1.core.sched import scheduler as sched_mod

    cls = sched_mod.Scheduler
    if getattr(cls.schedule, "_dsv41_prefill_cap", None) is not None:
        return "already installed"
    try:
        src = inspect.getsource(cls.schedule)
        params = tuple(inspect.signature(cls.schedule).parameters)
    except (OSError, TypeError, ValueError) as exc:
        problem = f"Scheduler.schedule source unavailable: {exc!r}"
    else:
        problem = contract_problem(src, params, "schedule" in vars(async_mod.AsyncScheduler),
                                   callable(getattr(cls, "get_request_counts", None)), steps)
    if problem is not None:
        print("dsv41: sched prefill cap DISABLED -> stock scheduling: %s" % problem, flush=True)
        return "refused"
    cls.schedule = wrap(cls.schedule, cap, steps)
    print("dsv41: sched prefill cap armed (%s)" % describe(cap, steps), flush=True)
    return "armed"
