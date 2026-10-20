"""SWA prefix-cache retention of generated tokens (DSV41_PREFIX_REPLY_RETAIN, default 1; 0 = stock e15).

A follow-up whose history repeats the assistant reply verbatim shares the previous
request's prompt and generated tokens. With e15's default --prefix-cache-retention-interval
0 (kv-pack: 4096) a sliding-window group keeps prefix-cache tails only around the prompt's
replay boundary and a shared-prefix junction (SlidingWindowManager.reachable_block_mask):
the blocks the reply fills during decode are never cached, and SlidingWindowManager frees
each one once the 128-token window has slid past it, so by the time the request finishes
the end of the reply cannot be cached any more. The follow-up then hits at the prompt end
at best and prefills the whole reply again.

With the lever, BlockPool.cache_full_blocks also caches every SWA block that holds rows at
or past the prompt end (decode-filled, or the block straddling the prompt end) as it fills,
which e15 does anyway with dense retention (interval None). A follow-up that repeats the
reply then hits within 192 tokens of the shared end. Decode rows run every layer even under
decoder replay, and the prompt rows of the straddling block lie in its valid tail, so every
block this caches is exact. Cost: one prefix-cache hash entry per 64 generated tokens and SWA
group; cached blocks stay free blocks (evicted on reuse).

Scheduler side (EngineCore). install() wraps the classes in every process, so every rank's
log carries LOG_ENGAGED. Top-level imports are stdlib only. sitecustomize calls install().
"""

from __future__ import annotations

import os

# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dsv41: prefix reply retain armed"
LOG_DISARMED = "dsv41: prefix reply retain DISABLED ->"

ENV = "DSV41_PREFIX_REPLY_RETAIN"
# Attribute on the scheduler's BlockPool: the SWA KV-cache group ids.
ATTR = "_dsv41_reply_groups"


def enabled(env=None) -> bool:
    """Unset/empty or 1 = on (the default), 0 = stock e15."""
    env = os.environ if env is None else env
    raw = (env.get(ENV, "") or "").strip()
    if raw not in ("", "0", "1"):
        raise ValueError(f"{ENV}={raw!r}: 1 (default) or 0")
    return raw != "0"


def reply_mask(given, num_cached_blocks: int, num_full_blocks: int, block_size: int, prompt_len: int):
    """``given`` (a cache_full_blocks block_mask over blocks[num_cached_blocks:num_full_blocks];
    None caches every block) with every block holding rows >= prompt_len switched on."""
    if given is None:
        return None
    first = prompt_len // block_size
    return [k or b >= first for k, b in zip(given, range(num_cached_blocks, num_full_blocks))]


def install(env=None) -> str:
    """Wrap Scheduler.__init__ (find the SWA groups) and BlockPool.cache_full_blocks. "off" at 0."""
    import functools
    import inspect

    if not enabled(env):
        return "off"
    from vllm.v1.core import block_pool as bp_mod
    from vllm.v1.core.sched import scheduler as sched_mod
    from vllm.v1.kv_cache_interface import SlidingWindowSpec, iter_layer_specs

    cache_full = bp_mod.BlockPool.cache_full_blocks
    if getattr(cache_full, "_dsv41_reply_retain", False):
        return "already installed"
    names = ("self", "request", "blocks", "num_cached_blocks", "num_full_blocks", "block_size",
             "kv_cache_group_id", "block_mask")
    if tuple(inspect.signature(cache_full).parameters) != names:
        reason = f"BlockPool.cache_full_blocks{inspect.signature(cache_full)} is not the pinned one"
        print("dsv41: prefix reply retain DISABLED -> %s; stock SWA prefix-cache retention" % reason, flush=True)
        raise RuntimeError(reason)
    sched_init = sched_mod.Scheduler.__init__

    @functools.wraps(sched_init)
    def scheduler_init(self, *args, **kwargs):
        sched_init(self, *args, **kwargs)
        kvm = getattr(self, "kv_cache_manager", None)
        if kvm is None:
            return
        groups = frozenset(
            i for i, g in enumerate(self.kv_cache_config.kv_cache_groups)
            if any(isinstance(s, SlidingWindowSpec) for s in iter_layer_specs(g.kv_cache_spec))
        )
        setattr(kvm.block_pool, ATTR, groups)
        print("dsv41: prefix reply retain: decode-filled blocks of %d SWA groups enter the prefix cache"
              % len(groups), flush=True)

    @functools.wraps(cache_full)
    def cache_full_blocks(self, request, blocks, num_cached_blocks, num_full_blocks, block_size,
                          kv_cache_group_id, block_mask=None):
        if kv_cache_group_id in getattr(self, ATTR, ()):
            block_mask = reply_mask(block_mask, num_cached_blocks, num_full_blocks, block_size,
                                    int(request.num_prompt_tokens))
        return cache_full(self, request, blocks, num_cached_blocks, num_full_blocks, block_size,
                          kv_cache_group_id, block_mask=block_mask)

    cache_full_blocks._dsv41_reply_retain = True
    sched_mod.Scheduler.__init__ = scheduler_init
    bp_mod.BlockPool.cache_full_blocks = cache_full_blocks
    print("dsv41: prefix reply retain armed (SWA blocks at or past the prompt end are cached as they fill)",
          flush=True)
    return "BlockPool.cache_full_blocks caches decode-filled SWA blocks"
