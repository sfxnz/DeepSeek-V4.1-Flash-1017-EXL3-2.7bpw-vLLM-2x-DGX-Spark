"""Prompt-end SWA prefix-cache fallback (DSV41_PREFIX_END_FALLBACK, default 1; 0 = stock e15).

A sliding-window KV-cache group serves a prefix hit at a 128-aligned P only from its 2
window blocks [P-128, P) plus the DSpark (EAGLE) peek block [P, P+64), all cached. e15
caches them only around retained boundaries (SlidingWindowManager.reachable_block_mask).
With the default --prefix-cache-retention-interval 0 (kv-pack's 4096 adds a checkpoint
every 4096 tokens) those are the replay boundary N-1 of an N-token prompt and a
shared-prefix junction J, each floored to 128: at a prompt's end P = (N-1)//128*128
only. A resend caps its hit at N-1, so it needs [P, P+64) below N-1: a tail N-P >= 65.
A follow-up that diverges at N needs the whole peek block: N-P >= 64. Every other
prompt length (about half) gets no SWA hit, so the hybrid hit is 0 and the whole prompt
is prefilled again (Round 34 s13: resend tails 10-58 missed, 68-127 hit).

The lever adds b - 128 to the reachable boundaries of every SWA cache_blocks call: the
tail one hit unit below the replay boundary (and below a junction) is cached too, 2 more
blocks per boundary and SWA group. A resend or a follow-up that diverges at N then hits P
or P-128, at most 192 tokens recomputed, for every N. Dense retention (interval None)
caches every block already; the lever changes nothing there.

With decoder replay (decoder_replay.py) the fallback window [P-256, P-128) reaches
N-P+256 rows back, 384 for N a multiple of 128: the default W = 2670 keeps it in the valid
tail for every N. (W = 2606 covers resends and follow-ups that diverge at N, which need
P-128 only when N-P <= 64, but not a prompt that diverges in the last tokens past P.)

Scheduler side (EngineCore); install() wraps the class in every process, so every rank's
log carries LOG_ENGAGED. Top-level imports are stdlib only. sitecustomize calls install().
"""

from __future__ import annotations

import os

# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dsv41: prefix-end fallback armed"
LOG_DISARMED = "dsv41: prefix-end fallback DISABLED ->"

ENV = "DSV41_PREFIX_END_FALLBACK"
_STATE = {"engaged": False}
# SlidingWindowManager.reachable_block_mask's parameters in the pinned e15 tree.
PARAMS = ("cls", "start_block", "end_block", "alignment_tokens", "kv_cache_spec", "use_eagle",
          "retention_interval", "reachable_boundaries", "dcp_world_size")


def enabled(env=None) -> bool:
    """Unset/empty or 1 = on (the default), 0 = stock e15."""
    env = os.environ if env is None else env
    raw = (env.get(ENV, "") or "").strip()
    if raw not in ("", "0", "1"):
        raise ValueError(f"{ENV}={raw!r}: 1 (default) or 0")
    return raw != "0"


def fallback_boundaries(boundaries, alignment: int) -> tuple:
    """The reachable boundaries plus, for each, the boundary one hit unit below (b - alignment),
    where that still floors to a boundary >= alignment (a hit at 0 is no hit)."""
    out = tuple(int(b) for b in boundaries)
    return out + tuple(b - alignment for b in out if b >= 2 * alignment)


def install(env=None) -> str:
    """Wrap SlidingWindowManager.reachable_block_mask. "off" when the env is 0."""
    import functools
    import inspect

    if not enabled(env):
        return "off"
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

    raw = SlidingWindowManager.__dict__.get("reachable_block_mask")
    stock = raw.__func__ if isinstance(raw, classmethod) else None
    if stock is not None and getattr(stock, "_dsv41_prefix_end_fallback", False):
        return "already installed"
    if stock is None or tuple(inspect.signature(stock).parameters) != PARAMS:
        reason = "SlidingWindowManager.reachable_block_mask is not the pinned classmethod"
        print("dsv41: prefix-end fallback DISABLED -> %s; stock SWA prefix-cache retention" % reason, flush=True)
        raise RuntimeError(reason)

    @functools.wraps(stock)
    def reachable_block_mask(cls, start_block, end_block, alignment_tokens, kv_cache_spec, use_eagle,
                             retention_interval=None, reachable_boundaries=(), dcp_world_size=1):
        if retention_interval is not None and alignment_tokens:
            wider = fallback_boundaries(reachable_boundaries, alignment_tokens)
            if len(wider) > len(reachable_boundaries) and not _STATE["engaged"]:
                _STATE["engaged"] = True
                print("dsv41: prefix-end fallback engaged: SWA boundaries %s also keep %s (interval %s)"
                      % (list(reachable_boundaries), list(wider[len(reachable_boundaries):]), retention_interval),
                      flush=True)
            reachable_boundaries = wider
        return stock(cls, start_block, end_block, alignment_tokens, kv_cache_spec, use_eagle,
                     retention_interval=retention_interval, reachable_boundaries=reachable_boundaries,
                     dcp_world_size=dcp_world_size)

    reachable_block_mask._dsv41_prefix_end_fallback = True
    SlidingWindowManager.reachable_block_mask = classmethod(reachable_block_mask)
    print("dsv41: prefix-end fallback armed (SWA prefix-cache tails at b and b-128 for every reachable boundary b)",
          flush=True)
    return "SlidingWindowManager.reachable_block_mask keeps b-128 next to every reachable boundary"
