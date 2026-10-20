"""Release cached allocator blocks after long prefill chunks (GB10 UMA hygiene).

Every prefill chunk of a DeepSeek-V4.1 sequence scores its queries against
the whole prefix, so the per-chunk indexer/attention transients grow with
the prefix and PyTorch's caching allocator cannot reuse the previous chunk's
slightly smaller blocks: reserved memory grows with the square of the
prompt. On a Spark that is host memory (her 100k prefill took the head from
4.9 GiB free to the 1.5 GiB guard, 2026-09-12). After any prefill step whose
longest sequence is >= DSV41_PREFILL_EMPTY_CACHE_TOKENS, release the unused
cached blocks — but only while the node's MemAvailable after the step is
below DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB (releasing every chunk cost
~20% prefill throughput at 50k tokens on her box, so the release is
adaptive: keep the blocks while there is headroom).

Source (READ ONLY reference): mia-exl3-ref overlay/patch_memory_log.py:153-266
(_install_prefill_empty_cache wrapping Worker.execute_model); her
.env.example:247-254 documents the two knobs.

Env guard: DSV41_PREFILL_EMPTY_CACHE_TOKENS (default 0 = no-op, our opt-in
convention; her default was 8192). Set both:
  DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192
  DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=2.5
(DSV41_PREFILL_END_EMPTY_CACHE — one extra release at prefill->decode handover
— measured no decode gain by mia and is intentionally NOT ported.)
Never breaks a step: all hook errors are swallowed after first report.

Hold while a prompt is in flight (DSV41_PREFILL_EMPTY_CACHE_HOLD=1, perf2).
The torch profile of the perf2 baseline 16K prefill (boot B0c; numbers in
perf2/prefill-mem/) shows the cost of a release. Chunk 1 mapped 82 x 20 MiB
expandable-segment pages (1,640 MiB) in 3.9 s of cuMemCreate on spark1
(median 15.9 ms, max 341 ms per page: kernel reclaim and swap at low
MemFree; the uncontended path is 0.6-1 ms per page). spark2 took 0.9 s for
the same pages. Chunk 2 mapped nothing. The B0c 64K prompt (8 chunks)
released 0.7-1.8 GiB at 6 chunk boundaries on spark1 and 4 on spark2 before
its last chunk (a MemAvailable saw-tooth), and each following chunk mapped
the pages again. A release
between two chunks of one prompt cannot lower the next chunk's peak: the
next chunk needs the same blocks again within a second. It only turns them
into page reclaim inside cuMemCreate, with the GPU idle. With the hold, a
long step after which some scheduled request is still short of its prompt
length releases only while MemAvailable is below
DSV41_PREFILL_EMPTY_CACHE_FLOOR_GIB (default 3), or when the held pool has
grown by more than DSV41_PREFILL_EMPTY_CACHE_GROWTH_MIB (default 150; 0 =
off) beyond what the growth of the steps' peak need explains (see grown()).
A step that leaves no prompt in flight applies the rule above.
DSV41_PREFILL_EMPTY_CACHE_HOLD=0 applies the rule above on every long step.
Prompt lengths come from the step's new requests (NewRequestData.prompt_len)
and are dropped with finished_req_ids. A request the hook never saw as new
counts as done with its prompt, so the rule above applies to it.

The growth rule bounds what holding can add to the pool. A release-every-step
policy regrows, in each step, the pool that step needs. A held pool that
cannot serve a step's blocks (fragmentation, a buffer that grows with the
context) maps new pages while its free blocks stay cached, so it can end up
larger than any single step needs. From the first prefill step after a
release (the hold period's start), every prefill step records torch reserved
bytes and its peak allocated bytes (the allocator's peak counter restarts
at each prefill step); growth of reserved since the period's start, minus
growth of the highest step peak since then, is pool the steps' need does
not explain. Above the cap a long in-flight step releases (action 'grown')
and the next prefill step starts a new period. Graph pools and other blocks
no step uses cancel in the differences.

DSV41_PREFILL_MEM_LOG=1 prints one line per prefill step (> 64 scheduled
tokens for one request) with: the step's wall time, the interval since the
previous prefill step, torch reserved bytes before and after the step
(growth means expandable-segment pages were mapped in that step), the
step's transient peak (torch max allocated during the step minus allocated
at its start; the counter restarts each prefill step), the hold period's
unexplained growth (grown()), MemAvailable, and the action taken.
DSV41_PREFILL_MEM_PROBE=1 adds prefill_mem_probe.py's 'probe step' line per
prefill step (MemFree outside the torch pool; driver-side stores), its flush
experiment and a MemFree series after each release; =2 the step lines only.

Available memory (DSV41_PREFILL_EMPTY_CACHE_COUNT_PCP=1, the default): the
MemAvailable rules above (MEMAVAIL_GIB, FLOOR_GIB) compare MemAvailable plus
the free pages on the kernel's per-CPU page lists (the pagesets' `count:`
lines of /proc/zoneinfo times the page size), read once per long step. On
kernels 6.7+ with vm.percpu_pagelist_high_fraction=0 each CPU's list is
auto-tuned up to 1/8 of the zone split over the CPUs (774 MiB per CPU on a
Spark); those pages are free but are not counted in MemFree / MemAvailable,
and go back to the buddy lists only when an allocation enters direct reclaim.
perf2 FINAL, spark1, 64K prefills: MemAvailable 1,724 MiB with 4,123 MiB on
the lists in the same second. =0 compares MemAvailable alone. The step line
keeps "MemAvailable X GiB action A" (raw; the campaign parsers read it) and
appends "lists Y GiB available Z GiB".
"""

from __future__ import annotations

import os

MARK = "dsv41-prefill-empty-cache"

# Boot-log markers for tools/engagement_audit.py (run.sh post-ready audit).
LOG_ENGAGED = "[dsv41-prefill-empty-cache] armed"
LOG_DISARMED = "[dsv41-prefill-empty-cache] hook error"

LONG_STEP_TOKENS = 64  # a step scheduling more than this for one request is a prefill step
MAX_TRACKED = 4096  # prompt-length records kept before the table is reset
try:
    PAGE_BYTES = os.sysconf("SC_PAGE_SIZE")
except (AttributeError, OSError, ValueError):
    PAGE_BYTES = 4096


def _memavail_gib() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def _pcp_gib(path: str = "/proc/zoneinfo") -> float:
    """Free pages on the kernel's per-CPU lists: the sum of zoneinfo's pageset `count:` lines, GiB (0.0 unread)."""
    pages = 0
    try:
        with open(path) as fh:
            for line in fh:
                if line.lstrip().startswith("count:"):
                    pages += int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return 0.0
    return pages * PAGE_BYTES / 2**30


def available(count_pcp: bool) -> tuple[float, float, float]:
    """(MemAvailable, free pages on the per-CPU lists, the value the release rules compare), GiB."""
    raw, pcp = _memavail_gib(), _pcp_gib()
    return raw, pcp, (raw + pcp) if count_pcp else raw


def decide(
    in_flight: bool, avail: float, memavail_gib: float, hold: bool, floor_gib: float, grown_over: bool = False
) -> str:
    """Action after a long step: 'hold', 'floor', 'grown', 'keep' or 'release'.

    'hold' / 'floor' / 'grown' happen only while a prompt is in flight and the hold is on;
    `avail`: available()'s compared value, GiB;
    `grown_over`: the held pool outgrew its need by more than the cap (grown()).
    'floor', 'grown' and 'release' call torch.cuda.empty_cache().
    """
    if hold and in_flight:
        if avail < floor_gib:
            return "floor"
        return "grown" if grown_over else "hold"
    if memavail_gib > 0 and avail >= memavail_gib:
        return "keep"
    return "release"


def grown(period: dict, reserved: int, peak: int) -> int:
    """Bytes the pool grew since the hold period's first prefill step beyond the growth of the
    highest step peak since then. An empty `period` (after a release) starts at this step: 0."""
    if not period:
        period.update(reserved=reserved, peak0=peak, peak=peak)
        return 0
    period["peak"] = max(period["peak"], peak)
    return (reserved - period["reserved"]) - max(0, period["peak"] - period["peak0"])


def scan(scheduler_output, prompt_len: dict) -> tuple[bool, int, bool]:
    """(prefill step, longest sequence after the step, a scheduled prompt still in flight after it).

    Records the new requests' prompt lengths in `prompt_len` and drops finished requests.
    """
    for rid in getattr(scheduler_output, "finished_req_ids", None) or ():
        prompt_len.pop(rid, None)
    nst = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
    new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None) or ()
    if len(prompt_len) + len(new_reqs) > MAX_TRACKED:
        prompt_len.clear()
    longest, in_flight = 0, False
    for req in new_reqs:
        rid = getattr(req, "req_id", "")
        plen = getattr(req, "prompt_len", None)
        if plen is None:
            plen = len(getattr(req, "prompt_token_ids", None) or ())
        prompt_len[rid] = int(plen)
        done = int(getattr(req, "num_computed_tokens", 0) or 0) + int(nst.get(rid, 0))
        longest = max(longest, done)
        in_flight = in_flight or done < int(plen)
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if cached is not None:
        for rid, nc in zip(cached.req_ids, cached.num_computed_tokens):
            done = int(nc) + int(nst.get(rid, 0))
            longest = max(longest, done)
            plen = prompt_len.get(rid)
            in_flight = in_flight or (plen is not None and done < plen)
    prefill = bool(nst) and max(nst.values()) > LONG_STEP_TOKENS
    return prefill, longest, in_flight


def install() -> bool:
    import functools
    import os
    import time

    threshold = int(os.environ.get("DSV41_PREFILL_EMPTY_CACHE_TOKENS", "0") or 0)
    if threshold <= 0:
        return False
    memavail_gib = float(os.environ.get("DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB", "2.5") or 0)
    hold = os.environ.get("DSV41_PREFILL_EMPTY_CACHE_HOLD", "0") == "1"
    floor_gib = float(os.environ.get("DSV41_PREFILL_EMPTY_CACHE_FLOOR_GIB", "3") or 0)
    growth_mib = float(os.environ.get("DSV41_PREFILL_EMPTY_CACHE_GROWTH_MIB", "150") or 0)
    count_pcp = (os.environ.get("DSV41_PREFILL_EMPTY_CACHE_COUNT_PCP", "") or "1") != "0"
    log_steps = os.environ.get("DSV41_PREFILL_MEM_LOG", "0") == "1"
    probe_mode = {"1": "full", "2": "passive"}.get(os.environ.get("DSV41_PREFILL_MEM_PROBE", "0"), "")
    probe_on = bool(probe_mode)  # prefill_mem_probe.py
    track = log_steps or (hold and growth_mib > 0) or probe_on  # reserved / peak bytes per prefill step

    from vllm.v1.worker.gpu_worker import Worker

    orig = getattr(Worker, "execute_model", None)
    if orig is None or getattr(orig, "_dsv41_empty_cache", False):
        return True  # idempotent: another copy is already armed

    state = {"calls": 0, "skipped": 0, "held": 0, "floor": 0, "grown": 0, "steps": 0, "errors": 0, "last_t": None}
    prompt_len: dict[str, int] = {}
    period: dict[str, int] = {}  # grown(): the hold period since the last release
    probe: list = []  # [prefill_mem_probe.Probe], made at the first prefill step (CUDA is up by then)

    def _say(msg: str) -> None:
        print(f"[{MARK}] {msg}", flush=True)

    def _said(mem) -> str:
        """The available reading behind a decision, as the policy lines print it."""
        if mem is None:
            return "available not read"
        raw, pcp, eff = mem
        lists = f"per-CPU lists {pcp:.2f}" + ("" if count_pcp else " not counted")
        return f"available {eff:.2f} GiB (MemAvailable {raw:.2f} + {lists})"

    def _error(exc: BaseException) -> None:
        state["errors"] += 1
        if state["errors"] == 1:
            print(f"[dsv41-prefill-empty-cache] hook error: {exc!r}", flush=True)

    def _reserved() -> int:
        import torch

        return int(torch.cuda.memory_reserved())

    def _peak_start() -> int:
        """Allocated bytes now; the allocator's peak counter restarts here."""
        import torch

        torch.cuda.reset_peak_memory_stats()
        return int(torch.cuda.memory_allocated())

    def _peak() -> int:
        import torch

        return int(torch.cuda.max_memory_allocated())

    def _probe():
        if probe_on and not probe:
            import torch
            import prefill_mem_probe

            probe.append(prefill_mem_probe.Probe(torch, prefill_mem_probe.Cudart.load(), _say,
                                                 passive=probe_mode == "passive"))
        return probe[0] if probe else None

    @functools.wraps(orig)
    def execute_model(self, scheduler_output, *args, **kwargs):
        info = None
        try:
            prefill, longest, in_flight = scan(scheduler_output, prompt_len)
            if prefill:
                info = (longest, in_flight, time.monotonic(), _reserved() if track else 0,
                        _peak_start() if track else 0)
                if _probe():
                    probe[0].start()
        except Exception as exc:  # never break a step
            _error(exc)
        out = orig(self, scheduler_output, *args, **kwargs)
        if info is None:
            return out
        try:
            longest, in_flight, t0, res0, alloc0 = info
            t1 = time.monotonic()
            res1 = _reserved() if track else 0
            free1 = probe[0].end_free() if probe else None  # MemFree before any release below
            peak_abs = _peak() if track else 0
            peak = peak_abs - alloc0 if track else 0
            excess = grown(period, res1, peak_abs) if track else 0
            action, mem, released = "-", None, False  # mem: available(), read once per step
            if longest >= threshold:
                if (hold and in_flight) or memavail_gib > 0:
                    mem = available(count_pcp)
                over = growth_mib > 0 and excess > growth_mib * 2**20
                action = decide(in_flight, mem[2] if mem else 0.0, memavail_gib, hold, floor_gib, over)
                if action == "hold":
                    state["held"] += 1
                    if state["held"] in (1, 100, 1000):
                        _say(
                            f"held #{state['held']} mid-prefill (longest seq {longest}, "
                            f"{_said(mem)} >= floor {floor_gib} GiB)"
                        )
                elif action == "keep":
                    state["skipped"] += 1
                    if state["skipped"] in (1, 100, 1000):
                        _say(
                            f"skipped #{state['skipped']} (longest seq {longest}, "
                            f"{_said(mem)} >= {memavail_gib} GiB)"
                        )
                else:
                    import torch

                    torch.cuda.empty_cache()
                    released = True
                    period.clear()
                    if probe:
                        probe[0].released(action, res1, _reserved())
                    state["calls"] += 1
                    if action in ("floor", "grown"):
                        state[action] += 1
                    if state["calls"] in (1, 10, 100) or (action != "release" and state[action] in (1, 10, 100)):
                        what = {
                            "floor": f", below the {floor_gib} GiB floor mid-prefill",
                            "grown": f", held pool grew {excess / 2**20:.0f} MiB beyond its need mid-prefill "
                            f"(cap {growth_mib:.0f} MiB)",
                        }.get(action, "")
                        _say(
                            f"empty_cache #{state['calls']} (longest seq {longest}, "
                            f"{_said(mem)}{what})"
                        )
            if log_steps:
                if mem is None:
                    mem = available(count_pcp)
                state["steps"] += 1
                last, state["last_t"] = state["last_t"], t1
                gap = f"{t1 - last:.2f} s" if last is not None else "-"
                after = f", {_reserved() / 2**30:.2f} GiB after release" if released else ""
                _say(
                    f"step {state['steps']} longest {longest} in_flight {int(in_flight)} wall {t1 - t0:.2f} s "
                    f"interval {gap} reserved {res0 / 2**30:.2f} -> {res1 / 2**30:.2f} GiB "
                    f"({(res1 - res0) / 2**20:+.0f} MiB{after}) peak +{peak / 2**20:.0f} MiB "
                    f"grown {excess / 2**20:+.0f} MiB MemAvailable {mem[0]:.2f} GiB action {action} "
                    f"lists {mem[1]:.2f} GiB available {mem[2]:.2f} GiB"
                )
            if probe and free1 is not None:
                probe[0].end(longest, res0, res1, free1)
        except Exception as exc:  # never break a step
            _error(exc)
        return out

    execute_model._dsv41_empty_cache = True
    Worker.execute_model = execute_model
    raw, pcp, eff = available(count_pcp)
    print(
        f"[dsv41-prefill-empty-cache] armed (threshold {threshold} tokens, release when available < "
        f"{memavail_gib} GiB{' [always]' if memavail_gib <= 0 else ''}"
        + (f"; hold mid-prefill, release there below {floor_gib} GiB" if hold else "")
        + (f" or on {growth_mib:.0f} MiB of pool growth the step peaks do not explain" if hold and growth_mib > 0 else "")
        + (f"; available = MemAvailable + per-CPU free lists, now {raw:.2f} + {pcp:.2f} = {eff:.2f} GiB" if count_pcp
           else f"; available = MemAvailable, now {raw:.2f} GiB (per-CPU free lists {pcp:.2f} GiB not counted)")
        + ("; per-step log" if log_steps else "")
        + (f"; memory probe ({probe_mode})" if probe_on else "")
        + ")",
        flush=True,
    )
    return True
