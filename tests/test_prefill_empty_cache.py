"""docker/patch/prefill_empty_cache.py release policy on synthetic scheduler outputs (CPU only, no torch).

A stub Worker and a fake torch count torch.cuda.empty_cache() calls; MemAvailable and the free pages on
the kernel's per-CPU lists (0 unless a test sets them) are mocked, and a fake /proc/zoneinfo checks the
lists' reader. The fake
allocator keeps reserved / allocated / peak bytes; a step's `pool` entry sets its peak need and any
growth its need does not explain (fragmentation). The scheduler outputs carry the fields the hook reads
in e15 (v1/core/sched/output.py): num_scheduled_tokens, scheduled_new_reqs (req_id, prompt_len,
num_computed_tokens), scheduled_cached_reqs (req_ids, num_computed_tokens), finished_req_ids.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docker/patch"))

import prefill_empty_cache as pec  # noqa: E402

CHUNK = 8192


def new_req(rid: str, prompt_len: int, computed: int = 0):
    return types.SimpleNamespace(req_id=rid, prompt_len=prompt_len, num_computed_tokens=computed)


def step(new=(), cached=(), sched=None, finished=()):
    """new: [NewRequestData-like]; cached: [(req_id, num_computed)]; sched: {req_id: tokens}."""
    return types.SimpleNamespace(
        num_scheduled_tokens=dict(sched or {}),
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=types.SimpleNamespace(
            req_ids=[r for r, _ in cached], num_computed_tokens=[c for _, c in cached]
        ),
        finished_req_ids=set(finished),
    )


def prompt_steps(rid: str, prompt_len: int, decode_steps: int = 2, k: int = 3):
    """Chunked prefill of one prompt at 8192 tokens a step, then DSpark decode steps (1 + k tokens)."""
    out, done = [], 0
    while done < prompt_len:
        n = min(CHUNK, prompt_len - done)
        if done == 0:
            out.append(step(new=[new_req(rid, prompt_len)], sched={rid: n}))
        else:
            out.append(step(cached=[(rid, done)], sched={rid: n}))
        done += n
    for i in range(decode_steps):
        out.append(step(cached=[(rid, done + i)], sched={rid: 1 + k}))
    return out


MIB, GIB = 2**20, 2**30
LIVE = 2 * GIB  # allocated between steps
FIXED = 1 * GIB  # reserved no release frees (graph pools)


class Harness:
    """`pool(step_index)` -> (peak transient bytes, unexplained growth bytes) for that step, or None."""

    def __init__(self, env: dict, memavail, pool=None, pcp=0.0):
        self.empty_calls = 0
        self.step_index = 0
        self.releases_at = []
        self.mem = {"reserved": LIVE + FIXED + GIB, "allocated": LIVE, "peak": LIVE}
        harness, mem = self, self.mem

        def empty_cache():
            harness.empty_calls += 1
            harness.releases_at.append(harness.step_index)
            mem["reserved"] = mem["allocated"] + FIXED

        def reset_peak_memory_stats():
            mem["peak"] = mem["allocated"]

        def execute_model(self, so):
            want = pool(harness.step_index) if pool else None
            if want:
                transient, frag = want
                mem["peak"] = max(mem["peak"], LIVE + transient)
                mem["reserved"] = max(mem["reserved"], LIVE + transient + FIXED) + frag
            return "out"

        cuda = types.SimpleNamespace(
            empty_cache=empty_cache,
            memory_reserved=lambda: mem["reserved"],
            memory_allocated=lambda: mem["allocated"],
            max_memory_allocated=lambda: mem["peak"],
            reset_peak_memory_stats=reset_peak_memory_stats,
        )
        self.torch = types.ModuleType("torch")
        self.torch.cuda = cuda
        self.worker = type("Worker", (), {"execute_model": execute_model})
        mods = {n: types.ModuleType(n) for n in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu_worker")}
        mods["vllm.v1.worker.gpu_worker"].Worker = self.worker
        mods["torch"] = self.torch
        base = {"DSV41_PREFILL_EMPTY_CACHE_TOKENS": "8192", "DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB": "4"}
        self.patches = [
            mock.patch.dict(sys.modules, mods),
            mock.patch.dict(os.environ, {**base, **env}),
            mock.patch.object(pec, "_memavail_gib", memavail if callable(memavail) else (lambda: memavail)),
            mock.patch.object(pec, "_pcp_gib", pcp if callable(pcp) else (lambda: pcp)),
        ]

    def __enter__(self):
        for p in self.patches:
            p.start()
        self.out = io.StringIO()
        with contextlib.redirect_stdout(self.out):
            assert pec.install()
        self.w = self.worker()
        return self

    def run(self, steps) -> None:
        with contextlib.redirect_stdout(self.out):
            for so in steps:
                self.step_index += 1
                self.last = self.w.execute_model(so)

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.stop()
        return False


HOLD = {"DSV41_PREFILL_EMPTY_CACHE_HOLD": "1", "DSV41_PREFILL_EMPTY_CACHE_FLOOR_GIB": "2"}
STOCK = {"DSV41_PREFILL_EMPTY_CACHE_HOLD": "0"}


class DecideTests(unittest.TestCase):
    def test_truth_table(self) -> None:
        d = pec.decide
        self.assertEqual(d(True, 3.0, 4.0, True, 2.0), "hold")
        self.assertEqual(d(True, 1.9, 4.0, True, 2.0), "floor")
        self.assertEqual(d(True, 3.0, 4.0, True, 2.0, True), "grown")
        self.assertEqual(d(True, 1.9, 4.0, True, 2.0, True), "floor")
        self.assertEqual(d(False, 4.5, 4.0, True, 2.0, True), "keep")  # the growth rule acts mid-prefill only
        self.assertEqual(d(True, 4.5, 4.0, False, 2.0, True), "keep")
        self.assertEqual(d(False, 3.0, 4.0, True, 2.0), "release")
        self.assertEqual(d(False, 4.5, 4.0, True, 2.0), "keep")
        self.assertEqual(d(True, 3.0, 4.0, False, 2.0), "release")  # hold off: the stock rule mid-prefill too
        self.assertEqual(d(True, 4.5, 4.0, False, 2.0), "keep")
        self.assertEqual(d(False, 9.0, 0.0, True, 2.0), "release")  # MEMAVAIL_GIB <= 0: always release
        self.assertEqual(d(True, 9.0, 0.0, True, 2.0), "hold")


class GrownTests(unittest.TestCase):
    def test_growth_the_peak_explains_is_not_counted(self) -> None:
        period = {}
        self.assertEqual(pec.grown(period, 100 * GIB, 98 * GIB), 0)  # starts the period
        self.assertEqual(pec.grown(period, 100 * GIB + 200 * MIB, 98 * GIB + 200 * MIB), 0)
        self.assertEqual(pec.grown(period, 100 * GIB + 200 * MIB, 98 * GIB), 0)  # a smaller step: no growth
        self.assertEqual(pec.grown(period, 100 * GIB + 500 * MIB, 98 * GIB + 220 * MIB), 280 * MIB)
        self.assertEqual(pec.grown(period, 99 * GIB, 98 * GIB), -1 * GIB - 220 * MIB)  # someone else released
        period.clear()
        self.assertEqual(pec.grown(period, 99 * GIB, 98 * GIB), 0)


class ScanTests(unittest.TestCase):
    def test_in_flight_follows_prompt_length_across_chunks(self) -> None:
        table = {}
        steps = prompt_steps("a", 20000)
        got = [pec.scan(s, table) for s in steps]
        self.assertEqual(got[0], (True, 8192, True))
        self.assertEqual(got[1], (True, 16384, True))
        self.assertEqual(got[2], (True, 20000, False))  # last chunk ends the prompt
        self.assertEqual(got[3][0], False)  # 4-token decode step is not a prefill step
        pec.scan(step(finished={"a"}), table)
        self.assertEqual(table, {})

    def test_unknown_cached_request_is_not_in_flight_and_prefix_hits_end_at_once(self) -> None:
        table = {}
        self.assertEqual(pec.scan(step(cached=[("x", 8192)], sched={"x": 8192}), table), (True, 16384, False))
        hit = step(new=[new_req("r", 32614, computed=32512)], sched={"r": 102})
        self.assertEqual(pec.scan(hit, table), (True, 32614, False))

    def test_prompt_len_falls_back_to_prompt_token_ids(self) -> None:
        req = types.SimpleNamespace(req_id="t", prompt_token_ids=[1] * 9000, num_computed_tokens=0)
        self.assertEqual(pec.scan(step(new=[req], sched={"t": 8192}), {}), (True, 8192, True))


class PolicyTests(unittest.TestCase):
    def test_64k_prompt_hold_releases_once_at_the_end(self) -> None:
        steps = prompt_steps("a", 65000)
        with Harness(HOLD, 3.0) as h:
            h.run(steps)
        self.assertEqual(h.empty_calls, 1)
        self.assertEqual(h.releases_at, [8])  # the 8th chunk ends the prompt
        self.assertIn("held #1 mid-prefill", h.out.getvalue())
        self.assertIn("hold mid-prefill, release there below 2.0 GiB", h.out.getvalue())

    def test_64k_prompt_stock_releases_after_every_long_chunk(self) -> None:
        with Harness(STOCK, 3.0) as h:
            h.run(prompt_steps("a", 65000))
        self.assertEqual(h.releases_at, [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertNotIn("hold mid-prefill", h.out.getvalue())

    def test_below_the_floor_mid_prefill_releases(self) -> None:
        with Harness(HOLD, 1.5) as h:
            h.run(prompt_steps("a", 65000))
        self.assertEqual(h.releases_at, [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertIn("below the 2.0 GiB floor mid-prefill", h.out.getvalue())

    def test_headroom_keeps_everything(self) -> None:
        with Harness(HOLD, 6.0) as h:
            h.run(prompt_steps("a", 65000))
        self.assertEqual(h.empty_calls, 0)
        self.assertIn("skipped #1", h.out.getvalue())

    def test_16k_prompt_unchanged_one_release_after_the_last_chunk(self) -> None:
        # The profiled 16K prompt: chunks of 8190 and 8001 tokens (B0c trace); chunk 1 is below the threshold.
        steps = [
            step(new=[new_req("a", 16191)], sched={"a": 8190}),
            step(cached=[("a", 8190)], sched={"a": 8001}),
            step(cached=[("a", 16191)], sched={"a": 4}),
        ]
        for env in (HOLD, STOCK):
            with Harness(env, 3.0) as h:
                h.run(steps)
            self.assertEqual(h.releases_at, [2], env)

    def test_no_release_while_another_prompt_is_still_in_flight(self) -> None:
        # a ends its prompt in step 2 while b (40000 tokens) is mid-prompt; b ends in step 6.
        steps = [
            step(new=[new_req("a", 12000)], sched={"a": 8192}),
            step(new=[new_req("b", 40000)], cached=[("a", 8192)], sched={"a": 3808, "b": 4384}),
            step(cached=[("a", 12000), ("b", 4384)], sched={"a": 4, "b": 8188}),
            step(cached=[("a", 12004), ("b", 12572)], sched={"a": 4, "b": 8188}),
            step(cached=[("a", 12008), ("b", 20760)], sched={"a": 4, "b": 8188}),
            step(cached=[("a", 12012), ("b", 28948)], sched={"a": 4, "b": 8188}),
            step(cached=[("a", 12016), ("b", 37136)], sched={"a": 4, "b": 2864}),
            step(cached=[("a", 12020), ("b", 40000)], sched={"a": 4, "b": 4}),
        ]
        with Harness(HOLD, 3.0) as h:
            h.run(steps)
        self.assertEqual(h.releases_at, [7])
        with Harness(STOCK, 3.0) as h:
            h.run(steps)
        self.assertEqual(h.releases_at, [1, 2, 3, 4, 5, 6, 7])

    def test_per_step_log_line(self) -> None:
        with Harness({**HOLD, "DSV41_PREFILL_MEM_LOG": "1"}, 3.0) as h:
            h.run(prompt_steps("a", 20000))
        lines = [ln for ln in h.out.getvalue().splitlines() if "] step " in ln]
        self.assertEqual(len(lines), 3, lines)
        self.assertIn("step 1 longest 8192 in_flight 1 wall ", lines[0])
        self.assertIn(
            "interval - reserved 4.00 -> 4.00 GiB (+0 MiB) peak +0 MiB grown +0 MiB MemAvailable 3.00 GiB action hold",
            lines[0],
        )
        self.assertIn("action release", lines[2])
        self.assertIn("GiB after release", lines[2])

    def test_context_growth_the_peak_explains_keeps_the_hold(self) -> None:
        # 1.5 GiB transient that grows 64 MiB a chunk with the context (indexer logits up to the cap).
        pool = lambda i: (1536 * MIB + 64 * MIB * min(i, 4), 0)  # noqa: E731
        with Harness(HOLD, 3.0, pool) as h:
            h.run(prompt_steps("a", 65000))
        self.assertEqual(h.releases_at, [8])
        self.assertIn("or on 150 MiB of pool growth the step peaks do not explain", h.out.getvalue())

    def test_held_pool_outgrowing_its_need_releases_once_then_holds_again(self) -> None:
        # chunk 4 maps 400 MiB its peak does not need (fragmented held blocks); later chunks fit the new pool.
        pool = lambda i: (1536 * MIB, 400 * MIB if i == 4 else 0)  # noqa: E731
        with Harness({**HOLD, "DSV41_PREFILL_MEM_LOG": "1"}, 3.0, pool) as h:
            h.run(prompt_steps("a", 65000))
        self.assertEqual(h.releases_at, [4, 8])
        out = h.out.getvalue()
        self.assertIn("held pool grew 400 MiB beyond its need mid-prefill (cap 150 MiB)", out)
        self.assertIn("grown +400 MiB MemAvailable 3.00 GiB action grown", out)
        self.assertRegex(out, r"step 5 longest 40960 in_flight 1 .*\(\+1536 MiB\) peak \+1536 MiB grown \+0 MiB .*action hold")  # regrowth: new period
        self.assertIn("(+400 MiB, 3.00 GiB after release)", out)

    def test_growth_below_the_cap_or_with_the_rule_off_keeps_the_hold(self) -> None:
        small = lambda i: (1536 * MIB, 100 * MIB if i == 4 else 0)  # noqa: E731
        with Harness(HOLD, 3.0, small) as h:
            h.run(prompt_steps("a", 65000))
        self.assertEqual(h.releases_at, [8])
        big = lambda i: (1536 * MIB, 400 * MIB if i == 4 else 0)  # noqa: E731
        with Harness({**HOLD, "DSV41_PREFILL_EMPTY_CACHE_GROWTH_MIB": "0"}, 3.0, big) as h:
            h.run(prompt_steps("a", 65000))
        self.assertEqual(h.releases_at, [8])
        self.assertNotIn("pool growth the step peaks do not explain", h.out.getvalue())
        with Harness(STOCK, 6.0, big) as h:  # hold off: the growth rule never acts
            h.run(prompt_steps("a", 65000))
        self.assertEqual(h.releases_at, [])

    def test_growth_accumulates_across_prompts_kept_at_the_end(self) -> None:
        # Two prompts kept at their ends (MemAvailable >= 4 there): 100 MiB unexplained in each; the
        # second prompt's growth crosses the cap, since nothing released the pool in between.
        steps = prompt_steps("a", 40000) + prompt_steps("b", 40000)
        pool = lambda i: (1536 * MIB, 100 * MIB if i in (2, 9) else 0)  # noqa: E731
        avail = iter([3.0] + [3.0, 3.0, 3.0, 3.0, 4.5] + [3.0] * 8)  # the first reading is the armed line's
        with Harness(HOLD, lambda: next(avail), pool) as h:
            h.run(steps)
        self.assertEqual(h.releases_at, [9, 12])

    def test_default_floor_is_3_gib(self) -> None:
        env = {"DSV41_PREFILL_EMPTY_CACHE_HOLD": "1"}
        with Harness(env, 2.9) as h:
            h.run(prompt_steps("a", 40000))
        self.assertEqual(h.releases_at, [1, 2, 3, 4, 5])
        self.assertIn("hold mid-prefill, release there below 3.0 GiB", h.out.getvalue())

    def test_hook_errors_are_reported_once_and_never_break_a_step(self) -> None:
        bad = types.SimpleNamespace(num_scheduled_tokens={"a": "x"})  # str > int raises inside scan()
        with Harness(HOLD, 3.0) as h:
            h.run([bad, bad])
            h.run(prompt_steps("b", 20000))  # the hook keeps working after an error
        self.assertEqual(h.out.getvalue().count(pec.LOG_DISARMED), 1)
        self.assertEqual(h.last, "out")
        self.assertEqual(h.releases_at, [5])

    def test_defaults_on_in_recipe_and_forwarded(self) -> None:
        run = (ROOT / "run.sh").read_text()
        recipe = (ROOT / "recipe.yaml").read_text()
        for name, value in (
            ("DSV41_PREFILL_EMPTY_CACHE_HOLD", "1"),
            ("DSV41_PREFILL_EMPTY_CACHE_FLOOR_GIB", "3"),
            ("DSV41_PREFILL_EMPTY_CACHE_GROWTH_MIB", "150"),
            ("DSV41_PREFILL_EMPTY_CACHE_COUNT_PCP", "1"),
        ):
            self.assertIn(f'{name}="${{{name}:-{value}}}"', run)
            self.assertIn(f"  {name}=\n", run)
            self.assertIn(f"    {name}: {value}\n", recipe)
        self.assertIn("  DSV41_PREFILL_MEM_LOG=0\n", run)


ZONEINFO = """Node 0, zone      DMA
  pages free     3001
        min      12
  pagesets
    cpu: 0
              count:    100
              high:     500
              batch:    63
              high_min: 124
              high_max: 1092
  vm stats threshold: 10
    cpu: 1
              count:    28
              high:     124
Node 0, zone   Normal
  pages free     23590814
        nr_free_pages 23590814
  pagesets
    cpu: 0
              count:    262144
              high:     300000
    cpu: 1
              count:    131072
              high:     198132
  node_unreclaimable:  0
  start_pfn:           1048576
"""


class PcpTests(unittest.TestCase):
    """The per-CPU free lists in the available reading (DSV41_PREFILL_EMPTY_CACHE_COUNT_PCP)."""

    def test_reads_only_the_pageset_count_lines(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "zoneinfo"
            path.write_text(ZONEINFO)
            pages = 100 + 28 + 262144 + 131072
            self.assertAlmostEqual(pec._pcp_gib(str(path)), pages * pec.PAGE_BYTES / GIB)
            self.assertEqual(pec._pcp_gib(str(Path(d) / "absent")), 0.0)

    def test_available_adds_the_lists_when_counted(self) -> None:
        with mock.patch.object(pec, "_memavail_gib", lambda: 1.5), mock.patch.object(pec, "_pcp_gib", lambda: 2.25):
            self.assertEqual(pec.available(True), (1.5, 2.25, 3.75))
            self.assertEqual(pec.available(False), (1.5, 2.25, 1.5))

    def test_lists_keep_the_hold_above_the_floor(self) -> None:
        # MemAvailable 1.5 GiB is below the 2 GiB floor; with 1 GiB on the lists available is 2.5 GiB.
        with Harness(HOLD, 1.5, pcp=1.0) as h:
            h.run(prompt_steps("a", 65000))
        self.assertEqual(h.releases_at, [8])  # only at the prompt end (2.5 < 4)
        with Harness({**HOLD, "DSV41_PREFILL_EMPTY_CACHE_COUNT_PCP": "0"}, 1.5, pcp=1.0) as h:
            h.run(prompt_steps("a", 65000))
        self.assertEqual(h.releases_at, [1, 2, 3, 4, 5, 6, 7, 8])  # raw MemAvailable: every chunk is a floor release
        self.assertIn("below the 2.0 GiB floor mid-prefill", h.out.getvalue())

    def test_lists_count_toward_the_end_threshold(self) -> None:
        with Harness(HOLD, 3.0, pcp=1.5) as h:  # 4.5 >= 4 at the prompt end: keep
            h.run(prompt_steps("a", 20000))
        self.assertEqual(h.releases_at, [])
        self.assertIn("skipped #1 (longest seq 20000, available 4.50 GiB (MemAvailable 3.00 + per-CPU lists 1.50) "
                      ">= 4.0 GiB)", h.out.getvalue())
        with Harness({**HOLD, "DSV41_PREFILL_EMPTY_CACHE_COUNT_PCP": "0"}, 3.0, pcp=1.5) as h:
            h.run(prompt_steps("a", 20000))
        self.assertEqual(h.releases_at, [3])

    def test_armed_and_step_lines_show_raw_and_available(self) -> None:
        with Harness({**HOLD, "DSV41_PREFILL_MEM_LOG": "1"}, 1.5, pcp=2.25) as h:
            h.run(prompt_steps("a", 20000))
        out = h.out.getvalue()
        self.assertIn("release when available < 4.0 GiB", out)
        self.assertIn("available = MemAvailable + per-CPU free lists, now 1.50 + 2.25 = 3.75 GiB", out)
        self.assertIn("held #1 mid-prefill (longest seq 8192, available 3.75 GiB (MemAvailable 1.50 + per-CPU lists "
                      "2.25) >= floor 2.0 GiB)", out)
        step1 = next(ln for ln in out.splitlines() if "] step 1 " in ln)
        self.assertIn("MemAvailable 1.50 GiB action hold lists 2.25 GiB available 3.75 GiB", step1)
        with Harness({**HOLD, "DSV41_PREFILL_EMPTY_CACHE_COUNT_PCP": "0", "DSV41_PREFILL_MEM_LOG": "1"}, 3.0,
                     pcp=2.25) as h:
            h.run(prompt_steps("a", 20000))
        out = h.out.getvalue()
        self.assertIn("available = MemAvailable, now 3.00 GiB (per-CPU free lists 2.25 GiB not counted)", out)
        self.assertIn("MemAvailable 3.00 GiB action hold lists 2.25 GiB available 3.00 GiB", out)
        self.assertIn("(MemAvailable 3.00 + per-CPU lists 2.25 not counted)", out)

    def test_one_reading_per_long_step(self) -> None:
        reads = {"mem": 0, "pcp": 0}

        def mem():
            reads["mem"] += 1
            return 3.0

        def pcp():
            reads["pcp"] += 1
            return 0.5

        with Harness({**HOLD, "DSV41_PREFILL_MEM_LOG": "1"}, mem, pcp=pcp) as h:
            h.run(prompt_steps("a", 20000))  # 3 prefill steps, 2 decode steps
        self.assertEqual(reads, {"mem": 1 + 3, "pcp": 1 + 3})  # the armed line, then one per prefill step


if __name__ == "__main__":
    unittest.main()
