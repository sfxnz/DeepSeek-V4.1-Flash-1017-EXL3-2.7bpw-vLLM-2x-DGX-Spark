"""decoder_replay (DSV41_DECODER_REPLAY_ROWS): row plan, exactness, prefix-cache guard, wiring.

Host (numpy, no torch):
- the window arithmetic for the 1017 config (cut 20, 19 layers after it, window 128):
  exact from 2414 rows; 2606 keeps a 320-row valid tail, the least that keeps every prefix hit
  e15 lands at a trimmed prompt's end;
- the per-step plan (end-anchored chunks, clamp-only steps, decodes, prompt logprobs, hits,
  the per-chunk approximate mode), the step kind both TP ranks derive from it, the kept-row
  gather, and run_tail's three branches on numpy stand-ins (a step with no kept row runs
  nothing past write_kv);
- e15's own SWA hit search and retention mask (pinned) on a fake block pool under the replay
  frontier: --prefix-cache-retention-interval dense, 0 (e15's default) and 4096 (kv-pack);
  resends, follow-ups, mid-prompt divergence, retention checkpoints and shared-prefix junctions;
- the sub-batch build's entry point and model-specific metadata (kv-pack's batching stays off);
- ToyCED: a decoder stack past the cut with integer arithmetic (bit-exact) whose rows reach
  older rows only through a per-layer sliding window, run chunked through plan_batch with
  garbage-initialised caches. It checks the exactness claims (last row, decode-window KV, DSpark
  taps, every row past the valid tail), minimality, that no unwritten KV is ever read, the
  scheduler's prefix-cache frontier under e15's SWA hit rule (2 blocks + the DSpark peek block,
  128-aligned boundaries) and that ignoring the mask breaks a hit;
- the DeepseekV4Model.forward rewrite on the pinned e15 source, executed with fakes;
- wiring: sitecustomize, run.sh FORWARD_ENVS, recipe default, audit gate, pinned e15 signatures.
Image (torch + vllm; skipped on the host): the torch clamp against the numpy reference, run_tail's
gather/scatter on fake layers, install() on the real classes.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import textwrap
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker" / "patch"
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(PATCH))
import decoder_replay as dr  # noqa: E402
import prefix_end_fallback as pef  # noqa: E402
import prefix_reply_retain as prr  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_VLLM = HAS_TORCH and importlib.util.find_spec("vllm") is not None

# 1017 text config (config.json of deepseek-ai/DeepSeek-V4.1-Flash-1017).
KV_SOURCES, LAYERS, WINDOW = [2, 8, 14, 20], 40, 128


def _method_src(path: Path, cls: str | None, name: str) -> str:
    """Verbatim source of cls.name (or a module function) from a pinned file, class-indented."""
    text = path.read_text()
    tree = ast.parse(text)
    scope = tree.body
    if cls is not None:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = text.splitlines(keepends=True)
    first = min([d.lineno for d in node.decorator_list] + [node.lineno])
    return "".join(lines[first - 1 : node.end_lineno])


def _pinned_e15(name: str) -> str:
    """A segment of tests/fixtures/decoder_replay_e15.pin.py by its '# ---- ... name' header."""
    text = (FIX / "decoder_replay_e15.pin.py").read_text()
    for part in text.split("\n# ---- ")[1:]:
        header, _, body = part.partition("\n")
        qual = header.split()[-1]
        if name in (qual, qual.rsplit(".", 1)[-1]):
            return body
    raise KeyError(name)


def _params(src: str) -> list[str]:
    fn = ast.parse(textwrap.dedent(src)).body[0]
    return [a.arg for a in fn.args.args + fn.args.kwonlyargs]


class WindowArithmeticTests(unittest.TestCase):
    def test_1017_geometry(self) -> None:
        self.assertEqual(dr.geometry(KV_SOURCES, LAYERS, WINDOW), (20, 19, 128))

    def test_exact_minimum_and_default(self) -> None:
        cut, after, w = dr.geometry(KV_SOURCES, LAYERS, WINDOW)
        self.assertEqual(dr.exact_min_rows(w, after), 128 + 18 * 127)
        self.assertEqual(dr.exact_min_rows(w, after), 2414)
        # 18*127 truncated rows + the deepest prompt-end window a hit reads: the fallback
        # P-128's [P-256, P-128), N - P <= 128 below N
        self.assertEqual(dr.DEFAULT_ROWS, 18 * 127 + 128 + 2 * 128)
        self.assertEqual(dr.valid_tail(2670, w, after), 384)
        self.assertEqual(dr.valid_tail(2606, w, after), 320)
        self.assertEqual(dr.valid_tail(2541, w, after), 255)
        self.assertEqual(dr.valid_tail(2414, w, after), 128)
        self.assertEqual(dr.valid_tail(2413, w, after), 128)  # approximate: the decode window
        self.assertEqual(dr.valid_tail(128, w, after), 128)
        self.assertEqual(dr.mode_for(2606, w, after), "exact")
        self.assertEqual(dr.mode_for(2414, w, after), "exact")
        self.assertEqual(dr.mode_for(2413, w, after), "approx")
        self.assertEqual(dr.mode_for(128, w, after), "approx")
        self.assertIsNone(dr.mode_for(0, w, after))
        with self.assertRaises(ValueError):
            dr.mode_for(127, w, after)

    def test_rows_from_env(self) -> None:
        self.assertEqual(dr.rows_from_env({}), 2670)
        self.assertEqual(dr.rows_from_env({dr.ENV: ""}), 2670)
        self.assertEqual(dr.rows_from_env({dr.ENV: "0"}), 0)
        self.assertEqual(dr.rows_from_env({dr.ENV: " 128 "}), 128)
        with self.assertRaises(ValueError):
            dr.rows_from_env({dr.ENV: "-1"})
        with self.assertRaises(ValueError):
            dr.rows_from_env({dr.ENV: "on"})

    def test_install_off_needs_no_vllm(self) -> None:
        self.assertEqual(dr.install({dr.ENV: "0"}), "off")

    def test_frontier(self) -> None:
        # trimmed cold prefill: SWA blocks from prompt_len - 320 on are cacheable
        self.assertEqual(dr.frontier(9000, 0, False, 2606, 320), 8680)
        self.assertEqual(dr.frontier(9000, 6528, False, 2606, 320), 0)  # suffix <= W: nothing skipped
        self.assertEqual(dr.frontier(9000, 0, True, 2606, 320), 0)  # prompt logprobs keep every row
        self.assertEqual(dr.frontier(2606, 0, False, 2606, 320), 0)
        self.assertEqual(dr.frontier(2607, 0, False, 2606, 320), 2287)
        self.assertEqual(dr.frontier(9000, 0, False, 0, 320), 0)

    def test_block_mask(self) -> None:
        self.assertIsNone(dr.block_mask(0, 10, 64, 0))
        self.assertEqual(dr.block_mask(0, 4, 64, 129), [False, False, False, True])
        self.assertEqual(dr.block_mask(0, 4, 64, 128), [False, False, True, True])
        self.assertIsNone(dr.block_mask(3, 6, 64, 128))  # every new block starts past the frontier
        self.assertEqual(dr.block_mask(1, 3, 64, 128), [False, True])
        self.assertEqual(dr.block_mask(1, 3, 64, 192), [False, False])
        self.assertIsNone(dr.block_mask(5, 5, 64, 1000))
        # a junction anchor at 512 (before = 256): blocks starting 256..448 and the peek block at 512
        got = dr.block_mask(0, 12, 64, 700, anchor=512, before=256)
        self.assertEqual([b for b, k in enumerate(got) if k], [4, 5, 6, 7, 8, 11])
        self.assertEqual(dr.block_mask(0, 12, 64, 700), [b >= 11 for b in range(12)])
        # an unaligned anchor exempts only whole blocks below it (no peek block starts at it)
        got = dr.block_mask(0, 12, 64, 700, anchor=520, before=256)
        self.assertEqual([b for b, k in enumerate(got) if k], [5, 6, 7, 11])
        self.assertEqual(dr.and_masks([True, False], [True, True]), [True, False])
        self.assertEqual(dr.and_masks(None, [False]), [False])
        self.assertEqual(dr.and_masks([True], None), [True])

class SchedulerGuardTests(unittest.TestCase):
    """scheduler_guard / CacheGuard with a scoped stand-in for vllm.v1.kv_cache_interface."""

    def _guard(self, rows=2606):
        from unittest import mock

        class SlidingWindowSpec:
            pass

        class Uniform:
            def __init__(self, specs):
                self.kv_cache_specs = specs

        kci = types.ModuleType("vllm.v1.kv_cache_interface")
        kci.SlidingWindowSpec = SlidingWindowSpec
        kci.iter_layer_specs = lambda s: s.kv_cache_specs.values() if isinstance(s, Uniform) else (s,)
        groups = [types.SimpleNamespace(kv_cache_spec=s) for s in (
            SlidingWindowSpec(), object(), Uniform({"a": object(), "b": SlidingWindowSpec()}), object())]
        cfg = types.SimpleNamespace(kv_source_layer_ids=KV_SOURCES, num_hidden_layers=LAYERS, sliding_window=WINDOW)
        vcfg = types.SimpleNamespace(model_config=types.SimpleNamespace(hf_text_config=cfg, hf_config=None))
        fakes = {name: types.ModuleType(name) for name in ("vllm", "vllm.v1")}
        fakes["vllm.v1.kv_cache_interface"] = kci
        with mock.patch.dict(sys.modules, fakes):
            return dr.scheduler_guard(vcfg, types.SimpleNamespace(kv_cache_groups=groups), rows)

    def test_guard_masks_only_swa_groups(self) -> None:
        guard = self._guard()
        self.assertEqual((guard.tail, guard.swa_groups), (320, frozenset({0, 2})))
        req = types.SimpleNamespace(num_prompt_tokens=9000, sampling_params=types.SimpleNamespace(prompt_logprobs=None))
        self.assertIsNone(guard.mask(req, 0, 4, 64, 1, None))  # compressed / indexer groups stay cached
        # never admitted through allocate_slots: assume no hit (conservative)
        self.assertEqual(guard.mask(req, 130, 141, 64, 0, None), [b >= 136 for b in range(130, 141)])
        setattr(req, dr.ATTR, guard.admit(req, 6528))  # suffix <= W: nothing skipped
        self.assertIsNone(guard.mask(req, 0, 141, 64, 2, None))
        setattr(req, dr.ATTR, guard.admit(req, 0))
        self.assertEqual(getattr(req, dr.ATTR), (8680, 0))
        self.assertEqual(guard.mask(req, 135, 138, 64, 2, [True, True, False]), [False, True, False])
        logprobs = types.SimpleNamespace(num_prompt_tokens=9000, sampling_params=types.SimpleNamespace(prompt_logprobs=1))
        self.assertEqual(guard.frontier(logprobs, 0), 0)
        pooling = types.SimpleNamespace(num_prompt_tokens=9000, sampling_params=None)
        self.assertEqual(guard.frontier(pooling, 0), 0)

    def test_junction_anchor_at_admission(self) -> None:
        guard = self._guard()
        self.assertEqual(guard.before, 256)
        req = types.SimpleNamespace(num_prompt_tokens=20000, shared_prefix_boundary=16384,
                                    sampling_params=types.SimpleNamespace(prompt_logprobs=None))
        self.assertEqual(guard.admit(req, 0), (19680, 16384))
        setattr(req, dr.ATTR, guard.admit(req, 0))
        mask = guard.mask(req, 250, 260, 64, 0, None)  # blocks 252..256 hold [J-256, J+64)
        self.assertEqual([250 + i for i, k in enumerate(mask) if k], [252, 253, 254, 255, 256])
        self.assertEqual(guard.admit(req, 16384), (19680, 0))  # junction at the hit: nothing to anchor
        req.shared_prefix_boundary = 0
        self.assertEqual(guard.admit(req, 0), (19680, 0))
        req.shared_prefix_boundary = 16384
        req.num_prompt_tokens = 18000  # suffix 18000 - 16384 <= W: no trim, no frontier, no anchor
        self.assertEqual(guard.admit(req, 16384), (0, 0))
        approx = self._guard(rows=128)
        self.assertEqual(approx.before, 0)
        req.num_prompt_tokens = 20000
        self.assertEqual(approx.admit(req, 0), (19872, 0))  # approximate W: no anchors

    def test_no_guard_when_off_or_below_the_window(self) -> None:
        self.assertIsNone(self._guard(rows=0))
        self.assertIsNone(self._guard(rows=64))
        self.assertEqual(self._guard(rows=128).tail, 128)


class _SWASpec:
    def __init__(self, sliding_window=WINDOW, block_size=64):
        self.sliding_window, self.block_size = sliding_window, block_size


class _Pool:
    """BlockPool stand-in for the pinned SWA hit search: block hash i is block i of the prefix
    a later prompt shares with the cached one; ``cached`` are the blocks registered."""

    hash_block_size = 64
    null_block = None

    def __init__(self, cached):
        self.cached = cached

    def get_cached_block(self, block_hash, group_ids):
        return [block_hash] * len(group_ids) if block_hash in self.cached else None


def _pinned_swa_manager():
    """SlidingWindowManager's hit-block count, retention mask and hit search, pinned from e15."""
    import collections.abc
    import typing

    ns = {"cdiv": lambda a, b: -(-a // b), "Sequence": collections.abc.Sequence, "ClassVar": typing.ClassVar,
          "BlockHashList": list, "BlockPool": _Pool, "KVCacheSpec": object, "KVCacheBlock": object,
          "SlidingWindowSpec": _SWASpec, "resolve_block_hashes": lambda hashes, *a, **k: hashes}
    body = "".join(_pinned_e15(f"SlidingWindowManager.{name}")
                   for name in ("_contiguous_blocks_for_hit", "reachable_block_mask", "find_longest_cache_hit"))
    src = "class SlidingWindowManager:\n    supports_fine_grained_hash_lookup = False\n\n" + body
    exec(compile(src, "pinned SlidingWindowManager", "exec"), ns)
    return ns["SlidingWindowManager"]


class RetentionTests(unittest.TestCase):
    """--prefix-cache-retention-interval under the replay frontier, on e15's own SWA code.

    An n-token prompt is prefilled cold (and trimmed); the pinned
    SlidingWindowManager.reachable_block_mask decides what an SWA group caches, ANDed with
    CacheGuard.mask as the cache_full_blocks wrapper does (rows=None: stock e15, no replay).
    A later prompt that shares its first ``shared`` tokens then hits where the pinned
    find_longest_cache_hit lands, with the coordinator's EAGLE margin (the DSpark peek block)
    and full-attention groups that hit every matched 128-aligned prefix. Modes: None = dense,
    0 = e15's default (only the replay boundary and shared-prefix junction tails), 4096 =
    kv-pack (plus a checkpoint every 4096 tokens). Blocks are 64 tokens, hits 128-aligned."""

    MODES = (None, 0, 4096)
    LENGTHS = range(9000, 9000 + 128)

    @classmethod
    def setUpClass(cls) -> None:
        cls.swm = _pinned_swa_manager()
        cls.spec = _SWASpec()

    def _cached(self, n, mode, rows, junction=0, *, fallback=False, hit=0, anchors=True, computed=None,
                reply=False):
        """Blocks an SWA group caches for an n-token prompt prefilled from ``hit``, with e15's
        junction ``junction`` (0 = none); ``fallback``: prefix_end_fallback's extra boundaries;
        ``anchors`` False: no junction anchors (their tails below the frontier stay masked);
        ``computed``: tokens with KV once decoding stops (default n); ``reply``:
        prefix_reply_retain's decode-filled blocks. Each block is decided once, when it fills,
        from the request's fixed boundaries, so one mask over every full block is the same."""
        first, nfull = hit // 64, (n if computed is None else computed) // 64
        bounds = [n - 1] + ([junction] if junction else [])
        if fallback:
            bounds = list(pef.fallback_boundaries(bounds, 128))
        mask = self.swm.reachable_block_mask(first, nfull, 128, self.spec, True, retention_interval=mode,
                                             reachable_boundaries=bounds)
        if reply:
            mask = prr.reply_mask(mask, first, nfull, 64, n)
        if rows is not None:
            guard = dr.CacheGuard(rows, dr.valid_tail(rows, WINDOW, 19), {0}, 2 * WINDOW if anchors else 0)
            req = types.SimpleNamespace(num_prompt_tokens=n, shared_prefix_boundary=junction,
                                        sampling_params=types.SimpleNamespace(prompt_logprobs=None))
            setattr(req, dr.ATTR, guard.admit(req, hit))
            mask = guard.mask(req, first, nfull, 64, 0, mask)
        return {b for b in range(first, nfull) if mask is None or mask[b - first]}

    def _hit(self, cached, shared, new_len):
        cap = new_len - 1  # max_cache_hit_length: the last token is recomputed
        full_attention = min(shared, cap) // 128 * 128
        max_length = min(full_attention + 64, cap)  # HybridKVCacheCoordinator's eagle margin
        pool = _Pool({b for b in cached if (b + 1) * 64 <= shared})
        _, hit = self.swm.find_longest_cache_hit(list(range(max_length // 64 + 1)), max_length, [0], pool,
                                                 self.spec, True, 128)
        return hit

    def test_pinned_coordinator_margin_and_default_interval(self) -> None:
        coord = _pinned_e15("HybridKVCacheCoordinator.find_longest_cache_hit")
        self.assertIn("curr_hit_length + eagle_margin, max_cache_hit_length", coord)
        self.assertIn("max_cache_hit_length = request.num_tokens - 1", _pinned_e15("KVCacheManager.get_computed_blocks"))
        self.assertIn("reachable_boundaries = [request.num_prompt_tokens - 1]",
                      _pinned_e15("SingleTypeKVCacheManager.cache_blocks"))
        # unset (this recipe sets none, kv-pack sets 4096): sparse retention, interval 0
        self.assertIn("return 0 if env_value is None else int(env_value)",
                      _pinned_e15("_get_prefix_cache_retention_interval"))

    def _stock_prompt_end(self):
        """Stock e15 hits after an N-token prompt, for a resend and for a follow-up that diverges
        at N: {(mode, case): {N: P}}; P = 0 (miss) is left out."""
        out = {}
        for mode in self.MODES:
            for case, extra in (("resend", 0), ("follow-up", 1000)):
                hits = {n: self._hit(self._cached(n, mode, None), n, n + extra) for n in self.LENGTHS}
                out[mode, case] = {n: p for n, p in hits.items() if p}
        return out

    def test_prompt_end_hits_reach_back_320_dense_and_256_sparse(self) -> None:
        """How far below N a prompt-end hit reads, N - (P - 128). The smallest W that keeps every
        one is 18*127 + the largest: 2606, set by dense resends with N = 64 mod 128; with sparse
        retention (e15's default 0, kv-pack's 4096) 2542, set by N a multiple of 128. Under 4096
        the prompt ends whose last boundary is not cached fall back to the 4096 checkpoint below:
        those are mid-prompt hits (test_checkpoints_below_the_frontier_never_hit)."""
        worst, where = {}, {}
        for (mode, case), hits in self._stock_prompt_end().items():
            back = {n: n - (p - 128) for n, p in hits.items() if n - p <= 192}
            checkpoints = {p for n, p in hits.items() if n - p > 192}
            self.assertEqual(checkpoints, {8192} if mode == 4096 else set(), (mode, case))
            worst[mode, case] = max(back.values())
            where[mode, case] = {n % 128 for n, b in back.items() if b == worst[mode, case]}
            if mode is not None:
                # sparse: only the replay boundary (N-1)//128*128, a hit when its peek block is
                # whole: tail N - P >= 65 for a resend (tools/warm_prefix.py, Round 34 s13: tails
                # 10-58 missed, 68-127 hit), >= 64 for a follow-up
                least = 65 if case == "resend" else 64
                self.assertEqual(sorted(back), [n for n in self.LENGTHS if n - (n - 1) // 128 * 128 >= least])
        self.assertEqual(worst, {(None, "resend"): 320, (None, "follow-up"): 319, (0, "resend"): 256,
                                 (0, "follow-up"): 256, (4096, "resend"): 256, (4096, "follow-up"): 256})
        self.assertEqual(where[None, "resend"], {64})
        self.assertEqual(where[0, "resend"], {0})
        self.assertEqual(18 * 127 + max(worst.values()), 2606)  # stock e15 alone; the fallback: 2670

    def test_the_default_keeps_every_prompt_end_hit(self) -> None:
        for (mode, case), hits in self._stock_prompt_end().items():
            for n in self.LENGTHS:
                new_len = n if case == "resend" else n + 1000
                stock = hits.get(n, 0)
                end_hit = stock if n - stock <= 192 else 0  # a 4096 checkpoint below is not one
                self.assertEqual(self._hit(self._cached(n, mode, 2606), n, new_len), end_hit, (mode, case, n))
                if mode is None:
                    continue
                self.assertEqual(self._hit(self._cached(n, mode, 2542), n, new_len), end_hit, (mode, case, n))
                lost = self._hit(self._cached(n, mode, 2541), n, new_len) != end_hit
                self.assertEqual(lost, n % 128 == 0, (mode, case, n))

    def test_checkpoints_below_the_frontier_never_hit(self) -> None:
        """After a trimmed prompt every hit lands at P >= frontier + 128 or is 0, wherever the
        later prompt diverges: retention checkpoints and a shared-prefix junction tail below the
        frontier are never cached, so mid-prompt divergence recomputes from the start (or from
        what another request cached), never from a stale checkpoint. Stock e15 would hit them."""
        tail = dr.valid_tail(dr.DEFAULT_ROWS, WINDOW, 19)
        junction = 4480  # a junction tail no worker anchored (anchors off)
        for mode in self.MODES:
            stale = 0
            for n in self.LENGTHS[::9]:
                front = n - tail
                guarded = self._cached(n, mode, dr.DEFAULT_ROWS, junction, fallback=True, anchors=False)
                stock = self._cached(n, mode, None, junction, fallback=True)
                for shared in sorted(set(range(64, n + 1, 61)) | {junction + 64, 4096 + 64, 8192 + 64, n}):
                    p = self._hit(guarded, shared, shared + 1000)
                    self.assertTrue(p == 0 or p - 128 >= front, (mode, n, shared, p))
                    if shared < n - tail + 192:  # no P with P - 128 >= front and P + 64 <= shared
                        self.assertEqual(p, 0, (mode, n, shared))
                    q = self._hit(stock, shared, shared + 1000)
                    stale += 0 < q and q - 128 < front
            self.assertGreater(stale, 0, mode)

    # -- (A) the prompt-end fallback, prefix_end_fallback.py

    def test_divergence_in_the_last_tokens_needs_2670(self) -> None:
        """A prompt sharing D tokens with a trimmed N-token one: P = (N-1)//128*128 needs its peek
        block shared (D >= P+64), the fallback P-128 needs D >= P-64 and reads [P-256, P-128),
        up to 384 rows back (N a multiple of 128). 2670 keeps every such hit, 2669 loses N a
        multiple of 128, 2606 every N with N - P > 64."""
        for mode in (0, 4096):
            for n in self.LENGTHS:
                p_r = (n - 1) // 128 * 128
                stock = self._cached(n, mode, None, fallback=True)
                for shared in (p_r - 64, p_r - 1, p_r + 1, p_r + 62, p_r + 63, p_r + 64, n):
                    if shared > n:
                        continue
                    key = (mode, n, shared)
                    want = p_r if shared >= p_r + 64 else p_r - 128
                    self.assertEqual(self._hit(stock, shared, shared + 1000), want, key)
                    self.assertEqual(self._hit(self._cached(n, mode, 2670, fallback=True), shared, shared + 1000),
                                     want, key)
                    for rows, lost in ((2669, n % 128 == 0), (2606, n - p_r > 64)):
                        got = self._hit(self._cached(n, mode, rows, fallback=True), shared, shared + 1000)
                        self.assertEqual(got != want, lost and want == p_r - 128, (rows,) + key)

    def test_boot_i1_prefix_cases(self) -> None:
        """Boot I1 (perf2/integration/I1/prefix_cache.json, prefix_junction.json; c0caffe at 2606,
        kv-pack retention 4096). a_novel: N = 32591 = 4 template tokens + a 32569-token doc
        (corpus counts it with BOS: 32570) + 18 question tokens. d_same_doc_new_question shares
        2 + 32569 + 3 ("\\n\\nQuestion:") = 32574 tokens: 62 past P = 32512, so P's peek block
        [32512, 32576) is not shared. e (32387 tokens) shared 16391 tokens with a; f repeats e.
        The replay_ab resend: a 20017-token turn (tail 49)."""
        n_a, d, n_e, shared_e, n_r = 32591, 32574, 32387, 16391, 20017
        for mode in (0, 4096):
            rows_cases = {None: "stock", 2606: "c0caffe W", 2670: "default W"}
            for rows in rows_cases:
                for fallback in (False, True):
                    key = (mode, rows, fallback)
                    a = self._cached(n_a, mode, rows, fallback=fallback)
                    self.assertEqual(self._hit(a, n_a, n_a), 32512, key)  # b_identical
                    self.assertEqual(self._hit(a, n_a + 50, n_a + 84), 32512, key)  # c_multiturn
                    # (1) d: needs the fallback 32384, its window [32256, 32384) 335 rows below N
                    want = {False: (28672 if mode == 4096 and rows is None else 0),
                            True: (0 if rows == 2606 else 32384)}[fallback]
                    self.assertEqual(self._hit(a, d, n_a), want, key)
                    # (2) e misses (a's tails are at a's end; kv-pack's 12288 checkpoint serves
                    # stock only); its junction is 16384
                    hit_e = self._hit(a, shared_e, n_e)
                    self.assertEqual(hit_e, 12288 if mode == 4096 and rows is None else 0, key)
                    e = self._cached(n_e, mode, rows, 16384, fallback=fallback, hit=hit_e)
                    # f resends e: tail 3, so e's own boundary 32384 is past the cap
                    want_f = 32256 if fallback else (28672 if mode == 4096 and rows is None else 16384)
                    self.assertEqual(self._hit(a | e, n_e, n_e), want_f, key)
                    if rows is not None and not fallback:  # c0caffe: no anchor, no fallback
                        bare = self._cached(n_e, mode, rows, 16384, hit=hit_e, anchors=False)
                        self.assertEqual(self._hit(a | bare, n_e, n_e), 0, key)
                    # (3) the replay_ab resend
                    r = self._cached(n_r, mode, rows, fallback=fallback)
                    want_r = 19840 if fallback else (16384 if mode == 4096 and rows is None else 0)
                    self.assertEqual(self._hit(r, n_r, n_r), want_r, key)

    def test_fallback_gives_every_prompt_end_a_hit(self) -> None:
        """Sparse retention keeps the replay boundary P = (N-1)//128*128 and, with the fallback,
        P-128. Every resend and every follow-up that diverges at N hits one of them (<= 192
        tokens recomputed); without it half the lengths miss. Dense retention is unchanged."""
        for mode in self.MODES:
            for n in self.LENGTHS:  # every N mod 128
                p_r = (n - 1) // 128 * 128
                for case, new_len, least in (("resend", n, 65), ("follow-up", n + 1000, 64)):
                    want = p_r if n - p_r >= least else p_r - 128
                    got = self._hit(self._cached(n, mode, None, fallback=True), n, new_len)
                    stock = self._hit(self._cached(n, mode, None), n, new_len)
                    if mode is None:
                        self.assertEqual((got, stock), (want, want), (mode, case, n))
                        continue
                    self.assertEqual(got, want, (mode, case, n))
                    self.assertLessEqual(n - got, 192)
                    # stock: the same P with a whole peek block, else a miss (or kv-pack's 8192)
                    self.assertEqual(stock, want if want == p_r else (8192 if mode == 4096 else 0), (mode, case, n))

    def test_fallback_under_replay_needs_2606(self) -> None:
        """With replay at 2606 every fallback hit stays (its window [P-256, P-128) is in the
        320-row valid tail whenever it is needed); at 2605 the resends with N = 64 mod 128 miss."""
        for mode in self.MODES:
            for n in self.LENGTHS:
                for new_len in (n, n + 1000):
                    want = self._hit(self._cached(n, mode, None, fallback=True), n, new_len)
                    self.assertEqual(self._hit(self._cached(n, mode, 2606, fallback=True), n, new_len), want,
                                     (mode, n, new_len))
                    lost = self._hit(self._cached(n, mode, 2605, fallback=True), n, new_len) != want
                    self.assertEqual(lost, new_len == n and n % 128 == 64, (mode, n, new_len))

    # -- (B) shared-prefix junction anchors

    def _siblings(self, mode, sys_len, q, rows, fallback, anchors=True):
        """Prefix hits of three requests that share only a sys_len-token system prompt (each with
        q tokens of its own), in order, as e15 admits them: its junction is the full-attention hit
        when the SWA groups lag it (KVCacheManager.get_computed_blocks)."""
        n, cached, hits = sys_len + q, set(), []
        for k in range(3):
            shared = sys_len if k else 0
            p = self._hit(cached, shared, n) if shared else 0
            full = min(shared, n - 1) // 128 * 128
            junction = full if p < full else 0
            hits.append(p)
            cached |= self._cached(n, mode, rows, junction, fallback=fallback, hit=p, anchors=anchors)
        return hits

    def test_sibling_hits_the_junction_on_the_third_request(self) -> None:
        """Stock e15: request 2 establishes the junction J = sys_len//128*128, request 3 hits J (a
        whole peek block inside the shared prompt) or, with the fallback, J - 128. With replay
        the anchor keeps request 3's hit identical to stock; without anchors it misses."""
        for mode in self.MODES:
            lengths = range(9000, 9128) if mode == 0 else range(9000, 9128, 9)
            for fallback in (False, True):
                for q in (500, 3000):
                    for sys_len in lengths:
                        j = sys_len // 128 * 128
                        stock = self._siblings(mode, sys_len, q, None, fallback)
                        replay = self._siblings(mode, sys_len, q, dr.DEFAULT_ROWS, fallback)
                        key = (mode, fallback, q, sys_len)
                        if sys_len - j >= 64:
                            want = j
                        elif fallback or mode is None:  # dense retention keeps J - 128 too
                            want = j - 128
                        else:  # kv-pack's last 4096 checkpoint below, else a miss
                            want = 8192 if mode == 4096 else 0
                        self.assertEqual(stock[2], want, key)
                        # replay drops checkpoints below the frontier: only kv-pack without the
                        # fallback differs (8192 -> 0)
                        lost = mode == 4096 and not fallback and sys_len - j < 64
                        self.assertEqual(replay[2], 0 if lost else stock[2], key)
                        self.assertEqual(replay[:2], [0, 0], key)  # dense / kv-pack's 8192 hits for 2 go
                        if mode == 0:
                            self.assertEqual(stock[:2], [0, 0], key)
                        if want and mode == 0 and sys_len % 3 == 0:
                            bare = self._siblings(mode, sys_len, q, dr.DEFAULT_ROWS, fallback, anchors=False)
                            self.assertEqual(bare[2], 0, key)

    def test_anchor_exemptions_serve_no_stale_window(self) -> None:
        """Request 2 (anchored at J) is exact on [J-256, J) and [N-320, N). Whatever a later prompt
        shares with it, a hit P reads its window [P-128, P) inside those rows."""
        tail = dr.valid_tail(dr.DEFAULT_ROWS, WINDOW, 19)
        for mode in self.MODES:
            for sys_len in (9000, 9040, 9080, 9127):
                for q in (500, 3000):
                    n = sys_len + q
                    j = sys_len // 128 * 128
                    cached = self._cached(n, mode, dr.DEFAULT_ROWS, j, fallback=True)
                    for shared in sorted(set(range(64, n + 1, 37)) | {j + 64, j, j - 64, n}):
                        p = self._hit(cached, shared, shared + 1000)
                        ok = p == 0 or p - 128 >= n - tail or (j - 256 <= p - 128 and p <= j)
                        self.assertTrue(ok, (mode, sys_len, q, shared, p))

    # -- (C) decode-filled blocks, prefix_reply_retain.py

    def test_reply_retain_hits_at_the_reply_end(self) -> None:
        """A follow-up that repeats an r-token reply verbatim shares the prompt and r - 1
        generated tokens with KV. Sparse retention caches no decode-filled SWA block, so it hits
        at the prompt end at best; with the lever within 192 tokens of the shared end, replay
        on or off. Every window it reads is exact (decode rows, or the prompt's valid tail)."""
        tail = dr.valid_tail(dr.DEFAULT_ROWS, WINDOW, 19)
        for mode in (0, 4096):
            for n in range(9000, 9128, 7):
                for r in (1, 63, 64, 65, 130, 200, 700):
                    shared = n + r - 1
                    for rows in (None, dr.DEFAULT_ROWS):
                        key = (mode, n, r, rows)
                        got = self._hit(self._cached(n, mode, rows, fallback=True, computed=shared, reply=True),
                                        shared, shared + 1000)
                        bare = self._hit(self._cached(n, mode, rows, fallback=True, computed=shared),
                                         shared, shared + 1000)
                        self.assertGreaterEqual(got, shared - 192, key)
                        self.assertLessEqual(bare, n, key)
                        if rows is not None:
                            self.assertGreaterEqual(got - 128, n - tail, key)

    def test_replay_ab_expectations_are_the_pinned_search(self) -> None:
        """tools/replay_ab.py prefix's expected hits equal this simulation of e15's own code for
        every case it sends (resend / at-N follow-up per residue, siblings, verbatim reply)."""
        sys.path.insert(0, str(ROOT / "tools"))
        import replay_ab as ab

        for mode in (0, 4096):
            for fallback in (False, True):
                for rows in (None, 2606, dr.DEFAULT_ROWS):
                    opts = dict(fallback=fallback, retention=mode, replay_rows=rows or 0)
                    for n in self.LENGTHS:
                        blocks, sim = ab.cached_blocks(n, n, **opts), self._cached(n, mode, rows, fallback=fallback)
                        key = (mode, fallback, rows, n)
                        self.assertEqual(ab.landing(blocks, n, n - 1), self._hit(sim, n, n), key)  # resend
                        self.assertEqual(ab.landing(blocks, n, n + 999), self._hit(sim, n, n + 1000), key)
                        p_r = (n - 1) // 128 * 128
                        for shared in range(p_r - 70, n, 7):  # a new final question (newq_r{R})
                            self.assertEqual(ab.landing(blocks, shared, n - 1), self._hit(sim, shared, n), key)
                    for sys_len in range(9000, 9128, 9):
                        for q in (500, 3000):
                            n, j = sys_len + q, sys_len // 128 * 128
                            hits = self._siblings(mode, sys_len, q, rows, fallback)
                            first = ab.cached_blocks(n, n, **opts)
                            self.assertEqual(ab.landing(first, sys_len, n - 1), hits[1], (mode, fallback, rows, sys_len))
                            both = first | ab.cached_blocks(n, n, junction=j if hits[1] < j else 0, hit=hits[1], **opts)
                            self.assertEqual(ab.landing(both, sys_len, n - 1), hits[2], (mode, fallback, rows, sys_len))
                    for n in range(9000, 9128, 11):
                        for r in (2, 64, 200, 512):
                            kv = n + r - 1
                            for reply in (False, True):
                                sim = self._cached(n, mode, rows, fallback=fallback, computed=kv, reply=reply)
                                blocks = ab.cached_blocks(n, kv, reply=reply, **opts)
                                self.assertEqual(ab.landing(blocks, n + r, n + r + 999),
                                                 self._hit(sim, n + r, n + r + 1000), (mode, fallback, rows, n, r))


ANCHOR = dr.anchor_rows(WINDOW, 19)


def _plan(mode, rows, prompt_len, hit, keeps, prefilling, start, qlen, anchor=0, with_lo=False):
    k, c, n, lo = dr.plan_batch(mode, rows, WINDOW, [prompt_len], [hit], [keeps], [prefilling], [start], [qlen],
                                anchor=[anchor], anchor_len=ANCHOR if mode == "exact" else 0)
    out = (int(k[0]), int(c[0]), bool(n[0]))
    return out + (int(lo[0]),) if with_lo else out


class PlanTests(unittest.TestCase):
    def test_chunks_before_the_window_skip_the_decoder(self) -> None:
        self.assertEqual(_plan("exact", 2541, 64000, 0, False, True, 0, 8192), (0, 61459, True))
        self.assertEqual(_plan("exact", 2541, 64000, 0, False, True, 57344, 6656), (2541, 61459, True))

    def test_window_spanning_two_chunks(self) -> None:
        # 9000-token prompt: chunk 0 keeps [6459, 8192), chunk 1 keeps everything and reads
        # back only to 8065 >= 6459, so it runs the plain path.
        self.assertEqual(_plan("exact", 2541, 9000, 0, False, True, 0, 8192), (1733, 6459, True))
        self.assertEqual(_plan("exact", 2541, 9000, 0, False, True, 8192, 808), (808, 6459, False))
        # window start in chunk 0's last 127 rows: chunk 1 keeps every row but needs the clamp
        self.assertEqual(_plan("exact", 2541, 10691, 0, False, True, 0, 8192), (42, 8150, True))
        self.assertEqual(_plan("exact", 2541, 10691, 0, False, True, 8192, 2499), (2499, 8150, True))

    def test_rows_that_never_trim(self) -> None:
        self.assertEqual(_plan("exact", 2541, 64000, 0, False, False, 63999, 4), (4, 0, False))  # decode
        self.assertEqual(_plan("exact", 2541, 64000, 0, True, True, 0, 8192), (8192, 0, False))  # logprobs
        self.assertEqual(_plan("exact", 2541, 9000, 6528, False, True, 6528, 2472), (2472, 0, False))  # hit
        self.assertEqual(_plan("exact", 2541, 2541, 0, False, True, 0, 2541), (2541, 0, False))
        self.assertEqual(_plan("exact", 0, 64000, 0, False, True, 0, 8192), (8192, 0, False))
        self.assertEqual(_plan(None, 2541, 64000, 0, False, True, 0, 8192), (8192, 0, False))

    def test_resumed_prefill_past_the_prompt_is_kept(self) -> None:
        # preempted after 100 outputs: the re-prefill covers 9100 rows, anchored at the prompt end
        self.assertEqual(_plan("exact", 2541, 9000, 0, False, True, 8192, 908), (908, 6459, False))

    def test_approximate_mode_is_per_chunk(self) -> None:
        self.assertEqual(_plan("approx", 128, 64000, 0, False, True, 0, 8192), (128, 8064, True))
        self.assertEqual(_plan("approx", 128, 64000, 0, False, True, 57344, 6656), (128, 63872, True))
        self.assertEqual(_plan("approx", 128, 64000, 0, False, True, 63950, 50), (50, 0, False))

    def test_mixed_batch_keeps_decodes_and_gathers_suffix_rows(self) -> None:
        # decode A (4 verify rows), prefill B chunk wholly before its window, prefill C final chunk
        qsl = np.array([0, 4, 8196, 9196])
        kept, clamp, needs, lo = dr.plan_batch(
            "exact", 2541, WINDOW,
            prompt_len=[500, 64000, 3000], hit=[0, 0, 0], keeps=[False] * 3,
            prefilling=[False, True, True], start=[496, 8192, 2000], qlen=np.diff(qsl),
        )
        self.assertEqual(kept.tolist(), [4, 0, 1000])
        self.assertEqual(clamp.tolist(), [0, 61459, 459])
        self.assertEqual(needs.tolist(), [False, True, False])
        self.assertEqual(lo.tolist(), [0, 8192, 0])
        rows = dr.kept_rows(qsl, kept)
        self.assertEqual(rows.tolist(), list(range(0, 4)) + list(range(8196, 9196)))
        self.assertEqual(dr.kept_rows(qsl, kept, lo).tolist(), rows.tolist())
        self.assertEqual(dr.kept_rows(qsl, [0, 0, 0]).size, 0)

    def test_junction_run(self) -> None:
        """A 64000-token prompt with a junction at 16384 also keeps [16384 - 2542, 16384), its
        window clamped at 13842: exact on [J - 256, J)."""
        self.assertEqual(ANCHOR, 2542)
        self.assertEqual(dr.kept_runs(2606, 64000, 0, 16384, ANCHOR), [(13842, 16384, 13842), (61394, None, 61394)])
        # chunk [8192, 16384) ends at J: a suffix
        self.assertEqual(_plan("exact", 2606, 64000, 0, False, True, 8192, 8192, anchor=16384, with_lo=True),
                         (2542, 13842, True, 5650))
        # J inside a chunk with trimmed rows after it: one segment, not a suffix
        self.assertEqual(_plan("exact", 2606, 64000, 0, False, True, 12000, 8192, anchor=16384, with_lo=True),
                         (2542, 13842, True, 1842))
        self.assertEqual(_plan("exact", 2606, 64000, 0, False, True, 16384, 8192, anchor=16384), (0, 61394, True))
        # the junction run starts at the hit: no clamp
        self.assertEqual(dr.kept_runs(2606, 64000, 15000, 16384, ANCHOR), [(15000, 16384, 0), (61394, None, 61394)])
        self.assertEqual(_plan("exact", 2606, 64000, 15000, False, True, 15000, 8192, anchor=16384, with_lo=True),
                         (1384, 0, True, 0))
        # without an anchor the same chunk keeps nothing
        self.assertEqual(_plan("exact", 2606, 64000, 0, False, True, 8192, 8192), (0, 61394, True))

    def test_junction_run_and_end_window_in_one_chunk(self) -> None:
        """J = 16384, prompt 19500: the end window starts at 16894, inside J's next chunk; with a
        prompt of 18600 it starts at 15994 < J and the two runs merge."""
        self.assertEqual(dr.kept_runs(2606, 19500, 0, 16384, ANCHOR), [(13842, 16384, 13842), (16894, None, 16894)])
        # chunk [12000, 19500) holds the end of J's run, the gap [16384, 16894) and the end window:
        # one span from 13842 to the chunk end under J's clamp
        self.assertEqual(_plan("exact", 2606, 19500, 0, False, True, 12000, 7500, anchor=16384, with_lo=True),
                         (5658, 13842, True, 1842))
        self.assertEqual(dr.kept_runs(2606, 18600, 0, 16384, ANCHOR), [(13842, None, 13842)])
        self.assertEqual(_plan("exact", 2606, 18600, 0, False, True, 8192, 8192, anchor=16384, with_lo=True),
                         (2542, 13842, True, 5650))
        # J past the end window's start: the merged run starts at J - 2542 or the window start
        self.assertEqual(dr.kept_runs(2606, 64000, 0, 63000, ANCHOR), [(60458, None, 60458)])
        self.assertEqual(dr.kept_runs(2606, 64000, 0, 63990, ANCHOR), [(61394, None, 61394)])


def _fake_core(n_layers=LAYERS, cut=20):
    """What ReplayRunner.__init__ reads of a DeepseekV4Model: config, end_layer, attention prefixes."""

    def attn(i):
        swa = types.SimpleNamespace(prefix=f"layers.{i}.attn.swa")
        if i == cut:
            return types.SimpleNamespace(swa_cache_layer=swa, compressed_cache_prefix=f"layers.{i}.attn.c",
                                         indexer=types.SimpleNamespace(k_cache=types.SimpleNamespace(prefix="idx")))
        return types.SimpleNamespace(swa_cache_layer=swa, compressed_cache_prefix=None, indexer=None)

    cfg = types.SimpleNamespace(kv_source_layer_ids=KV_SOURCES, num_hidden_layers=n_layers, sliding_window=WINDOW)
    return types.SimpleNamespace(config=cfg, end_layer=n_layers,
                                 layers=[types.SimpleNamespace(attn=attn(i)) for i in range(n_layers)])


def _input_batch(reqs, padded=None):
    """InputBatch host arrays for [(req_index, start, qlen, prefilling)], decodes first."""
    qsl = np.concatenate([[0], np.cumsum([q for _, _, q, _ in reqs])]).astype(np.int32)
    return types.SimpleNamespace(
        num_reqs=len(reqs),
        query_start_loc_np=qsl,
        idx_mapping_np=np.array([i for i, _, _, _ in reqs], dtype=np.int32),
        is_prefilling_np=np.array([p for _, _, _, p in reqs], dtype=bool),
        num_computed_tokens_np=np.array([s for _, s, _, _ in reqs], dtype=np.int32),
        num_tokens_after_padding=int(qsl[-1]) if padded is None else padded,
    )


class StepDecisionTests(unittest.TestCase):
    """What a step does, from ReplayRunner.plan_step and step_kind (host arrays only)."""

    def _ranks(self):
        """Two ranks' runners, fed the same NewRequestData (the scheduler broadcasts one output)."""
        new = [types.SimpleNamespace(prompt_len=n, num_computed_tokens=0,
                                     sampling_params=types.SimpleNamespace(prompt_logprobs=None))
               for n in (64000, 30000, 500, 10756)]
        runners = []
        for _ in range(2):
            r = dr.ReplayRunner(None, _fake_core(), 2606, 8, "cpu")
            for i, d in enumerate(new):
                r.add_request(i, d)
            runners.append(r)
        return runners

    def _decide(self, runner, reqs, padded=None):
        plan = runner.plan_step(_input_batch(reqs, padded))
        return None if plan is None else (plan[3], plan[1].tolist(), plan[2].tolist())

    def test_step_kind(self) -> None:
        self.assertEqual(dr.step_kind([0, 0], [8192, 100]), "skip")
        self.assertEqual(dr.step_kind([8192, 100], [8192, 100]), "clamp")
        self.assertEqual(dr.step_kind([4, 0], [4, 8188]), "trim")

    def test_both_ranks_take_the_same_branch(self) -> None:
        steps = [
            [(0, 0, 8192, True), (1, 0, 8192, True)],  # chunks wholly before both windows
            [(2, 499, 1, False), (0, 8192, 8192, True)],  # a decode keeps its row
            [(3, 8192, 2564, True)],  # 10756's last chunk: every row kept, window clamped at 8150
            [(1, 24576, 5424, True)],  # 30000's last chunk: keeps its last 2606 rows
            [(2, 499, 1, False)],  # decode only: no replay path
        ]
        want = [
            ("skip", [0, 0], [61394, 27394]),
            ("trim", [1, 0], [0, 61394]),
            ("clamp", [2564], [8150]),
            ("trim", [2606], [27394]),
            None,
        ]
        a, b = self._ranks()
        for step, expect in zip(steps, want):
            self.assertEqual(self._decide(a, step), expect, step)
            self.assertEqual(self._decide(b, step), self._decide(a, step), step)

    def test_padded_or_untrimmed_steps_run_every_layer(self) -> None:
        a, _ = self._ranks()
        self.assertIsNone(self._decide(a, [(0, 0, 8192, True)], padded=8200))
        off = dr.ReplayRunner(None, _fake_core(), 0, 8, "cpu")
        off.add_request(0, types.SimpleNamespace(prompt_len=64000, num_computed_tokens=0,
                                                 sampling_params=types.SimpleNamespace(prompt_logprobs=None)))
        self.assertIsNone(self._decide(off, [(0, 0, 8192, True)]))
        a.add_request(4, types.SimpleNamespace(prompt_len=64000, num_computed_tokens=0,
                                               sampling_params=types.SimpleNamespace(prompt_logprobs=1)))
        self.assertIsNone(self._decide(a, [(4, 0, 8192, True)]))  # prompt logprobs keep every row

    def test_the_scheduler_anchor_reaches_the_plan(self) -> None:
        """NewRequestData's anchor (both ranks get the same pickled object) adds the junction run;
        a warmup request built on the worker carries none."""
        ranks = self._ranks()
        for r in ranks:
            anchored = types.SimpleNamespace(prompt_len=64000, num_computed_tokens=0,
                                             sampling_params=types.SimpleNamespace(prompt_logprobs=None))
            setattr(anchored, dr.ANCHOR_ATTR, 16384)
            r.add_request(5, anchored)
            r.add_request(6, types.SimpleNamespace(prompt_len=64000, num_computed_tokens=0,
                                                   sampling_params=types.SimpleNamespace(prompt_logprobs=None)))
        a, b = ranks
        self.assertEqual((int(a.anchor[5]), int(a.anchor[6])), (16384, 0))
        step = [(5, 8192, 8192, True), (6, 8192, 8192, True)]
        plan = a.plan_step(_input_batch(step))
        self.assertEqual((plan[3], plan[1].tolist(), plan[2].tolist(), plan[4].tolist()),
                         ("trim", [2542, 0], [13842, 61394], [5650, 8192]))
        self.assertEqual(dr.kept_rows(plan[0], plan[1], plan[4]).tolist(), list(range(5650, 8192)))
        other = b.plan_step(_input_batch(step))
        for x, y in zip(plan, other):
            self.assertEqual(np.asarray(x).tolist() if not isinstance(x, str) else x,
                             np.asarray(y).tolist() if not isinstance(y, str) else y)
        approx = dr.ReplayRunner(None, _fake_core(), 128, 8, "cpu")
        anchored = types.SimpleNamespace(prompt_len=64000, num_computed_tokens=0,
                                         sampling_params=types.SimpleNamespace(prompt_logprobs=None))
        setattr(anchored, dr.ANCHOR_ATTR, 16384)
        approx.add_request(0, anchored)
        self.assertEqual(int(approx.anchor[0]), 0)  # approximate W: no junction runs


class AnchorTransportTests(unittest.TestCase):
    """The junction anchor rides from the scheduler to every TP rank on NewRequestData."""

    @staticmethod
    def _new_request_data(mod) -> None:
        """exec the pinned NewRequestData into ``mod`` (which must be in sys.modules)."""
        import dataclasses

        mod.dataclass = dataclasses.dataclass
        mod.strip_covered_mm_data = lambda feats, num_computed, **kw: feats
        # output.py imports its annotation types under TYPE_CHECKING only (lazy annotations)
        src = "from __future__ import annotations\n" + textwrap.dedent(_pinned_e15("NewRequestData"))
        exec(compile(src, "pinned output.py", "exec"), mod.__dict__)

    def test_anchor_survives_the_executor_pickle(self) -> None:
        from unittest import mock

        mod = types.ModuleType("dsv41_pinned_sched_output")
        with mock.patch.dict(sys.modules, {mod.__name__: mod}):
            self._new_request_data(mod)
            self._round_trip(mod.NewRequestData)

    def _round_trip(self, cls) -> None:
        import copyreg
        import io
        import pickle

        self.assertFalse(cls.__dataclass_params__.frozen)
        self.assertNotIn("__slots__", cls.__dict__)
        self.assertIsInstance(cls.__dict__["from_request"], classmethod)
        req = types.SimpleNamespace(request_id="r1", prompt_token_ids=[1, 2, 3], mm_features=[], num_computed_tokens=0,
                                    sampling_params=None, pooling_params=None, lora_request=None, prompt_embeds=None,
                                    prompt_is_token_ids=None)
        setattr(req, dr.ATTR, (19680, 16384))
        data = cls.from_request(req, ([7],))
        setattr(data, dr.ANCHOR_ATTR, dr.anchor_of(req))  # install()'s from_request wrapper
        # MessageQueue.enqueue / recv (shm for local readers, ZMQ for the headless rank): a
        # Pickler over copyreg's dispatch table, HIGHEST_PROTOCOL, out-of-band buffers.
        enq, recv = _pinned_e15("MessageQueue.enqueue"), _pinned_e15("MessageQueue.recv")
        for needle in ("dispatch_table = dict(copyreg.dispatch_table)", "pickle.Pickler(",
                       "protocol=pickle.HIGHEST_PROTOCOL", "pickler.dump(obj)"):
            self.assertIn(needle, enq)
        self.assertIn("pickle.loads(recv, buffers=recv_oob)", recv)
        with io.BytesIO() as bio:
            pickler = pickle.Pickler(bio, protocol=pickle.HIGHEST_PROTOCOL, buffer_callback=lambda b: True)
            pickler.dispatch_table = dict(copyreg.dispatch_table)
            pickler.dump(("execute_model", (types.SimpleNamespace(scheduled_new_reqs=[data]),), {}, 0))
            got = pickle.loads(bio.getvalue())
        sent = got[1][0].scheduled_new_reqs[0]
        self.assertEqual((sent.req_id, sent.prompt_len, getattr(sent, dr.ANCHOR_ATTR)), ("r1", 3, 16384))
        self.assertEqual(dr.anchor_of(types.SimpleNamespace()), 0)

    def test_e15_sets_the_junction_before_allocating_and_resends_resumed_requests_new(self) -> None:
        sched = _pinned_e15("Scheduler.schedule")
        junction = sched.index("request.shared_prefix_boundary,")
        self.assertIn("self.kv_cache_manager.allocate_slots(", sched[junction:])
        self.assertIn("scheduled_new_reqs.extend(scheduled_resumed_reqs)", sched)  # V2 runner
        self.assertIn("NewRequestData.from_request(", sched)
        self.assertIn("shared_prefix_boundary = (\n            num_new_computed_tokens + num_uncached if num_uncached else 0",
                      _pinned_e15("KVCacheManager.get_computed_blocks"))


class PrefixEndFallbackTests(unittest.TestCase):
    """prefix_end_fallback.py (DSV41_PREFIX_END_FALLBACK)."""

    def test_env(self) -> None:
        self.assertTrue(pef.enabled({}))
        self.assertTrue(pef.enabled({pef.ENV: "1"}))
        self.assertFalse(pef.enabled({pef.ENV: " 0 "}))
        with self.assertRaises(ValueError):
            pef.enabled({pef.ENV: "2"})
        self.assertEqual(pef.install({pef.ENV: "0"}), "off")

    def test_fallback_boundaries(self) -> None:
        self.assertEqual(pef.fallback_boundaries([8999], 128), (8999, 8871))
        self.assertEqual(pef.fallback_boundaries([8999, 4480], 128), (8999, 4480, 8871, 4352))
        self.assertEqual(pef.fallback_boundaries([200], 128), (200,))  # P - 128 would be 0: no hit
        self.assertEqual(pef.fallback_boundaries([], 128), ())

    def test_install_wraps_the_pinned_class(self) -> None:
        from unittest import mock

        swm = _pinned_swa_manager()
        mod = types.ModuleType("vllm.v1.core.single_type_kv_cache_manager")
        mod.SlidingWindowManager = swm
        fakes = {name: types.ModuleType(name) for name in ("vllm", "vllm.v1", "vllm.v1.core")}
        fakes[mod.__name__] = mod
        stock = swm.reachable_block_mask
        spec = _SWASpec()
        with mock.patch.dict(sys.modules, fakes):
            self.assertIn("b-128", pef.install({}))
            self.assertEqual(pef.install({}), "already installed")
        for mode in (None, 0, 4096):
            got = swm.reachable_block_mask(0, 140, 128, spec, True, retention_interval=mode, reachable_boundaries=[8999])
            want = stock(0, 140, 128, spec, True, retention_interval=mode, reachable_boundaries=[8999, 8871])
            self.assertEqual(got, want, mode)
        self.assertEqual(sorted(i for i, k in enumerate(swm.reachable_block_mask(
            0, 140, 128, spec, True, retention_interval=0, reachable_boundaries=[8999])) if k),
            [136, 137, 138, 139])  # P = 8960: 138-139 (its peek 140 is past end_block); P-128: 136-138


class _NT:
    """numpy stand-in for the torch.Tensor methods run_tail calls."""

    def __init__(self, a):
        self.a = np.asarray(a)

    @property
    def shape(self):
        return self.a.shape

    def index_select(self, dim, idx):
        assert dim == 0
        return _NT(self.a[idx.a])

    def index_copy_(self, dim, idx, src):
        assert dim == 0
        self.a[idx.a] = src.a
        return self

    def index_copy(self, dim, idx, src):
        return _NT(self.a.copy()).index_copy_(dim, idx, src)

    def mean(self, dim):
        return _NT(self.a.mean(axis=dim))


class RunTailHostTests(unittest.TestCase):
    """run_tail's three branches with numpy stand-ins, fake layers and a fake vllm."""

    N, HC, HID, CUT, END = 10, 4, 6, 2, 5

    def setUp(self) -> None:
        self.log = []
        log = self.log

        class Layer:
            def __init__(self, i):
                self.i = i

            def __call__(self, h, pos, ids, pre, pm, rm, r, eh, em):
                log.append(("layer", self.i, h.shape[0]))
                return _NT(h.a * 2 + self.i), _NT(r.a + 1), pm, rm, _NT(pre.a + self.i)

            def write_kv(self, h, pos, ids, pre, pm, rm, r):
                log.append(("write_kv", h.shape[0]))
                return _NT(r.a + 100.0)  # the stream at the cut

        self.core = types.SimpleNamespace(layers=[Layer(i) for i in range(self.END)], end_layer=self.END,
                                          aux_hidden_state_layers=(4, 5))
        n, hc, hid = self.N, self.HC, self.HID
        self.h = _NT(np.arange(n * hid, dtype=np.float64).reshape(n, hid))
        self.r = _NT(np.ones((n, hc, hid)))
        self.pm, self.rm = _NT(np.zeros((n, hc, 1))), _NT(np.zeros((n, hc, hc)))
        self.pre = _NT(np.full((n, hc), 0.25))
        self.pos = _NT(np.arange(100, 100 + n))

    def _run(self, kind, rows):
        from contextlib import contextmanager
        from unittest import mock

        log = self.log

        @contextmanager
        def override_forward_context(ctx):
            log.append(("context", ctx))
            yield

        def build(pos):
            log.append(("build", pos.a.tolist()))
            return "sub-batch context"

        fc = types.ModuleType("vllm.forward_context")
        fc.override_forward_context = override_forward_context
        fake_model = types.SimpleNamespace(mhc_post_tilelang=lambda h, r, pm, rm: _NT(r.a + h.a[:, None, :]))
        self.core._dsv41_replay = types.SimpleNamespace(cut=self.CUT, batch=dr.ReplayBatch(None, rows, kind, build))
        # "skip" must not import vllm at all (a None entry makes the import raise)
        mods = {"vllm": None, "vllm.forward_context": None} if kind == "skip" else {
            "vllm": types.ModuleType("vllm"), "vllm.forward_context": fc}
        with mock.patch.dict(sys.modules, mods), mock.patch.object(dr, "_model_module", lambda: fake_model):
            out = dr.run_tail(self.core, self.h, self.pos, None, self.pre, self.pm, self.rm, self.r)
        self.assertIsNone(self.core._dsv41_replay.batch)
        return out

    def test_skip_runs_nothing_past_write_kv(self) -> None:
        h, pre, aux = self._run("skip", None)
        self.assertEqual(self.log, [("write_kv", self.N)])  # no layer, build or context
        stream = self.r.a + 100.0
        np.testing.assert_array_equal(h.a, stream)
        self.assertIs(pre, self.pre)
        self.assertEqual(len(aux), 2)  # inputs of layers 4 and 5 (the end)
        for a in aux:
            np.testing.assert_array_equal(a.a, stream.mean(axis=1))

    def test_trim_runs_the_kept_rows_and_fills_the_rest_from_the_cut(self) -> None:
        rows = _NT(np.array([3, 4, 8, 9]))
        h, pre, aux = self._run("trim", rows)
        self.assertEqual(self.log, [("write_kv", self.N), ("build", [103, 104, 108, 109]),
                                    ("context", "sub-batch context"), ("layer", 2, 4), ("layer", 3, 4), ("layer", 4, 4)])
        hk, rk = self.h.a[rows.a], self.r.a[rows.a]
        for i in range(self.CUT, self.END):
            hk, rk = hk * 2 + i, rk + 1
            if i + 1 == 4:
                aux4 = (rk + hk[:, None, :]).mean(axis=1)
        want = rk + hk[:, None, :]
        trimmed = np.setdiff1d(np.arange(self.N), rows.a)
        stream = self.r.a + 100.0
        np.testing.assert_array_equal(h.a[rows.a], want)
        np.testing.assert_array_equal(h.a[trimmed], stream[trimmed])
        np.testing.assert_array_equal(pre.a[rows.a], np.full((4, self.HC), 0.25 + 2 + 3 + 4))
        np.testing.assert_array_equal(pre.a[trimmed], self.pre.a[trimmed])
        np.testing.assert_array_equal(self.pre.a, np.full((self.N, self.HC), 0.25))  # input untouched
        self.assertEqual(len(aux), 2)
        np.testing.assert_array_equal(aux[0].a[rows.a], aux4)
        np.testing.assert_array_equal(aux[1].a[rows.a], want.mean(axis=1))
        for a in aux:
            np.testing.assert_array_equal(a.a[trimmed], stream.mean(axis=1)[trimmed])

    def test_clamp_runs_every_row_without_write_kv(self) -> None:
        h, pre, aux = self._run("clamp", _NT(np.arange(self.N)))
        self.assertEqual([e for e in self.log if e[0] != "context"],
                         [("build", list(range(100, 100 + self.N)))] + [("layer", i, self.N) for i in range(2, 5)])
        hk, rk = self.h.a, self.r.a
        for i in range(self.CUT, self.END):
            hk, rk = hk * 2 + i, rk + 1
        np.testing.assert_array_equal(h.a, rk + hk[:, None, :])
        self.assertEqual(len(aux), 2)


class ClampReferenceTests(unittest.TestCase):
    def test_clamped_rows_are_the_suffix_of_the_stock_rows(self) -> None:
        w, width = 8, 8
        positions = np.array([3, 10, 11, 12, 20, 21])
        clamp = np.array([0, 9, 9, 0, 18, 30])
        slots = lambda p: 1000 + p  # noqa: E731 - stand-in for block_table * block_size + offset
        idx = np.full((6, width), -1, dtype=np.int32)
        lens = np.zeros(6, dtype=np.int32)
        for t, p in enumerate(positions):
            start = max(p - (w - 1), 0)
            row = [slots(q) for q in range(start, p + 1)]
            idx[t, : len(row)], lens[t] = row, len(row)
        lens[5] = 0  # an invalid (padding) token keeps len 0
        idx[5] = -1
        out_i, out_l = dr.clamp_rows_reference(idx, lens, positions, clamp, w)
        for t, p in enumerate(positions[:5]):
            lo = max(p - (w - 1), 0, clamp[t])
            want = [slots(q) for q in range(lo, p + 1)]
            self.assertEqual(out_l[t], len(want))
            self.assertEqual(out_i[t, : len(want)].tolist(), want)
            self.assertTrue((out_i[t, len(want) :] == -1).all())
        self.assertEqual(out_l[5], 0)


M = 2_147_483_647


class ToyCED:
    """Decoder layers 0..after (0 = the cut layer, whose KV is written for every row).

    KV_j(p) = f_j(H_j(p)); H_{j+1}(p) = g_j(H_j(p), KV_j over [max(lo, p-w+1), p]) where lo is
    the clamp (layers past the cut only). Integer arithmetic mod a prime: equal means bit-equal.
    """

    def __init__(self, w: int, after: int, seed: int = 0) -> None:
        self.w, self.after, self.n = w, after, after + 1
        rng = np.random.default_rng(seed)
        self.a, self.b, self.c, self.d = (rng.integers(1, M, self.n, dtype=np.int64) for _ in range(4))

    def kv(self, j, h):
        return (h * self.a[j] + self.b[j]) % M

    def out(self, j, h, cache, p, lo):
        q = np.arange(max(lo, p - self.w + 1, 0), p + 1)
        return int((h * self.c[j] + int(((cache[q] * (p - q + 7)) % M).sum()) + self.d[j]) % M)

    def full(self, h0):
        n = len(h0)
        hs, kvs = [h0.copy()], []
        for j in range(self.n):
            kv = self.kv(j, hs[j])
            kvs.append(kv)
            hs.append(np.array([self.out(j, hs[j][p], kv, p, 0) for p in range(n)], dtype=np.int64))
        return hs, kvs

    def serve(self, h0, rows, mode, chunk, *, hit=0, cache=None, garbage=1, clamp_on=True, eager=None, anchor=0):
        """Chunked prefill through dr.plan_batch, rows [hit, len(h0)). ``eager(s)`` False runs a
        chunk the plain way (a graph step: never trims, never clamps). ``anchor``: a junction J."""
        n = len(h0)
        g = np.random.default_rng(garbage)
        kvs = [c.copy() for c in cache] if cache is not None else [g.integers(0, M, n, dtype=np.int64)
                                                                   for _ in range(self.n)]
        hs = [np.full(n, -1, dtype=np.int64) for _ in range(self.n + 1)]
        anchor_len = dr.anchor_rows(self.w, self.after) if mode == "exact" else 0
        for s in range(hit, n, chunk):
            e = min(n, s + chunk)
            kept, cl, needs, lo = dr.plan_batch(mode, rows, self.w, [n], [hit], [False], [True], [s], [e - s],
                                                anchor=[anchor], anchor_len=anchor_len)
            if eager is not None and not eager(s):
                needs = np.array([False])
            if needs[0]:
                a, b, c = s + int(lo[0]), s + int(lo[0]) + int(kept[0]), (int(cl[0]) if clamp_on else 0)
            else:
                a, b, c = s, e, 0
            kvs[0][s:e] = self.kv(0, h0[s:e])  # write_kv: the cut layer's KV, every row
            h = h0[a:b].copy()
            for j in range(self.n):
                hs[j][a:b] = h
                kvs[j][a:b] = self.kv(j, h)
                h = np.array([self.out(j, h[i], kvs[j], p, c if j else 0)
                              for i, p in enumerate(range(a, b))], dtype=np.int64)
            hs[self.n][a:b] = h
        return hs, kvs


class ToyExactnessTests(unittest.TestCase):
    W, AFTER = 4, 6  # exact from 4 + 5*3 = 19 (the 2414 analog)

    def setUp(self) -> None:
        self.toy = ToyCED(self.W, self.AFTER)
        self.min = dr.exact_min_rows(self.W, self.AFTER)
        # the 2541 analog (W + AFTER*(W-1) = 22, tail 7) and the default's (2286 + 128 + 192:
        # 15 + the window + 3 blocks = 25, tail 10)
        self.half = self.W + self.AFTER * (self.W - 1)
        self.default = (self.AFTER - 1) * (self.W - 1) + self.NEED * self.BS + 3 * self.BS

    def _h0(self, n, seed=5):
        return np.random.default_rng(seed).integers(0, M, n, dtype=np.int64)

    def _assert_decode_state_exact(self, n, rows, served, ref, tail):
        hs, kvs = served
        fh, fk = ref
        self.assertEqual(hs[self.toy.n][n - 1], fh[self.toy.n][n - 1], "last row output")
        for j in range(self.toy.n):
            np.testing.assert_array_equal(kvs[j][n - tail :], fk[j][n - tail :], f"layer {j} KV tail")
        for j in range(self.toy.n - 3, self.toy.n):  # DSpark taps: inputs of the last 3 layers
            np.testing.assert_array_equal(hs[j][n - self.W :], fh[j][n - self.W :], f"taps {j}")

    def test_end_anchored_window_is_exact_from_the_minimum(self) -> None:
        self.assertEqual(self.min, 19)
        for n, chunk in ((61, 16), (64, 16), (70, 9), (45, 64), (100, 7)):
            h0 = self._h0(n)
            ref = self.toy.full(h0)
            for rows in (self.min, self.half, self.default, self.default + 5):
                tail = dr.valid_tail(rows, self.W, self.AFTER)
                served = self.toy.serve(h0, rows, "exact", chunk)
                self._assert_decode_state_exact(n, rows, served, ref, tail)

    def test_one_row_less_is_not_exact(self) -> None:
        n = 61
        h0 = self._h0(n)
        ref = self.toy.full(h0)
        served = self.toy.serve(h0, self.min - 1, "exact", 16)
        self.assertNotEqual(served[0][self.toy.n][n - 1], ref[0][self.toy.n][n - 1])

    def test_needed_rows_never_read_unwritten_kv(self) -> None:
        n = 61
        h0 = self._h0(n)
        for rows, mode in ((self.default, "exact"), (self.W, "approx"), (10, "approx")):
            a = self.toy.serve(h0, rows, mode, 16, garbage=1)
            b = self.toy.serve(h0, rows, mode, 16, garbage=2)
            tail = dr.valid_tail(rows, self.W, self.AFTER)
            self.assertEqual(a[0][self.toy.n][n - 1], b[0][self.toy.n][n - 1], (rows, mode))
            for j in range(self.toy.n):
                np.testing.assert_array_equal(a[1][j][n - tail :], b[1][j][n - tail :])

    def test_graph_steps_that_skip_the_plan_stay_exact(self) -> None:
        """A chunk run the plain way (no trim, no clamp) reads unwritten KV only inside the zone."""
        for n, chunk in ((61, 3), (64, 5)):
            h0 = self._h0(n)
            ref = self.toy.full(h0)
            for rows in (self.min, self.default):
                served = self.toy.serve(h0, rows, "exact", chunk, eager=lambda s: s % 2 == 0)
                self._assert_decode_state_exact(n, rows, served, ref, dr.valid_tail(rows, self.W, self.AFTER))

    def test_approximate_mode_differs_but_is_defined(self) -> None:
        n = 61
        h0 = self._h0(n)
        ref = self.toy.full(h0)
        served = self.toy.serve(h0, self.W, "approx", 16)
        self.assertNotEqual(served[0][self.toy.n][n - 1], ref[0][self.toy.n][n - 1])

    BS, NEED = 2, 2  # the toy's 64-row blocks / 2 window blocks (128-row window), scaled down

    def _hit_ok(self, p, match_end, front):
        """e15's rule for an SWA group with the DSpark peek block, scaled: P aligned to 2 blocks,
        window blocks [P - NEED*BS, P) and the peek block [P, P + BS) matched and cached; a block
        is cached when it starts at or past the frontier."""
        bs = self.BS
        if p % (2 * bs) or p - self.NEED * bs < 0 or p + bs > match_end:
            return False
        return all(b * bs >= front for b in range((p - self.NEED * bs) // bs, (p + bs) // bs))

    def _request_b(self, h_a, kv_a, p, n_new=30, rows=None, keep=None):
        """B = A's first ``keep`` rows + n_new new rows, prefix-hit at p on A's cached window blocks."""
        rows = self.default if rows is None else rows
        h_b = np.concatenate([h_a[:keep], self._h0(n_new, seed=13)])
        cache = []
        for j in range(self.toy.n):
            c = np.random.default_rng(99).integers(0, M, len(h_b), dtype=np.int64)  # unwritten
            c[p - self.NEED * self.BS : p] = kv_a[j][p - self.NEED * self.BS : p]
            cache.append(c)
        return h_b, self.toy.serve(h_b, rows, "exact", 16, hit=p, cache=cache)

    def test_prefix_hits_under_the_frontier_are_exact(self) -> None:
        hits = 0
        for rows in (self.half, self.default):
            tail = dr.valid_tail(rows, self.W, self.AFTER)
            for n_a in range(56, 72):
                h_a = self._h0(n_a, seed=11)
                kv_a = self.toy.serve(h_a, rows, "exact", 16)[1]
                front = dr.frontier(n_a, 0, False, rows, tail)
                ok = [p for p in range(n_a + 1) if self._hit_ok(p, n_a, front)]
                if rows == self.default:
                    self.assertTrue(ok, n_a)  # every prompt end keeps a hit (dense caching)
                if not ok:
                    continue
                for n_new in (10, 30):  # B's suffix shorter than W (reads A's window KV) / longer (trims)
                    hits += 1
                    h_b, served_b = self._request_b(h_a, kv_a, max(ok), n_new=n_new, rows=rows)
                    self._assert_decode_state_exact(len(h_b), rows, served_b, self.toy.full(h_b), tail)
        self.assertGreater(hits, 32)

    def test_a_hit_the_mask_forbids_is_wrong(self) -> None:
        # B resends A's first 50 rows; p = 36 is aligned and its window [32, 36) lies below A's
        # window (S = 61 - 25 = 36), so A never wrote decoder KV there. B's 14-row suffix is
        # shorter than the receptive field (19), so B's last row reads that unwritten KV. (A
        # suffix of at least the receptive field never reads below p in the decoder layers.)
        n_a, p = 61, 36
        h_a = self._h0(n_a, seed=11)
        kv_a = self.toy.serve(h_a, self.default, "exact", 16)[1]
        front = dr.frontier(n_a, 0, False, self.default, dr.valid_tail(self.default, self.W, self.AFTER))
        self.assertFalse(self._hit_ok(p, 50, front))
        h_b, served_b = self._request_b(h_a, kv_a, p, n_new=0, keep=50)
        self.assertNotEqual(served_b[0][self.toy.n][len(h_b) - 1], self.toy.full(h_b)[0][self.toy.n][len(h_b) - 1])

    def test_junction_run_is_exact_on_the_windows_of_j_and_j_minus_one_unit(self) -> None:
        """B (a junction J below its end window) keeps [J - anchor_rows, J): every layer's KV on
        [J - 2W, J) is the full forward's, so a third request C that hits J or J - 2*BS (the
        fallback) on B's cached window blocks serves exactly; without the anchor it does not."""
        before, unit = 2 * self.W, self.NEED * self.BS  # the 256 and 128 analogs
        self.assertEqual(dr.anchor_rows(self.W, self.AFTER), 15 + before)
        cases = 0
        for n_b, j_at, chunk in ((90, 40, 16), (90, 44, 16), (80, 52, 16), (66, 36, 16), (70, 40, 7), (61, 28, 64)):
            h_b = self._h0(n_b, seed=17)
            ref_b = self.toy.full(h_b)
            served_b = self.toy.serve(h_b, self.default, "exact", chunk, anchor=j_at)
            self._assert_decode_state_exact(n_b, self.default, served_b, ref_b, dr.valid_tail(self.default, self.W, self.AFTER))
            for j in range(self.toy.n):
                np.testing.assert_array_equal(served_b[1][j][j_at - before : j_at], ref_b[1][j][j_at - before : j_at],
                                              f"layer {j} KV under J={j_at}")
            bare = self.toy.serve(h_b, self.default, "exact", chunk)[1]
            for p in (j_at, j_at - unit):  # C hits J, or the fallback one unit below
                # C's suffix (<= W) does not trim, so every decoder layer reads B's window KV
                h_c = np.concatenate([h_b[: j_at + 1], self._h0(10, seed=19)])
                for kv_src, exact in ((served_b[1], True), (bare, False)):
                    cache = []
                    for j in range(self.toy.n):
                        c = np.random.default_rng(99).integers(0, M, len(h_c), dtype=np.int64)  # unwritten
                        c[p - unit : p] = kv_src[j][p - unit : p]
                        cache.append(c)
                    got = self.toy.serve(h_c, self.default, "exact", 16, hit=p, cache=cache)
                    same = got[0][self.toy.n][len(h_c) - 1] == self.toy.full(h_c)[0][self.toy.n][len(h_c) - 1]
                    self.assertEqual(same, exact, (n_b, j_at, chunk, p))
                    cases += exact
        self.assertEqual(cases, 12)


class _Rec:
    """A traced value: records ops for the fake forward."""

    def __init__(self, tag):
        self.tag = tag

    def mean(self, dim):
        return ("mean", self.tag, dim)

    def __eq__(self, other):
        return isinstance(other, _Rec) and other.tag == self.tag

    def __repr__(self):
        return f"R({self.tag!r})"


POS = types.SimpleNamespace(shape=(5,))  # positions: forward reads only .shape[0] here


def _forward_ns(log, replay):
    pp = types.SimpleNamespace(is_first_rank=True, is_last_rank=True)
    return {
        "torch": types.SimpleNamespace(Tensor=object, int64="int64"),
        "IntermediateTensors": object,
        "islice": __import__("itertools").islice,
        "typing": __import__("typing"),
        "get_pp_group": lambda: pp,
        "is_forward_context_available": lambda: True,
        "envs": types.SimpleNamespace(VLLM_MOE_SKIP_PADDING=False),
        "mhc_post_tilelang": lambda h, r, pm, rm: (log.append(("post", h)), _Rec(("post", h)))[1],
        "hc_collapse_triton": lambda h, pm: ("collapse", h, pm),
        "_dsv41_decoder_replay": replay,
    }


class _Layer:
    def __init__(self, i, log):
        self.i, self.log = i, log

    def __call__(self, h, positions, input_ids, pre, post, res, residual, eh, em):
        self.log.append(("layer", self.i))
        return f"h{self.i}", f"r{self.i}", f"pm{self.i}", f"rm{self.i}", f"pre{self.i}"


def _fake_model(log, n_layers=6, aux=(4, 5)):
    m = types.SimpleNamespace(
        layers=[_Layer(i, log) for i in range(n_layers)], start_layer=0, end_layer=n_layers,
        engram_hash=None, use_mega_moe=False, use_sequence_parallel=False, _mtp_hidden_buffer=None,
        aux_hidden_state_layers=aux,
    )
    m.embed_input_ids = lambda ids: "embed"
    m.norm = lambda x: ("norm", x)
    return m


class ForwardRewriteTests(unittest.TestCase):
    def _forward(self, src, ns):
        local: dict = {}
        exec(compile(textwrap.dedent(src), "forward", "exec", dont_inherit=True), ns, local)
        return local["forward"]

    def setUp(self) -> None:
        self.src = _method_src(FIX / "dsv41_model_e13.pin.py", "DeepseekV4Model", "forward")

    def test_pinned_model_is_the_e15_file(self) -> None:
        e15 = Path("/home/sfxnz/projects/data/dsv41-1017/perf2/src/vllm-e15/models/deepseek_v4_1/nvidia/model.py")
        if not e15.is_file():
            self.skipTest("extracted e15 tree not on this host")
        self.assertEqual((FIX / "dsv41_model_e13.pin.py").read_text(), e15.read_text())

    def test_anchors_apply_once_and_drift_refuses(self) -> None:
        out = dr.patch_forward_source(self.src)
        self.assertEqual(out.count("_dsv41_decoder_replay.stop_layer(self)"), 1)
        self.assertEqual(out.count("_dsv41_decoder_replay.run_tail("), 1)
        compile(textwrap.dedent(out), "forward", "exec", dont_inherit=True)
        with self.assertRaises(ValueError):
            dr.patch_forward_source(self.src.replace("self.end_layer),", "self.end_layer ),"))
        with self.assertRaises(ValueError):
            dr.patch_forward_source(out)

    def test_original_function_under_wrappers_and_code_swap(self) -> None:
        """install() finds the image's forward under other patches' wrappers (functools.wraps,
        or a bare closure like the step census) and swaps its code in place, so they keep
        calling it."""
        import functools

        ns: dict = {}
        exec(compile("def forward(self, x):\n    return ('stock', x)\n", "/img/model.py", "exec"), ns)
        orig = ns["forward"]

        @functools.wraps(orig)
        def wrapped(*a, **k):  # ar_l2_prefetch style: __wrapped__, copied qualname
            return ("l2pf", orig(*a, **k))

        def census(fn):  # sm120_page.wrap_gpu style: a closure, no __wrapped__
            def inner(*a, **k):
                return ("census", fn(*a, **k))

            return inner

        outer = census(wrapped)
        self.assertIs(dr.original_function(outer, ("/img/model.py",), "forward"), orig)
        self.assertIs(dr.original_function(orig, ("/img/model.py",), "forward"), orig)
        self.assertIsNone(dr.original_function(outer, ("/other.py",), "forward"))
        new: dict = {}
        exec(compile("def forward(self, x):\n    return ('replay', x)\n", "/img/model.py [x]", "exec"), new)
        orig.__code__ = new["forward"].__code__
        self.assertEqual(outer(None, 1), ("census", ("l2pf", ("replay", 1))))
        self.assertIs(dr.original_function(outer, ("/img/model.py", "/img/model.py [x]"), "forward"), orig)

    def test_no_replay_step_runs_the_stock_loop(self) -> None:
        def run(src):
            log = []
            replay = types.SimpleNamespace(
                stop_layer=lambda model: model.end_layer,
                run_tail=lambda *a: self.fail("run_tail on a plain step"),
            )
            out = self._forward(src, _forward_ns(log, replay))(_fake_model(log), "ids", POS, None)
            return out, log

        self.assertEqual(run(self.src), run(dr.patch_forward_source(self.src)))

    def test_replay_step_stops_at_the_cut_and_uses_the_tail(self) -> None:
        log, calls = [], []

        def run_tail(model, h, positions, ids, pre, post, res, residual):
            calls.append((h, positions, ids, pre, post, res, residual))
            return "tail_h", "tail_pre", ["tail_aux5"]

        replay = types.SimpleNamespace(stop_layer=lambda model: 3, run_tail=run_tail)
        fwd = self._forward(dr.patch_forward_source(self.src), _forward_ns(log, replay))
        hidden, aux = fwd(_fake_model(log, aux=(2, 5)), "ids", POS, None)
        self.assertEqual([e for e in log if e[0] == "layer"], [("layer", 0), ("layer", 1), ("layer", 2)])
        self.assertEqual(calls, [("h2", POS, "ids", "pre2", "pm2", "rm2", "r2")])
        self.assertEqual(hidden, ("norm", ("collapse", "tail_h", "tail_pre")))
        self.assertEqual(aux, [("mean", ("post", "h1"), 1), "tail_aux5"])


class WiringTests(unittest.TestCase):
    def test_sitecustomize_registers_the_patch(self) -> None:
        src = (PATCH / "sitecustomize.py").read_text()
        self.assertIn('_patch("decoder_replay", _p_decoder_replay)', src)
        self.assertIn('raise _PatchSkip(f"{decoder_replay.ENV}=0")', src)

    def test_run_sh_forwards_and_defaults_the_env(self) -> None:
        run = (ROOT / "run.sh").read_text()
        self.assertIn(f"  {dr.ENV}=\n", run)
        self.assertIn(f'{dr.ENV}="${{{dr.ENV}:-{dr.DEFAULT_ROWS}}}"', run)
        self.assertIn(f"    {dr.ENV}: {dr.DEFAULT_ROWS}\n", (ROOT / "recipe.yaml").read_text())

    def test_audit_expects_the_armed_line_at_the_default(self) -> None:
        sys.path.insert(0, str(ROOT / "tools"))
        import engagement_audit as ea

        self.assertEqual(ea.PATCHES["decoder_replay.py"], {dr.ENV: str(dr.DEFAULT_ROWS)})
        expected, disarm = ea.expectations({dr.ENV: str(dr.DEFAULT_ROWS)})
        self.assertIn(("decoder_replay.py", dr.LOG_ENGAGED), expected)
        self.assertIn(dr.LOG_DISARMED, disarm)
        expected, _ = ea.expectations({dr.ENV: "128"})
        self.assertNotIn(("decoder_replay.py", dr.LOG_ENGAGED), expected)
        line = "(Worker_TP1 pid=7) dsv41: decoder replay DISABLED -> drafter mtp (only DSpark); every step runs every layer"
        problems = ea.audit({"worker": line}, {dr.ENV: "0"})
        self.assertEqual([q for q in problems if "decoder replay" in q], ["worker: " + line[:240]])

    def test_pinned_e15_signatures(self) -> None:
        self.assertEqual(
            _params(_pinned_e15("KVCacheManager.allocate_slots"))[:7],
            ["self", "request", "num_new_tokens", "num_new_computed_tokens", "new_computed_blocks",
             "num_lookahead_tokens", "num_external_computed_tokens"],
        )
        self.assertEqual(
            _params(_pinned_e15("BlockPool.cache_full_blocks")),
            ["self", "request", "blocks", "num_cached_blocks", "num_full_blocks", "block_size",
             "kv_cache_group_id", "block_mask"],
        )
        self.assertIn("for_capture", _params(_pinned_e15("DefaultModelState.prepare_attn")))
        self.assertIn("block_mask=block_mask", _pinned_e15("SingleTypeKVCacheManager.cache_blocks"))
        alloc = _pinned_e15("KVCacheManager.allocate_slots")
        self.assertIn("self.coordinator.cache_blocks(request, num_tokens_to_cache)", alloc)
        init = _pinned_e15("Scheduler.__init__")
        for needle in ("self.vllm_config = vllm_config", "self.kv_cache_config = kv_cache_config",
                       "self.kv_cache_manager = KVCacheManager("):
            self.assertIn(needle, init)
        build = _params(_pinned_e15("build_attn_metadata"))
        for name in ("attn_groups", "num_reqs", "num_tokens", "query_start_loc_gpu", "query_start_loc_cpu",
                     "max_query_len", "seq_lens", "max_seq_len", "block_tables", "slot_mappings",
                     "kv_cache_config", "seq_lens_cpu_upper_bound", "positions", "is_prefilling"):
            self.assertIn(name, build)
        ctx = _params(_pinned_e15("create_forward_context"))
        for name in ("cudagraph_runtime_mode", "batch_descriptor", "slot_mapping", "additional_kwargs",
                     "skip_compiled"):
            self.assertIn(name, ctx)

    def test_pinned_window_layouts(self) -> None:
        """What the clamp and the DSpark window argument rely on, in the e15 kernels."""
        swa = _pinned_e15("_compute_swa_indices_and_lens_kernel")
        self.assertIn("start_pos = tl.maximum(pos - (window_size - 1) - left_add, 0)", swa)
        self.assertIn("swa_len = end_pos - start_pos", swa)
        self.assertIn("pos_offset = start_pos + offset", swa)
        draft = _pinned_e15("ComputeDSparkNoncausalSWAIndicesKernel.kernel")  # DSpark non-causal: context [prefix - window, prefix) + block
        self.assertIn("start_pos = tl.maximum(prefix_len - window_size, 0)", draft)
        aux = _pinned_e15("get_eagle3_aux_layers_from_config")
        self.assertIn("layer_ids = list(dspark_layer_ids)", aux)  # v4.1 taps = inputs of 37-39
        hit = _pinned_e15("_contiguous_blocks_for_hit")
        self.assertIn("blocks = cdiv(window_size - 1, block_size)", hit)
        self.assertIn("blocks += 1", hit)

    def test_replay_ab_compare(self) -> None:
        sys.path.insert(0, str(ROOT / "tools"))
        import replay_ab as ab

        self.assertEqual(ab.first_divergence(list("abcd"), list("abxd")), 2)
        self.assertEqual(ab.first_divergence(list("ab"), list("abc")), 2)
        a = {"prompt_tokens": 9, "tokens": ["x", "y", "z"], "logprobs": [-0.1, -0.5, -1.0],
             "top": [["x", "q"], ["y", "r"], ["z", "s"]], "hits": 0}
        b = {"prompt_tokens": 9, "tokens": ["x", "y", "w"], "logprobs": [-0.1, -0.4, -2.0],
             "top": [["x", "q"], ["y", "t"], ["w", "s"]], "hits": 128}
        row = ab.compare_item(a, b)
        self.assertEqual(row["first_divergence"], 2)
        self.assertAlmostEqual(row["max_abs_dlogprob"], 0.1)
        self.assertEqual(row["min_top5_overlap"], 0.5)
        self.assertEqual(row["hits"], (0, 128))

    def test_markers_are_printed(self) -> None:
        src = (PATCH / "decoder_replay.py").read_text()
        self.assertGreaterEqual(src.count(dr.LOG_ENGAGED), 2)
        self.assertGreaterEqual(src.count(dr.LOG_DISARMED), 2)

    def test_prefix_end_fallback_wiring(self) -> None:
        site = (PATCH / "sitecustomize.py").read_text()
        self.assertIn('_patch("prefix_end_fallback", _p_prefix_end_fallback)', site)
        self.assertLess(site.index('_patch("prefix_end_fallback"'), site.index('_patch("decode_levers"'))
        run = (ROOT / "run.sh").read_text()
        self.assertIn(f"  {pef.ENV}=\n", run)
        self.assertIn(f'{pef.ENV}="${{{pef.ENV}:-1}}"', run)
        self.assertIn(f"    {pef.ENV}: 1\n", (ROOT / "recipe.yaml").read_text())
        sys.path.insert(0, str(ROOT / "tools"))
        import engagement_audit as ea

        self.assertEqual(ea.PATCHES["prefix_end_fallback.py"], {pef.ENV: "1"})
        expected, disarm = ea.expectations({pef.ENV: "1"})
        self.assertIn(("prefix_end_fallback.py", pef.LOG_ENGAGED), expected)
        self.assertIn(pef.LOG_DISARMED, disarm)
        expected, _ = ea.expectations({pef.ENV: "0"})
        self.assertNotIn(("prefix_end_fallback.py", pef.LOG_ENGAGED), expected)

    def test_prefix_reply_retain_wiring(self) -> None:
        site = (PATCH / "sitecustomize.py").read_text()
        self.assertIn('_patch("prefix_reply_retain", _p_prefix_reply_retain)', site)
        self.assertLess(site.index('_patch("prefix_reply_retain"'), site.index('_patch("decode_levers"'))
        run = (ROOT / "run.sh").read_text()
        self.assertIn(f"  {prr.ENV}=\n", run)
        self.assertIn(f'{prr.ENV}="${{{prr.ENV}:-1}}"', run)
        self.assertIn(f"    {prr.ENV}: 1\n", (ROOT / "recipe.yaml").read_text())
        sys.path.insert(0, str(ROOT / "tools"))
        import engagement_audit as ea

        self.assertEqual(ea.PATCHES["prefix_reply_retain.py"], {prr.ENV: "1"})
        expected, disarm = ea.expectations({prr.ENV: "1"})
        self.assertIn(("prefix_reply_retain.py", prr.LOG_ENGAGED), expected)
        self.assertIn(prr.LOG_DISARMED, disarm)


class PrefixReplyRetainTests(unittest.TestCase):
    """prefix_reply_retain.py (DSV41_PREFIX_REPLY_RETAIN) on stand-ins for the e15 classes."""

    def test_env_and_mask(self) -> None:
        self.assertTrue(prr.enabled({}))
        self.assertFalse(prr.enabled({prr.ENV: "0"}))
        with self.assertRaises(ValueError):
            prr.enabled({prr.ENV: "yes"})
        self.assertEqual(prr.install({prr.ENV: "0"}), "off")
        # prompt 300 tokens: blocks 4 (rows 256..319, holds 300..319) and on are decode-filled
        self.assertEqual(prr.reply_mask([False, True, False, False], 2, 6, 64, 300), [False, True, True, True])
        self.assertIsNone(prr.reply_mask(None, 2, 6, 64, 300))  # every block cached already
        self.assertEqual(prr.reply_mask([False, False], 0, 2, 64, 9000), [False, False])

    def test_install_on_stand_ins(self) -> None:
        from unittest import mock

        calls = []

        class BlockPool:
            def cache_full_blocks(self, request, blocks, num_cached_blocks, num_full_blocks, block_size,
                                  kv_cache_group_id, block_mask=None):
                calls.append((kv_cache_group_id, block_mask))

        class SlidingWindowSpec:
            pass

        class Scheduler:
            def __init__(self, groups):
                self.kv_cache_config = types.SimpleNamespace(kv_cache_groups=groups)
                self.kv_cache_manager = types.SimpleNamespace(block_pool=BlockPool())

        mods = {name: types.ModuleType(name) for name in (
            "vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.block_pool", "vllm.v1.core.sched",
            "vllm.v1.core.sched.scheduler", "vllm.v1.kv_cache_interface")}
        mods["vllm.v1.core.block_pool"].BlockPool = BlockPool
        mods["vllm.v1.core.sched.scheduler"].Scheduler = Scheduler
        mods["vllm.v1.kv_cache_interface"].SlidingWindowSpec = SlidingWindowSpec
        mods["vllm.v1.kv_cache_interface"].iter_layer_specs = lambda spec: (spec,)
        mods["vllm.v1.core"].block_pool = mods["vllm.v1.core.block_pool"]
        mods["vllm.v1.core.sched"].scheduler = mods["vllm.v1.core.sched.scheduler"]
        with mock.patch.dict(sys.modules, mods):
            self.assertIn("decode-filled", prr.install({}))
            self.assertEqual(prr.install({}), "already installed")
        sched = Scheduler([types.SimpleNamespace(kv_cache_spec=SlidingWindowSpec()),
                           types.SimpleNamespace(kv_cache_spec=object())])
        pool = sched.kv_cache_manager.block_pool
        self.assertEqual(getattr(pool, prr.ATTR), frozenset({0}))
        req = types.SimpleNamespace(num_prompt_tokens=300)
        pool.cache_full_blocks(req, [], 2, 6, 64, 0, block_mask=[False] * 4)
        pool.cache_full_blocks(req, [], 2, 6, 64, 1, block_mask=[False] * 4)  # full attention: untouched
        pool.cache_full_blocks(req, [], 2, 6, 64, 0)
        self.assertEqual(calls, [(0, [False, False, True, True]), (1, [False] * 4), (0, None)])


class SubBatchBuildTests(unittest.TestCase):
    """The mid-forward sub-batch build and kv-pack's SWA batching (swa_meta_batched.py wraps the
    build_attn_metadata name in vllm.v1.worker.gpu.model_states.default and batches only calls
    without model_specific_attn_metadata)."""

    def _build_fn(self):
        tree = ast.parse((PATCH / "decoder_replay.py").read_text())
        runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ReplayRunner")
        return next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "_build")

    def test_entry_point_and_model_specific_metadata(self) -> None:
        fn = self._build_fn()
        imports = {(n.module, a.name) for n in ast.walk(fn) if isinstance(n, ast.ImportFrom) for a in n.names}
        self.assertIn(("vllm.v1.worker.gpu.attn_utils", "build_attn_metadata"), imports)
        self.assertIn(("vllm.v1.worker.gpu.model_states.interface", "ModelSpecificAttnMetadata"), imports)
        self.assertNotIn("vllm.v1.worker.gpu.model_states.default", {m for m, _ in imports})
        common = next(n.value for n in ast.walk(fn) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "common" for t in n.targets))
        kw = {k.arg: ast.unparse(k.value) for k in common.keywords}
        self.assertEqual(kw["model_specific_attn_metadata"], "ModelSpecificAttnMetadata()")
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "build_attn_metadata"]
        self.assertEqual(len(calls), 2)  # the sub-batch and the cut layer's unclamped SWA copy
        for call in calls:
            self.assertEqual([ast.unparse(k.value) for k in call.keywords if k.arg is None], ["common"])

    def test_base_model_specific_metadata_adds_nothing(self) -> None:
        """e15's build_attn_metadata (pinned) hands every builder the same metadata and kwargs
        with the base ModelSpecificAttnMetadata as with None."""
        import collections.abc
        import typing

        iface: dict = {"Any": typing.Any}
        src = "class ModelSpecificAttnMetadata:\n" + "".join(
            _pinned_e15(f"ModelSpecificAttnMetadata.{m}") for m in ("get_extra_common_attn_kwargs", "get_extra_attn_kwargs"))
        exec(compile(src, "pinned interface", "exec"), iface)
        calls = []

        class Common:
            def __init__(self, **kw):
                self.kw = kw

        class Builder:
            def build(self, common_prefix_len, common_attn_metadata, **extra):
                calls.append((common_prefix_len, common_attn_metadata.kw, extra))
                return object()

        ns = {"CommonAttentionMetadata": Common, "Sequence": collections.abc.Sequence,
              "Mapping": collections.abc.Mapping, "torch": types.SimpleNamespace(Tensor=object), "Any": typing.Any,
              "AttentionGroup": object, "KVCacheConfig": object,
              "ModelSpecificAttnMetadata": iface["ModelSpecificAttnMetadata"]}
        exec(compile(textwrap.dedent(_pinned_e15("build_attn_metadata")), "pinned attn_utils", "exec"), ns)
        group = types.SimpleNamespace(get_metadata_builder=lambda i: Builder(), layer_names=["layers.21.attn.swa"])
        kwargs = dict(attn_groups=[[group], []], num_reqs=1, num_tokens=3, query_start_loc_gpu="qsl",
                      query_start_loc_cpu="qsl_cpu", max_query_len=3, seq_lens=[9], max_seq_len=9,
                      block_tables=["bt0", "bt1"], slot_mappings=["sm0", "sm1"],
                      kv_cache_config=types.SimpleNamespace(kv_cache_groups=[None, None]),
                      seq_lens_cpu_upper_bound=[9], positions="pos", is_prefilling="pf")
        plain = ns["build_attn_metadata"](**kwargs)
        based = ns["build_attn_metadata"](**kwargs, model_specific_attn_metadata=iface["ModelSpecificAttnMetadata"]())
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(list(plain), list(based))

    @unittest.skipUnless((PATCH / "swa_meta_batched.py").is_file(), "kv-pack's swa_meta_batched.py is not merged here")
    def test_kv_pack_batching_refuses_the_sub_batch_build(self) -> None:
        import swa_meta_batched as smb

        kwargs = {"attn_groups": [[]], "kv_cache_config": object(), "block_tables": (), "slot_mappings": object()}
        saved = dict(smb._STATE)
        try:
            smb._STATE["armed"] = True
            self.assertTrue(smb.call_eligible(kwargs))
            self.assertFalse(smb.call_eligible({**kwargs, "model_specific_attn_metadata": object()}))
        finally:
            smb._STATE.clear()
            smb._STATE.update(saved)
        # Outside the wrapped call (prepare_attn's) no plan is published: rows_written() is False
        # for every build, so the sub-batch's SWA builders run their own launches.
        self.assertIsNone(smb._CALL["plan"])


@unittest.skipUnless(HAS_TORCH, "torch: run in the image")
class TorchClampTests(unittest.TestCase):
    def test_torch_clamp_matches_the_reference(self) -> None:
        import torch

        rng = np.random.default_rng(3)
        w, width, n = 128, 128, 257
        positions = rng.integers(0, 4000, n)
        clamp = np.where(rng.random(n) < 0.6, positions - rng.integers(0, 200, n), 0).clip(0)
        idx = np.full((n, 1, width), -1, dtype=np.int32)
        lens = np.zeros(n, dtype=np.int32)
        for t, p in enumerate(positions):
            start = max(int(p) - (w - 1), 0)
            row = np.arange(start, p + 1) * 3 + 7
            idx[t, 0, : row.size], lens[t] = row, row.size
        lens[-1], idx[-1] = 0, -1
        want_i, want_l = dr.clamp_rows_reference(idx[:, 0], lens, positions, clamp, w)
        ti, tl = torch.from_numpy(idx.copy()), torch.from_numpy(lens.copy())
        dr.clamp_swa_rows(torch, ti, tl, torch.from_numpy(positions), torch.from_numpy(clamp.astype(np.int32)),
                          w, verify=True)
        np.testing.assert_array_equal(ti[:, 0].numpy(), want_i)
        np.testing.assert_array_equal(tl.numpy(), want_l)
        bad = torch.from_numpy(lens.copy())
        bad[0] += 1
        with self.assertRaises(RuntimeError):
            dr.clamp_swa_rows(torch, torch.from_numpy(idx.copy()), bad, torch.from_numpy(positions),
                              torch.from_numpy(clamp.astype(np.int32)), w, verify=True)


@unittest.skipUnless(HAS_VLLM, "vllm: run in the image")
class TorchTailTests(unittest.TestCase):
    def test_run_tail_gathers_runs_and_scatters(self) -> None:
        import torch

        n, hc, hid, cut, end = 10, 4, 6, 2, 5
        rows = torch.tensor([3, 4, 8, 9])
        built = []

        class Layer:
            def __init__(self, i):
                self.i = i

            def __call__(self, h, pos, ids, pre, pm, rm, r, eh, em):
                return h * 2 + self.i, r + 1, pm, rm, pre + self.i

            def write_kv(self, *a):
                built.append(("write_kv", a[0].shape[0]))
                return a[6] + 100  # the stream at the cut, a fresh tensor

        core = types.SimpleNamespace(layers=[Layer(i) for i in range(end)], end_layer=end,
                                     aux_hidden_state_layers=(4, 5))
        runner = types.SimpleNamespace(cut=cut, batch=dr.ReplayBatch(None, rows, "trim",
                                                                     lambda pos: built.append(pos.tolist())))
        core._dsv41_replay = runner
        fake_mod = types.SimpleNamespace(mhc_post_tilelang=lambda h, r, pm, rm: r + h[:, None, :])
        old = dr._model_module
        dr._model_module = lambda: fake_mod
        try:
            h = torch.arange(n * hid, dtype=torch.float32).view(n, hid)
            r = torch.ones(n, hc, hid)
            pm, rm, pre = torch.zeros(n, hc, 1), torch.zeros(n, hc, hc), torch.zeros(n, hc)
            pos = torch.arange(100, 100 + n)
            out_h, out_pre, aux = dr.run_tail(core, h, pos, None, pre, pm, rm, r)
        finally:
            dr._model_module = old
        self.assertIsNone(runner.batch)
        self.assertEqual(built[0], ("write_kv", n))
        self.assertEqual(built[1], [103, 104, 108, 109])
        hk, rk = h[rows], r[rows]
        for i in range(cut, end):
            hk, rk = hk * 2 + i, rk + 1
            if i + 1 == 4:
                aux4 = (rk + hk[:, None, :]).mean(dim=1)
        want = rk + hk[:, None, :]
        trimmed = torch.ones(n, dtype=torch.bool)
        trimmed[rows] = False
        stream = r + 100
        torch.testing.assert_close(out_h[rows], want)
        torch.testing.assert_close(out_h[trimmed], stream[trimmed])  # trimmed rows carry the cut's stream
        self.assertEqual(len(aux), 2)
        torch.testing.assert_close(aux[0][rows], aux4)
        torch.testing.assert_close(aux[1][rows], want.mean(dim=1))
        for a in aux:
            torch.testing.assert_close(a[trimmed], stream.mean(dim=1)[trimmed])
        torch.testing.assert_close(out_pre[rows], torch.full((4, hc), float(2 + 3 + 4)))
        torch.testing.assert_close(out_pre[trimmed], pre[trimmed])
        self.assertTrue((pre == 0).all())  # the forward's pre-mix is not written

    def test_install_on_the_image_classes(self) -> None:
        env = dict(os.environ)
        env[dr.ENV] = str(dr.DEFAULT_ROWS)
        detail = dr.install(env)
        if detail != "already installed":  # the image's sitecustomize installs it at interpreter start
            self.assertIn(f"W={dr.DEFAULT_ROWS}", detail)
        self.assertEqual(dr.install(env), "already installed")
        from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
        from vllm.models.deepseek_v4_1.nvidia import model as m
        from vllm.v1.core.block_pool import BlockPool

        self.assertTrue(m.DeepseekV4Model.forward._dsv41_replay)
        self.assertTrue(hasattr(m.DeepseekV4DecoderLayer, "write_kv"))
        self.assertTrue(hasattr(DeepseekV4Attention, "forward_kv"))
        self.assertIs(m._dsv41_decoder_replay, dr)
        self.assertIn("block_mask", BlockPool.cache_full_blocks.__wrapped__.__code__.co_varnames)


if __name__ == "__main__":
    unittest.main()
