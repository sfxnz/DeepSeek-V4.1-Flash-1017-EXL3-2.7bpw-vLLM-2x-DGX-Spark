"""engram_prefill_ahead.py on the host (numpy, gcc; no torch).

- hash_segment() against a scalar transcription of _hash_ids_kernel (the batch kernel)
  with the real constants' scale (compressed vocab 99,092, primes above 16M, the
  multiplier formula), image (dead) tokens, positions 0..2, unknown window ids, and the
  chunking invariance the read-ahead relies on.
- plan / cover / run arithmetic.
- eng_gather_bf16 flag bit 1: the same bytes, and only the pages the call brought into the
  page cache are dropped (mincore).
- AheadCore + Gather end to end against the stock row semantics on synthetic table files,
  over scripted scheduler steps: chunked prefill, mixed prefill + decode at 4 sequences, a
  prefix-cache start, preemption and re-admission, abort, two prefills in one batch, a
  shorter chunk than read ahead, both TP ranks, and a wrong read-ahead (disarm, rows still
  exact).
The torch glue (Ahead, the stage wrapper, H2D) runs in the image:
kernel_study/fusion_host/engram_prefill_ahead_check.py.
"""

from __future__ import annotations

import ast
import ctypes
import mmap
import os
import random
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker" / "patch"
sys.path.insert(0, str(PATCH))
import engram_native_stage as ens  # noqa: E402
import engram_prefill_ahead as epa  # noqa: E402

DIM, SB = 256, 8
W_OFF = 664  # the real shard's w offset: rows are not page aligned
DEAD = (129264, 129265)  # mm_preprocess IMAGE_SENTINEL_BASE_ID, IMAGE_PAD_ID
VOCAB = 129280
COMPRESSED = 99092
PAD_TOKEN = 2
NGRAM, HEADS, LAYER_IDS = 4, 8, (1, 14)


# ------------------------------------------------------------------ reference pieces


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def layout_primes(vocab: int) -> np.ndarray:
    """EngramLayout's prime draw: [layers, (ngram - 1) * heads], never reused."""
    seen, out = set(), []
    for _ in LAYER_IDS:
        layer = []
        for _ in range(NGRAM - 1):
            current = vocab - 1
            for _ in range(HEADS):
                current += 1
                while not _is_prime(current) or current in seen:
                    current += 1
                seen.add(current)
                layer.append(current)
        out.append(layer)
    return np.array(out, dtype=np.int64)


def make_consts(vocab: int, seed: int) -> dict:
    primes = layout_primes(vocab)
    offs = np.array([np.cumsum([0, *p[:-1]]) for p in primes], dtype=np.int64)
    bound = max(1, (np.iinfo(np.int64).max // COMPRESSED) // 2)
    mult = np.stack([np.random.default_rng(10007 * lid).integers(0, bound, size=(NGRAM,), dtype=np.int64) * 2 + 1
                     for lid in LAYER_IDS])
    tm = np.random.default_rng(seed).integers(0, COMPRESSED, size=VOCAB, dtype=np.int64)
    return epa.hash_consts(tm, mult, primes.reshape(len(LAYER_IDS), NGRAM - 1, HEADS), offs, int(tm[PAD_TOKEN]), DEAD)


def kernel_ref(c, input_ids, positions, qsl, lookback) -> np.ndarray:
    """_hash_ids_kernel (engram.py), one token and layer at a time, V2 runner (no slot cache)."""
    tm, mult, primes, offs, pad = c["tm"], c["mult"], c["primes"], c["offs"], c["pad"]
    dead = {int(x) for x in c["dead"]}
    nl, ngram = mult.shape
    heads = primes.shape[1] // (ngram - 1)
    depth = len(lookback[0]) if len(lookback) else 0
    n, rows = len(input_ids), len(qsl) - 1
    out = np.zeros((n, nl, primes.shape[1]), dtype=np.int64)
    for layer in range(nl):
        for token in range(n):
            lo, hi = 0, rows
            while lo < hi:  # upper bound in query_start_loc[1:]
                mid = (lo + hi) // 2
                if token >= qsl[mid + 1]:
                    lo = mid + 1
                else:
                    hi = mid
            req = min(lo, rows - 1)
            chunk_start = positions[min(qsl[req], n - 1)]
            position = positions[token]
            blocked, rolling = False, 0
            for shift in range(ngram):
                lb = position - shift
                in_batch = lb >= chunk_start
                if in_batch:
                    tok = int(input_ids[max(token - shift, 0)])
                    source = -1 if tok in dead else int(tm[tok])
                else:
                    col = chunk_start - 1 - lb
                    known = 0 <= col < depth and lookback[req][col] >= 0
                    if known:
                        tok = int(lookback[req][col])
                        source = -1 if tok in dead else int(tm[tok])
                    else:
                        source = pad
                blocked = blocked or lb < 0 or source == -1
                value = pad if blocked else source
                rolling ^= value * int(mult[layer, shift])
                if shift > 0:
                    for head in range(heads):
                        col_h = (shift - 1) * heads + head
                        out[token, layer, col_h] = rolling % int(primes[layer, col_h]) + int(offs[layer, col_h])
    return out


def window_of(tokens, pos0: int, depth: int = NGRAM - 1) -> list:
    return [int(tokens[pos0 - 1 - j]) if pos0 - 1 - j >= 0 else -1 for j in range(depth)]


def e4m3fn_lut() -> np.ndarray:
    """torch's float8_e4m3fn -> float32 (0x7F/0xFF -> 0x7FF00000/0xFFF00000)."""
    out = np.zeros(256, np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 3) & 0xF, b & 7
        out[b] = s * (m / 8.0) * 2.0**-6 if e == 0 else s * (1 + m / 8.0) * 2.0 ** (e - 7)
    bits = out.view(np.uint32)
    bits[0x7F], bits[0xFF] = 0x7FF00000, 0xFFF00000
    return out


LUT = e4m3fn_lut()


def bf16_bits(x: np.ndarray) -> np.ndarray:
    u = x.view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(x)
    r[nan] = ((u[nan] >> 16) | 0x40).astype(np.uint16)
    return r


def ref_dequant(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    scale = (s.astype(np.uint32) << 23).view(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        prod = LUT[w].reshape(len(w), SB, -1) * scale[:, :, None]
    return bf16_bits(prod.reshape(len(w), DIM).astype(np.float32))


def stock_rows(w, s, ids: np.ndarray, local_heads: int, v0: int, v1: int) -> np.ndarray:
    """The stock stage's rows (disk_file_rows_owned + gather_dequant) as bf16 bits [n, lh, DIM]."""
    n, hv = ids.shape
    local = np.full((n, local_heads), -1, np.int64)
    local[:, :hv] = ids
    flat = local.reshape(-1)
    owned = (flat >= v0) & (flat < v1)
    rows = np.where(owned, flat, 0)
    out = ref_dequant(w[rows], s[rows])
    out[~owned] = 0
    return out.reshape(n, local_heads, DIM)


def build_lib(tmp: Path):
    if shutil.which("gcc") is None:
        raise unittest.SkipTest("gcc not on PATH")
    return ens.load(ens.build(ens.SRC, tmp / "cache"))


def write_table(path: Path, rows: int, seed: int):
    """[W_OFF pad][w rows x DIM][40 pad][s rows x SB]; returns (w, s, s_off)."""
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 256, (rows, DIM), dtype=np.uint8)
    s = rng.integers(100, 140, (rows, SB), dtype=np.uint8)
    s_off = W_OFF + rows * DIM + 40
    with open(path, "wb") as fh:
        fh.write(b"\0" * W_OFF)
        fh.write(w.tobytes())
        fh.write(b"\0" * 40)
        fh.write(s.tobytes())
    return w, s, s_off


# ----------------------------------------------------------------------------- tests


class EnvTests(unittest.TestCase):
    def test_env_parsing(self) -> None:
        self.assertFalse(epa.enabled({}))
        self.assertFalse(epa.enabled({"DSV41_ENGRAM_PREFILL_AHEAD": "0"}))
        self.assertTrue(epa.enabled({"DSV41_ENGRAM_PREFILL_AHEAD": "1"}))
        self.assertEqual(epa.cap_tokens(8192, {}), 8192)
        self.assertEqual(epa.cap_tokens(8192, {"DSV41_ENGRAM_PREFILL_AHEAD_TOKENS": "4096"}), 4096)
        self.assertEqual(epa.cap_tokens(4096, {"DSV41_ENGRAM_PREFILL_AHEAD_TOKENS": "8192"}), 4096)
        with self.assertRaises(ValueError):
            epa.cap_tokens(8192, {"DSV41_ENGRAM_PREFILL_AHEAD_TOKENS": "-1"})
        self.assertEqual(epa.threads({}), 8)
        self.assertEqual(epa.threads({"DSV41_ENGRAM_PREFILL_AHEAD_THREADS": "2"}), 2)
        with self.assertRaises(ValueError):
            epa.threads({"DSV41_ENGRAM_PREFILL_AHEAD_THREADS": "0"})
        self.assertEqual(epa.verify_calls({}), 4)
        self.assertEqual(epa.verify_calls({"DSV41_ENGRAM_PREFILL_AHEAD_VERIFY": "0"}), 1)
        self.assertTrue(epa.dontneed({}))
        self.assertTrue(epa.dontneed({"DSV41_ENGRAM_PREFILL_AHEAD_DONTNEED": ""}))
        self.assertFalse(epa.dontneed({"DSV41_ENGRAM_PREFILL_AHEAD_DONTNEED": "0"}))
        self.assertIsNone(epa.cpus({}))
        self.assertEqual(epa.cpus({"DSV41_ENGRAM_PREFILL_AHEAD_CPUS": "0-4,10-14"}),
                         set(range(0, 5)) | set(range(10, 15)))
        self.assertEqual(epa.cpus({"DSV41_ENGRAM_PREFILL_AHEAD_CPUS": " 7 "}), {7})
        with self.assertRaises(ValueError):
            epa.cpus({"DSV41_ENGRAM_PREFILL_AHEAD_CPUS": "0-4,"})

    def test_top_level_imports_are_stdlib_only(self) -> None:
        tree = ast.parse(Path(epa.__file__).read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {a.name for n in top if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in top if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"__future__", "ctypes", "os", "queue", "threading", "time", "concurrent.futures"})

    def test_markers_and_wiring(self) -> None:
        src = Path(epa.__file__).read_text()
        for marker in (epa.LOG_ENGAGED, epa.LOG_DISARMED):
            self.assertGreaterEqual(src.count(marker), 2, marker)
        levers = (PATCH / "decode_levers.py").read_text()
        # before the native stage: its wrapper must stay outermost (decode calls skip this lever)
        self.assertLess(levers.index('("engram-prefill-ahead"'), levers.index('("engram-native-stage"'))
        # the wrapper keeps the stock signature the native stage checks for
        self.assertIn("def stage(self, input_ids, positions, query_start_loc, lookback_token_ids, num_tokens):", src)
        self.assertEqual(ens.STOCK_STAGE_PARAMS,
                         ("self", "input_ids", "positions", "query_start_loc", "lookback_token_ids", "num_tokens"))
        for name in ens.REWRITE_METHODS:
            self.assertNotIn(f"def {name}(", src)

    def test_hash_statements_match_the_stock_stage(self) -> None:
        # Ahead.stage re-implements the stock hash + DtoH + sync (self -> st) and
        # _stock_rows the stock per-table gather; catches drift in STAGE_FAST.
        import engram_stage_fast

        src, stock = Path(epa.__file__).read_text(), engram_stage_fast.STAGE_FAST
        for theirs, ours in (
            ("ids = input_ids[:n]", "ids = input_ids[:n]"),
            ("positions[:n],", "positions[:n],"),
            ("image_sentinel_mask(ids),", "mask(ids),"),
            ("image_sentinel_mask(lookback_token_ids),", "mask(lookback),"),
            ("host = self.hash_host[:n]", "host = st.hash_host[:n]"),
            ("host.copy_(hashes[:, :, self.head_start : self.head_end], non_blocking=True)",
             "host.copy_(hashes[:, :, st.head_start : st.head_end], non_blocking=True)"),
            ("self.hashes_ready.record()", "st.hashes_ready.record()"),
            ("self.hashes_ready.synchronize()", "st.hashes_ready.synchronize()"),
            ("local = host[:, engram.layer_hash_index, :].to(torch.int64)",
             "local = host_rows[:, engram.layer_hash_index, :].to(torch.int64)"),
            ("file_rows, owned = engram.embed_tokens.disk_file_rows_owned(local)",
             "file_rows, owned = engram.embed_tokens.disk_file_rows_owned(local)"),
            ("rows = engram.embed_tokens.disk.gather_dequant(file_rows, owned)",
             "rows = engram.embed_tokens.disk.gather_dequant(file_rows, owned)"),
            ("dest = engram._staged_rows_for_ubatch()[:n]", "dest = engram._staged_rows_for_ubatch()"),
            ("self.num_staged = n", "st.num_staged = n"),
        ):
            self.assertIn(theirs, stock, theirs)
            self.assertIn(ours, src, ours)


class HashMirrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.c = make_consts(16_000_000, seed=5)  # the 1017 config: engram_vocab_size 16M

    def _prompt(self, rng, n: int) -> list:
        toks = [rng.randrange(VOCAB) for _ in range(n)]
        for _ in range(max(1, n // 200)):  # image spans
            a = rng.randrange(n)
            for i in range(a, min(n, a + rng.randrange(1, 9))):
                toks[i] = DEAD[rng.randrange(2)]
        return toks

    def test_segment_matches_the_kernel(self) -> None:
        rng = random.Random(1)
        for pos0 in (0, 1, 2, 3, 7, 1000, 123457):
            for n in (1, 2, 5, 64, 300):
                hist = self._prompt(rng, pos0 + n)
                ids = hist[pos0:]
                win = window_of(hist, pos0)
                ref = kernel_ref(self.c, ids, list(range(pos0, pos0 + n)), [0, n], [win])
                for h0, h1 in ((0, 12), (12, 24)):
                    got = epa.hash_segment(self.c, ids, win, pos0, h0, h1)
                    self.assertEqual(got.dtype, np.int32)
                    self.assertTrue(np.array_equal(got, ref[:, :, h0:h1]), (pos0, n, h0))

    def test_unknown_window_ids_take_the_pad_fallback(self) -> None:
        rng = random.Random(2)
        ids = self._prompt(rng, 40)
        for win in ([-1, -1, -1], [5, -1, 7], [DEAD[0], 9, -1]):
            ref = kernel_ref(self.c, ids, list(range(50, 90)), [0, 40], [win])
            self.assertTrue(np.array_equal(epa.hash_segment(self.c, ids, win, 50, 0, 24), ref), win)

    def test_chunking_does_not_change_a_positions_hash(self) -> None:
        # The read-ahead hashes chunk i+1 before the scheduler cuts it: any cut gives
        # the same ids at the same positions.
        rng = random.Random(3)
        prompt = self._prompt(rng, 900)
        whole = epa.hash_segment(self.c, prompt, [-1, -1, -1], 0, 0, 24)
        for _ in range(4):
            pos, parts = 0, []
            while pos < len(prompt):
                n = min(len(prompt) - pos, rng.choice((1, 2, 3, 7, 64, 129, 300)))
                parts.append(kernel_ref(self.c, prompt[pos:pos + n], list(range(pos, pos + n)), [0, n],
                                        [window_of(prompt, pos)]))
                pos += n
            self.assertTrue(np.array_equal(np.concatenate(parts), whole))
        for pos0 in (1, 2, 3, 455):
            seg = epa.hash_segment(self.c, prompt[pos0:pos0 + 200], window_of(prompt, pos0), pos0, 0, 24)
            self.assertTrue(np.array_equal(seg, whole[pos0:pos0 + 200]), pos0)

    def test_a_prefill_segment_inside_a_mixed_batch(self) -> None:
        rng = random.Random(4)
        ids, pos, qsl, wins = [], [], [0], []
        for _ in range(3):  # decode verify segments: bonus + 3 drafts
            hist = self._prompt(rng, 500)
            p0 = rng.randrange(10, 400)
            ids += [rng.randrange(VOCAB) for _ in range(4)]
            pos += list(range(p0, p0 + 4))
            qsl.append(qsl[-1] + 4)
            wins.append(window_of(hist, p0))
        prompt = self._prompt(rng, 700)
        ids += prompt[256:600]
        pos += list(range(256, 600))
        qsl.append(qsl[-1] + 344)
        wins.append(window_of(prompt, 256))
        ref = kernel_ref(self.c, ids, pos, qsl, wins)
        seg = epa.hash_segment(self.c, prompt[256:600], window_of(prompt, 256), 256, 12, 24)
        self.assertTrue(np.array_equal(seg, ref[12:, :, 12:24]))


class PlanTests(unittest.TestCase):
    def test_plan_continues_a_chunked_prefill(self) -> None:
        plan = epa.plan_ahead(["a"], [3], [8192], [8192], [30000], [True], cap=8192)
        self.assertEqual(plan, [("a", 3, 16384, 8192, 0)])
        plan = epa.plan_ahead(["a"], [3], [16384], [8192], [30000], [True], cap=8192)
        self.assertEqual(plan, [("a", 3, 24576, 5424, 0)])

    def test_plan_skips_finishing_decoding_and_short_continuations(self) -> None:
        self.assertEqual(epa.plan_ahead(["a"], [0], [0], [500], [500], [True], cap=8192), [])
        self.assertEqual(epa.plan_ahead(["a"], [0], [0], [460], [500], [True], cap=8192), [])  # 40 < 64
        self.assertEqual(epa.plan_ahead(["d"], [0], [900], [4], [100], [False], cap=8192), [])
        plan = epa.plan_ahead(["d", "p"], [0, 1], [900, 0], [4, 300], [100, 2000], [False, True], cap=8192)
        self.assertEqual(plan, [("p", 1, 300, 1700, 0)])

    def test_plan_shares_the_cap_in_batch_order(self) -> None:
        plan = epa.plan_ahead(["a", "b", "c"], [0, 1, 2], [0, 0, 0], [100, 100, 100], [5000, 5000, 5000],
                              [True, True, True], cap=1000)
        self.assertEqual(plan, [("a", 0, 100, 1000, 0)])
        plan = epa.plan_ahead(["a", "b"], [0, 1], [0, 0], [100, 100], [500, 5000], [True, True], cap=1000)
        self.assertEqual(plan, [("a", 0, 100, 400, 0), ("b", 1, 100, 600, 400)])
        plan = epa.plan_ahead(["a", "b"], [0, 1], [0, 0], [100, 100], [1000 - 30 + 100, 5000], [True, True], cap=1000)
        self.assertEqual(plan, [("a", 0, 100, 970, 0)])  # b would get 30 < 64

    def test_covered_rows(self) -> None:
        plan = [("p", 1, 1000, 384, 0), ("q", 2, 50, 100, 384)]
        # decode d first, then p's chunk exactly as read ahead
        cov = epa.covered_rows(plan, ["d", "p"], [77, 1000], [False, True], [0, 4, 388])
        self.assertEqual(cov, [(4, 388, 0)])
        # a shorter chunk: its prefix
        self.assertEqual(epa.covered_rows(plan, ["p"], [1000], [True], [0, 200]), [(0, 200, 0)])
        # the chunk starts earlier (recompute) / later than read ahead
        self.assertEqual(epa.covered_rows(plan, ["p"], [900], [True], [0, 300]), [(100, 300, 0)])
        self.assertEqual(epa.covered_rows(plan, ["p"], [1100], [True], [0, 384]), [(0, 284, 100)])
        # no overlap, other requests, decoding rows
        self.assertEqual(epa.covered_rows(plan, ["p"], [0], [True], [0, 384]), [])
        self.assertEqual(epa.covered_rows(plan, ["x"], [1000], [True], [0, 384]), [])
        self.assertEqual(epa.covered_rows(plan, ["p"], [1000], [False], [0, 4]), [])
        self.assertEqual(epa.covered_rows(plan, ["q", "p"], [60, 1000], [True, True], [0, 90, 474]),
                         [(0, 90, 394), (90, 474, 0)])

    def test_segment_runs_partition_every_batch(self) -> None:
        rng = random.Random(9)
        for _ in range(300):
            nreq = rng.randrange(1, 6)
            lens = [rng.randrange(1, 50) for _ in range(nreq)]
            qsl = [0]
            for x in lens:
                qsl.append(qsl[-1] + x)
            pref = [rng.random() < 0.6 for _ in range(nreq)]
            covered = []
            for i in range(nreq):
                if rng.random() < 0.5:
                    a = rng.randrange(qsl[i], qsl[i + 1])
                    b = rng.randrange(a + 1, qsl[i + 1] + 1)
                    covered.append((a, b, rng.randrange(100)))
            runs = epa.segment_runs(qsl, covered, pref)
            mark = np.zeros(qsl[-1], np.int32)
            for a, b, _ in covered:
                mark[a:b] += 1
            for a, b, p in runs:
                self.assertLess(a, b)
                mark[a:b] += 1
                i = max(j for j in range(nreq) if qsl[j] <= a)
                self.assertLessEqual(b, qsl[i + 1])
                self.assertEqual(p, pref[i])
            self.assertTrue(np.all(mark == 1))

    def test_batch_meta_ok(self) -> None:
        self.assertTrue(epa.batch_meta_ok([0, 4, 10], 2, 10))
        self.assertFalse(epa.batch_meta_ok([0, 4, 10], 1, 10))
        self.assertFalse(epa.batch_meta_ok([0, 4, 9], 2, 10))
        self.assertFalse(epa.batch_meta_ok([1, 4, 10], 2, 10))
        self.assertFalse(epa.batch_meta_ok([0, 6, 4, 10], 3, 10))

    def test_token_window(self) -> None:
        row = np.arange(100, 200, dtype=np.int32)
        ids, win = epa.token_window(row, 0, 5, 3)
        self.assertEqual((ids.tolist(), win.tolist()), ([100, 101, 102, 103, 104], [-1, -1, -1]))
        ids, win = epa.token_window(row, 2, 3, 3)
        self.assertEqual((ids.tolist(), win.tolist()), ([102, 103, 104], [101, 100, -1]))
        ids, win = epa.token_window(row, 10, 2, 3)
        self.assertEqual((ids.tolist(), win.tolist()), ([110, 111], [109, 108, 107]))
        row[:] = 0
        self.assertEqual(ids.tolist(), [110, 111])  # a copy: the slot may be rewritten later


class GatherFlagTests(unittest.TestCase):
    NROWS = 6000

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.lib = build_lib(tmp)
        cls.path = tmp / "table.bin"
        cls.w, cls.s, cls.s_off = write_table(cls.path, cls.NROWS, seed=11)
        os.sync()
        cls.fd = os.open(cls.path, os.O_RDONLY)
        os.posix_fadvise(cls.fd, 0, 0, os.POSIX_FADV_RANDOM)  # as DiskEngramTable._open: no readahead
        cls.size = os.fstat(cls.fd).st_size
        cls.libc = ctypes.CDLL(None, use_errno=True)
        cls.libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]

    @classmethod
    def tearDownClass(cls) -> None:
        os.close(cls.fd)
        cls._tmp.cleanup()

    def gather(self, ids: np.ndarray, flags: int):
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        n, hv = ids.shape
        out = np.zeros((n * hv, DIM), np.uint16)
        sw = np.zeros((n * hv, DIM), np.uint8)
        ss = np.zeros((n * hv, SB), np.uint8)
        st = (ctypes.c_int64 * ens.NSTATS)()
        rc = self.lib.eng_gather_bf16(self.fd, W_OFF, self.fd, self.s_off, DIM, SB, 0, self.NROWS,
                                      ids.ctypes.data, hv, n, hv, hv, LUT.ctypes.data, out.ctypes.data,
                                      sw.ctypes.data, ss.ctypes.data, flags, st)
        self.assertEqual(rc, 0)
        return out, list(st)

    def resident(self) -> np.ndarray:
        """Page-cache residency of every 4 KiB page of the table file (mincore)."""
        m = mmap.mmap(self.fd, self.size, access=mmap.ACCESS_COPY)
        try:
            buf = ctypes.c_char.from_buffer(m)
            vec = np.zeros((self.size + 4095) // 4096, np.uint8)
            rc = self.libc.mincore(ctypes.addressof(buf), self.size, vec.ctypes.data)
            del buf
            self.assertEqual(rc, 0, os.strerror(ctypes.get_errno()))
            return vec & 1
        finally:
            m.close()

    def drop_all(self) -> None:
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)
        if self.resident().any():
            self.skipTest("page cache does not drop on this filesystem")

    def pages_of(self, rows) -> tuple[set, set]:
        w = {(W_OFF + r * DIM) // 4096 for r in rows} | {(W_OFF + r * DIM + DIM - 1) // 4096 for r in rows}
        s = {(self.s_off + r * SB) // 4096 for r in rows} | {(self.s_off + r * SB + SB - 1) // 4096 for r in rows}
        return w, s

    def test_dontneed_keeps_the_bytes(self) -> None:
        rng = np.random.default_rng(1)
        ids = rng.integers(0, self.NROWS, (64, 12), dtype=np.int32)
        self.drop_all()
        a, _ = self.gather(ids, 1)
        self.drop_all()
        b, st = self.gather(ids, 3)
        ref = ref_dequant(self.w[ids.reshape(-1)], self.s[ids.reshape(-1)])
        self.assertTrue(np.array_equal(a, ref))
        self.assertTrue(np.array_equal(b, ref))
        self.assertGreater(st[2], 0)  # misses: the file was cold

    def test_dontneed_drops_only_the_pages_the_call_brought_in(self) -> None:
        rng = np.random.default_rng(2)
        ids = np.unique(rng.integers(0, self.NROWS, 400)).astype(np.int32)
        # A row straddling a page boundary (1 in 16 at the real w offset) that misses
        # drops both pages, a cached neighbour included: exact equality needs whole rows.
        ids = ids[(W_OFF + ids.astype(np.int64) * DIM) % 4096 <= 4096 - DIM]
        hot = ids[: len(ids) // 3]
        hot_w, hot_s = self.pages_of(hot.tolist())
        # DONTNEED is best effort: the kernel skips a page that is busy at that moment, which left one page of a
        # missed row cached in 4 of 220 runs. Three tries; a page cached before the call must survive each.
        for _ in range(3):
            self.drop_all()
            for r in hot:  # cached before the call (a decode row's pages)
                os.pread(self.fd, DIM, W_OFF + int(r) * DIM)
                os.pread(self.fd, SB, self.s_off + int(r) * SB)
            before = self.resident()
            self.assertTrue(all(before[p] for p in hot_w | hot_s))
            _, st = self.gather(ids.reshape(-1, 1), 3)
            self.assertGreater(st[2], 0)
            after = self.resident()
            dropped = np.nonzero((before == 1) & (after == 0))[0].tolist()
            self.assertEqual(dropped, [], "flag 2 must keep the pages that were cached before the call")
            diff = np.nonzero(after != before)[0].tolist()
            if not diff:
                break
        self.assertEqual(diff, [], "flag 2 must leave the page cache as it found it")
        self.gather(ids.reshape(-1, 1), 1)  # without the flag the read rows stay cached
        cold_w, cold_s = self.pages_of(ids.tolist())
        after1 = self.resident()
        self.assertTrue(all(after1[p] for p in cold_w | cold_s))


class _Req:
    def __init__(self, rid: str, prompt: list, idx: int, computed: int):
        self.rid, self.idx = rid, idx
        self.tokens = list(prompt)
        self.prefill_len = len(prompt)
        self.computed = computed


class _Sim:
    """A scheduler, the V2 runner's batch fields and all_token_ids, the GPU hash
    (kernel_ref) and the lever's consume / sync gather / read-ahead, step by step."""

    K = 3  # DSpark drafts per verify step

    def __init__(self, t: "EndToEndTests", rank: int, budget: int, seed: int):
        self.t, self.rank, self.budget = t, rank, budget
        self.rng = random.Random(seed)
        self.h0, self.h1 = 12 * rank, 12 * rank + 12
        self.lh = 12
        self.uva = np.zeros((8, 8192), np.int32)
        self.free = list(range(8))[::-1]
        self.running: list[_Req] = []
        cap = budget
        self.pf_hash = np.zeros((cap, 2, 12), np.int32)
        self.pf_rows = [np.zeros((cap, self.lh, DIM), np.uint16) for _ in range(2)]
        self.rows_host = [np.zeros((budget, self.lh, DIM), np.uint16) for _ in range(2)]
        tables = []
        for layer in range(2):
            v0, v1 = int(t.c["offs"][layer][self.h0]), int(t.c["offs"][layer][self.h0] + t.c["primes"][layer][self.h0:self.h1].sum())
            tables.append((t.fds[layer], W_OFF, t.fds[layer], t.s_offs[layer], DIM, SB, v0, v1, layer))
        self.tables = tables
        self.gather = epa.Gather(t.lib, tables, 2, 12, self.lh, DIM, LUT.ctypes.data, nthreads=3)
        self.core = epa.AheadCore(self.gather, t.c, self.h0, self.h1, NGRAM - 1, cap, self.pf_hash,
                                  [p.ctypes.data for p in self.pf_rows], lambda: None,
                                  epa.FLAG_NOWAIT | epa.FLAG_DONTNEED)
        self.steps = []

    def close(self) -> None:
        self.core.close()
        self.gather.close()

    def add(self, rid: str, prompt: list, computed: int = 0) -> None:
        idx = self.free.pop()
        self.uva[idx, : len(prompt)] = prompt
        self.running.append(_Req(rid, prompt, idx, computed))

    def remove(self, rid: str) -> _Req:
        r = next(x for x in self.running if x.rid == rid)
        self.running.remove(r)
        self.free.append(r.idx)
        return r

    def step(self, budget: int | None = None) -> dict:
        budget = self.budget if budget is None else budget
        dec, pre, left = [], [], budget
        for r in self.running:
            if r.computed >= r.prefill_len and left >= 1 + self.K:
                dec.append((r, 1 + self.K))
                left -= 1 + self.K
        for r in self.running:
            if r.computed < r.prefill_len and left > 0:
                n = min(r.prefill_len - r.computed, left)
                pre.append((r, n))
                left -= n
        segs = dec + sorted(pre, key=lambda x: x[1])  # sort_batch_req_ids: verify first, then by size
        ids, pos, qsl, wins, drafts = [], [], [0], [], {}
        for r, n in segs:
            p0 = r.computed
            if r.computed < r.prefill_len:
                ids += r.tokens[p0 : p0 + n]
            else:
                drafts[r.rid] = [self.rng.randrange(VOCAB) for _ in range(self.K)]
                ids += [r.tokens[-1]] + drafts[r.rid]
            pos += list(range(p0, p0 + n))
            qsl.append(qsl[-1] + n)
            wins.append(window_of(r.tokens, p0))
        n_tok = qsl[-1]
        batch = SimpleNamespace(
            req_ids=[r.rid for r, _ in segs], num_reqs=len(segs),
            idx_mapping_np=np.array([r.idx for r, _ in segs], np.intp),
            num_computed_tokens_np=np.array([r.computed for r, _ in segs], np.int32),
            num_scheduled_tokens=np.array([n for _, n in segs], np.int32),
            prefill_len_np=np.array([r.prefill_len for r, _ in segs], np.int32),
            is_prefilling_np=np.array([r.computed < r.prefill_len for r, _ in segs]),
            num_tokens=n_tok,
        )
        gpu = np.ascontiguousarray(kernel_ref(self.t.c, ids, pos, qsl, wins)[:, :, self.h0:self.h1].astype(np.int32))
        covered, runs, _, _ = self.core.consume(batch, gpu, qsl, n_tok)
        if runs:
            self.gather.run(gpu.ctypes.data, runs, [b.ctypes.data for b in self.rows_host])
        for layer in range(2):
            staged = np.full((n_tok, self.lh, DIM), 0xDEAD, np.uint16)
            for a, b, s in covered:
                staged[a:b] = self.pf_rows[layer][s : s + b - a]
            for a, b, _ in runs:
                staged[a:b] = self.rows_host[layer][a:b]
            w, s_ = self.t.tables[layer]
            ref = stock_rows(w, s_, gpu[:, layer, :], self.lh, self.tables[layer][6], self.tables[layer][7])
            self.t.assertTrue(np.array_equal(staged, ref), f"step {len(self.steps)} layer {layer}")
        if epa._STATE["armed"]:
            plan = self.core.plan(batch)
            if plan:
                self.core.start(plan, self.uva)
            else:
                self.core.drop()
        pre_rows = sum(n for r, n in segs if r.computed < r.prefill_len)
        info = {"n": n_tok, "ahead": sum(b - a for a, b, _ in covered), "prefill": pre_rows,
                "segs": [(r.rid, n) for r, n in segs]}
        for r, n in segs:  # the step's outputs
            if r.computed < r.prefill_len:
                r.computed += n
                if r.computed == r.prefill_len:
                    r.tokens.append(self.rng.randrange(VOCAB))
            else:
                a = self.rng.randrange(self.K + 1)
                r.tokens += drafts[r.rid][:a] + [self.rng.randrange(VOCAB)]
                r.computed += 1 + a
            self.uva[r.idx, : len(r.tokens)] = r.tokens
        self.steps.append(info)
        return info


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.lib = build_lib(tmp)
        cls.c = make_consts(500, seed=21)  # small primes: ~12k rows per table file
        cls.tables, cls.fds, cls.s_offs = [], [], []
        for layer in range(2):
            rows = int(cls.c["primes"][layer].sum())
            w, s, s_off = write_table(tmp / f"table{layer}.bin", rows, seed=31 + layer)
            cls.tables.append((w, s))
            cls.fds.append(os.open(tmp / f"table{layer}.bin", os.O_RDONLY))
            cls.s_offs.append(s_off)

    @classmethod
    def tearDownClass(cls) -> None:
        for fd in cls.fds:
            os.close(fd)
        cls._tmp.cleanup()

    def setUp(self) -> None:
        self.saved = dict(epa._STATE)
        epa._STATE.update(armed=True, verify_left=4, engaged=False)
        self.prng = random.Random(77)

    def tearDown(self) -> None:
        epa._STATE.clear()
        epa._STATE.update(self.saved)

    def prompt(self, n: int) -> list:
        toks = [self.prng.randrange(VOCAB) for _ in range(n)]
        for _ in range(max(1, n // 300)):
            a = self.prng.randrange(n)
            toks[a : a + 5] = [DEAD[0]] * len(toks[a : a + 5])
        return toks

    def sim(self, rank: int = 0, budget: int = 384, seed: int = 0) -> _Sim:
        s = _Sim(self, rank, budget, seed)
        self.addCleanup(s.close)
        return s

    def test_chunked_prefill_reads_every_chunk_but_the_first_ahead(self) -> None:
        for rank in (0, 1):
            s = self.sim(rank=rank)
            s.add("a", self.prompt(1950))  # 5 x 384 + 30: the 30-token tail is below MIN_AHEAD
            while any(r.computed < r.prefill_len for r in s.running):
                s.step()
            ahead = [x["ahead"] for x in s.steps]
            self.assertEqual(ahead, [0, 384, 384, 384, 384, 0], rank)

    def test_mixed_prefill_and_decode_at_four_sequences(self) -> None:
        s = self.sim()
        for rid in ("d1", "d2", "d3"):
            s.add(rid, self.prompt(50))
        s.step()  # three short prefills
        s.add("p", self.prompt(1600))  # 4 x 372 + 112
        while any(r.computed < r.prefill_len for r in s.running):
            info = s.step()
            self.assertEqual(sum(n for rid, n in info["segs"] if rid.startswith("d")), 12)
        ahead = [x["ahead"] for x in s.steps[1:]]
        # p gets 372-token chunks (384 - 12 decode rows), each the prefix of a 384-token
        # read-ahead; every chunk after its first is read ahead, decode rows never
        self.assertEqual(ahead, [0, 372, 372, 372, 112])
        self.assertTrue(all(a == x["prefill"] for a, x in zip(ahead[1:], s.steps[2:])), ahead)
        for _ in range(3):
            self.assertEqual(s.step()["ahead"], 0)  # decode only: nothing to read ahead

    def test_prefix_cache_start(self) -> None:
        s = self.sim()
        s.add("a", self.prompt(1500), computed=640)  # 640 tokens hit the prefix cache
        while any(r.computed < r.prefill_len for r in s.running):
            s.step()
        self.assertEqual([x["ahead"] for x in s.steps], [0, 384, 92])

    def test_preemption_and_readmission(self) -> None:
        s = self.sim()
        prompt = self.prompt(1600)
        s.add("a", prompt)
        s.step()
        s.step()  # [384, 768) read ahead; [768, 1152) is being read ahead now
        s.remove("a")  # preempted: its blocks are freed, num_computed back to 0
        s.add("b", self.prompt(200))
        info = s.step()  # the read-ahead of a matches nothing in this batch: dropped
        self.assertEqual(info["ahead"], 0)
        s.add("a", prompt)  # re-admitted from position 0 (prefill = prompt, nothing generated yet)
        while any(r.computed < r.prefill_len for r in s.running):
            s.step()
        ahead = [x["ahead"] for x in s.steps]
        self.assertEqual(ahead[:3], [0, 384, 0])
        self.assertEqual(ahead[3], 0)  # a's first chunk again
        self.assertGreater(sum(ahead[4:]), 0)

    def test_abort_with_a_job_in_flight(self) -> None:
        s = self.sim()
        s.add("a", self.prompt(2000))
        s.step()
        s.remove("a")  # aborted
        info = s.step() if s.running else None
        self.assertIsNone(info)
        s.add("b", self.prompt(900))
        self.assertEqual(s.step()["ahead"], 0)
        self.assertEqual(s.step()["ahead"], 384)

    def test_two_prefills_in_one_batch(self) -> None:
        s = self.sim()
        s.add("a", self.prompt(500))
        s.add("b", self.prompt(1500))
        while any(r.computed < r.prefill_len for r in s.running):
            s.step()
        self.assertEqual(s.steps[1]["segs"], [("a", 116), ("b", 268)])
        self.assertEqual(s.steps[1]["ahead"], 116)  # a's tail was read ahead, b's head was not
        self.assertEqual(s.steps[2]["segs"], [("a", 4), ("b", 380)])  # a now decodes
        self.assertEqual(s.steps[2]["ahead"], 380)

    def test_a_shorter_chunk_than_read_ahead(self) -> None:
        s = self.sim()
        s.add("a", self.prompt(1500))
        s.step()
        self.assertEqual(s.step(budget=100)["ahead"], 100)  # the budget shrank: its prefix
        self.assertEqual(s.step()["ahead"], 384)  # the next read-ahead starts where it stopped

    def test_a_wrong_read_ahead_disarms_and_the_rows_stay_exact(self) -> None:
        s = self.sim()
        s.add("a", self.prompt(1500))
        s.step()
        real = epa.hash_segment

        def wrong(*a, **k):
            out = real(*a, **k)
            out[7, 1, 3] ^= 1
            return out

        with mock.patch.object(epa, "hash_segment", wrong), redirect_stdout(StringIO()) as log:
            s.step()  # launches the corrupted read-ahead
            info = s.step()  # consumes it: mismatch -> sync rows, lever off
        self.assertIn(epa.LOG_DISARMED, log.getvalue())
        self.assertFalse(epa._STATE["armed"])
        self.assertEqual(info["ahead"], 0)

    def test_a_cancelled_job_is_dropped(self) -> None:
        s = self.sim()
        s.add("a", self.prompt(1500))
        s.step()
        job = s.core.job
        self.assertIsNotNone(job)
        s.core.drop()
        self.assertTrue(job.cancel.is_set())
        self.assertIsNone(s.core.job)
        self.assertEqual(s.step()["ahead"], 0)
        self.assertEqual(s.step()["ahead"], 384)


if __name__ == "__main__":
    unittest.main()
