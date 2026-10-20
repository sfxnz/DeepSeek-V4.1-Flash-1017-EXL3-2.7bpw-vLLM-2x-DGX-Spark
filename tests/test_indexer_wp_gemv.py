"""indexer_wp_gemv (DSV41_INDEXER_WP_GEMV): serving rules and wiring, host only.

The GPU proof (bit-identical to cuBLAS on all 8 layers' real weights at M 2..8,
the wrapper's self-test / verify / fallback, 20 graph replays; graph replay m=4
39.0 -> 12.3 us cold) is kernel_study/fusion_host/wp_gemv_bench.py.
"""

from __future__ import annotations

import ast
import sys
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker" / "patch"
sys.path.insert(0, str(PATCH))
import decode_levers as dl  # noqa: E402
import indexer_wp_gemv as wpg  # noqa: E402

TORCH = types.SimpleNamespace(bfloat16="bf16", float16="fp16")


class T:
    def __init__(self, shape, dtype="bf16", cuda=True, contiguous=True, stride1=1):
        self.shape = shape
        self.dtype = dtype
        self.is_cuda = cuda
        self._c = contiguous
        self._s1 = stride1

    def dim(self):
        return len(self.shape)

    def stride(self, i):
        return self._s1 if i == 1 else self.shape[1]

    def is_contiguous(self):
        return self._c


class UsableTests(unittest.TestCase):
    def setUp(self):
        wpg._STATE.update(armed=True, verify_left=0, engaged=False, wide=False)
        self.w = T((32, 5120))

    def test_decode_rows_are_served(self):
        for m in range(wpg.MIN_M, wpg.MAX_M + 1):
            self.assertTrue(wpg.usable(TORCH, T((m, 5120)), self.w, None), m)

    def test_everything_else_is_stock(self):
        cases = {
            "m=1 (cuBLAS takes another kernel)": (T((1, 5120)), self.w, None),
            "m=9": (T((9, 5120)), self.w, None),
            "bias": (T((4, 5120)), self.w, object()),
            "fp16 x": (T((4, 5120), "fp16"), self.w, None),
            "fp16 w": (T((4, 5120)), T((32, 5120), "fp16"), None),
            "cpu": (T((4, 5120), cuda=False), self.w, None),
            "strided x": (T((4, 5120), stride1=2), self.w, None),
            "non-contiguous w": (T((4, 5120)), T((32, 5120), contiguous=False), None),
            "k mismatch": (T((4, 4096)), self.w, None),
            "k not a block multiple": (T((4, 5000)), T((32, 5000)), None),
            "3-d x": (T((1, 4, 5120)), self.w, None),
        }
        for label, (x, w, b) in cases.items():
            self.assertFalse(wpg.usable(TORCH, x, w, b), label)
        wpg._STATE["armed"] = False
        self.assertFalse(wpg.usable(TORCH, T((4, 5120)), self.w, None))

    def test_wide_rows_only_after_their_self_test(self):
        for m in range(wpg.MAX_M + 1, wpg.WIDE_MAX_M + 1):
            self.assertFalse(wpg.usable(TORCH, T((m, 5120)), self.w, None), m)
        wpg._STATE["wide"] = True
        for m in range(wpg.MIN_M, wpg.WIDE_MAX_M + 1):
            self.assertTrue(wpg.usable(TORCH, T((m, 5120)), self.w, None), m)
        for m in (1, wpg.WIDE_MAX_M + 1):
            self.assertFalse(wpg.usable(TORCH, T((m, 5120)), self.w, None), m)
        # the self-test of a range asks for that range explicitly
        wpg._STATE["wide"] = False
        self.assertTrue(wpg.usable(TORCH, T((12, 5120)), self.w, None, max_m=wpg.WIDE_MAX_M))
        self.assertEqual(wpg.WIDE_MAX_M, 12)


class LaunchTests(unittest.TestCase):
    class Rows:
        """A 2-D row slice stand-in: rows [lo, hi) of a [m, k] tensor."""

        def __init__(self, m, k, lo=0):
            self.m, self.k, self.lo = m, k, lo
            self.shape = (m, k)

        def __getitem__(self, sl):
            lo, hi = sl.start, min(sl.stop, self.m)
            return LaunchTests.Rows(hi - lo, self.k, self.lo + lo)

        def stride(self, i):
            return self.k if i == 0 else 1

    def test_one_launch_per_16_row_tile(self):
        calls = []

        class Kernel:
            def __getitem__(self, grid):
                def run(x, w, o, m, n, k, *strides, **kw):
                    calls.append((grid, x.lo, o.lo, m, n, k, kw["BM"], kw["BK"]))

                return run

        triton = types.SimpleNamespace(cdiv=lambda a, b: -(-a // b))
        w = types.SimpleNamespace(shape=(32, 5120), stride=lambda i: 5120)
        for m, want in ((2, [(0, 2)]), (8, [(0, 8)]), (16, [(0, 16)]), (17, [(0, 16), (16, 1)]), (24, [(0, 16), (16, 8)])):
            calls.clear()
            wpg.launch(Kernel(), triton, self.Rows(m, 5120), w, self.Rows(m, 32))
            self.assertEqual([(lo, mm) for _, lo, olo, mm, *_ in calls], want, m)
            for grid, lo, olo, mm, n, k, bm, bk in calls:
                self.assertEqual((grid, olo, n, k, bm, bk), ((2,), lo, 32, 5120, 16, wpg.BK))


class ContractTests(unittest.TestCase):
    def test_env(self):
        self.assertFalse(wpg.enabled({}))
        self.assertTrue(wpg.enabled({"DSV41_INDEXER_WP_GEMV": "1"}))
        self.assertEqual(wpg.verify_calls({}), 8)
        self.assertEqual(wpg.verify_calls({"DSV41_INDEXER_WP_VERIFY": "0"}), 1)

    def test_kernel_keeps_one_k_ordered_mma_chain(self):
        src = (PATCH / "indexer_wp_gemv.py").read_text()
        body = src[src.index("def _wp_gemv_kernel(") : src.index("return _wp_gemv_kernel")]
        self.assertIn("for k0 in range(0, K, BK):", body)
        self.assertIn("acc = tl.dot(a, b, acc)", body)
        self.assertIn("acc.to(tl.bfloat16)", body)
        self.assertNotIn("atomic", body)  # no split-K: the order would change
        self.assertEqual((wpg.BM, wpg.BN), (16, 16))  # cuBLAS's 16x16 tiles
        self.assertEqual(5120 % wpg.BK, 0)

    def test_decode_levers_step(self):
        self.assertIn('("indexer-wp-gemv", _install_indexer_wp_gemv)', (PATCH / "decode_levers.py").read_text())
        calls = []
        fake = types.ModuleType("indexer_wp_gemv")
        fake.install = lambda: calls.append(1) or "armed"
        with mock.patch.dict(sys.modules, {"indexer_wp_gemv": fake}), redirect_stdout(StringIO()) as out:
            dl._install_indexer_wp_gemv({})
            dl._install_indexer_wp_gemv({"DSV41_INDEXER_WP_GEMV": "0"})
            self.assertEqual(calls, [])
            dl._install_indexer_wp_gemv({"DSV41_INDEXER_WP_GEMV": "1"})
        self.assertEqual(calls, [1])
        self.assertIn("indexer weights_proj: armed", out.getvalue())

    def test_top_level_imports_are_stdlib_only(self):
        tree = ast.parse((PATCH / "indexer_wp_gemv.py").read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {a.name for n in top if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in top if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"__future__", "os", "types"})


if __name__ == "__main__":
    unittest.main()
