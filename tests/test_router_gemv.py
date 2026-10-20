"""router_gemv (DSV41_ROUTER_GEMV): split-K replication logic, serving rules and wiring, host only.

The GPU proof (bitwise vs cuBLAS on the pack's router / draft / compressor weights at M 1..24, eager
and inside CUDA graphs, timing, sabotage) is kernel_study/router_gemv/bench.py. The kernel's loop, mask
and slice-order logic is checked on the CPU by kernel_study/router_gemv/interp_check.py (Triton
interpreter, fp32 operands, == a numpy mirror of router_gemv.schedule bit for bit: 78/78 cases).
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import tempfile
import textwrap
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker" / "patch"
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(PATCH))
import decode_levers as dl  # noqa: E402
import router_gemv as rg  # noqa: E402

TORCH = types.SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32")


class T:
    def __init__(self, shape, dtype="bf16", cuda=True, contiguous=True, ptr=1 << 20):
        self.shape = shape
        self.dtype = dtype
        self.is_cuda = cuda
        self._c = contiguous
        self._p = ptr

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self._c

    def data_ptr(self):
        return self._p


class SplitCandidateTests(unittest.TestCase):
    def test_trace_configs_are_candidates(self):
        c = rg.split_candidates(5120)
        for ks in (5120, 320, 576):  # S = 1, 16, 9 seen in the B0 traces
            self.assertIn(ks, c)

    def test_first_wins_and_no_duplicates(self):
        c = rg.split_candidates(5120, (576, 320, 576))
        self.assertEqual(c[:2], [576, 320])
        self.assertEqual(len(c), len(set(c)))
        self.assertTrue(all(ks % 8 == 0 and 8 <= ks <= 5120 for ks in c))

    def test_rounding_matches_cutlass_split_rule(self):
        # CUTLASS GemmUniversal: gemm_k_size = round_up(ceil(K / S), kAlignK), kAlignK 8 for bf16
        for s in rg.SPLIT_SEARCH:
            self.assertIn(rg.cdiv(rg.cdiv(5120, s), 8) * 8, rg.split_candidates(5120))


class ScheduleTests(unittest.TestCase):
    def check(self, k, ks, bk, order):
        sched = rg.schedule(k, ks, bk, order)
        slices = [s for s, _, _, first in sched if first]
        n_slices = rg.cdiv(k, ks)
        expect = list(range(n_slices))
        self.assertEqual(slices, expect[::-1] if order == rg.REV else expect)
        covered = {}
        for s, k0, kend, _ in sched:
            self.assertEqual(kend, min(s * ks + ks, k))
            for kk in range(k0, min(k0 + bk, kend)):
                covered.setdefault(s, []).append(kk)
        for s in range(n_slices):
            # each slice's elements exactly once, ascending (one MMA chain per slice)
            self.assertEqual(covered[s], list(range(s * ks, min(s * ks + ks, k))), (k, ks, bk, s))

    def test_every_slice_once_in_order(self):
        for ks in (5120, 2560, 1712, 856, 576, 320, 64):
            for bk in (64, 128, 256):
                for order in (rg.SEQ, rg.REV):
                    self.check(5120, ks, bk, order)

    def test_serve_configs(self):
        self.assertEqual(rg.serve_config(5120, 5120), rg.SERVE_UNSPLIT)  # one slice
        self.assertEqual(rg.serve_config(5120, 320), rg.SERVE_SPLIT)  # 3 blocks, last one half masked
        self.assertEqual(rg.serve_config(5120, 120)[0], 64)
        for k, ks in ((5120, 5120), (5120, 320), (5120, 576), (5120, 856), (5120, 120)):
            for m in (1, 4, 16, 17, 24):
                bk, warps, stages = rg.serve_config(k, ks, m)
                self.assertTrue(bk >= 16 and bk & (bk - 1) == 0 and warps in (1, 2, 4, 8) and stages >= 1)
                # the bf16 operand ring fits 99 KiB of shared memory
                self.assertLessEqual(stages * (rg.bm_for(m) * bk + bk * rg.BN) * 2, 99 * 1024, (ks, m))
                self.check(k, ks, bk, rg.SEQ)
        self.assertEqual((rg.bm_for(16), rg.bm_for(17), rg.bm_for(24)), (16, 32, 32))


def chain_model(x, w, ks, order=rg.SEQ):
    """fp32 model of cuBLAS split-K: per slice a k-ascending chain of 16-wide fp32 dot steps from 0,
    then the slice partials summed in order. Not the tensor core's rounding, only its structure."""
    m, k = x.shape
    out = np.zeros((m, w.shape[0]), np.float32)
    slices = range(rg.cdiv(k, ks))
    for s in (reversed(slices) if order == rg.REV else slices):
        acc = np.zeros_like(out)
        for c in range(s * ks, min(s * ks + ks, k), 16):
            acc = (acc + (x[:, c:c + 16].astype(np.float64) @ w[:, c:c + 16].T.astype(np.float64))).astype(np.float32)
        out = (out + acc).astype(np.float32)
    return out


class DiscoverTests(unittest.TestCase):
    def test_split_configs_give_different_bits(self):
        # the premise of the search: another slicing or order changes fp32 bits on random rows
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 5120)).astype(np.float32)
        w = rng.standard_normal((32, 5120)).astype(np.float32)
        ref = chain_model(x, w, 320)
        for ks, order in ((576, rg.SEQ), (5120, rg.SEQ), (320, rg.REV), (640, rg.SEQ)):
            self.assertFalse(np.array_equal(ref.view(np.int32), chain_model(x, w, ks, order).view(np.int32)), ks)
        self.assertTrue(np.array_equal(ref.view(np.int32), chain_model(x, w, 320).view(np.int32)))

    def test_table_and_stock_rows(self):
        # stand-in cuBLAS: m=1 a GEMV (matches nothing), m 2..8 unsplit, m 9..12 nine REV slices,
        # m 13..24 sixteen slices
        def truth(m):
            if m == 1:
                return ("gemv",)
            if m <= 8:
                return (5120, rg.SEQ)
            if m <= 12:
                return (576, rg.REV)
            return (320, rg.SEQ)

        calls = []

        def run_mine(m, ks, order):
            calls.append((m, ks, order))
            return [(ks, order), (ks, order)]

        table, stock = rg.discover(range(1, 25), lambda m: [truth(m), truth(m)], run_mine,
                                   lambda a, b: a == b, 5120, rg.LIKELY_KS)
        self.assertEqual(stock, [1])
        for m in range(2, 25):
            self.assertEqual(table[m], truth(m), m)
        # the previous M's config is tried first: m 3..8 take one call each
        self.assertEqual([c for c in calls if c[0] == 5], [(5, 5120, rg.SEQ)])
        # one slice is never re-tried in the reverse order
        self.assertNotIn((1, 5120, rg.REV), calls)

    def test_all_probes_must_match(self):
        table, stock = rg.discover([4], lambda m: ["a", "b"], lambda m, ks, o: ["a", "x"],
                                   lambda a, b: a == b, 5120)
        self.assertEqual((table, stock), ({}, [4]))


class ServingRuleTests(unittest.TestCase):
    def test_usable(self):
        w = T((384, 5120))
        for m in (1, 4, 16, 24):
            self.assertTrue(rg.usable(TORCH, T((m, 5120)), w), m)
        cases = {
            "m=25": (T((25, 5120)), w),
            "m=0": (T((0, 5120)), w),
            "fp16 x": (T((4, 5120), "fp16"), w),
            "fp32 w": (T((4, 5120)), T((384, 5120), "fp32")),
            "cpu": (T((4, 5120), cuda=False), w),
            "strided x": (T((4, 5120), contiguous=False), w),
            "misaligned x": (T((4, 5120), ptr=(1 << 20) + 8), w),
            "k mismatch": (T((4, 4096)), w),
            "3-d x": (T((1, 4, 5120)), w),
        }
        for label, (x, ww) in cases.items():
            self.assertFalse(rg.usable(TORCH, x, ww), label)

    def test_gate_tiers(self):
        x = T((4, 5120))
        base = dict(allow_ll_bf16_gemm=False, allow_fp32_router_gemm=False,
                    allow_bf16x3_router_gemm=False, allow_cublas_router_gemm=True)
        self.assertTrue(rg.gate_takes_cublas(types.SimpleNamespace(**base), x, TORCH))
        for flag in ("allow_ll_bf16_gemm", "allow_fp32_router_gemm", "allow_bf16x3_router_gemm"):
            self.assertFalse(rg.gate_takes_cublas(types.SimpleNamespace(**{**base, flag: True}), x, TORCH), flag)
        self.assertFalse(rg.gate_takes_cublas(types.SimpleNamespace(**{**base, "allow_cublas_router_gemm": False}),
                                              x, TORCH))
        self.assertFalse(rg.gate_takes_cublas(types.SimpleNamespace(**base), T((4, 5120), "fp32"), TORCH))
        # the cuteDSL tier only takes M <= 16: above it the cuBLAS tier runs
        self.assertTrue(rg.gate_takes_cublas(types.SimpleNamespace(**{**base, "allow_ll_bf16_gemm": True}),
                                             T((17, 5120)), TORCH))

    def test_gate_forward_routes(self):
        stock = mock.Mock(return_value=("stock", None))
        fwd = rg.make_gate_forward(TORCH, stock)
        gate = types.SimpleNamespace(weight="W", allow_cublas_router_gemm=True)
        x = T((4, 5120))
        rg._STATE["armed"] = True
        with mock.patch.object(rg, "mm_f32", return_value="mine") as mm:
            self.assertEqual(fwd(gate, x), ("mine", None))
            self.assertEqual(mm.call_args[0][:2], (x, "W"))
            mm.call_args[0][2](x)  # its stock fallback is the original forward's output
            stock.assert_called_with(gate, x)
            stock.reset_mock()
            rg._STATE["armed"] = False
            self.assertEqual(fwd(gate, x), ("stock", None))
            rg._STATE["armed"] = True
            gate.allow_cublas_router_gemm = False
            self.assertEqual(fwd(gate, x), ("stock", None))
        self.assertTrue(fwd._dsv41_router_gemv)

    def test_env(self):
        self.assertFalse(rg.enabled({}))
        self.assertTrue(rg.enabled({"DSV41_ROUTER_GEMV": "1"}))
        self.assertEqual(rg.verify_calls({}), 8)
        self.assertEqual(rg.verify_calls({"DSV41_ROUTER_GEMV_VERIFY": "0"}), 1)
        self.assertIsNone(rg.served_n({}))
        self.assertIsNone(rg.served_n({"DSV41_ROUTER_GEMV_N": ""}))
        self.assertEqual(rg.served_n({"DSV41_ROUTER_GEMV_N": "128, 1024"}), frozenset({128, 1024}))

    def test_shape_filter_leaves_cublas(self):
        fake_torch = types.SimpleNamespace(
            bfloat16="bf16", cuda=types.SimpleNamespace(is_current_stream_capturing=lambda: False))
        w = T((384, 5120))
        saved = dict(rg._STATE)
        try:
            rg._STATE.update(tables={}, shapes=frozenset({128}))
            with mock.patch.object(rg, "selftest") as st, redirect_stdout(StringIO()) as out:
                self.assertEqual(rg._table(fake_torch, None, None, w, None), {})
            st.assert_not_called()
            self.assertIn("left to cuBLAS (DSV41_ROUTER_GEMV_N)", out.getvalue())
            self.assertNotIn(rg.LOG_DISARMED, out.getvalue())
        finally:
            rg._STATE.clear()
            rg._STATE.update(saved)


class KernelContractTests(unittest.TestCase):
    def body(self):
        src = (PATCH / "router_gemv.py").read_text()
        return src[src.index("def _mm_f32_split_kernel(") : src.index("return _mm_f32_split_kernel")]

    def test_one_chain_per_slice_summed_in_order(self):
        b = self.body()
        self.assertIn("for j in range(0, S * NB):", b)
        self.assertIn("total = total + tl.where(first, acc, 0.0)", b)
        self.assertIn("acc = tl.where(first, 0.0, acc)", b)
        self.assertIn("acc = tl.dot(a, w, acc)", b)
        self.assertIn("total = total + acc", b)
        self.assertNotIn("atomic", b)  # no cross-CTA reduction: the order is fixed in-register
        self.assertNotIn("bfloat16", b)  # fp32 out, no rounding step
        self.assertLess(b.index("acc = tl.where(first, 0.0, acc)"), b.index("acc = tl.dot(a, w, acc)"))
        self.assertEqual(rg.BN, 16)

    def test_host_schedule_mirrors_kernel_indexing(self):
        b = self.body()
        for line in ("q = j // NB", "b = j - q * NB", "s = S - 1 - q", "s = q",
                     "k0 = s * KS + b * BK", "kend = tl.minimum(s * KS + KS, K)", "first = b == 0"):
            self.assertIn(line, b)
        src = (PATCH / "router_gemv.py").read_text()
        sched = src[src.index("def schedule("): src.index("def discover(")]
        for line in ("q, b = divmod(j, nb)", "s = s_count - 1 - q if order == REV else q",
                     "k0 = s * ks + b * bk", "min(s * ks + ks, k), b == 0"):
            self.assertIn(line, sched)


def _method_source(path: Path, cls: str, name: str) -> str:
    src = path.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    lines = src.splitlines(keepends=True)
                    return "".join(lines[f.lineno - 1 : f.end_lineno])
    raise KeyError(name)


class KvRebuildTests(unittest.TestCase):
    """The compressor kv_score swap on the pinned image text (fixture == e15 attention.py)."""

    def module(self, method_src: str):
        stub = (
            "class _Envs:\n    VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD = 1024\n"
            "envs = _Envs()\n"
            "class _Torch:\n    float32 = 'fp32'\n    class Tensor: pass\n"
            "    @staticmethod\n    def mm(*a, **k):\n        raise AssertionError('stock torch.mm called')\n"
            "torch = _Torch()\n"
            "def execute_in_parallel(main, aux, start, done, streams, enable=True):\n"
            "    return main(), [f() if f else None for f in aux]\n"
            "class DeepseekV4Attention:\n"
        )
        d = tempfile.mkdtemp()
        path = Path(d) / "attn_stub.py"
        path.write_text(stub + method_src)
        spec = importlib.util.spec_from_file_location("attn_stub", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["attn_stub"] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_anchor_once_in_image_text(self):
        src = _method_source(FIX / "dsv41_attention_e13.pin.py", "DeepseekV4Attention",
                             "_run_parallel_input_projections")
        self.assertEqual(src.count(rg.KV_ANCHOR), 1)
        self.assertEqual(rg.KV_ANCHOR.count("\n"), rg.KV_REPLACEMENT.count("\n"))

    def test_rebuilt_method_calls_the_helper(self):
        src = _method_source(FIX / "dsv41_attention_e13.pin.py", "DeepseekV4Attention",
                             "_run_parallel_input_projections")
        mod = self.module(src)
        seen = []

        def helper(x, w):
            seen.append((x, w))
            return "kv"

        self.assertEqual(rg.rebuild_kv_projection(mod.DeepseekV4Attention, helper), "compressor kv_score rebuilt")
        self.assertEqual(rg.rebuild_kv_projection(mod.DeepseekV4Attention, helper), "already rebuilt")
        weights_proj = mock.Mock(return_value=("iw", None))
        self_ = types.SimpleNamespace(
            aux_stream_list=None, ln_events=[None, None, None],
            compressor=types.SimpleNamespace(fused_wkv_wgate=types.SimpleNamespace(weight="Wkv")),
            indexer=types.SimpleNamespace(weights_proj=weights_proj),
            _fused_wqa_wkv_gemm=lambda h: "qr_kv",
        )
        hidden = types.SimpleNamespace(shape=(4, 5120))
        out = mod.DeepseekV4Attention._run_parallel_input_projections(self_, hidden)
        self.assertEqual(out, ("qr_kv", "kv", "iw"))
        self.assertEqual(seen, [(hidden, "Wkv")])
        fn = mod.DeepseekV4Attention._run_parallel_input_projections
        self.assertEqual(fn.__code__.co_filename, str(Path(sys.modules["attn_stub"].__file__)))

    def test_refuses_without_anchor(self):
        src = _method_source(FIX / "dsv41_attention_e13.pin.py", "DeepseekV4Attention",
                             "_run_parallel_input_projections").replace("out_dtype=torch.float32", "out_dtype=None")
        mod = self.module(src)
        with self.assertRaises(RuntimeError):
            rg.rebuild_kv_projection(mod.DeepseekV4Attention, lambda x, w: None)


class WiringTests(unittest.TestCase):
    def test_decode_levers_step(self):
        self.assertIn('("router-gemv", _install_router_gemv)', (PATCH / "decode_levers.py").read_text())
        calls = []
        fake = types.ModuleType("router_gemv")
        fake.install = lambda: calls.append(1) or "armed"
        with mock.patch.dict(sys.modules, {"router_gemv": fake}), redirect_stdout(StringIO()) as out:
            dl._install_router_gemv({})
            dl._install_router_gemv({"DSV41_ROUTER_GEMV": "0"})
            self.assertEqual(calls, [])
            dl._install_router_gemv({"DSV41_ROUTER_GEMV": "1"})
        self.assertEqual(calls, [1])
        self.assertIn("router gemv: armed", out.getvalue())

    def test_forwarded_default_off(self):
        run = (ROOT / "run.sh").read_text()
        self.assertIn("  DSV41_ROUTER_GEMV=0\n", run)
        self.assertIn("  DSV41_ROUTER_GEMV_VERIFY=\n", run)
        self.assertIn("  DSV41_ROUTER_GEMV_N=\n", run)

    def test_top_level_imports_are_stdlib_only(self):
        tree = ast.parse((PATCH / "router_gemv.py").read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {a.name for n in top if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in top if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"__future__", "os"})


if __name__ == "__main__":
    unittest.main()
