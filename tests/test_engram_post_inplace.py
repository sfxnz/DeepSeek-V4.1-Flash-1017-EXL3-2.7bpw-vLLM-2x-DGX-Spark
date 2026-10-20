"""docker/patch/engram_post_inplace.py on the pinned e15 Engram.forward (CPU only: numpy, no torch).

The stock path is the fixture's own Engram.forward (tests/fixtures/engram_e12.pin.py, the e15 text) in a
temporary module. Tensors are numpy arrays with torch's stride() / flatten(start_dim); the post-wkv
kernel is a numpy transcription of _fused_engram_post_wkv_kernel (per program (t, h): load the row,
reduce, store the row), and checks the strides and grid it is launched with. Integer-valued data keep
the stand-in wkv GEMM exact for any row split, unless a test makes it depend on M.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker/patch"
PIN = ROOT / "tests/fixtures/engram_e12.pin.py"
sys.path.insert(0, str(PATCH))

import attn_out_defer  # noqa: E402
import engram_post_inplace as epi  # noqa: E402

HC, DIM, HEADS, EDIM = 4, 8, 3, 2  # hc_mult, dim, embedding heads x dim per head
HEAD = '''from __future__ import annotations

torch = None
triton = None
_fused_engram_post_wkv_kernel = None


def get_tensor_model_parallel_world_size():
    return 1


def get_tensor_model_parallel_rank():
    return 0


class Engram:
'''


class T(np.ndarray):
    """numpy array with torch's stride(dim) (elements) and flatten(start_dim)."""

    def stride(self, dim):
        return self.strides[dim] // self.itemsize

    def flatten(self, start_dim=0):
        s = start_dim % self.ndim
        return self.reshape(self.shape[:s] + (-1,))


def t(a) -> T:
    return np.ascontiguousarray(a, dtype=np.float32).view(T)


class FakeTorch:
    float32 = np.dtype(np.float32)
    bfloat16 = float16 = np.dtype(np.float16)
    int32, int16 = np.dtype(np.int32), np.dtype(np.int16)

    def __init__(self, capturing=False):
        self.cuda = types.SimpleNamespace(is_current_stream_capturing=lambda: capturing)

    @staticmethod
    def empty_like(x):
        return np.full_like(x, np.nan)  # unwritten values show up

    @staticmethod
    def equal(a, b):
        return bool(np.array_equal(a, b))


class Kernel:
    """_fused_engram_post_wkv_kernel; safe=False also reads the neighbouring hc row (not in-place safe)."""

    def __init__(self, safe=True):
        self.safe, self.launches = safe, []

    def __getitem__(self, grid):
        def launch(hidden, kv, q_w, k_w, mask, out, num_kv, h0, h1, h2, kv0, kv1, q0, q1, k0, k1, m0, o0, o1, o2,
                   eps, clamp, DIM, HC_MULT, BLOCK_SIZE, HAS_MASK, num_warps):
            assert grid == (hidden.shape[0] * HC_MULT,), grid
            assert (h0, h1, h2) == tuple(hidden.stride(i) for i in range(3))
            assert (kv0, kv1) == (kv.stride(0), kv.stride(1)) and (o0, o1, o2) == tuple(out.stride(i) for i in range(3))
            assert (q0, q1, k0, k1) == (q_w.stride(0), q_w.stride(1), k_w.stride(0), k_w.stride(1))
            assert BLOCK_SIZE >= DIM and num_warps == 4 and m0 == (mask.stride(0) if HAS_MASK else 0)
            self.launches.append((hidden.shape[0], num_kv, out is hidden or np.shares_memory(out, hidden)))
            for pid in range(grid[0]):
                ti, hi = divmod(pid, HC_MULT)
                row = np.array(hidden[ti, hi, :DIM], dtype=np.float32)
                if not self.safe:
                    row = row + np.float32(0.5) * hidden[ti, (hi + 1) % HC_MULT, :DIM]
                valid = ti < num_kv
                key = kv[ti, hi * DIM:(hi + 1) * DIM] if valid else np.zeros(DIM, np.float32)
                hidden_rms = 1 / np.sqrt(np.sum(row * row) / DIM + eps)
                key_rms = 1 / np.sqrt(np.sum(key * key) / DIM + eps)
                dot = np.sum(row * q_w[hi] * k_w[hi] * key) * hidden_rms * key_rms / np.sqrt(np.float32(DIM))
                gate_in = np.sqrt(max(abs(dot), clamp)) * (-1 if dot < 0 else 1)
                gate = np.float32(1 / (1 + np.exp(-gate_in)))
                if HAS_MASK and not (valid and mask[ti]):
                    gate = np.float32(0)
                value = kv[ti, HC_MULT * DIM:(HC_MULT + 1) * DIM] if valid else np.zeros(DIM, np.float32)
                out[ti, hi, :DIM] = row + gate * value

        return launch


def pinned_forward(src: str) -> str:
    lines = src.splitlines(keepends=True)
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "Engram")
    fn = next(f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "forward")
    return "".join(lines[fn.lineno - 1:fn.end_lineno])


def build(forward_src: str | None = None, kernel: Kernel | None = None):
    """The engram module stand-in (torch, triton, kernel, Engram with the pinned forward)."""
    path = Path(tempfile.mkdtemp()) / "pinned_engram.py"
    path.write_text(HEAD + (forward_src or pinned_forward(PIN.read_text())))
    spec = importlib.util.spec_from_file_location(f"pinned_engram_{id(path)}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.torch = FakeTorch()
    mod.triton = types.SimpleNamespace(next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
    mod._fused_engram_post_wkv_kernel = kernel or Kernel()
    return mod


class Wkv:
    """[M, HEADS*EDIM] -> [M, (HC+1)*DIM]; m_dependent adds 1 to every output when M differs from `full`."""

    def __init__(self, full=None):
        rng = np.random.default_rng(1)
        self.w = rng.integers(-3, 4, size=(HEADS * EDIM, (HC + 1) * DIM)).astype(np.float32)
        self.full, self.ms = full, []

    def __call__(self, x):
        self.ms.append(x.shape[0])
        out = t(x @ self.w)
        if self.full is not None and x.shape[0] != self.full:
            out += 1
        return out


def engram(mod, wkv=None):
    e = mod.Engram.__new__(mod.Engram)
    rng = np.random.default_rng(2)
    e.hc_mult, e.dim, e.use_sequence_parallel, e.eps, e.clamp_value = HC, DIM, False, 1e-6, 1e-4
    e.q_weight = t(rng.integers(1, 4, size=(HC, DIM)))
    e.k_weight = t(rng.integers(1, 4, size=(HC, DIM)))
    e.embeds = []
    e.embed = lambda ids: (e.embeds.append(ids.shape[0]), t(ids[:, :, None] % 5 - 2 + np.zeros((1, 1, EDIM))))[1]
    e.wkv = wkv or Wkv()
    return e


def inputs(n, seed=3, masked=True):
    rng = np.random.default_rng(seed)
    hidden = t(rng.integers(-4, 5, size=(n, HC, DIM)))
    ids = rng.integers(0, 1000, size=(n, HEADS))
    mask = (rng.random(n) > 0.2).view(T) if masked else None
    return hidden, ids, mask


def bits(a):
    return np.ascontiguousarray(a).view(np.int32)


def patched(mod, inplace=True, rows=0):
    out = io.StringIO()
    fwd, state = epi.make(mod.Engram, mod, mod.torch, inplace, rows, say=lambda m: print(m, file=out))
    mod.Engram.forward = fwd
    return state, out


class PinTests(unittest.TestCase):
    def test_pinned_forward_is_the_e15_text(self) -> None:
        self.assertEqual(attn_out_defer.fn_text(pinned_forward(PIN.read_text())), epi.EXPECTED_FORWARD)


class InPlaceTests(unittest.TestCase):
    def stock(self, n, masked=True, kernel=None, wkv=None, seed=3):
        mod = build(kernel=kernel)
        hidden, ids, mask = inputs(n, seed, masked)
        return mod.Engram.forward(engram(mod, wkv), hidden, ids, mask)

    def test_in_place_equals_the_stock_kernel_and_writes_the_input(self) -> None:
        for masked in (True, False):
            mod = build()
            state, log = patched(mod)
            e = engram(mod)
            for call, seed in enumerate((3, 4, 5, 6)):  # 2 self-check calls, then the plain path
                hidden, ids, mask = inputs(200, seed, masked)
                out = mod.Engram.forward(e, hidden, ids, mask)
                self.assertTrue(np.shares_memory(out, hidden), call)
                np.testing.assert_array_equal(bits(out), bits(self.stock(200, masked, seed=seed)))
            self.assertEqual(state["verify"], 0)
            self.assertIn("dsv41: engram post self-check bit-exact on 2 eager calls (in place on, wkv rows full)",
                          log.getvalue())
            # self-check calls: stock into a fresh buffer, then in place; later calls: in place only
            launches = mod._fused_engram_post_wkv_kernel.launches
            self.assertEqual([x[2] for x in launches], [False, True, False, True, True, True])

    def test_row_chunks_equal_the_full_gemm_and_embed_once(self) -> None:
        mod = build()
        wkv = Wkv()
        state, log = patched(mod, inplace=True, rows=64)
        e = engram(mod, wkv)
        for seed in (3, 4, 5):
            hidden, ids, mask = inputs(200, seed)
            out = mod.Engram.forward(e, hidden, ids, mask)
            np.testing.assert_array_equal(bits(out), bits(self.stock(200, seed=seed)))
        self.assertEqual(wkv.ms, [200, 64, 64, 64, 8] * 2 + [64, 64, 64, 8])
        self.assertEqual(e.embeds, [200, 200, 200])
        self.assertEqual(state["rows"], 64)
        self.assertIn("wkv rows 64)", log.getvalue())
        chunked = [x for x in mod._fused_engram_post_wkv_kernel.launches[-4:]]
        self.assertEqual([(r, k) for r, k, _ in chunked], [(64, 64), (64, 64), (64, 64), (8, 8)])

    def test_an_m_dependent_gemm_disarms_row_chunks(self) -> None:
        mod = build()
        state, log = patched(mod, inplace=False, rows=64)
        e = engram(mod, Wkv(full=200))
        hidden, ids, mask = inputs(200)
        out = mod.Engram.forward(e, hidden, ids, mask)
        np.testing.assert_array_equal(bits(out), bits(self.stock(200, wkv=Wkv(full=200))))
        self.assertEqual(state["rows"], 0)
        self.assertIn("dsv41: engram wkv rows DISABLED: wkv on rows 0:64 of 200", log.getvalue())
        self.assertEqual(e.wkv.ms, [200, 64, 64, 64, 8])  # every slice compared before disarming
        hidden, ids, mask = inputs(200, 4)
        out = mod.Engram.forward(e, hidden, ids, mask)
        np.testing.assert_array_equal(bits(out), bits(self.stock(200, wkv=Wkv(full=200), seed=4)))
        self.assertNotIn("self-check bit-exact", log.getvalue())

    def test_a_kernel_that_is_not_in_place_safe_disarms_in_place(self) -> None:
        mod = build(kernel=Kernel(safe=False))
        state, log = patched(mod)
        e = engram(mod)
        for seed in (3, 4, 5):
            hidden, ids, mask = inputs(200, seed)
            out = mod.Engram.forward(e, hidden, ids, mask)
            self.assertFalse(np.shares_memory(out, hidden))
            np.testing.assert_array_equal(bits(out), bits(self.stock(200, kernel=Kernel(safe=False), seed=seed)))
        self.assertFalse(state["inplace"])
        self.assertIn("dsv41: engram post in place DISABLED", log.getvalue())

    def test_small_steps_sequence_parallel_and_capture_run_the_stock_forward(self) -> None:
        for n, sp, capturing in ((epi.MIN_ROWS, False, False), (200, True, False), (200, False, True)):
            mod = build()
            mod.torch = FakeTorch(capturing)
            state, _ = patched(mod)
            e = engram(mod)
            e.use_sequence_parallel = sp  # world size 1 in the stand-in module: the shard is every row
            hidden, ids, mask = inputs(n)
            out = mod.Engram.forward(e, hidden, ids, mask)
            self.assertFalse(np.shares_memory(out, hidden), (n, sp, capturing))
            self.assertEqual(state["verify"], epi.VERIFY_CALLS, (n, sp, capturing))
            self.assertEqual(len(mod._fused_engram_post_wkv_kernel.launches), 1)


class InstallTests(unittest.TestCase):
    def test_install_engaged_idempotent_and_refusals(self) -> None:
        mod = build()
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertEqual(epi.install(mod.Engram, mod, {"DSV41_ENGRAM_POST_INPLACE": "1"}, mod.torch), "engaged")
            self.assertEqual(epi.install(mod.Engram, mod, {"DSV41_ENGRAM_POST_INPLACE": "1"}, mod.torch),
                             "already installed")
        self.assertIn("dsv41: engram post engaged (in place on, wkv rows full, above 64 eager rows", log.getvalue())
        changed = pinned_forward(PIN.read_text()).replace("output = torch.empty_like(hidden_states)",
                                                          "output = torch.empty_like(hidden_states) * 1")
        no_kernel = build()
        del no_kernel._fused_engram_post_wkv_kernel
        for bad, why in ((build(changed), "differs from the pinned e15 source"), (no_kernel, "no _fused_engram")):
            stock = bad.Engram.forward
            with contextlib.redirect_stdout(io.StringIO()) as log:
                detail = epi.install(bad.Engram, bad, {"DSV41_ENGRAM_POST_INPLACE": "1"}, bad.torch)
            self.assertTrue(detail.startswith("disabled: ") and why in detail, detail)
            self.assertIn(epi.LOG_DISARMED[0], log.getvalue())
            self.assertIs(bad.Engram.forward, stock)

    def test_settings(self) -> None:
        self.assertEqual(epi.settings({}), (False, 0))
        self.assertEqual(epi.settings({"DSV41_ENGRAM_POST_INPLACE": "1", "DSV41_ENGRAM_WKV_ROWS": "2048"}), (True, 2048))
        self.assertEqual(epi.settings({"DSV41_ENGRAM_WKV_ROWS": "x"}), (False, 0))
        self.assertTrue(epi.enabled({"DSV41_ENGRAM_WKV_ROWS": "1024"}))
        self.assertFalse(epi.enabled({"DSV41_ENGRAM_POST_INPLACE": "0", "DSV41_ENGRAM_WKV_ROWS": "0"}))

    def test_defaults_in_recipe_and_forwarded(self) -> None:
        run = (ROOT / "run.sh").read_text()
        self.assertIn('DSV41_ENGRAM_POST_INPLACE="${DSV41_ENGRAM_POST_INPLACE:-1}"', run)
        self.assertIn("  DSV41_ENGRAM_POST_INPLACE=\n", run)
        self.assertIn("  DSV41_ENGRAM_WKV_ROWS=0\n", run)
        self.assertIn("    DSV41_ENGRAM_POST_INPLACE: 1\n", (ROOT / "recipe.yaml").read_text())
        self.assertIn('_patch("engram_post_inplace", _p_engram_post_inplace)', (PATCH / "sitecustomize.py").read_text())


if __name__ == "__main__":
    unittest.main()
