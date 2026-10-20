"""docker/patch/fat_scratch_shared.py against the pinned vllm_exl3 _fat_scratch (CPU only, no torch).

The stock function comes from tests/fixtures/exl3_e14.pin.py (the e15-kbits image has the same
_fat_scratch; its kbits rewrite touches only apply_exl3_batched_fat). A fake torch records every
torch.empty, so the stock function and the replacement can be run and compared on the host.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import itertools
import math
import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker/patch"
PIN = ROOT / "tests/fixtures/exl3_e14.pin.py"
sys.path.insert(0, str(PATCH))

import fat_scratch_shared as fss  # noqa: E402

HIDDEN, INTER = 5120, 1152  # per TP rank on DeepSeek-V4.1-Flash-1017 (moe_intermediate 2304 / 2)
IN_TILES, OUT_TILES = HIDDEN // 16, INTER // 16
BUCKETS = range(512, 2049, 256)  # fat experts have 257..2048 rows
K_GU = (2, 3)  # 2.7bpw-mcg-hq expert_bits K_gu values
MIB = 2**20


def _stock_source() -> str:
    m = re.search(r"^def _fat_scratch\(.*?(?=^def )", PIN.read_text(), re.S | re.M)
    assert m is not None
    return m.group(0)


class _Storage:
    ids = itertools.count()

    def __init__(self, numel: int, dtype: str) -> None:
        self.id, self.numel, self.dtype = next(self.ids), numel, dtype


class FakeTensor:
    def __init__(self, shape, dtype, storage=None) -> None:
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.storage = storage if storage is not None else _Storage(math.prod(self.shape), dtype)

    def numel(self) -> int:
        return math.prod(self.shape)

    def element_size(self) -> int:
        return fss.ITEMSIZE[self.dtype]

    def __getitem__(self, s):
        assert isinstance(s, slice) and s.start is None and s.step is None, s
        return FakeTensor((min(s.stop, self.shape[0]),) + self.shape[1:], self.dtype, self.storage)

    def view(self, *shape):
        assert math.prod(shape) == self.numel(), (shape, self.shape)
        return FakeTensor(shape, self.dtype, self.storage)


def fake_torch():
    torch = types.ModuleType("torch")
    torch.float16, torch.float32, torch.int16 = "float16", "float32", "int16"
    torch.Tensor = FakeTensor
    torch.allocated = []

    def empty(shape, dtype=None, device=None):
        t = FakeTensor((shape,) if isinstance(shape, int) else shape, dtype)
        torch.allocated.append(t)
        return t

    torch.empty = empty
    return torch


def gate(k_gu: int):
    return types.SimpleNamespace(
        in_features=HIDDEN, out_features=INTER, trellis=FakeTensor((IN_TILES, OUT_TILES, 16 * k_gu), "int16")
    )


def stock_module(torch):
    """The pinned stock _fat_scratch, run against `torch` (a fake)."""
    ns = {"torch": torch, "_FAT_SCRATCH_CACHE": {}, "Any": object}
    exec(compile(_stock_source(), str(PIN), "exec"), ns)
    return ns


def nbytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


class StockShapeTests(unittest.TestCase):
    def test_expected_stock_matches_the_pinned_source(self) -> None:
        self.assertIsNone(fss.stock_mismatch(_stock_source()))

    def test_a_changed_stock_function_is_refused(self) -> None:
        src = _stock_source()
        variants = {
            "dtype": src.replace('"act": torch.empty(\n            (bucketed_cap, intermediate), dtype=torch.float32',
                                 '"act": torch.empty(\n            (bucketed_cap, intermediate), dtype=torch.float16'),
            "new field": src.replace('"u_tmp": torch.empty(', '"v_tmp": torch.empty(1, device=device),\n        "u_tmp": torch.empty('),
            "bucket": src.replace("+ 255) // 256) * 256", "+ 511) // 512) * 512"),
            "key": src.replace("key = (str(device), bucketed_cap, intermediate, hidden, k_words)",
                               "key = (str(device), bucketed_cap, intermediate, hidden)"),
        }
        for name, bad in variants.items():
            self.assertNotEqual(bad, src, name)
            self.assertIsNotNone(fss.stock_mismatch(bad), name)

    def test_byte_model_equals_what_the_stock_function_allocates(self) -> None:
        for cap, k in itertools.product((256,) + tuple(BUCKETS), K_GU + (4,)):
            torch = fake_torch()
            ns = stock_module(torch)
            ns["_fat_scratch"]("cuda:0", cap, gate(k))
            self.assertEqual(
                nbytes(torch.allocated), fss.stock_bytes(cap, HIDDEN, INTER, IN_TILES, OUT_TILES, 16 * k), (cap, k)
            )

    def test_model_reproduces_prefill_moe_measured_holds(self) -> None:
        # perf2/prefill-moe/runs/pm-check-all.log, stock_cold "held" MiB for single census calls; the
        # bucket sets are the ones that reproduce them (to 0.5 MiB), not measured themselves.
        cases = {
            "2:2 census m=2048 held 202": (202, 2, (512, 768)),
            "2:3 census m=2048 held 328": (328, 2, (512, 768, 1024)),
            "3:3 census m=2048 held 205": (205, 3, (512, 768)),
            "3:4 census m=8192 held 637": (637, 3, (512, 768, 1024, 1280, 1536)),
        }
        for label, (held, k, caps) in cases.items():
            mib = sum(fss.stock_bytes(c, HIDDEN, INTER, IN_TILES, OUT_TILES, 16 * k) for c in caps) / MIB
            self.assertLess(abs(mib - held), 0.6, (label, mib))

    def test_all_fourteen_keys_versus_one_shared_set(self) -> None:
        stock = sum(fss.stock_bytes(c, HIDDEN, INTER, IN_TILES, OUT_TILES, 16 * k) for c in BUCKETS for k in K_GU)
        shared = fss.set_bytes(2048, HIDDEN, INTER, IN_TILES * 2 * OUT_TILES * fss.PACKED_K_WORDS)
        self.assertAlmostEqual(stock / MIB, 2009.3, places=1)
        self.assertAlmostEqual(shared / MIB, 195.9, places=1)


class ReplacementTests(unittest.TestCase):
    def _pair(self):
        torch_s, torch_r = fake_torch(), fake_torch()
        stock = stock_module(torch_s)
        exl3 = types.SimpleNamespace(_FAT_SCRATCH_CACHE={}, TEMP_ROWS_FUSED=2048)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            shared = fss.make_fat_scratch(exl3, torch_r)
        return stock, torch_s, exl3, torch_r, shared, out

    def test_every_key_gets_the_stock_shapes_from_one_set(self) -> None:
        stock, torch_s, exl3, torch_r, shared, _ = self._pair()
        storages = {}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            for k, cap in itertools.product(K_GU, BUCKETS):
                for rows in (cap - 255, cap):
                    want = stock["_fat_scratch"]("cuda:0", rows, gate(k))
                    got = shared("cuda:0", rows, gate(k))
                    self.assertEqual(set(got), set(want), (k, cap))
                    for name in want:
                        self.assertEqual((got[name].shape, got[name].dtype), (want[name].shape, want[name].dtype),
                                         (k, cap, name))
                        storages.setdefault(name, set()).add(got[name].storage.id)
                    self.assertIs(shared("cuda:0", rows, gate(k)), got)  # cached per key, like stock
        self.assertEqual({n: len(s) for n, s in storages.items()}, {n: 1 for n in storages})
        self.assertEqual(len(stock["_FAT_SCRATCH_CACHE"]), 14)
        self.assertAlmostEqual(nbytes(torch_s.allocated) / MIB, 2009.3, places=1)
        self.assertAlmostEqual(nbytes(torch_r.allocated) / MIB, 195.9, places=1)
        lines = out.getvalue().splitlines()
        self.assertEqual(sum("one set of 2048 rows" in ln for ln in lines), 1, lines)
        self.assertEqual(sum("stock key" in ln for ln in lines), 14, lines)
        self.assertIn("would hold 2009.3 MiB in 14 sets, shared 195.9 MiB", lines[-1])

    def test_a_cleared_cache_rebuilds_one_set(self) -> None:
        _, _, exl3, torch_r, shared, _ = self._pair()
        with contextlib.redirect_stdout(io.StringIO()):
            shared("cuda:0", 600, gate(3))
            exl3._FAT_SCRATCH_CACHE.clear()  # what moe_prefill_tc does after its stock reference call
            again = shared("cuda:0", 1500, gate(2))
        self.assertEqual(again["h"].shape, (1536, HIDDEN))
        self.assertEqual(len(torch_r.allocated), 2 * (1 + len(fss.ROW_FIELDS) + len(fss.FIXED_FIELDS)))

    def test_a_wider_packed13_regrows_and_drops_the_old_views(self) -> None:
        _, _, exl3, torch_r, shared, _ = self._pair()
        with contextlib.redirect_stdout(io.StringIO()):
            shared("cuda:0", 600, gate(3))
            n = len(torch_r.allocated)
            wide = shared("cuda:0", 600, gate(5))  # wider than PACKED_K_WORDS: regrow
        self.assertEqual(wide["packed13"].shape, (IN_TILES, 2 * OUT_TILES, 80))
        self.assertGreater(len(torch_r.allocated), n)
        keys = [k for k in exl3._FAT_SCRATCH_CACHE if k[0] != fss.BASE_TAG]
        self.assertEqual(keys, [("cuda:0", 768, INTER, HIDDEN, 80)])

    def test_rows_above_the_shared_capacity_regrow(self) -> None:
        _, _, exl3, torch_r, shared, _ = self._pair()
        with contextlib.redirect_stdout(io.StringIO()):
            shared("cuda:0", 300, gate(2))
            big = shared("cuda:0", 3000, gate(2))
        self.assertEqual(big["down"].shape, (3072, HIDDEN))
        base = exl3._FAT_SCRATCH_CACHE[(fss.BASE_TAG, "cuda:0", INTER, HIDDEN)]
        self.assertEqual(base["rows"], 3072)


class InstallTests(unittest.TestCase):
    def _module(self, d: str, src: str):
        path = Path(d) / "fake_exl3.py"
        head = "from __future__ import annotations\n\nfrom typing import Any\n\nimport torch\n\n"
        path.write_text(head + "TEMP_ROWS_FUSED = 2048\n_FAT_SCRATCH_CACHE = {}\n\n\n" + src)
        spec = importlib.util.spec_from_file_location("fake_exl3", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_install_engages_once_and_refuses_a_different_function(self) -> None:
        torch = fake_torch()
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(sys.modules, {"torch": torch}):
            mod = self._module(d, _stock_source())
            stock_fn = mod._fat_scratch
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(fss.install(mod), "engaged")
                self.assertEqual(fss.install(mod), "already installed")
            self.assertIsNot(mod._fat_scratch, stock_fn)
            self.assertIn(fss.LOG_ENGAGED, out.getvalue())
            self.assertNotIn(fss.LOG_DISARMED, out.getvalue())
            with contextlib.redirect_stdout(io.StringIO()):
                got = mod._fat_scratch("cuda:0", 700, gate(3))
            self.assertEqual(got["gate_up"].shape, (768, 2 * INTER))

            bad = self._module(d, _stock_source().replace('"down": torch.empty(\n            (bucketed_cap, hidden), dtype=torch.float32',
                                                          '"down": torch.empty(\n            (bucketed_cap, hidden), dtype=torch.float16'))
            orig = bad._fat_scratch
            with contextlib.redirect_stdout(io.StringIO()) as out:
                status = fss.install(bad)
            self.assertTrue(status.startswith("disabled"), status)
            self.assertIs(bad._fat_scratch, orig)
            self.assertIn(fss.LOG_DISARMED, out.getvalue())

    def test_env_gate(self) -> None:
        with mock.patch.dict(os.environ, {"DSV41_FAT_SCRATCH_SHARED": "1"}):
            self.assertTrue(fss.enabled())
        for value in ("0", ""):
            with mock.patch.dict(os.environ, {"DSV41_FAT_SCRATCH_SHARED": value}):
                self.assertFalse(fss.enabled())
        env = {k: v for k, v in os.environ.items() if k != "DSV41_FAT_SCRATCH_SHARED"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(fss.enabled())

    def test_sitecustomize_installs_it_after_the_plugin_load(self) -> None:
        site = (PATCH / "sitecustomize.py").read_text()
        plugin = site.index('_patch("load_general_plugins", _p_load_general_plugins)')
        ours = site.index('_patch("fat_scratch_shared", _p_fat_scratch_shared)')
        self.assertGreater(ours, plugin)
        body = site[site.index("def _p_fat_scratch_shared"):ours]
        self.assertIn("raise _PatchSkip", body)
        self.assertIn("raise RuntimeError(detail)", body)

    def test_defaults_on_in_recipe_and_forwarded(self) -> None:
        run = (ROOT / "run.sh").read_text()
        self.assertIn('DSV41_FAT_SCRATCH_SHARED="${DSV41_FAT_SCRATCH_SHARED:-1}"', run)
        self.assertIn("  DSV41_FAT_SCRATCH_SHARED=\n", run)
        self.assertIn("    DSV41_FAT_SCRATCH_SHARED: 1\n", (ROOT / "recipe.yaml").read_text())


if __name__ == "__main__":
    unittest.main()
