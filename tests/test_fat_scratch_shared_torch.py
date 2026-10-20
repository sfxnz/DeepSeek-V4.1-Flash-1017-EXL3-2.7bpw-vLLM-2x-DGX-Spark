"""fat_scratch_shared.py with real torch on CPU, through e15's own apply_exl3_batched_fat (image test).

e15's exl3.py is rebuilt from tests/fixtures/exl3_e14.pin.py with widen_p2b_kbits.patch_py (equal to the
image file). Its fat-expert loop runs with CPU stand-ins for the ExLlamaV3 kernels (each writes its whole
output, as the CUDA kernels do), once with the stock per-bucket cache and once with the shared set. Between
calls every scratch buffer is poisoned (NaN / 12345), so a read of anything the current expert did not write
shows up as a mismatch. Skips on hosts without torch; run it in the image:

  docker run --rm --network none --memory 8g --cpus 4 -v "$PWD":/w -w /w/tests --entrypoint python3 \\
    dsv41-flash-exl3-sm121:e15-kbits -m unittest test_fat_scratch_shared_torch -q
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
import types
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    raise unittest.SkipTest("torch not installed on this host")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docker/patch"))

import fat_scratch_shared as fss  # noqa: E402
import widen_p2b_kbits  # noqa: E402

H, I = 256, 128  # small hidden / intermediate: the loop logic does not depend on the sizes
FAT_CAP = 256


def _e15_functions(src: str) -> str:
    out = []
    for name in ("_fat_scratch", "apply_exl3_batched_fat"):
        m = re.search(rf"^def {name}\(.*?(?=^def )", src, re.S | re.M)
        assert m is not None, name
        out.append(m.group(0))
    return "\n\n".join(out)


class FakeExt:
    @staticmethod
    def had_r_128(inp, out, pre, post, scale):
        t = inp.float()
        if pre is not None:
            t = t * pre.float()
        if post is not None:
            t = t * post.float()
        out.copy_((t * scale).to(out.dtype))

    @staticmethod
    def reconstruct(w, trellis, k, mcg, mul1):
        a, b = trellis.shape[0], trellis.shape[1]
        v = trellis[:, :, :1].float().reshape(a, 1, b, 1).expand(a, 16, b, 16).reshape(a * 16, b * 16)
        w.copy_((v * 1e-3 + k * 1e-2).to(w.dtype))

    @staticmethod
    def hgemm(a, b, c):
        c.copy_((a.float() @ b.float()).to(c.dtype))


def _namespace():
    src = _e15_functions(widen_p2b_kbits.patch_py((ROOT / "tests/fixtures/exl3_e14.pin.py").read_text()))
    ns = {
        "torch": torch,
        "Any": object,
        "_FAT_SCRATCH_CACHE": {},
        "TEMP_ROWS_FUSED": 2048,
        "_load_native_exl3_ext": lambda: None,
        "load_exllamav3_ext": lambda: FakeExt,
    }
    exec(compile(src, "exl3_e15_fat_path", "exec"), ns)
    return ns


def _proj(g, rows_in, cols_out, k):
    return types.SimpleNamespace(
        in_features=rows_in,
        out_features=cols_out,
        trellis=torch.randint(-300, 300, (rows_in // 16, cols_out // 16, 16 * k), generator=g, dtype=torch.int16),
        suh=(torch.rand(rows_in, generator=g) + 0.5).half(),
        svh=(torch.rand(cols_out, generator=g) + 0.5).half(),
        K=k,
        mcg=True,
        mul1=False,
    )


def _layer(seed: int, k_gu: int, k_d: int, counts: list[int], tokens: int = 2048):
    g = torch.Generator().manual_seed(seed)
    inners = []
    for e in range(len(counts)):
        inners.append({
            "gate": _proj(g, H, I, k_gu),
            "up": _proj(g, H, I, k_gu),
            "down": _proj(g, I, H, k_d),
            "_exl3_gate_up_shared_suh": e % 3 == 0,  # mostly distinct suh, as on the 1017 pack
        })
    xh = (torch.randn(tokens, H, generator=g) * 0.5).half()
    token_sorted = torch.cat([torch.randperm(tokens, generator=g)[:n] for n in counts])
    weight_sorted = torch.rand(int(sum(counts)), generator=g).half()
    return xh, token_sorted, weight_sorted, counts, inners


LAYERS = [  # (K_gu, K_d, rows per expert): buckets 512..2048, both K_gu, thin experts skipped (<= 256)
    (2, 2, [300, 10, 700, 2048, 0, 257, 1000, 1300]),
    (3, 3, [1800, 256, 513, 900, 1536, 40]),
    (2, 3, [1024, 1025, 600, 2000]),
    (3, 4, [257, 1792, 1100, 30, 1280]),
    (2, 2, [512, 400, 2047, 768]),
]


def _poison(cache: dict) -> None:
    for entry in cache.values():
        for t in entry.values() if isinstance(entry, dict) else ():
            if isinstance(t, torch.Tensor):
                t.fill_(12345 if t.dtype == torch.int16 else float("nan"))


def _run(shared: bool):
    ns = _namespace()
    if shared:
        exl3 = types.SimpleNamespace(_FAT_SCRATCH_CACHE=ns["_FAT_SCRATCH_CACHE"], TEMP_ROWS_FUSED=2048)
        ns["_fat_scratch"] = fss.make_fat_scratch(exl3, torch)
    outs = []
    for i, (k_gu, k_d, counts) in enumerate(LAYERS):
        xh, ts, ws, ch, inners = _layer(100 + i, k_gu, k_d, counts)
        out = torch.zeros(xh.shape[0], H, dtype=torch.float32)
        ns["apply_exl3_batched_fat"](xh, ts, ws, ch, inners, None, FAT_CAP, out, use_kernel=True)
        outs.append(out)
        _poison(ns["_FAT_SCRATCH_CACHE"])
    return outs, ns["_FAT_SCRATCH_CACHE"]


class SharedFatScratchTorchTests(unittest.TestCase):
    def test_outputs_are_bit_identical_to_the_stock_cache(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            stock, stock_cache = _run(False)
            shared, shared_cache = _run(True)
        for i, (a, b) in enumerate(zip(stock, shared)):
            self.assertFalse(torch.isnan(a).any(), i)
            self.assertTrue(torch.equal(a, b), f"layer {i}: max |diff| {(a - b).abs().max()}")
        stock_bytes = sum(t.numel() * t.element_size() for e in stock_cache.values() for t in e.values())
        base = shared_cache[(fss.BASE_TAG, "cpu", I, H)]
        shared_bytes = sum(t.numel() * t.element_size() for t in base.values() if isinstance(t, torch.Tensor))
        self.assertGreater(len(stock_cache), 8)
        self.assertLess(shared_bytes, stock_bytes / 4)

    def test_views_match_stock_shapes_strides_and_start_at_the_shared_base(self) -> None:
        ns = _namespace()
        exl3 = types.SimpleNamespace(_FAT_SCRATCH_CACHE={}, TEMP_ROWS_FUSED=2048)
        with contextlib.redirect_stdout(io.StringIO()):
            shared = fss.make_fat_scratch(exl3, torch)
            for k in (2, 3):
                for rows in (257, 700, 2048):
                    gate = _proj(torch.Generator().manual_seed(k), H, I, k)
                    want = ns["_fat_scratch"](torch.device("cpu"), rows, gate)
                    got = shared(torch.device("cpu"), rows, gate)
                    base = exl3._FAT_SCRATCH_CACHE[(fss.BASE_TAG, "cpu", I, H)]
                    for name, t in want.items():
                        g = got[name]
                        self.assertEqual((g.shape, g.dtype, g.stride()), (t.shape, t.dtype, t.stride()), name)
                        self.assertTrue(g.is_contiguous(), name)
                        src = base["packed"] if name == "packed13" else base[name]
                        self.assertEqual(g.data_ptr(), src.data_ptr(), name)


if __name__ == "__main__":
    unittest.main()
