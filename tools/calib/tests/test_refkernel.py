"""refkernel vs the official TileLang kernels (inference/kernel.py), plus CPU checks.

GPU tests need CUDA and tilelang (dsv41-quant151 image has tilelang 0.1.12). Set CALIB_REF_DIR to the
1017 `inference/` dir if it is not at the default HF-cache snapshot path.
Run: python3 -m unittest discover -s tools/calib/tests -p 'test_refkernel.py' -v
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import refkernel as rk  # noqa: E402

REF_DIR = Path(os.environ.get("CALIB_REF_DIR", Path.home() / ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170/inference"))


def load_tl():
    if not torch.cuda.is_available():
        return None
    try:
        spec = importlib.util.spec_from_file_location("ref_tl_kernel", REF_DIR / "kernel.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:  # tilelang missing or broken on this arch
        print("tilelang unavailable:", type(e).__name__, e)
        return None


TL = load_tl()


class CpuChecks(unittest.TestCase):
    def test_round_fp4_grid_and_ties(self):
        v = torch.tensor([0.0, 0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.01, 6.0, -0.75, -2.5, -5.0])
        want = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, -1.0, -2.0, -4.0])
        self.assertTrue(torch.equal(rk.round_fp4(v), want))

    def test_pow2_ceil(self):
        x = torch.tensor([1.0, 1.5, 0.75, 3e-7, 448.0, 2.0**-127])
        want = torch.tensor([1.0, 2.0, 1.0, 2.0**-21, 512.0, 2.0**-126])
        got = rk.pow2_ceil_scale(x)
        self.assertTrue(torch.equal(got, want), got)

    def test_fp4_dequant_matches_convert_table(self):
        torch.manual_seed(0)
        w = torch.randint(-128, 127, (64, 32), dtype=torch.int8)
        s = torch.randint(120, 130, (64, 2), dtype=torch.uint8).view(torch.float8_e8m0fnu)
        got = rk.dequant_fp4_weight(w, s, torch.float32)
        tbl = torch.tensor(rk.FP4_VALUES)
        u = w.view(torch.uint8)
        ref = torch.stack([tbl[(u & 15).long()], tbl[(u >> 4).long()]], -1).view(64, 64)
        ref = ref * (2.0 ** (s.view(torch.uint8).float() - 127)).repeat_interleave(32, 1)
        self.assertTrue(torch.equal(got, ref))


@unittest.skipIf(TL is None, "needs CUDA + tilelang")
class TileLangEquivalence(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        torch.set_default_dtype(torch.bfloat16)
        self.dev = "cuda"

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def rand(self, *shape, scale=1.0):
        x = torch.randn(*shape, device=self.dev, dtype=torch.float32) * scale
        x[..., ::97] *= 40  # outliers so block scales vary
        return x.to(torch.bfloat16)

    def test_act_quant_inplace_bitexact(self):
        x = self.rand(300, 4096)
        a, b = x.clone(), x.clone()
        TL.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu, True)
        rk.act_quant(b, 32, "ue8m0", torch.float8_e8m0fnu, True)
        self.assertTrue(torch.equal(a, b), (a.float() - b.float()).abs().max())

    def test_act_quant_out_bitexact(self):
        x = self.rand(256, 5120)
        ya, sa = TL.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        yb, sb = rk.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        self.assertTrue(torch.equal(ya.view(torch.uint8), yb.view(torch.uint8)))
        self.assertTrue(torch.equal(sa.view(torch.uint8), sb.view(torch.uint8)))

    def test_fp4_quant_e8m0_bitexact(self):
        x = self.rand(128, 32, 128)
        a, b = x.clone(), x.clone()
        TL.fp4_act_quant(a, 32, True)
        rk.fp4_act_quant(b, 32, True)
        self.assertTrue(torch.equal(a, b), (a.float() - b.float()).abs().max())

    def test_fp4_quant_e4m3_bitexact(self):
        x = self.rand(512, 512)
        a, b = x.clone(), x.clone()
        TL.fp4_act_quant(a, 16, True, scale_dtype=torch.float8_e4m3fn)
        rk.fp4_act_quant(b, 16, True, scale_dtype=torch.float8_e4m3fn)
        diff = (a.float() != b.float()).float().mean().item()
        self.assertEqual(diff, 0.0, f"mismatch fraction {diff}")

    def test_fp8_gemm(self):
        x = self.rand(200, 1280)
        w = torch.randn(1024, 1280, device=self.dev, dtype=torch.float32) * 0.05
        wq, ws2 = self.block_quant(w)
        xa, xs = TL.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        a = TL.fp8_gemm(xa, xs, wq, ws2, torch.float8_e8m0fnu, 32)
        b = rk.fp8_gemm(xa, xs, wq, ws2, torch.float8_e8m0fnu, 32)
        rel = ((a.float() - b.float()).norm() / a.float().norm()).item()
        print(f"{self.id().split('.')[-1]} rel diff {rel:.3e}, bit-equal frac {(a == b).float().mean().item():.4f}")
        self.assertLess(rel, 2e-3, rel)

    def block_quant(self, w):
        N, K = w.shape
        wb = w.view(N // 32, 32, K // 32, 32)
        amax = wb.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-4)
        s = rk.pow2_ceil_scale(amax / 448.0)
        q = (wb / s).clamp(-448, 448).to(torch.float8_e4m3fn).view(N, K)
        return q, s.view(N // 32, K // 32).to(torch.float8_e8m0fnu)

    def test_fp4_gemm(self):
        x = self.rand(150, 5120)
        w = torch.randint(-128, 127, (2304, 2560), dtype=torch.int8, device=self.dev)
        s = torch.randint(118, 124, (2304, 160), dtype=torch.uint8, device=self.dev).view(torch.float8_e8m0fnu)
        xa, xs = TL.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        a = TL.fp4_gemm(xa, xs, w.view(torch.float4_e2m1fn_x2), s, torch.float8_e8m0fnu, 32)
        b = rk.fp4_gemm(xa, xs, w, s, torch.float8_e8m0fnu, 32)
        rel = ((a.float() - b.float()).norm() / a.float().norm()).item()
        print(f"{self.id().split('.')[-1]} rel diff {rel:.3e}, bit-equal frac {(a == b).float().mean().item():.4f}")
        self.assertLess(rel, 2e-3, rel)

    def test_sparse_attn(self):
        # The model shape (h=64, d=512) needs 141 KB dynamic smem in the TileLang kernel, above the
        # sm_121 limit, so the official kernel cannot launch on GB10. Compare at shapes that fit.
        for h, d in ((16, 256), (32, 256), (16, 128)):
            with self.subTest(h=h, d=d):
                self._sparse_attn(h, d)

    def _sparse_attn(self, h, d):
        b, m, n, k = 1, 300, 700, 640
        q = self.rand(b, m, h, d, scale=0.3)
        kv = self.rand(b, n, d, scale=0.3)
        sink = torch.randn(h, device=self.dev, dtype=torch.float32)
        idx = torch.randint(0, n, (b, m, k), device=self.dev, dtype=torch.int32)
        idx[:, :, ::3] = -1
        idx[:, 0, :] = -1  # an all-empty row must give zeros
        a = TL.sparse_attn(q, kv, sink, idx, d ** -0.5)
        c = rk.sparse_attn(q, kv, sink, idx, d ** -0.5)
        self.assertTrue(torch.all(c[:, 0] == 0))
        rel = ((a.float() - c.float()).norm() / a.float().norm()).item()
        print(f"sparse_attn h={h} d={d} rel diff {rel:.3e}, max abs {(a.float() - c.float()).abs().max().item():.3e}, "
              f"bit-equal frac {(a == c).float().mean().item():.4f}")
        self.assertLess(rel, 5e-3, rel)

    def test_hc_sinkhorn(self):
        mixes = torch.randn(2, 100, 24, device=self.dev, dtype=torch.float32) * 3
        scale = torch.randn(3, device=self.dev, dtype=torch.float32)
        base = torch.randn(24, device=self.dev, dtype=torch.float32)
        a = TL.hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
        b = rk.hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
        for x, y in zip(a, b):
            self.assertLess((x - y).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
