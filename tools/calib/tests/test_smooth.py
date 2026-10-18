"""Smooth-mode helpers of native.py (CPU or GPU, no weights): split GEMM accuracy, fake-quant switch,
grouped wo_a equivalence, noise draw sharing across r."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NEED_CUDA = unittest.skipUnless(DEV == "cuda", "torch.mm(out_dtype=...) is CUDA-only")


class SmoothTest(unittest.TestCase):
    @NEED_CUDA
    def test_mmw_split_accuracy(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(64, 512, generator=g).to(DEV)
        w = torch.randn(96, 512, generator=g).to(torch.bfloat16).to(DEV)
        ref = (x.double() @ w.double().t())
        got = native.mmw(x, w)
        self.assertEqual(got.dtype, torch.float32)
        rel = ((got.double() - ref).norm() / ref.norm()).item()
        bf = ((x.to(torch.bfloat16).double() @ w.double().t() - ref).norm() / ref.norm()).item()
        self.assertLess(rel, 1e-5)          # ~2^-17 activation rounding + fp32 accumulation
        self.assertLess(rel, bf / 100)      # far below plain bf16 activations (~2^-9)
        xb = x.to(torch.bfloat16)
        self.assertTrue(torch.equal(native.mmw(xb, w), torch.matmul(xb, w.t())))  # same dtype: plain GEMM

    @NEED_CUDA
    def test_mmw_batched_shape(self):
        x = torch.randn(3, 5, 64, device=DEV)
        w = torch.randn(32, 64, device=DEV).to(torch.bfloat16)
        self.assertEqual(tuple(native.mmw(x, w).shape), (3, 5, 32))

    @NEED_CUDA
    def test_split_sparse_attn(self):
        g = torch.Generator().manual_seed(1)
        m, n, h, d, k = 300, 200, 4, 64, 48
        q = torch.randn(1, m, h, d, generator=g).to(DEV)
        kv = torch.randn(1, n, d, generator=g).to(DEV)
        idx = torch.randint(-1, n, (1, m, k), generator=g).int().to(DEV)
        sink = torch.randn(h, generator=g).to(DEV)
        prev = native.SPLIT_ATTN
        try:
            native.SPLIT_ATTN = False
            ref = native.sparse_attn_f32(q, kv, sink, idx, d ** -0.5, chunk=128)
            native.SPLIT_ATTN = True
            got = native.sparse_attn_f32(q, kv, sink, idx, d ** -0.5, chunk=128)
        finally:
            native.SPLIT_ATTN = prev
        self.assertLess(((got - ref).norm() / ref.norm()).item(), 1e-5)

    def test_sparse_attn_win_matches(self):
        g = torch.Generator().manual_seed(2)
        for s, clen, k, win in ((300, 150, 64, 128), (90, 0, 0, 128), (500, 500, 512, 128)):
            h, d = 4, 64
            q = torch.randn(s, h, d, generator=g).to(DEV)
            kv = torch.randn(s + clen, d, generator=g).to(DEV)
            sink = torch.randn(h, generator=g).to(DEV)
            widx = native.window_idxs(s, win, DEV)
            if k:
                pos = torch.arange(s)[:, None] // 2  # causal-ish: only compressed slots < t/2 valid
                cid = torch.rand(s, clen, generator=g).argsort(-1)[:, :k]  # unique per query, like top-k
                cid = torch.where(cid <= pos, cid + s, -1).int().to(DEV)
                idx = torch.cat([widx, cid], -1)
            else:
                cid, idx = None, widx
            ref = native.sparse_attn_f32(q[None], kv[None], sink, idx[None], d ** -0.5, 64)[0]
            for dm in (0, 64, 1024):  # gather path, mixed, dense path
                got = native.sparse_attn_win(q, kv, sink, cid, win, d ** -0.5, chunk=48, dense_max=dm)
                self.assertLess(((got - ref).norm() / ref.norm()).item(), 1e-5, (s, clen, k, dm))

    def test_aq8_switch(self):
        x = torch.randn(4, 64, device=DEV)
        prev = native.FAKE_QUANT
        try:
            native.FAKE_QUANT = False
            self.assertIs(native.aq8(x), x)
            native.FAKE_QUANT = True
            self.assertEqual(native.aq8(x).dtype, torch.bfloat16)
        finally:
            native.FAKE_QUANT = prev

    @NEED_CUDA
    def test_grouped_wo_a_matches_einsum(self):
        s, gr, d, r = 7, 8, 64, 16
        o = torch.randn(s, gr, d, device=DEV)
        wo = torch.randn(gr, r, d, device=DEV).to(torch.bfloat16)
        ref = torch.einsum("sgd,grd->sgr", o.double(), wo.double())
        hi, lo = native.split16(o.transpose(0, 1))
        wt = wo.transpose(1, 2)
        got = (torch.bmm(hi, wt, out_dtype=torch.float32) + torch.bmm(lo, wt, out_dtype=torch.float32)).transpose(0, 1)
        self.assertLess(((got.double() - ref).norm() / ref.norm()).item(), 1e-5)

    def test_shared_noise_draw(self):
        hz = native.Harness.__new__(native.Harness)  # only the attributes _variant_weights needs
        hz.dev = torch.device(DEV)
        hz.act_dtype = torch.float32
        w13 = torch.randn(8, 32, device=DEV).to(torch.bfloat16)
        w2 = torch.randn(32, 4, device=DEV).to(torch.bfloat16)
        va = native.Variant("L3-GU-r0.1", {3: ("noise", "GU", 0.1)}, noise_key="L3-GU")
        vb = native.Variant("L3-GU-r0.2", {3: ("noise", "GU", 0.2)}, noise_key="L3-GU")
        draws = {}
        a13, a2, _ = hz._variant_weights(3, 5, va, w13, w2, va.mods[3], draws)
        b13, b2, _ = hz._variant_weights(3, 5, vb, w13, w2, vb.mods[3], draws)
        self.assertEqual(len(draws), 1)  # one draw, reused
        da, db = a13 - w13.float(), b13 - w13.float()
        self.assertTrue(torch.allclose(db, 2 * da, rtol=1e-4, atol=1e-6))  # same direction, scaled by r
        self.assertEqual(a13.dtype, torch.float32)
        self.assertTrue(torch.equal(a2, w2))  # GU leaves w2 untouched
        z13, _, _ = hz._variant_weights(3, 5, native.Variant("z", {3: ("noise", "GU", 0.0)}), w13, w2,
                                        ("noise", "GU", 0.0))
        self.assertTrue(torch.equal(z13, w13.float()))  # r = 0 is exactly the native weight

    def test_joint_per_group_r(self):
        """group ALL with r = (r_gu, r_d) == the GU-only draw at r_gu and the D-only draw at r_d (same key/seed)."""
        hz = native.Harness.__new__(native.Harness)
        hz.dev = torch.device(DEV)
        hz.act_dtype = torch.float32
        w13 = torch.randn(8, 32, device=DEV).to(torch.bfloat16)
        w2 = torch.randn(32, 4, device=DEV).to(torch.bfloat16)
        mk = lambda mod: native.Variant("j", {7: mod}, seed=3, noise_key="joint")
        j13, j2, _ = hz._variant_weights(7, 2, mk(("noise", "ALL", (0.1, 0.3))), w13, w2, ("noise", "ALL", (0.1, 0.3)))
        g13, _, _ = hz._variant_weights(7, 2, mk(("noise", "GU", 0.1)), w13, w2, ("noise", "GU", 0.1))
        _, d2, _ = hz._variant_weights(7, 2, mk(("noise", "D", 0.3)), w13, w2, ("noise", "D", 0.3))
        self.assertTrue(torch.equal(j13, g13))
        self.assertTrue(torch.equal(j2, d2))
        a13, a2, _ = hz._variant_weights(7, 2, mk(("noise", "ALL", 0.2)), w13, w2, ("noise", "ALL", 0.2))
        t13, t2, _ = hz._variant_weights(7, 2, mk(("noise", "ALL", (0.2, 0.2))), w13, w2, ("noise", "ALL", (0.2, 0.2)))
        self.assertTrue(torch.equal(a13, t13) and torch.equal(a2, t2))  # scalar r == (r, r)


if __name__ == "__main__":
    unittest.main()
