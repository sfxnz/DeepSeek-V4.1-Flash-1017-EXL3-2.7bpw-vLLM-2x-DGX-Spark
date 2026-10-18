"""HessianV1 (decision B-3) on CPU with a toy layer: files match the formula computed directly in fp64.
Run in dsv41-quant151 (torch + safetensors), no GPU:  python3 -m unittest tools/calib/tests/test_hessian_v1.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from hooks import HessianV1, read_meta  # noqa: E402
from safetensors import safe_open  # noqa: E402


def toy(T=300, E=6, k=2, dim=16, inter=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, dim, generator=g).to(torch.bfloat16)
    scores = torch.rand(T, E, generator=g)
    scores[:, E - 1] = -1  # expert E-1 never routed
    w, idx = scores.topk(k, dim=-1)
    w = w / w.sum(-1, keepdim=True) * 1.5
    act = {e: torch.randn(T, inter, generator=g) for e in range(E)}  # fake "native expert" activations
    return x, w, idx, act


def drive(hook, x, w, idx, act, layer=3, rows=(100, 120, 80)):
    E = 6
    hz = SimpleNamespace(cfg=SimpleNamespace(n_routed_experts=E), dev=torch.device("cpu"))
    hook.begin(hz, [SimpleNamespace(n=n, variant=None) for n in rows])
    hook.routing(layer, idx, w, torch.zeros(x.size(0), dtype=torch.long))
    hook.layer_tokens(layer, x, None)
    prior = hook.prior_tokens(layer)
    flat = idx.flatten()
    order = torch.argsort(flat, stable=True)
    tok_all, slot_all = order // idx.size(1), order % idx.size(1)
    counts = torch.bincount(flat, minlength=E).tolist()
    cur = 0
    for e in range(E):
        hook.expert_prior(layer, e, act[e][prior])
        n = counts[e]
        if n:
            tok, slot = tok_all[cur:cur + n], slot_all[cur:cur + n]
            cur += n
            hook.expert(layer, e, tok, w[tok, slot], x[tok], act[e][tok], None, None)
    hook.end_layer(layer)
    hook.finish()
    return prior


class HessianV1Test(unittest.TestCase):
    def test_formula_full_and_holdout(self):
        x, w, idx, act = toy()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            hook = HessianV1([3], d / "h", wpow=2.0, m0_mult=256, prior_tokens=50, holdout_mod=3,
                             train_out=d / "ht", dump_factors=10, factors_out=d / "f", log=lambda *a: None)
            prior = drive(hook, x, w, idx, act)
            meta = json.loads((d / "h" / "layer_03" / "meta.json").read_text())
            self.assertTrue(meta["complete"])
            self.assertEqual(meta["format"], "dsv41-hessian-v1")
            self.assertEqual(meta["experts_written"], 6)
            row = torch.cat([torch.full((n,), i) for i, n in enumerate((100, 120, 80))])
            train = row % 3 != 0
            self.assertTrue(bool(train[prior].all()), "holdout: prior tokens from the train fold")
            for name, mask in (("h", torch.ones_like(train)), ("ht", train)):
                xd = x.double()
                H_all = (xd[mask].t() @ xd[mask]) / int(mask.sum())
                picks = w.double()[mask].pow(2)
                m0 = 256 * picks.mean().item()
                for e in range(6):
                    sel = (idx == e) & mask.unsqueeze(-1)
                    t, s = sel.nonzero(as_tuple=True)
                    g = w.double()[t, s].pow(2)
                    S = (xd[t] * g.unsqueeze(-1)).t() @ xd[t]
                    Sa = (act[e].double()[t] * g.unsqueeze(-1)).t() @ act[e].double()[t]
                    pa = act[e].double()[prior]
                    P = pa.t() @ pa / pa.size(0)
                    den = g.sum().item() + m0
                    want13, want2 = (S + m0 * H_all) / den, (Sa + m0 * P) / den
                    with safe_open(str(d / name / "layer_03" / f"expert_{e:03d}.safetensors"), "pt") as fh:
                        h13, h2 = fh.get_tensor("w13_in"), fh.get_tensor("w2_in")
                        md = read_meta(fh)
                    self.assertEqual(h13.dtype, torch.float32)
                    self.assertTrue(torch.equal(h13, h13.t()) and torch.equal(h2, h2.t()))
                    self.assertLess(((h13.double() - want13).norm() / want13.norm()).item(), 1e-6, (name, e))
                    self.assertLess(((h2.double() - want2).norm() / want2.norm()).item(), 1e-6, (name, e))
                    self.assertEqual(md["n"], int(t.numel()))
                    if e == 5:
                        self.assertEqual(md["n"], 0)
                        self.assertLess(((h13.double() - H_all).norm() / H_all.norm()).item(), 1e-6)
            with safe_open(str(d / "f" / "layer_03" / "expert_000.safetensors"), "pt") as fh:
                self.assertEqual(fh.get_tensor("x").shape[0], 10)


if __name__ == "__main__":
    unittest.main()
