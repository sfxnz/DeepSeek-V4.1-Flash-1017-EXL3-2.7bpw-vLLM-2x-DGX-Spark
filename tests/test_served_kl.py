#!/usr/bin/env python3
"""tools/served_kl.py on synthetic rows (CPU only, no serve)."""
from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import served_kl as sk  # noqa: E402


def _norm(logits: list[float]) -> list[float]:
    m = max(logits)
    lse = m + math.log(sum(math.exp(x - m) for x in logits))
    return [x - lse for x in logits]


class KlTopk(unittest.TestCase):
    def test_identical_is_zero(self):
        lps = _norm([3.0, 2.0, 1.0, 0.0, -1.0])
        ids = [7, 3, 9, 1, 4]
        q = dict(zip(ids, lps))
        kl, miss = sk.kl_topk(ids, lps, q, 3)
        self.assertAlmostEqual(kl, 0.0, places=9)
        self.assertEqual(miss, 0)

    def test_matches_bucketed_kl(self):
        p = _norm([2.0, 1.0, 0.0, -1.0])
        qv = _norm([1.0, 1.5, 0.0, -0.5])
        ids = [0, 1, 2, 3]
        kl, miss = sk.kl_topk(ids, p, dict(zip(ids, qv)), 2)
        pt = 1 - math.exp(p[0]) - math.exp(p[1])
        qt = 1 - math.exp(qv[0]) - math.exp(qv[1])
        want = sum(math.exp(a) * (a - b) for a, b in zip(p[:2], qv[:2])) + pt * math.log(pt / qt)
        self.assertAlmostEqual(kl, want, places=9)
        self.assertEqual(miss, 0)

    def test_missing_uses_served_kth_not_actual_token(self):
        p = _norm([2.0, 1.0, 0.0])
        q = {0: math.log(0.6), 5: math.log(0.3), 9: math.log(1e-6)}  # 9 = actual token, rank > k
        kl, miss = sk.kl_topk([0, 1, 2], p, q, 2)
        self.assertEqual(miss, 1)
        lq = [math.log(0.6), math.log(0.3)]  # token 1 imputed at the served 2nd logprob
        pt, qt = 1 - math.exp(p[0]) - math.exp(p[1]), 1 - 0.6 - 0.3
        want = sum(math.exp(a) * (a - b) for a, b in zip(p[:2], lq)) + pt * math.log(pt / qt)
        self.assertAlmostEqual(kl, max(want, 0.0), places=9)


class Rows(unittest.TestCase):
    def test_reader_and_alignment(self):
        # One row: tokens 0..5, scored positions 2 and 3 (predicting tokens 3 and 4).
        tokens = [0, 11, 12, 13, 14, 15]
        pos = [2, 3]
        k = 3
        ids = [[13, 1, 2], [9, 14, 2]]
        lps = [_norm([2.0, 0.0, -1.0]), _norm([1.0, 0.5, -1.0])]
        nll = [-lps[0][0], -lps[1][1]]
        tensors = {"r/pos": ("I32", [2], pos), "r/ids": ("I32", [2, k], sum(ids, [])),
                   "r/logprobs": ("F32", [2, k], sum(lps, [])), "r/nll": ("F32", [2], nll)}
        hdr, blob = {}, b""
        for name, (dt, shape, vals) in tensors.items():
            raw = struct.pack(f"<{len(vals)}{'i' if dt == 'I32' else 'f'}", *vals)
            hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [len(blob), len(blob) + len(raw)]}
            blob += raw
        h = json.dumps(hdr).encode()
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            f.write(struct.pack("<Q", len(h)) + h + blob)
        nat = sk.TopK(f.name)
        self.assertEqual(nat.rows(), ["r"])
        row = nat.row("r")
        self.assertEqual(row["pos"], pos)
        plp = [None] * len(tokens)
        for j, p in enumerate(pos):
            plp[p + 1] = dict(zip(row["ids"][j], row["lps"][j]))
        s = sk.score_row({"tokens": tokens}, row, plp, k)
        self.assertEqual(s["n"], 2)
        self.assertEqual(s["top1"], 2)
        self.assertAlmostEqual(s["dnll"], 0.0, places=5)
        self.assertAlmostEqual(s["kl"], 0.0, places=5)
        summ = sk.summarize([s])
        self.assertEqual(summ["n"], 2)
        self.assertAlmostEqual(summ["nll_native"], sum(nll) / 2, places=5)
        Path(f.name).unlink()


if __name__ == "__main__":
    unittest.main()
