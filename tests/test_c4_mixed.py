"""tools/c4_mixed.py arithmetic: ITL windows, stalls, identity verdicts (no server)."""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import c4_mixed as cm  # noqa: E402


class WindowTests(unittest.TestCase):
    def test_intervals_and_the_open_stall(self) -> None:
        tl = {"s1": [(1.0, 3), (1.1, 2), (1.2, 4), (4.2, 3)], "s2": [(1.05, 1), (1.15, 2)]}
        st = cm.window_stats(tl, 1.0, 5.0)
        # s1: 0.1, 0.1, 3.0 (its chunk at t0 is the start, not a sample); s2: 0.1
        self.assertEqual(st["chunks"], 4)
        self.assertAlmostEqual(st["itl_p50_s"], 0.1)
        self.assertAlmostEqual(st["itl_p99_s"], 3.0)
        # s2 has been silent since 1.15: a 3.85 s stall at the window end beats s1's 3.0 s gap
        self.assertAlmostEqual(st["max_stall_s"], 3.85)
        self.assertAlmostEqual(st["decode_tok_s_per_stream"]["s1"], 9 / 4.0)
        self.assertAlmostEqual(st["decode_tok_s_per_stream_min"], 3 / 4.0)  # s2: 1 + 2 tokens in (1, 5]

    def test_a_window_without_chunks_is_one_long_stall(self) -> None:
        st = cm.window_stats({"s1": [(0.5, 2)]}, 1.0, 9.0)
        self.assertEqual(st["chunks"], 0)
        self.assertIsNone(st["itl_p50_s"])
        self.assertAlmostEqual(st["max_stall_s"], 8.5)

    def test_percentile_is_nearest_rank(self) -> None:
        self.assertIsNone(cm.percentile([], 50))
        self.assertEqual(cm.percentile([3, 1, 2], 50), 2)
        self.assertEqual(cm.percentile(list(range(1, 101)), 99), 99)
        self.assertEqual(cm.percentile([5], 99), 5)


class IdentityTests(unittest.TestCase):
    def rec(self, tokens, top2=None, logprobs=None):
        return {"tokens": tokens, "top2": top2 or [[-0.1, -3.0]] * len(tokens),
                "logprobs": logprobs or [-0.1] * len(tokens)}

    def test_verdicts(self) -> None:
        a = {"same": self.rec(list("abcd")), "short": self.rec(list("abcdef")), "tie": self.rec(list("abcx")),
             "far": self.rec(list("abcx")), "only_a": self.rec(list("a"))}
        b = {"same": self.rec(list("abcd")), "short": self.rec(list("abc")),
             "tie": self.rec(list("abcy"), top2=[[-0.1, -3.0]] * 3 + [[-0.70, -0.75]]), "far": self.rec(list("abcy"))}
        rows = cm.compare_requests(a, b, tie=0.25)
        self.assertEqual({k: r["verdict"] for k, r in rows.items()},
                         {"same": "identical", "short": "prefix", "tie": "near_tie", "far": "diverged"})
        self.assertEqual(rows["far"]["first_divergence"], 3)
        self.assertAlmostEqual(rows["tie"]["margin"][1], 0.05)
        self.assertEqual(rows["same"]["max_abs_dlogprob"], 0.0)

    def test_compare_mode_on_solo_files(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            base = {"mode": "solo", "requests": {"s1.0": self.rec(list("abc")), "p65536": self.rec(list("xy"))}}
            moved = json.loads(json.dumps(base))
            moved["requests"]["s1.0"]["logprobs"][1] = -0.1000001
            (p / "a.json").write_text(json.dumps(base))
            (p / "b.json").write_text(json.dumps(base))
            (p / "c.json").write_text(json.dumps(moved))
            ok = cm.compare_mode(SimpleNamespace(files=[p / "a.json", p / "b.json"], tie=0.25))
            self.assertTrue(ok["solo_exact"])
            off = cm.compare_mode(SimpleNamespace(files=[p / "a.json", p / "c.json"], tie=0.25))
            self.assertFalse(off["solo_exact"])  # same tokens, a logprob moved: not exact
            self.assertEqual(off["verdicts"], {"identical": 2})


class HygieneTests(unittest.TestCase):
    def test_stdlib_only_at_top_level(self) -> None:
        tree = ast.parse((ROOT / "tools" / "c4_mixed.py").read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {a.name for n in top if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in top if isinstance(n, ast.ImportFrom)}
        self.assertLessEqual(mods, {"__future__", "argparse", "json", "math", "statistics", "subprocess", "sys",
                                    "threading", "time", "urllib.request", "pathlib"})

    def test_every_request_carries_a_cache_salt_and_greedy(self) -> None:
        src = (ROOT / "tools" / "c4_mixed.py").read_text()
        self.assertIn('"cache_salt": salt', src)
        self.assertIn('"temperature": 0', src)
        self.assertIn('"top_logprobs": 2', src)


if __name__ == "__main__":
    unittest.main()
