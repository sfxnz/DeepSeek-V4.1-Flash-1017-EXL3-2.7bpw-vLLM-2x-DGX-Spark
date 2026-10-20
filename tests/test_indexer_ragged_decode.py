"""indexer_ragged_decode: vLLM 0.32rc1's needs_padded_path guard on e15's sparse indexer (host only)."""

from __future__ import annotations

import ast
import re
import sys
import textwrap
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker" / "patch"
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(PATCH))
import indexer_ragged_decode as ird  # noqa: E402

MARKER = "# --- excerpt ---\n"


def pin(name: str) -> str:
    text = (FIX / name).read_text()
    return text[text.index(MARKER) + len(MARKER):]


E15 = pin("sparse_attn_indexer_decode_e15.pin.py")
V032 = pin("sparse_attn_indexer_ragged_032rc1.pin.txt")


def guard_expr(src: str) -> str:
    m = re.search(r"needs_padded_path = (.*?\n        \))", src, re.S)
    return " ".join(m.group(1).split())


class RewriteTests(unittest.TestCase):
    def test_three_anchors_once_each_and_idempotent(self) -> None:
        out = ird.patch_source(E15)
        self.assertEqual(out.count("needs_padded_path"), 3)
        self.assertEqual(out.count("requires_padding"), 1)  # only inside the guard now
        self.assertEqual(out.count(ird.MARK), 1)
        self.assertEqual(ird.patch_source(out), out)
        ast.parse("def f():\n" + out)

    def test_guard_is_the_032rc1_expression(self) -> None:
        self.assertEqual(guard_expr(ird.patch_source(E15)), guard_expr(V032))
        self.assertIn("elif needs_padded_path:", V032)
        self.assertIn("if needs_padded_path:\n            # if padded, we need to unpack", V032)

    def test_only_the_anchors_change(self) -> None:
        out = ird.patch_source(E15)
        for old, new, _ in ird.HUNKS:
            out = out.replace(new, old)
        self.assertEqual(out, E15)

    def test_drift_is_refused(self) -> None:
        for old, _, label in ird.HUNKS:
            with self.assertRaises(ValueError, msg=label):
                ird.patch_source(E15.replace(old, "", 1))
            with self.assertRaises(ValueError, msg=label):
                ird.patch_source(E15 + old)

    def test_guard_semantics(self) -> None:
        body = textwrap.dedent(ird.GUARD_NEW).split("if num_decode_tokens == 0:")[0]
        for requires_padding, tokens, decodes, want in (
            (False, 8, 2, False),   # uniform 4+4: the stock uniform reshape
            (False, 16, 4, False),  # c=4, k=3
            (False, 8, 6, True),    # upstream SM120 TP=2 case: would crash the reshape
            (False, 9, 4, True),
            (True, 8, 2, True),     # ragged lengths: already padded
            (False, 0, 1, False),   # no decode tokens: the stock first branch
        ):
            ns = {"decode_metadata": types.SimpleNamespace(requires_padding=requires_padding,
                                                           decode_lens=types.SimpleNamespace(shape=(decodes,))),
                  "num_decode_tokens": tokens}
            exec(compile(body, "guard", "exec"), ns)
            self.assertIs(bool(ns["needs_padded_path"]), want, (requires_padding, tokens, decodes))


class WiringTests(unittest.TestCase):
    def test_sitecustomize_rewrites_the_indexer_file_required(self) -> None:
        site = (PATCH / "sitecustomize.py").read_text()
        self.assertIn('_patch("indexer_ragged_decode", _p_indexer_ragged_decode, required=True)', site)
        self.assertIn("indexer_ragged_decode.REL", site)
        self.assertEqual(ird.REL, "model_executor/layers/sparse_attn_indexer.py")

    def test_top_level_imports_are_stdlib_only(self) -> None:
        tree = ast.parse((PATCH / "indexer_ragged_decode.py").read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertEqual({n.module for n in top if isinstance(n, ast.ImportFrom)}, {"__future__"})


if __name__ == "__main__":
    unittest.main()
