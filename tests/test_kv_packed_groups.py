"""kv_packed_groups (DSV41_KV_PACKED_GROUPS): PR #60996 on the e15 planner, source level (host only).

The behaviour (50 groups, 46,080-byte blocks, logged KV tokens) is the image's own planner run on
CPU with the rewrite applied; see the kv-pack report and perf2/memory/kvmodel/. Here: the four PR
hunks rebuild the PR text from the pinned e15 region, the anchors refuse drift, the wiring.
"""

from __future__ import annotations

import ast
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker" / "patch"
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(PATCH))
import kv_packed_groups as kvp  # noqa: E402

MARKER = "# --- excerpt ---\n"


def pin(name: str) -> str:
    text = (FIX / name).read_text()
    return text[text.index(MARKER) + len(MARKER):]


E15 = pin("kv_cache_utils_packing_e15.pin.py")
PR = pin("kv_cache_utils_packing_pr60996.pin.py")


def rewrite(path, fn):
    """sitecustomize._rewrite's contract: 'rewritten' / 'already applied'."""
    path = Path(path)
    src = path.read_text()
    out = fn(src)
    if out == src:
        return "already applied"
    path.write_text(out)
    return "rewritten"


class HunkTests(unittest.TestCase):
    def test_pr_hunks_rebuild_the_upstream_pr_region(self) -> None:
        self.assertEqual(kvp.apply_hunks(E15, kvp.PR_HUNKS), PR)

    def test_patch_adds_only_the_planner_log_line(self) -> None:
        out = kvp.patch_source(E15)
        self.assertEqual(out.replace(kvp.LOG_NEW, kvp.LOG_OLD), PR)
        self.assertEqual(out.count(kvp.MARK), kvp.LOG_NEW.count(kvp.MARK))  # only in the added lines
        self.assertIn(kvp.LOG_PLANNER, out)
        ast.parse(out)
        self.assertEqual(kvp.patch_source(out), out)  # idempotent

    def test_the_old_split_is_gone_and_the_new_search_is_in(self) -> None:
        out = kvp.patch_source(E15)
        self.assertNotIn("_approximate_gcd", out)
        self.assertIn("_max_memory_usage_bytes_from_groups(vllm_config, build_groups(repeats))", out)
        self.assertIn("range(1, max_repeats_per_group + 1)", out)

    def test_anchor_drift_and_duplicates_are_refused(self) -> None:
        for old, _, label in kvp.PR_HUNKS:
            with self.assertRaises(ValueError, msg=label):
                kvp.patch_source(E15.replace(old, old.replace("    ", "  ", 1), 1))
        with self.assertRaises(ValueError):
            kvp.patch_source(E15 + kvp.PR_HUNKS[1][0])

    def test_fixture_regions_are_python(self) -> None:
        ast.parse(E15)
        ast.parse(PR)
        self.assertTrue(E15.startswith("def _approximate_gcd("))
        self.assertTrue(PR.startswith("def _get_packed_kv_cache_groups("))


class InstallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "v1/core").mkdir(parents=True)
        self.file = self.root / kvp.REL
        self.file.write_text("# head\n" + E15 + "# tail\n")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_rewrites_once_and_prints_engaged(self) -> None:
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(kvp.install(self.root, rewrite, modules={}), "kv_cache_utils rewritten")
            self.assertEqual(kvp.install(self.root, rewrite, modules={kvp.MODULE: object()}),
                             "kv_cache_utils already applied")
        self.assertEqual(err.getvalue().count(kvp.LOG_ENGAGED), 2)
        self.assertIn(kvp.MARK, self.file.read_text())

    def test_module_imported_before_the_rewrite_fails(self) -> None:
        with redirect_stderr(io.StringIO()) as err, self.assertRaises(RuntimeError):
            kvp.install(self.root, rewrite, modules={kvp.MODULE: object()})
        self.assertIn(kvp.LOG_DISARMED, err.getvalue())

    def test_anchor_miss_fails_loud(self) -> None:
        self.file.write_text("def _get_packed_kv_cache_groups(): pass\n")
        with redirect_stderr(io.StringIO()) as err, self.assertRaises(ValueError):
            kvp.install(self.root, rewrite, modules={})
        self.assertIn(kvp.LOG_DISARMED, err.getvalue())

    def test_knob_off_refuses_a_rewritten_file(self) -> None:
        kvp.refuse_stale(self.root)  # stock file: fine
        with redirect_stderr(io.StringIO()):
            kvp.install(self.root, rewrite, modules={})
        with self.assertRaises(RuntimeError):
            kvp.refuse_stale(self.root)
        kvp.refuse_stale(self.root / "missing")  # no file: nothing to refuse


class WiringTests(unittest.TestCase):
    def test_env(self) -> None:
        self.assertFalse(kvp.enabled({}))
        self.assertFalse(kvp.enabled({"DSV41_KV_PACKED_GROUPS": "0"}))
        self.assertTrue(kvp.enabled({"DSV41_KV_PACKED_GROUPS": "1"}))

    def test_sitecustomize_rewrites_before_anything_imports_vllm(self) -> None:
        site = (PATCH / "sitecustomize.py").read_text()
        call = site.index('_patch("kv_packed_groups", _p_kv_packed_groups, required=True)')
        for needle in ("from vllm", "import vllm", "load_general_plugins"):
            self.assertLess(call, site.index(needle), needle)

    def test_forwarded_default_on_and_audited(self) -> None:
        run = (ROOT / "run.sh").read_text()
        self.assertIn("  DSV41_KV_PACKED_GROUPS=\n", run)
        self.assertIn('DSV41_KV_PACKED_GROUPS="${DSV41_KV_PACKED_GROUPS:-1}"\n', run)
        sys.path.insert(0, str(ROOT / "tools"))
        import engagement_audit as ea

        self.assertEqual(ea.PATCHES["kv_packed_groups.py"], {"DSV41_KV_PACKED_GROUPS": "1"})

    def test_top_level_imports_are_stdlib_only(self) -> None:
        tree = ast.parse((PATCH / "kv_packed_groups.py").read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {a.name for n in top if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in top if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"__future__", "os", "sys"})


if __name__ == "__main__":
    unittest.main()
