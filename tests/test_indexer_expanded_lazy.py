"""indexer_expanded_lazy (DSV41_INDEXER_EXPANDED_TABLE_LAZY): who reads the buffer, the rewrite, the property.

Host only. The e15 indexer module is pinned whole (tests/fixtures/mla_indexer_e15.pin.py), so the
consumer proof the patch rests on is checked here, not just stated.
"""

from __future__ import annotations

import ast
import io
import sys
import tempfile
import textwrap
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker" / "patch"
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(PATCH))
import indexer_expanded_lazy as iel  # noqa: E402

E15 = (FIX / "mla_indexer_e15.pin.py").read_text()
NEXT_N = (FIX / "deep_gemm_native_next_n_e15.pin.py").read_text()
BUILDER = "DeepseekV32IndexerMetadataBuilder"


def node(tree, kind, name):
    return next(n for n in ast.walk(tree) if isinstance(n, kind) and n.name == name)


def rewrite(path, fn):
    path = Path(path)
    src = path.read_text()
    out = fn(src)
    if out == src:
        return "already applied"
    path.write_text(out)
    return "rewritten"


class ConsumerProofTests(unittest.TestCase):
    """Every reader of expanded_block_table_buffer in e15 sits in the flattened decode branch."""

    def setUp(self) -> None:
        self.tree = ast.parse(E15)
        self.cls = node(self.tree, ast.ClassDef, BUILDER)

    def test_one_allocation_and_five_reads_all_in_the_not_use_native_branch(self) -> None:
        uses = [n for n in ast.walk(self.tree) if isinstance(n, ast.Attribute) and n.attr == "expanded_block_table_buffer"]
        self.assertEqual(len(uses), 6)
        init = node(self.cls, ast.FunctionDef, "__init__")
        prep = node(self.cls, ast.FunctionDef, "_prepare_decode_tensors")
        branch = next(n for n in prep.body if isinstance(n, ast.If) and ast.unparse(n.test) == "not use_native")
        stores = [u for u in uses if isinstance(u.ctx, ast.Store)]
        self.assertEqual(len(stores), 1)
        self.assertTrue(init.lineno <= stores[0].lineno <= init.end_lineno)
        for u in uses:
            if u is not stores[0]:
                self.assertTrue(branch.body[0].lineno <= u.lineno <= branch.body[-1].end_lineno, u.lineno)

    def test_no_subclass_and_build_is_the_only_caller(self) -> None:
        bases = [ast.unparse(b) for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef) for b in n.bases]
        self.assertNotIn(BUILDER, bases)
        callers = []
        for fn in (n for n in ast.walk(self.cls) if isinstance(n, ast.FunctionDef)):
            for call in (c for c in ast.walk(fn) if isinstance(c, ast.Call)):
                if isinstance(call.func, ast.Attribute) and call.func.attr == "_prepare_decode_tensors":
                    callers.append(fn.name)
        self.assertEqual(callers, ["build"])
        self.assertNotIn("__dict__", ast.unparse(self.cls))
        self.assertNotIn("vars(", ast.unparse(self.cls))

    def test_native_path_holds_on_sm120(self) -> None:
        build = node(self.cls, ast.FunctionDef, "build")
        src = ast.unparse(build)
        self.assertIn("use_native = not (self.use_flattening or self.supports_varlen) and max_decode_len <= next_n "
                      "and step_next_n_ok", src)
        self.assertIn("step_next_n_ok = max_decode_len <= 1 or _supports_native_decode(max_decode_len)", src)
        self.assertIn("decode_threshold=self.decode_threshold", src)  # decodes are <= next_n by construction
        native = ast.unparse(node(self.tree, ast.FunctionDef, "_supports_native_decode"))
        self.assertIn("if current_platform.is_device_capability_family(120):\n        return native_next_n_supported(next_n)",
                      native)
        flat = ast.unparse(node(self.tree, ast.FunctionDef, "_use_flattening"))
        self.assertIn("not _supports_native_decode(next_n) or (speculative_config is not None and "
                      "speculative_config.enable_adaptive_verification and _supports_flattened_device_query_lens())", flat)
        self.assertIn("is_device_capability_family(90)", ast.unparse(node(self.tree, ast.FunctionDef,
                                                                          "_supports_flattened_device_query_lens")))
        self.assertIn("is_device_capability_family(100)", ast.unparse(node(self.tree, ast.FunctionDef,
                                                                           "_supports_varlen_paged_mqa_logits")))
        self.assertIn("if current_platform.is_device_capability_family(90):\n        return next_n in (1, 2, 4)\n"
                      "    return True", NEXT_N)


class RewriteTests(unittest.TestCase):
    def test_both_anchors_once_and_idempotent(self) -> None:
        out = iel.patch_source(E15)
        self.assertNotEqual(out, E15)
        self.assertEqual(iel.patch_source(out), out)
        tree = ast.parse(out)
        cls = node(tree, ast.ClassDef, BUILDER)
        prop = node(cls, ast.FunctionDef, "expanded_block_table_buffer")
        self.assertEqual([ast.unparse(d) for d in prop.decorator_list], ["property"])
        init = ast.unparse(node(cls, ast.FunctionDef, "__init__"))
        self.assertNotIn("self.expanded_block_table_buffer =", init)
        self.assertIn("self._dsv41_expanded_shape = (scheduler_config.max_num_batched_tokens, block_table_width)", init)
        self.assertIn("self._dsv41_expanded_block_table = None", init)

    def test_lazy_buffer_is_the_stock_one(self) -> None:
        stock = ast.unparse(ast.parse(textwrap.dedent(iel.ALLOC_OLD)))
        self.assertIn("torch.zeros((scheduler_config.max_num_batched_tokens, block_table_width), "
                      "dtype=torch.int32, device=self.device)", stock)
        self.assertIn("torch.zeros((rows, cols), dtype=torch.int32, device=self.device)", iel.PROP_NEW)

    def test_only_the_anchors_change(self) -> None:
        out = iel.patch_source(E15)
        for old, new, _ in iel.HUNKS:
            out = out.replace(new, old)
        self.assertEqual(out, E15)

    def test_drift_is_refused(self) -> None:
        for old, _, label in iel.HUNKS:
            with self.assertRaises(ValueError, msg=label):
                iel.patch_source(E15.replace(old, "", 1))
            with self.assertRaises(ValueError, msg=label):
                iel.patch_source(E15 + old)


class PropertyTests(unittest.TestCase):
    def make(self):
        calls = {"zeros": [], "warning": []}
        torch = types.SimpleNamespace(
            int32="int32", Tensor=object,
            zeros=lambda shape, dtype, device: calls["zeros"].append((shape, dtype, device)) or object())
        logger = types.SimpleNamespace(warning=lambda msg, *a: calls["warning"].append(msg % a))
        body = iel.PROP_NEW[iel.PROP_NEW.index("    @property"):]
        ns = {"torch": torch, "logger": logger}
        exec(compile("class B:\n" + body, "prop", "exec"), ns)
        b = ns["B"]()
        b._dsv41_expanded_shape, b._dsv41_expanded_block_table = (8192, 3907), None
        b.device, b.layer_names = "cuda:0", ["model.layers.2.attn.indexer.k_cache"]
        return b, calls

    def test_allocates_once_on_first_read_and_says_so(self) -> None:
        b, calls = self.make()
        self.assertEqual(calls["zeros"], [])  # nothing until a reader asks
        first = b.expanded_block_table_buffer
        self.assertIs(b.expanded_block_table_buffer, first)
        self.assertEqual(calls["zeros"], [((8192, 3907), "int32", "cuda:0")])
        self.assertEqual(len(calls["warning"]), 1)
        self.assertTrue(calls["warning"][0].startswith(iel.LOG_ALLOCATED))
        self.assertIn("8192 x 3907 int32, 122.1 MiB, model.layers.2.attn.indexer.k_cache", calls["warning"][0])


class InstallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "v1/attention/backends/mla").mkdir(parents=True)
        (self.root / iel.REL).write_text(E15)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_engaged_once_rewritten_then_already_applied(self) -> None:
        with redirect_stderr(io.StringIO()) as err:
            self.assertEqual(iel.install(self.root, rewrite, modules={}), "indexer rewritten")
            self.assertEqual(iel.install(self.root, rewrite, modules={iel.MODULE: object()}), "indexer already applied")
        self.assertEqual(err.getvalue().count(iel.LOG_ENGAGED), 2)

    def test_imported_before_rewrite_fails_and_stale_file_is_refused(self) -> None:
        with redirect_stderr(io.StringIO()) as err, self.assertRaises(RuntimeError):
            iel.install(self.root, rewrite, modules={iel.MODULE: object()})
        self.assertIn(iel.LOG_DISARMED[0], err.getvalue())
        with self.assertRaises(RuntimeError):
            iel.refuse_stale(self.root)  # rewritten above, knob now off
        (self.root / iel.REL).write_text(E15)
        iel.refuse_stale(self.root)


class WiringTests(unittest.TestCase):
    def test_env(self) -> None:
        self.assertFalse(iel.enabled({}))
        self.assertFalse(iel.enabled({"DSV41_INDEXER_EXPANDED_TABLE_LAZY": "0"}))
        self.assertTrue(iel.enabled({"DSV41_INDEXER_EXPANDED_TABLE_LAZY": "1"}))

    def test_sitecustomize_rewrites_before_the_indexer_is_imported(self) -> None:
        site = (PATCH / "sitecustomize.py").read_text()
        call = site.index('_patch("indexer_expanded_lazy", _p_indexer_expanded_lazy, required=True)')
        self.assertLess(call, site.index("from vllm.v1.attention.backends.mla.indexer import"))
        for needle in ("from vllm", "import vllm", "load_general_plugins"):
            self.assertLess(call, site.index(needle), needle)

    def test_forwarded_default_on_and_audited(self) -> None:
        run = (ROOT / "run.sh").read_text()
        self.assertIn("  DSV41_INDEXER_EXPANDED_TABLE_LAZY=\n", run)
        self.assertIn('DSV41_INDEXER_EXPANDED_TABLE_LAZY="${DSV41_INDEXER_EXPANDED_TABLE_LAZY:-1}"\n', run)
        sys.path.insert(0, str(ROOT / "tools"))
        import engagement_audit as ea

        self.assertEqual(ea.PATCHES["indexer_expanded_lazy.py"], {"DSV41_INDEXER_EXPANDED_TABLE_LAZY": "1"})
        found = ea.markers(PATCH / "indexer_expanded_lazy.py")
        self.assertIn(iel.LOG_ALLOCATED, found["LOG_DISARMED"])  # the saving not holding is an audit line

    def test_top_level_imports_are_stdlib_only(self) -> None:
        tree = ast.parse((PATCH / "indexer_expanded_lazy.py").read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {a.name for n in top if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in top if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"__future__", "os", "sys"})


if __name__ == "__main__":
    unittest.main()
