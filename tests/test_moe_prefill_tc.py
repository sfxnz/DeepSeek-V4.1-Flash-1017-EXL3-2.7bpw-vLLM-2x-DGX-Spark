"""moe_prefill_tc (DSV41_MOE_PREFILL_TC): wiring, dispatch wrapper and kernel-source invariants, host only.

The GPU proof (tc vs the stock path and vs an fp32 reference, the A2 intermediate, edge routings, per-layer
time at m = 2048 / 4096 / 8192 for K 2/2, 2/3, 3/3, 3/4) is kernel_study/moe_prefill/bench_prefill_tc.py;
kernel_study/moe_prefill/host_compile.sh checks registers / spills of every kernel instance with nvcc.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker" / "patch"
sys.path.insert(0, str(PATCH))
import decode_levers as dl  # noqa: E402
import moe_prefill_tc as tc  # noqa: E402

CUH = (PATCH / "moe_prefill_tc.cuh").read_text()
CU = (PATCH / "moe_prefill_tc.cu").read_text()
PIN = ROOT / "tests" / "fixtures" / "exl3_fused_e15.pin.py"


def pinned_function(name: str) -> str:
    src = PIN.read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, fn)


def fake_exl3_module(tmp: str):
    """A module file holding the pinned e15 functions (so inspect.getsource works) and the globals they name."""
    path = Path(tmp) / "fake_vllm_exl3_exl3.py"
    path.write_text("from __future__ import annotations\n_FUSED_TEMP_CACHE = {}\nTEMP_ROWS_FUSED = 2048\n\n"
                    + pinned_function("build_exl3_fused_state") + "\n")
    spec = importlib.util.spec_from_file_location("fake_vllm_exl3_exl3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cuh_constant(name: str) -> int:
    m = re.search(rf"constexpr int {name} = ([^;]+);", CUH)
    return int(eval(m.group(1), {}, {k: cuh_constant(k) for k in re.findall(r"k[A-Z]\w*", m.group(1))}))  # noqa: S307


class WiringTests(unittest.TestCase):
    def test_env(self) -> None:
        self.assertFalse(tc.enabled({}))
        self.assertFalse(tc.enabled({"DSV41_MOE_PREFILL_TC": "0"}))
        self.assertTrue(tc.enabled({"DSV41_MOE_PREFILL_TC": "1"}))
        self.assertEqual(tc.min_rows({}), 9)  # every call above the native p2b cap (pm-time-small)
        self.assertEqual(tc.min_rows({"DSV41_MOE_PREFILL_TC_MIN_ROWS": "1024"}), 1024)
        # never below the native decode cap: <= 8 rows stay on p2b
        self.assertEqual(tc.min_rows({"DSV41_MOE_PREFILL_TC_MIN_ROWS": "1"}), 9)
        self.assertEqual(tc.verify_calls({}), 2)
        self.assertEqual(tc.verify_calls({"DSV41_MOE_PREFILL_TC_VERIFY": "0"}), 0)

    def test_decode_levers_step(self) -> None:
        self.assertIn('("moe-prefill-tc", _install_moe_prefill_tc)', (PATCH / "decode_levers.py").read_text())
        calls = []
        fake = types.ModuleType("moe_prefill_tc")
        fake.install = lambda: calls.append(1) or "wrapped"
        with mock.patch.dict(sys.modules, {"moe_prefill_tc": fake}), redirect_stdout(StringIO()) as out:
            dl._install_moe_prefill_tc({})
            dl._install_moe_prefill_tc({"DSV41_MOE_PREFILL_TC": "0"})
            self.assertEqual(calls, [])
            dl._install_moe_prefill_tc({"DSV41_MOE_PREFILL_TC": "1"})
        self.assertEqual(calls, [1])
        self.assertIn("moe prefill tc: wrapped", out.getvalue())

    def test_top_level_imports_are_stdlib_only(self) -> None:
        tree = ast.parse((PATCH / "moe_prefill_tc.py").read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {a.name for n in top if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in top if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"__future__", "hashlib", "importlib.machinery", "importlib.util", "os", "threading",
                                "pathlib"})

    def test_run_sh_forwards_the_lever_on_by_default(self) -> None:
        run = (ROOT / "run.sh").read_text()
        body = re.search(r"^FORWARD_ENVS=\((.*?)^\)", run, re.M | re.S).group(1).split()
        envs = dict(item.split("=", 1) for item in body)
        self.assertEqual(envs["DSV41_MOE_PREFILL_TC"], "")  # recipe.yaml default, rendered into run.sh
        self.assertIn('DSV41_MOE_PREFILL_TC="${DSV41_MOE_PREFILL_TC:-1}"', run)
        self.assertEqual(envs["DSV41_MOE_PREFILL_TC_MIN_ROWS"], "")
        self.assertEqual(envs["DSV41_MOE_PREFILL_TC_VERIFY"], "")

    def test_engagement_audit_expects_the_self_check_line_when_on(self) -> None:
        sys.path.insert(0, str(ROOT / "tools"))
        import engagement_audit as ea

        on, disarm = ea.expectations({"DSV41_MOE_PREFILL_TC": "1"})
        self.assertIn(("moe_prefill_tc.py", tc.LOG_ENGAGED), on)
        self.assertIn(tc.LOG_DISARMED, disarm)
        off, _ = ea.expectations({"DSV41_MOE_PREFILL_TC": "0"})
        self.assertNotIn(("moe_prefill_tc.py", tc.LOG_ENGAGED), off)

    def test_default_image_prebuilds_the_kernel(self) -> None:
        """docker/Dockerfile (from scratch) and docker/Dockerfile.e16 (on e15-kbits) both build e16-rows with the
        kernel prebuilt for the shipped sources, after the extensions it links against and before sitecustomize."""
        self.assertFalse((ROOT / "docker/Dockerfile.e16p").exists())
        self.assertEqual(str(tc.PREBUILT_ROOT), "/opt/dsv41-moe-prefill-tc")
        for name in ("docker/Dockerfile", "docker/Dockerfile.e16"):
            df = (ROOT / name).read_text()
            self.assertIn("moe_prefill_tc.py \\\n      --prebuild /opt/dsv41-moe-prefill-tc --arch 121a", df, name)
            self.assertIn("print('moe_prefill_tc prebuilt ok', tag)", df, name)
            self.assertLess(df.index("pip install --no-build-isolation --no-deps"), df.index("--prebuild /opt/dsv41-moe-prefill-tc"), name)
            self.assertLess(df.index("--prebuild /opt/dsv41-moe-prefill-tc"), df.index("sitecustomize.py"), name)
            label = re.search(r'LABEL dsv41\.recipe\.patches="([^"]+)"', df).group(1).split(",")
            self.assertEqual(label[-1], "moe_prefill_tc", name)

    def test_source_tag_covers_both_sources(self) -> None:
        tag = tc.source_tag()
        self.assertRegex(tag, r"^[0-9a-f]{12}$")
        self.assertEqual(tc.SOURCES, ("moe_prefill_tc.cu", "moe_prefill_tc.cuh"))


class KernelSourceTests(unittest.TestCase):
    def test_binding_argument_order_matches_the_python_side(self) -> None:
        sig = re.search(r"void moe_prefill_tc\((.*?)\)\n\{", CU, re.S).group(1)
        names = re.findall(r"(?:at::Tensor|int64_t|double) (\w+)", sig)
        self.assertEqual(names, ["x", "out", "token_sorted", "weight_sorted", "counts",
                                 "gate_t", "gate_suh", "gate_svh", "up_t", "up_suh", "up_svh",
                                 "down_t", "down_suh", "down_svh", "a2", "items1", "items2", "ctrl",
                                 "k_gu", "k_d", "limit"])
        self.assertEqual(tc.PTR_NAMES, ("gate_trellis", "gate_suh", "gate_svh", "up_trellis", "up_suh", "up_svh",
                                        "down_trellis", "down_suh", "down_svh"))
        self.assertIn('m.attr("ABI") = 1;', CU)

    def test_shared_memory_fits_sm12x(self) -> None:
        mt1, mt2, pitch = cuh_constant("kMt1"), cuh_constant("kMt2"), cuh_constant("kCpitch")
        gu_a, dn_a = 2 * 2 * mt1 * 256, 2 * mt2 * 256
        for k in (2, 3, 4):
            gu = gu_a + 2 * (2 * 4 * 16 * 32 * k) + mt1 * 4 + 16
            dn = dn_a + 2 * (8 * 8 * 32 * k) + mt2 * 4 + mt2 * 2 + 16
            self.assertLessEqual(gu, 101376, k)
            self.assertLessEqual(dn, 101376, k)
            # the fp32 epilogue tiles reuse the K-loop buffers below s_tok
            self.assertLessEqual(2 * mt1 * pitch * 4, gu_a + 2 * (2 * 4 * 16 * 32 * k))
            self.assertLessEqual(mt2 * pitch * 4, dn_a + 2 * (8 * 8 * 32 * k))

    def test_rotation_constants_and_codebook(self) -> None:
        self.assertIn("constexpr float kRScale = 0.088388347648f;", CUH)  # ExLlamaV3's 1 / sqrt(128)
        self.assertIn("constexpr int kCbMcg = 1;", CUH)
        self.assertEqual(CUH.count("dq_dispatch<KG, kCbMcg>("), 2)
        self.assertEqual(CUH.count("dq_dispatch<KD, kCbMcg>("), 1)

    def test_shuffles_never_sit_under_a_lane_divergent_branch(self) -> None:
        # Row tests around the Hadamard shuffles are warp votes (else nvcc emulates the collective per
        # shuffle: WARPSYNC + ENDCOLLECTIVE, ~2x the instructions; host_compile.sh checks the SASS).
        self.assertEqual(CUH.count("__any_sync(0xffffffffu, 8 * warp < nrows)"), 1)
        self.assertEqual(CUH.count("__any_sync(0xffffffffu, kRowsW2 * warp < nrows)"), 1)
        self.assertNotIn("if (r < nrows) {", CUH)

    def test_work_list_capacity(self) -> None:
        # Host capacity (moe_prefill_tc.run and the C++ check): ceil(slots / MT) + n_exp >= sum ceil(n_e / MT).
        import random

        rng = random.Random(3)
        for _ in range(200):
            n_exp = rng.choice((1, 7, 384, 1024))
            counts = [rng.choice((0, 0, 1, 63, 64, 65, 128, 129, 2000)) for _ in range(n_exp)]
            slots = sum(counts) + rng.randrange(0, 50)  # invalid-id rows sort into the sentinel bin
            for mt in (cuh_constant("kMt1"), cuh_constant("kMt2")):
                self.assertGreaterEqual(-(-slots // mt) + n_exp, sum(-(-n // mt) for n in counts))


class WrapperTests(unittest.TestCase):
    """install() on a fake vllm_exl3.exl3 / torch: which calls take the kernel, and the disarm paths."""

    def setUp(self) -> None:
        tc._STATE.update(armed=True, bypass=False, checked=False, reasons=set(), lazy_temps=False, temps_logged=False)
        self.calls = []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.exl3 = fake_exl3_module(self.tmp.name)

        def orig(x2d, ids, weights, layer, inners, expert_map, limit=None):
            self.calls.append(("stock", x2d.shape[0], tc._STATE["bypass"]))
            return f"stock{x2d.shape[0]}"

        self.exl3.apply_exl3_fused_moe = orig
        self.exl3.map_topk_to_local = lambda ids, n, emap: ids
        pkg = types.ModuleType("vllm_exl3")
        pkg.exl3 = self.exl3
        self.capturing = False
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_current_stream_capturing=lambda: self.capturing)
        torch.empty = lambda shape, dtype=None, device=None: types.SimpleNamespace(shape=shape)
        torch.float16 = "fp16"
        torch.device = lambda d: d
        self.mods = mock.patch.dict(sys.modules, {"torch": torch, "vllm_exl3": pkg, "vllm_exl3.exl3": self.exl3})
        self.mods.start()
        self.addCleanup(self.mods.stop)
        self.patches = [
            mock.patch.object(tc, "ext", lambda: "mod"),
            mock.patch.object(tc, "unsupported", lambda layer, x2d, inners: None),
            mock.patch.object(tc, "synthetic_case", lambda *a, **k: (types.SimpleNamespace(shape=(256, 5120)), "ids", "w")),
            mock.patch.object(tc, "run", self.fake_run),
            mock.patch.object(tc, "compare", lambda out, ref: (True, 1e-3, 5e-4)),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.env = mock.patch.dict("os.environ", {"DSV41_MOE_PREFILL_TC_MIN_ROWS": "", "DSV41_MOE_PREFILL_TC_VERIFY": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def fake_run(self, mod, x2d, *args):
        self.calls.append(("tc", x2d.shape[0]))
        return f"tc{x2d.shape[0]}"

    def call(self, rows: int, layer=None):
        x = types.SimpleNamespace(shape=(rows, 5120), dtype="bf16", device="cuda")
        layer = layer or types.SimpleNamespace(_exl3_k=3, _exl3_k_d=3)
        return self.exl3.apply_exl3_fused_moe(x, "ids", "w", layer, [], None, 10.0)

    def keyed_layer(self):
        return types.SimpleNamespace(_exl3_k=3, _exl3_k_d=3, _exl3_fused_temps=None,
                                     _dsv41_temps_key=("cuda:0", 5120, 1152, 6))

    def install(self) -> str:
        with redirect_stdout(StringIO()) as out:
            msg = tc.install()
        self.out = out
        return msg

    def test_routes_by_rows_self_checks_once_and_verifies(self) -> None:
        self.assertIn("min rows 9, verify 1; self-check at the first eager call; exl3_moe scratch lazy", self.install())
        self.assertTrue(self.exl3.apply_exl3_fused_moe._dsv41_moe_prefill_tc)
        self.assertTrue(self.exl3.build_exl3_fused_state._dsv41_lazy_temps)
        with redirect_stdout(StringIO()) as out:
            self.assertEqual(self.call(8), "stock8")      # decode-sized: stock
            self.assertEqual(self.call(8192), "tc8192")   # self-check + verify against stock (bypassed)
            self.assertEqual(self.call(4096), "tc4096")   # verify budget spent: kernel only
        self.assertIn(tc.LOG_ENGAGED, out.getvalue())
        self.assertEqual(self.calls, [("stock", 8, False), ("tc", 256), ("stock", 256, True), ("tc", 8192),
                                      ("stock", 8192, True), ("tc", 4096)])
        self.assertEqual(self.install(), "already installed")

    def test_references_borrow_scratch_and_fallbacks_keep_it(self) -> None:
        self.install()
        lay = self.keyed_layer()
        with redirect_stdout(StringIO()) as out:
            self.assertEqual(self.call(8192, lay), "tc8192")  # self-check + verify borrow, then release
        self.assertEqual(self.exl3._FUSED_TEMP_CACHE, {})
        self.assertIsNone(lay._exl3_fused_temps)
        self.assertNotIn(tc.LOG_TEMPS, out.getvalue())
        with redirect_stdout(StringIO()) as out:
            self.assertEqual(self.call(8, lay), "stock8")  # a genuine stock call: allocated once, kept
            self.assertEqual(self.call(8, self.keyed_layer()), "stock8")
        self.assertEqual(list(self.exl3._FUSED_TEMP_CACHE), [("cuda:0", 5120, 1152, 6)])
        self.assertIsNotNone(lay._exl3_fused_temps)
        self.assertEqual(out.getvalue().count(tc.LOG_TEMPS), 1)

    def test_capture_and_unsupported_calls_stay_stock(self) -> None:
        self.install()
        self.capturing = True
        self.assertEqual(self.call(8192), "stock8192")
        self.capturing = False
        with mock.patch.object(tc, "unsupported", lambda layer, x2d, inners: "codebook flags (MCG only)"), \
                redirect_stdout(StringIO()) as out:
            self.assertEqual(self.call(8192), "stock8192")
            self.assertEqual(self.call(8192), "stock8192")
        self.assertEqual(out.getvalue().count(f"{tc.LOG_DISARMED} this call: codebook flags"), 1)
        self.assertTrue(tc._STATE["armed"])

    def test_self_check_failure_disarms(self) -> None:
        self.install()
        with mock.patch.object(tc, "compare", lambda out, ref: (False, 0.5, 0.2)), redirect_stdout(StringIO()) as out:
            self.assertEqual(self.call(8192), "stock8192")
            self.assertEqual(self.call(8192), "stock8192")
        self.assertIn(f"{tc.LOG_DISARMED} self-check vs stock failed", out.getvalue())
        self.assertFalse(tc._STATE["armed"])

    def test_verify_mismatch_returns_stock_and_disarms(self) -> None:
        self.install()
        verdicts = iter([(True, 1e-3, 5e-4), (False, 0.3, 0.1)])
        with mock.patch.object(tc, "compare", lambda out, ref: next(verdicts)), redirect_stdout(StringIO()) as out:
            self.assertEqual(self.call(8192), "stock8192")
        self.assertIn(f"{tc.LOG_DISARMED} verify of a 8192-row call failed", out.getvalue())
        self.assertFalse(tc._STATE["armed"])

    def test_kernel_exception_disarms_to_stock(self) -> None:
        self.install()

        def boom(*a):
            raise RuntimeError("no kernel image")

        with mock.patch.object(tc, "ext", boom), redirect_stdout(StringIO()) as out:
            self.assertEqual(self.call(8192), "stock8192")
        self.assertIn(f"{tc.LOG_DISARMED} RuntimeError: no kernel image", out.getvalue())
        self.assertFalse(tc._STATE["armed"])


class LazyTempsTests(unittest.TestCase):
    """exl3_moe's scratch: no longer allocated at load; first stock call allocates, references borrow."""

    KEY = ("cuda:0", 5120, 1152, 6)

    def setUp(self) -> None:
        tc._STATE.update(temps_logged=False)
        self.made = []

        def empty(shape, dtype=None, device=None):
            self.made.append(shape)
            return types.SimpleNamespace(shape=shape)

        torch = types.ModuleType("torch")
        torch.empty = empty
        torch.float16 = "fp16"
        torch.device = lambda d: d
        self.mods = mock.patch.dict(sys.modules, {"torch": torch})
        self.mods.start()
        self.addCleanup(self.mods.stop)
        self.exl3 = types.SimpleNamespace(_FUSED_TEMP_CACHE={}, TEMP_ROWS_FUSED=2048)

    def layer(self):
        return types.SimpleNamespace(_exl3_fused_temps=None, _dsv41_temps_key=self.KEY)

    def test_rewrite_only_drops_the_allocation(self) -> None:
        src = pinned_function("build_exl3_fused_state")
        out = tc.patch_build_source(src)
        compile(out, "x", "exec")
        self.assertEqual(out.replace(tc.TEMPS_NEW, tc.TEMPS_OLD), src)
        self.assertNotIn("torch.empty((concurrency, TEMP_ROWS_FUSED", out)
        self.assertIn("layer._exl3_fused_temps = _FUSED_TEMP_CACHE.get(key)", out)
        self.assertIn("layer._exl3_fused_concurrency = concurrency", out)  # the boot log line keeps its value
        with self.assertRaises(ValueError):
            tc.patch_build_source(src.replace("TEMP_ROWS_FUSED, hidden)", "TEMP_ROWS_FUSED, hidden + 1)"))

    def test_bytes_per_rank(self) -> None:
        self.assertEqual(tc.temps_bytes(self.KEY), 308281344)  # 294.0 MiB: 6 x 2048 x (2 x 5120 + 2 x 1152) fp16

    def test_first_stock_call_allocates_once_and_logs(self) -> None:
        a, b = self.layer(), self.layer()
        with redirect_stdout(StringIO()) as out:
            self.assertTrue(tc.ensure_temps(self.exl3, a, 64))
            self.assertFalse(tc.ensure_temps(self.exl3, b, 16))   # shared buffer, no second allocation
            self.assertFalse(tc.ensure_temps(self.exl3, a, 16))   # already held
        self.assertEqual(self.made, [(6, 2048, 5120), (6, 2048, 5120), (6, 2048, 1152), (6, 2048, 1152)])
        self.assertIs(a._exl3_fused_temps, b._exl3_fused_temps)
        self.assertEqual(out.getvalue().count(tc.LOG_TEMPS), 1)
        self.assertIn("(294 MiB per rank; a 64-row call)", out.getvalue())

    def test_reference_borrow_is_released(self) -> None:
        a = self.layer()
        with redirect_stdout(StringIO()) as out:
            self.assertTrue(tc.ensure_temps(self.exl3, a, 256, borrow=True))
            tc.release_temps(self.exl3, a)
        self.assertEqual(out.getvalue(), "")
        self.assertIsNone(a._exl3_fused_temps)
        self.assertEqual(self.exl3._FUSED_TEMP_CACHE, {})

    def test_layers_without_a_key_are_left_alone(self) -> None:
        lay = types.SimpleNamespace(_exl3_fused_temps=None)
        self.assertFalse(tc.ensure_temps(self.exl3, lay, 64))
        self.assertEqual(self.made, [])


if __name__ == "__main__":
    unittest.main()
