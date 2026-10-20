#!/usr/bin/env python3
"""widen_p2b_rows.py (native p2b decode MoE for 9..24 rows: the dataflow kernel with a 144-pair sort cap):
patch anchors, idempotence, the edits and nothing else, the wide kernel = the dataflow kernel's text with
the listed substitutions only, host gates, control-word layout, the plugin's row cap, and Dockerfile.e16
sync. CPU only; the GPU gates are kernel_study/p2b_rows/ (sass_identity_rows.sh, bench_rows.py).
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PIN_CU = ROOT / "tests/fixtures/p2b_moe.pin.cu"
PIN_PY = ROOT / "tests/fixtures/exl3_e14.pin.py"
CHAIN = ("shapes", "mrow", "cfg1", "codebook", "fshift", "srcsort", "coop", "dataflow", "kbits")

sys.path.insert(0, str(ROOT / "docker/patch"))


def _load(name: str, path: Path | None = None):
    path = path or ROOT / "docker/patch" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _block(src: str, start: str, end: str) -> str:
    a = src.index(start)
    return src[a : src.index(end, a)]


class RowsCudaPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _load("widen_p2b_rows")
        src = PIN_CU.read_text()
        for name in CHAIN:
            src = _load(f"widen_p2b_{name}").patch_cu(src)
        cls.kb = src
        cls.patched = cls.rows.patch_cu(src)

    def test_applies_after_kbits_and_is_idempotent(self) -> None:
        self.assertNotEqual(self.patched, self.kb)
        self.assertEqual(self.rows.patch_cu(self.patched), self.patched)
        self.assertEqual(self.patched.count("grid.sync()"), self.kb.count("grid.sync()") + 2)
        src = PIN_CU.read_text()
        for name in CHAIN[:-1]:
            src = _load(f"widen_p2b_{name}").patch_cu(src)
        with self.assertRaises(SystemExit):
            self.rows.patch_cu(src)  # no kbits stage

    def test_undoing_the_edits_gives_the_kbits_source_back(self) -> None:
        r, out = self.rows, self.patched
        wide = r.wide_kernel(self.kb)
        self.assertEqual(out.count(wide), 1)
        out = out.replace(wide, "", 1)
        for old, new in ((r.LAUNCH_OLD, r.LAUNCH_NEW), (r.ROWS_OLD, r.ROWS_NEW), (r.DF_FLAG_OLD, r.DF_FLAG_NEW)):
            self.assertEqual(out.count(new), 1)
            out = out.replace(new, old, 1)
        self.assertEqual(out, self.kb)

    def test_wide_kernel_is_the_dataflow_kernel_with_the_listed_substitutions(self) -> None:
        r = self.rows
        df = r._function(self.kb, r.DF_KERNEL_START)
        dfw = _block(self.patched, "template <int BITS, int CB>\n__global__ __launch_bounds__(256, BITS == 2 ? 3 : 2)"
                     "  // widen_p2b_kbits: K != 2 at 2 blocks/SM\nvoid p2b_coop_dfw_kernel(", "\n}\n") + "\n}\n"
        back = dfw
        for old, new, n in r.DFW_SUBS:
            self.assertEqual(back.count(new), n, new)
            back = back.replace(new, old)
        self.assertEqual(back, df)
        # same tiles, epilogue, output and barrier count as the dataflow kernel
        for call in ("p2b_df_tile<p2b_kgu(BITS)>(", "p2b_df_tile<p2b_kd(BITS)>(", "p2b_df_act(", "p2b_df_out6(",
                     "p2b_df_out(", "p2b_df_prefetch_tile<p2b_kgu(BITS)>("):
            self.assertEqual(dfw.count(call), df.count(call), call)
        self.assertEqual(dfw.count("grid.sync()"), 2)
        self.assertNotIn("P2B_SORT_CAP", dfw)
        self.assertNotIn("_smem_k<BITS>()", dfw)
        self.assertNotIn("p2b_df_red_k", dfw)
        build = _block(self.patched, "__device__ __forceinline__ void p2b_dfw_build(", "\n}\n") + "\n}\n"
        back = build
        for old, new, n in r.BUILD_SUBS:
            self.assertEqual(back.count(new), n, new)
            back = back.replace(new, old)
        self.assertEqual(back, r._function(self.kb, r.DF_BUILD_START))
        # the chunks keep <= 8 members (MMA rows 0..7): the tile code is the dataflow kernel's
        self.assertIn("chunk[P2B_WIDE_CAP + u] = min(P2B_COOP_ROWS, run_len[s] - off);", build)

    def test_the_e15_kernels_are_not_touched(self) -> None:
        for start in ("void p2b_coop_df_kernel(", "void p2b_moe_batched_kernel(", "void p2b_df_tile(",
                      "void p2b_df_build_k(", "void p2b_df_out6("):
            self.assertEqual(_block(self.patched, start, "\n}\n"), _block(self.kb, start, "\n}\n"), start)
        # the wide kernel's shared arrays are its own templates
        for name in ("p2b_dfw_sort_smem", "p2b_dfw_chunk_smem", "p2b_dfw_red"):
            body = _block(self.patched, f"{name}()\n{{", "\n}\n")
            self.assertIn("__shared__", body)
            self.assertNotIn("return p2b_", body)

    def test_constants_and_control_words(self) -> None:
        r, p = self.rows, self.patched
        self.assertEqual((r.WIDE_ROWS, r.WIDE_CAP), (24, 144))
        self.assertIn("constexpr int P2B_WIDE_ROWS = 24;", p)
        self.assertIn("constexpr int P2B_WIDE_CAP = 144;", p)
        self.assertIn("constexpr int P2B_DFW_ACT = P2B_DF_READY + P2B_WIDE_CAP;", p)
        self.assertIn("return P2B_DFW_ACT + P2B_WIDE_CAP * (inter / 128);", p)
        self.assertIn("return m * e > P2B_SORT_CAP && m * e <= P2B_WIDE_CAP && p2b_dfw_ctl_words(inter) <= m * hidden;", p)
        ready = int(re.search(r"constexpr int P2B_DF_READY = (\d+);", p).group(1))
        sort_cap = int(re.search(r"constexpr int P2B_SORT_CAP = (\d+);", p).group(1))
        coop_rows = int(re.search(r"constexpr int P2B_COOP_ROWS = (\d+);", p).group(1))
        self.assertEqual((ready, sort_cap, coop_rows), (32, 64, 8))
        hidden, inter, topk = 5120, 1152, 6
        act = ready + r.WIDE_CAP
        for m in range(1, r.WIDE_ROWS + 1):
            pairs = m * topk
            wide = sort_cap < pairs <= r.WIDE_CAP
            self.assertEqual(wide, m >= 11, m)  # m 9..10 still fit the dataflow kernel's 64 pairs
            if wide:
                # units <= pairs: READY slots [ready, ready + units) end before ACT; ACT fits accum
                self.assertLessEqual(ready + pairs, act)
                self.assertLessEqual(act + r.WIDE_CAP * (inter // 128), m * hidden)
        # shared memory of the wide kernel at 3 blocks / SM (K = 2): sort + chunk + red + s_df
        smem = 4 * (3 * r.WIDE_CAP) + 4 * (2 * r.WIDE_CAP + 1) + 8 * 8 * 64 * 4 + 8
        self.assertLessEqual(3 * smem, 99 * 1024)

    def test_host_gates(self) -> None:
        p, r = self.patched, self.rows
        sel = _block(p, "    if constexpr (CB == 1) {  // widen_p2b_kbits", "    cudaOccupancyMaxActiveBlocksPerMultiprocessor")
        self.assertLess(sel.index("kernel = (void*) p2b_coop_df_kernel<BITS, CB>;"),
                        sel.index("kernel = (void*) p2b_coop_dfw_kernel<BITS, CB>;"))
        self.assertIn("if (p2b_coop_mode() == 2 && p2b_dfw_rows_ok(m, e, hidden, inter)) {", sel)
        self.assertIn(f'std::fprintf(stderr, "%s\\n", "{r.LOG_ENGAGED}");', sel)
        self.assertEqual(p.count("cudaLaunchCooperativeKernel("), 1)
        host = p[p.index("at::Tensor p2b_fused_moe_cuda(") :]
        self.assertIn("x.size(0) >= 1 && x.size(0) <= P2B_WIDE_ROWS &&", host)
        self.assertIn('"fused MoE requires 1-24 input rows whose hidden width', host)
        self.assertNotIn("x.size(0) <= 8", host)
        flag = _block(host, "    const bool df = ", ";\n")
        self.assertIn("p2b_dfw_rows_ok(m, e, hidden, inter)", flag)
        self.assertIn("mcg && p2b_coop_mode() == 2 && ((m * e <= P2B_SORT_CAP && p2b_df_ctl_words(inter) <= m * hidden)", flag)
        # the accum memset is skipped for both dataflow kernels (they zero their control words)
        self.assertIn("auto accum = df ? at::empty({m, hidden}", host)
        # every BITS the dataflow kernel has gets the wide kernel (same dispatch path)
        self.assertEqual(set(r.WIDE_BITS), {2, 3, 4} | {10 * a + b for a, b in ((2, 3), (2, 4), (3, 2), (3, 4), (4, 2), (4, 3))})

    def test_bindings(self) -> None:
        r = self.rows
        b = '#include "exl3_fat_gemm.cuh"\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n    m.attr("P2B_MOE_ABI_VERSION") = 2;\n}\n'
        kb = _load("widen_p2b_kbits")
        b = kb.patch_bindings(b)
        out = r.patch_bindings(b)
        self.assertIn('m.attr("P2B_WIDE_ROWS") = 24;', out)
        self.assertEqual(r.patch_bindings(out), out)
        self.assertIn('m.def("exl3_moe_kmix"', out)
        with self.assertRaises(SystemExit):
            r.patch_bindings("PYBIND11_MODULE(x, m) {}\n")

    def test_bench_sources(self) -> None:
        mb = _load("make_bench_rows", ROOT / "kernel_study/p2b_rows/make_bench_rows.py")
        srcs = mb.build()
        self.assertEqual(srcs["chain_kbits.cu"], self.kb)
        self.assertEqual(srcs["chain_rows.cu"], self.patched)
        bench = srcs["bench_rows.cu"]
        self.assertIn("static int p2b_coop_mode()\n{\n    return g_bench_variant;\n", bench)
        self.assertIn("if (p2b_coop_mode() == 2 && p2b_dfw_rows_ok(m, e, hidden, inter)) {", bench)
        self.assertIn('mod.def("set_variant"', bench)


class RowsPluginPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _load("widen_p2b_rows")
        cls.kb = _load("widen_p2b_kbits").patch_py(PIN_PY.read_text())
        cls.patched = cls.rows.patch_py(cls.kb)

    def test_idempotent_and_edits_only(self) -> None:
        self.assertEqual(self.rows.patch_py(self.patched), self.patched)
        out = self.patched
        for old, new, _ in reversed(self.rows.PY_EDITS):
            self.assertEqual(out.count(new), 1, new)
            out = out.replace(new, old, 1)
        self.assertEqual(out, self.kb)
        with self.assertRaises(SystemExit):
            self.rows.patch_py(PIN_PY.read_text())  # needs the kbits plugin

    def test_runtime_levers_still_apply(self) -> None:
        mp = _load("moe_prep_fused")
        out = mp.patch_experts_source(mp.patch_source(self.patched))
        self.assertIn("_dsv41_moe_prep(ids, weights, x2d, n_exp, expert_map)", out)
        pg = _load("pfg8_loader_reindex")
        self.assertNotEqual(pg.patch(self.patched), self.patched)

    def _module(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "exl3_rows.py"
            path.write_text(self.patched)
            return _load("exl3_rows", path)

    def test_row_cap(self) -> None:
        try:
            mod = self._module()
        except Exception as exc:  # pragma: no cover - host without the module's soft deps
            self.skipTest(f"plugin module not importable here: {exc!r}")

        class X:
            is_cuda = True

            def __init__(self, rows):
                self.shape = (rows, 5120)

            def dim(self):
                return 2

        def layer(mcg=True):
            return types.SimpleNamespace(_exl3_hidden_size=5120, _exl3_intermediate_local=1152, _exl3_k=3, _exl3_k_d=2,
                                         _exl3_codebook_flags=(mcg, not mcg, mcg, not mcg, mcg, not mcg))

        ext = types.SimpleNamespace(P2B_WIDE_ROWS=24)
        old_ext = types.SimpleNamespace()  # an e15 vllm_exl3_c: no P2B_WIDE_ROWS

        def ok(rows, env, module=ext, lay=None):
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(mod, "_load_native_exl3_ext", lambda: module):
                return mod._native_moe_dimensions_supported(X(rows), lay or layer(), [object()], 10.0)

        on = {"DSV41_P2B_COOP": "2"}
        for rows in (1, 4, 8):
            for env in ({}, on, {"DSV41_P2B_COOP": "2", "DSV41_P2B_WIDE": "0"}):
                self.assertTrue(ok(rows, env), (rows, env))
        for rows in (9, 12, 16, 20, 24):
            self.assertTrue(ok(rows, on), rows)
            self.assertTrue(ok(rows, on | {"DSV41_P2B_WIDE": "1"}), rows)
            self.assertFalse(ok(rows, {}), rows)  # COOP off: the e15 contract
            self.assertFalse(ok(rows, {"DSV41_P2B_COOP": "1"}), rows)
            self.assertFalse(ok(rows, on | {"DSV41_P2B_WIDE": "0"}), rows)
            self.assertFalse(ok(rows, on, module=old_ext), rows)
            self.assertFalse(ok(rows, on, module=None), rows)
            self.assertFalse(ok(rows, on, lay=layer(mcg=False)), rows)  # mul1 codebook
        self.assertFalse(ok(25, on))
        # a row-cap override below 8 keeps its cap; the measured per-K cap (K = 3 -> 1) too
        self.assertFalse(ok(9, on | {"VLLM_EXL3_NATIVE_MOE_MAX_ROWS": "4"}))
        self.assertFalse(ok(5, on | {"VLLM_EXL3_NATIVE_MOE_MAX_ROWS": "4"}))
        self.assertFalse(ok(2, on | {"VLLM_EXL3_NATIVE_MOE_MEASURED_CAP": "1"}))
        self.assertFalse(ok(12, on | {"VLLM_EXL3_NATIVE_MOE_MEASURED_CAP": "1"}))

    def test_env_is_forwarded(self) -> None:
        run = (ROOT / "run.sh").read_text()
        block = run[run.index("FORWARD_ENVS=(\n") : run.index("\n)\n", run.index("FORWARD_ENVS=(\n"))]
        self.assertIn("\n  DSV41_P2B_WIDE=\n", block + "\n")  # empty default: unset = on (code default)


class RowsImageTests(unittest.TestCase):
    def test_dockerfile_e16(self) -> None:
        e16 = (ROOT / "docker/Dockerfile.e16").read_text()
        e15 = (ROOT / "docker/Dockerfile.e15").read_text()
        self.assertIn("FROM dsv41-flash-exl3-sm121:e15-kbits\n", e16)
        ref = re.search(r"ARG VLLM_EXL3_REF=(\w+)", e15).group(1)
        self.assertIn(f"ARG VLLM_EXL3_REF={ref}\n", e16)
        stages = [re.findall(r"(widen_p2b_\w+)\.py /opt/vllm-exl3", df) for df in (e15, e16)]
        self.assertEqual(stages[1], stages[0] + ["widen_p2b_rows"])
        self.assertLess(e16.index("widen_p2b_rows.py /opt/vllm-exl3"),
                        e16.index("pip install --no-build-isolation --no-deps --force-reinstall /opt/vllm-exl3"))
        checks = [re.search(r'python3 -c "(import torch, vllm_exl3, vllm_exl3_c;.*?)"', df).group(1) for df in (e15, e16)]
        self.assertTrue(checks[1].startswith(checks[0].rsplit("; print(", 1)[0]), "e16 keeps every e15 build assert")
        for needle in ("b'p2b_coop_dfw_kernelILi23ELi1E' in so", "b'p2b_coop_dfw_kernelILi2ELi1E' in so",
                       "vllm_exl3_c.P2B_WIDE_ROWS == 24", "hasattr(e, '_native_moe_wide_rows')"):
            self.assertIn(needle, checks[1])
        self.assertNotIn("widen_b12x_rows16", e16)  # dropped: no measured gain (rows16_check.py)
        labels = [re.search(r'LABEL dsv41.recipe.patches="([^"]+)"', df).group(1).split(",") for df in (e15, e16)]
        want = list(labels[0])
        want.insert(want.index("kbits") + 1, "rows")
        self.assertEqual(labels[1], want)
        self.assertIn("cp /opt/dsv41-patch/sitecustomize.py /usr/lib/python3.12/sitecustomize.py", e16)
        self.assertIn('LABEL dsv41.recipe.git_rev="$DSV41_GIT_REV" dsv41.recipe.tree_diff_sha256="$DSV41_TREE_DIFF"', e16)
        self.assertIn("--memory 24g", e16)


if __name__ == "__main__":
    unittest.main()
