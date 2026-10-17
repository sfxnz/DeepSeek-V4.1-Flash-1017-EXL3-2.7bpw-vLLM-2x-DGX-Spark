#!/usr/bin/env python3
"""widen_p2b_kbits.py (K = 3 / 4 and K_gu != K_d on the coop / dataflow p2b kernels, expert_bits in the
plugin): patch anchors, idempotence, the edits and nothing else, K decode of the template argument,
the plugin config, the runtime levers that rewrite exl3.py after it, and Dockerfile.e15 sync. CPU only.

tests/fixtures/exl3_e14.pin.py is /usr/local/lib/python3.12/dist-packages/vllm_exl3/exl3.py of
dsv41-flash-exl3-sm121:canonical-e14 (sha256 afc6da3e...), which equals the vllm-exl3 pin d3cfd394
after widen_p2b_shapes / mrow / codebook patch_py.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_CU = ROOT / "tests/fixtures/p2b_moe.pin.cu"
PIN_PY = ROOT / "tests/fixtures/exl3_e14.pin.py"
CHAIN = ("shapes", "mrow", "cfg1", "codebook", "fshift", "srcsort", "coop", "dataflow")
MIXED = ((2, 3), (2, 4), (3, 2), (3, 4), (4, 2), (4, 3))

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


def kgu(bits: int) -> int:  # p2b_kgu
    return bits // 10 if bits >= 10 else bits


def kd(bits: int) -> int:  # p2b_kd
    return bits % 10 if bits >= 10 else bits


class KbitsCudaPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = _load("widen_p2b_kbits")
        src = PIN_CU.read_text()
        for name in CHAIN:
            src = _load(f"widen_p2b_{name}").patch_cu(src)
        cls.df = src
        cls.patched = cls.kb.patch_cu(src)

    def test_applies_after_dataflow_and_is_idempotent(self) -> None:
        self.assertNotEqual(self.patched, self.df)
        self.assertEqual(self.kb.patch_cu(self.patched), self.patched)
        self.assertEqual(self.patched.count("grid.sync()"), self.df.count("grid.sync()"))
        src = PIN_CU.read_text()
        for name in CHAIN[:-1]:
            src = _load(f"widen_p2b_{name}").patch_cu(src)
        with self.assertRaises(SystemExit):
            self.kb.patch_cu(src)  # no dataflow stage

    def test_undoing_the_edits_gives_the_dataflow_source_back(self) -> None:
        kb, out = self.kb, self.patched
        builders = {n: kb._builder_k(self.df, s, n, nn, subs) for s, n, nn, subs in kb.BUILDERS}
        for text in (kb.HELPERS + builders["p2b_coop_build"], kb.DF_SMEM + builders["p2b_df_build"]):
            self.assertEqual(out.count(text), 1)
            out = out.replace(text, "", 1)
        old_tile = self.df[self.df.index(kb.TILE_START) : self.df.index(kb.TILE_END)]
        self.assertEqual(out.count(kb.TILE_NEW), 1)
        out = out.replace(kb.TILE_NEW, old_tile, 1)
        for start, subs in (("void p2b_moe_batched_kernel(", kb.SORT_KERNEL_SMEM), ("void p2b_coop_df_kernel(", kb.DF_KERNEL_SMEM)):
            out = kb._region_subs(out, start, tuple((new, old, n) for old, new, n in subs))
        for old, new, _ in kb.SORT_KERNEL_EDITS + kb.DF_EDITS + kb.HOST_EDITS:
            self.assertEqual(out.count(new), 1, new)
            out = out.replace(new, old, 1)
        self.assertEqual(out.count(kb.DISPATCH_NEW), 1)
        out = out.replace(kb.DISPATCH_NEW, "", 1)
        self.assertEqual(out, self.df)

    def test_template_k_decode(self) -> None:
        self.assertIn("constexpr int p2b_kgu(int bits) { return bits >= 10 ? bits / 10 : bits; }", self.patched)
        self.assertIn("constexpr int p2b_kd(int bits) { return bits >= 10 ? bits % 10 : bits; }", self.patched)
        for k in (2, 3, 4):
            self.assertEqual((kgu(k), kd(k)), (k, k))
        for a, b in MIXED:
            self.assertEqual((kgu(10 * a + b), kd(10 * a + b)), (a, b))
        host = self.patched[self.patched.index("at::Tensor p2b_fused_moe_cuda(") :]
        for a, b in MIXED:
            self.assertIn(f"        case {10 * a + b}: launch_moe_batched<{10 * a + b}, 1>(", host)
        # the mixed switch runs before the uniform chain, which still dispatches on kg only
        self.assertLess(host.index("    if (kg != kd) {"), host.index("    if (mcg && kg == 2) launch_moe_batched<2, 1>("))
        self.assertIn("        return out;\n    }\n", host)

    def test_gate_up_and_down_tiles_take_their_own_k(self) -> None:
        k = _block(self.patched, "void p2b_moe_batched_kernel(", "// --- widen_p2b_dataflow")
        self.assertEqual(k.count("run_gemv_tile<p2b_kgu(BITS), CB, 1>(B32, A2, C, kslices_gate,"), 1)
        self.assertEqual(k.count("run_gemv_tile<p2b_kd(BITS), CB, 1>(B32, A2, C, kslices_down,"), 1)
        self.assertEqual(k.count("run_gemv_tile_coop<p2b_kgu(BITS), CB, 1>(B32, is_up ? had_up : had_gate,"), 1)
        self.assertEqual(k.count("run_gemv_tile_coop<p2b_kd(BITS), CB, 1>(B32, had_down, down,"), 1)
        self.assertNotIn("<BITS, CB, 1>", k)
        d = _block(self.patched, "void p2b_coop_df_kernel(", "template <int BITS, int CB>\nstatic void launch_moe_batched(")
        self.assertEqual(d.count("p2b_df_tile<p2b_kgu(BITS)>("), 1)
        self.assertEqual(d.count("p2b_df_tile<p2b_kd(BITS)>("), 1)
        self.assertIn("if constexpr (p2b_kgu(BITS) == p2b_kd(BITS))", d)
        self.assertNotIn("p2b_df_tile(", d)
        self.assertNotIn("p2b_df_prefetch_tile(", d)

    def test_df_tile_k_lines_are_run_gemv_tile_coops(self) -> None:
        coop = _block(self.patched, "__device__ __forceinline__ void run_gemv_tile_coop(", "template <int BITS, int CB, int SORT = 0>")
        tile = _block(self.patched, "__device__ __forceinline__ void p2b_df_tile(", "// L2 prefetch of the first n k-slices")
        for line in (
            "constexpr int TWORDS = 8 * bits;",
            "constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;",
            "constexpr int LSTRIDE = bits == 3 ? 24 : 32;",
            "b1 = (t_offset + 257) * 3;",
            "i0 = (b1 - 16) / 32;",
            "i2 = (b2 - 1) / 32;",
            "x_s2 = (i2 + 1) * 32 - b2;",
            "x_src_a = i0 % 24;",
            "x_src_b = i2 % 24;",
            "return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0",
            "uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);",
            "uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);",
            "uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);",
            "exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);",
            "acc0[t][f].x += __low2float(ch[t][f][0]);",
            "sum += sh_red[j][r][c];",
        ):
            self.assertIn(line, coop, line)
            self.assertIn(line, tile, line)
        for a, b in (("exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);", "exl3_gemv_ns::dq8_regs_4bits<1>(aw, bw[t], f0, f1);"),
                     ("exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);", "exl3_gemv_ns::dq8_regs_3bits<1>(awv, bwv, x_s2, f0, f1);"),
                     ("bench_fshift::dq8_regs_2bits_fs<cb>(awv, bwv, lane << 3, f0, f1);",
                      "bench_fshift::dq8_regs_2bits_fs<1>(awv, bwv, lane << 3, f0, f1);")):
            self.assertIn(a, coop)
            self.assertIn(b, tile)
        # L2 prefetch: bits 128-byte lines per k-slice (K = 2: lanes 0..1 at 32-word steps, as before)
        self.assertIn("constexpr int PFLINES = bits;", tile)
        self.assertIn("lane * PFSTRIDE;", tile)
        pf = _block(self.patched, "void p2b_df_prefetch_tile(", "// Output (row, 128-block hb)")
        self.assertIn("if (lane < bits) {", pf)
        self.assertIn("const size_t stride = (size_t) ntiles * (8 * bits);", pf)
        self.assertIn("group * (32 * bits) + lane * 32;", pf)
        self.assertNotIn("atomicAdd", tile)

    def test_shared_arrays_per_bits(self) -> None:
        p = self.patched
        for name in ("p2b_coop_smem", "p2b_coop_red", "p2b_df_sort_smem", "p2b_df_chunk_smem", "p2b_df_red"):
            k = _block(p, f"{name}_k()\n{{", "\n}\n")
            self.assertIn(f"return {name}();", k)  # BITS == 2: the round-2/3 array
            self.assertIn("__shared__", k)  # other BITS: their own
        for kernel, names in (("void p2b_moe_batched_kernel(", ("p2b_coop_smem()", "p2b_coop_red()", "p2b_coop_build(")),
                              ("void p2b_coop_df_kernel(", ("p2b_df_sort_smem()", "p2b_df_chunk_smem()", "p2b_df_red()",
                                                            "p2b_df_build("))):
            body = p[p.index(kernel) : p.index("\n}\n", p.index(kernel))]
            for n in names:
                self.assertNotIn(n, body, (kernel, n))

    def test_host_gates(self) -> None:
        p = self.patched
        self.assertIn("if constexpr (CB == 1) {  // widen_p2b_kbits", p)
        self.assertNotIn("if constexpr (BITS == 2 && CB == 1)", p)
        self.assertIn("TORCH_CHECK(kg == ku && (kg == 2 || kg == 3 || kg == 4) && (kd == 2 || kd == 3 || kd == 4) && (kd == kg || mcg),", p)
        self.assertIn("const bool df = mcg && p2b_coop_mode() == 2 && m * e <= P2B_SORT_CAP && p2b_df_ctl_words(inter) <= m * hidden;", p)
        self.assertNotIn("kg == 2 && p2b_coop_mode()", p)
        # the launch picks the dataflow kernel under the same conditions as the host's df flag (CB 1 = mcg)
        sel = _block(p, "    if constexpr (CB == 1) {", "    cudaOccupancyMaxActiveBlocksPerMultiprocessor")
        self.assertIn("if (p2b_coop_mode() == 2 && m * e <= P2B_SORT_CAP && p2b_df_ctl_words(inter) <= m * hidden) {", sel)
        self.assertEqual(p.count("cudaLaunchCooperativeKernel("), 1)
        self.assertIn("__global__ __launch_bounds__(256, BITS == 2 ? 3 : 2)  // widen_p2b_kbits: K != 2 at 2 blocks/SM\n"
                      "void p2b_coop_df_kernel(", p)

    def test_bench_sources(self) -> None:
        mb = _load("make_bench_kbits", ROOT / "kernel_study/p2b_kbits/make_bench_kbits.py")
        srcs = mb.build()
        self.assertEqual(srcs["chain_kbits.cu"], self.patched)
        self.assertEqual(srcs["chain_r3.cu"], self.df)
        bench = srcs["bench_kbits.cu"]
        self.assertIn("g_bench_variant == 1 && m * e <= P2B_SORT_CAP", bench)
        self.assertIn("static int p2b_coop_mode()\n{\n    return g_bench_variant;\n", bench)
        self.assertEqual(bench.count("p2b_coop_mode() == 2"), self.patched.count("p2b_coop_mode() == 2"))
        self.assertIn("__launch_bounds__(256, 3)\nvoid p2b_coop_df_kernel(", srcs["bench_kbits_lb3.cu"])


class KbitsPluginPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = _load("widen_p2b_kbits")
        cls.src = PIN_PY.read_text()
        cls.patched = cls.kb.patch_py(cls.src)

    def test_idempotent_and_edits_only(self) -> None:
        self.assertEqual(self.kb.patch_py(self.patched), self.patched)
        out = self.patched
        for old, new, _ in reversed(self.kb.PY_EDITS):
            self.assertEqual(out.count(new), 1, new)
            out = out.replace(new, old, 1)
        self.assertEqual(out, self.src)

    def test_runtime_levers_still_apply(self) -> None:
        mp = _load("moe_prep_fused")
        out = mp.patch_experts_source(mp.patch_source(self.patched))
        self.assertIn("k_d,\n        mcg,", out)
        pg = _load("pfg8_loader_reindex")
        g8 = pg.patch(self.patched)
        # the G8 alloc keeps w2 at K_d: k_words is rebound before the w2 allocation
        self.assertLess(g8.index("k_words = self.bits_d * 16"), g8.index("_w2_shape = ("))

    def test_k_triples(self) -> None:
        p = self.patched
        native = _block(p, "def _apply_native_fused_moe(", "\ndef ")
        self.assertIn('k_d = int(getattr(layer, "_exl3_k_d", k))', native)
        self.assertIn("        k,\n        k,\n        k_d,\n        mcg,\n", native)
        self.assertIn("if k_d != k and not mcg:\n        return None", native)
        fused = _block(p, "def apply_exl3_fused_moe(", "\ndef ")
        self.assertIn("        MOE_ACT_SILU,\n        k,\n        k,\n        k_d,\n", fused)
        fat = _block(p, "def apply_exl3_batched_fat(", "\ndef ")
        self.assertIn('k_d = int(getattr(down, "K", k))', fat)
        self.assertIn("ext.reconstruct(w2, down.trellis, k_d, mcg, mul1)", fat)
        self.assertIn("            and k_d == 4\n", fat)
        self.assertIn("ext.reconstruct(w13, packed13, k, mcg, mul1)", fat)  # gate/up keep K_gu
        dims = _block(p, "def _native_moe_dimensions_supported(", "\n# Measured")
        self.assertIn("and bits_d in (2, 3, 4)", dims)
        cw = _block(p, "    def create_weights(", "    def _load_exl3(")
        self.assertLess(cw.index("k_words = self.bits * 16"), cw.index("w13_trellis = Parameter("))
        self.assertLess(cw.index("w13_trellis = Parameter("), cw.index("k_words = self.bits_d * 16"))
        self.assertLess(cw.index("k_words = self.bits_d * 16"), cw.index("w2_trellis = Parameter("))
        self.assertIn("layer._exl3_bits_d = self.bits_d", cw)
        self.assertEqual(p.count('layer._exl3_k_d = int(getattr(layer, "_exl3_bits_d", layer._exl3_bits))'), 2)

    def _module(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "exl3_kbits.py"
            path.write_text(self.patched)
            return _load("exl3_kbits", path)

    def test_expert_bits_config(self) -> None:
        try:
            mod = self._module()
        except Exception as exc:  # pragma: no cover - host without the module's soft deps
            self.skipTest(f"plugin module not importable here: {exc!r}")
        cfg = mod.Exl3Config.from_config({"quant_method": "exl3", "bits": 2, "codebook": "mcg",
                                          "layer_bits": {"5": 4},
                                          "expert_bits": {"0": [3, 2], "7": [2, 4], "39": [4, 4]}})
        self.assertEqual(cfg.expert_bits, {0: (3, 2), 7: (2, 4), 39: (4, 4)})
        self.assertEqual(cfg.expert_bits_for_prefix("model.layers.0.ffn.experts"), (3, 2))
        self.assertEqual(cfg.expert_bits_for_prefix("layers.7.ffn.experts"), (2, 4))
        self.assertEqual(cfg.expert_bits_for_prefix("model.layers.5.ffn.experts"), (4, 4))  # layer_bits
        self.assertEqual(cfg.expert_bits_for_prefix("model.layers.6.ffn.experts"), (2, 2))  # bits
        self.assertEqual(cfg.expert_bits_for_prefix("model.layers.70.ffn.experts"), (2, 2))
        self.assertNotIn("expert_bits", cfg.raw_config)
        plain = mod.Exl3Config.from_config({"quant_method": "exl3", "bits": 2, "codebook": "mcg"})
        self.assertEqual(plain.expert_bits, {})
        self.assertEqual(plain.expert_bits_for_prefix("model.layers.3.ffn.experts"), (2, 2))
        for bad in ({"1": [5, 2]}, {"1": [2, 1]}, {"1": [3]}, {"1": "32"}):
            with self.assertRaises(ValueError, msg=bad):
                mod.Exl3Config.from_config({"quant_method": "exl3", "bits": 2, "expert_bits": bad})


class KbitsKmixTests(unittest.TestCase):
    """exl3_moe with compile-time (K_gu, K_d) in vllm_exl3_c (prefill at mixed K)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = _load("widen_p2b_kbits")
        cls.src = (ROOT / "tests/fixtures/exl3_moe_kernel_e14.pin.cuh").read_text()
        cls.out = cls.kb.kmix_kernel(cls.src)

    def test_kernel_rewrite_is_the_listed_edits_only(self) -> None:
        out = self.out.split("\n", 1)[1]  # generated-by line
        out = out.replace("exl3_gemm_kernel_inner<t_bits_gu,", "exl3_gemm_kernel_inner<t_bits,").replace(
            "exl3_gemm_kernel_inner<t_bits_d,", "exl3_gemm_kernel_inner<t_bits,")
        out = out.replace("if constexpr (t_bits_gu)", "if constexpr (t_bits)").replace("if constexpr (t_bits_d)", "if constexpr (t_bits)")
        for old, new, _ in self.kb.KMIX_KERNEL_EDITS:
            out = out.replace(new, old)
        self.assertEqual(out, self.src)
        gu = _block(self.out, "auto gemm_up", "auto had_guad")
        d = _block(self.out, "auto gemm_down", "auto had_d_out")
        self.assertIn("exl3_gemm_kernel_inner<t_bits_gu, false, cb, SHAPE_ARGS, false>(ARGS);", gu)
        self.assertIn("exl3_gemm_kernel_inner<t_bits_d, false, cb, SHAPE_ARGS, false>(ARGS);", d)
        self.assertNotIn("t_bits_d", gu)
        self.assertNotIn("t_bits_gu", d)
        self.assertIn("gemm_up(temp_state_u, temp_intermediate_u, exp_up_trellis, K_up);", self.out)
        self.assertIn("gemm_down(temp_intermediate_g, temp_state_g, exp_down_trellis, K_down);", self.out)

    def test_host_entry(self) -> None:
        cu = self.kb.KMIX_CU
        for a, b in MIXED:
            self.assertIn(f"KMIX({a}, {b});", cu)
        self.assertIn("exl3_moe_kmix_kernel<A, B, 256, 1> : exl3_moe_kmix_kernel<A, B, 128, 1>", cu)
        self.assertIn("(MAX_TILES_C + MAX_BARRIERS * 2 + MOE_SCHED_INTS) * sizeof(int)", cu)  # = DevCtx locks
        self.assertIn("cuda_check(cudaMemset(locks[device], 0, size));", cu)
        self.assertIn("TORCH_CHECK(gate_mcg && up_mcg && down_mcg && !gate_mul1 && !up_mul1 && !down_mul1,", cu)
        # exllamav3 exl3_moe's launch rule
        self.assertIn("int num_groups = MIN(concurrency, MOE_MAX_GROUPS);", cu)
        self.assertIn("group_size = MIN(num_sms / num_groups, MOE_MAX_SMS_PER_EXPERT);", cu)
        self.assertIn("cudaLaunchKernel((void*) kernel, grid_dim, block_dim, kernelArgs, SMEM_MAX, stream)", cu)

    def test_setup_and_bindings(self) -> None:
        kb = self.kb
        fix = _load("fix_vllm_exl3_setup")
        setup = fix.patch("""                sources=[
                    str(ROOT / "csrc" / "bindings.cpp"),
                    str(ROOT / "csrc" / "exl3_gemv.cu"),
                    str(ROOT / "csrc" / "p2b_batched.cu"),
                    str(ROOT / "csrc" / "p2b_moe.cu"),
                    str(ROOT / "csrc" / "exl3_gemm.cu"),
                    str(ROOT / "csrc" / "exl3_fat_gemm.cu"),
                ],""")
        out = kb.patch_setup(setup)
        self.assertIn('"csrc/exl3_moe_kmix.cu",', out)
        self.assertEqual(kb.patch_setup(out), out)
        b = '#include "exl3_fat_gemm.cuh"\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n    m.attr("P2B_MOE_ABI_VERSION") = 2;\n}\n'
        bo = kb.patch_bindings(b)
        self.assertIn('m.def("exl3_moe_kmix", &exl3_moe_kmix,', bo)
        self.assertIn("void exl3_moe_kmix(", bo)
        self.assertEqual(kb.patch_bindings(bo), bo)

    def test_plugin_picks_kmix_for_mixed_k(self) -> None:
        p = self.kb.patch_py(PIN_PY.read_text())
        fused = _block(p, "def apply_exl3_fused_moe(", "\ndef ")
        self.assertIn('fn = getattr(_load_native_exl3_ext(), "exl3_moe_kmix", None) or fn', fused)
        self.assertLess(fused.index("fn = exllamav3_ext.exl3_moe"), fused.index("exl3_moe_kmix"))
        self.assertLess(fused.index("exl3_moe_kmix"), fused.index("    args = ("))
        self.assertIn("if k_d != k and bool(", fused)


class KbitsImageTests(unittest.TestCase):
    def test_dockerfile_e15(self) -> None:
        e15 = (ROOT / "docker/Dockerfile.e15").read_text()
        e14 = (ROOT / "docker/Dockerfile.e14").read_text()
        self.assertIn("FROM dsv41-flash-exl3-sm121:canonical-e14\n", e15)
        ref = re.search(r"ARG VLLM_EXL3_REF=(\w+)", e14).group(1)
        self.assertIn(f"ARG VLLM_EXL3_REF={ref}\n", e15)
        stages = [re.findall(r"(widen_p2b_\w+)\.py /opt/vllm-exl3", df) for df in (e14, e15)]
        self.assertEqual(stages[1], stages[0] + ["widen_p2b_kbits"])
        self.assertLess(e15.index("widen_p2b_kbits.py /opt/vllm-exl3"),
                        e15.index("pip install --no-build-isolation --no-deps --force-reinstall /opt/vllm-exl3"))
        labels = [re.search(r'LABEL dsv41.recipe.patches="([^"]+)"', df).group(1).split(",") for df in (e14, e15)]
        want = list(labels[0])
        want.insert(want.index("dataflow") + 1, "kbits")
        self.assertEqual(labels[1], want)
        needle = _load("decode_levers").P2B_DATAFLOW_NEEDLE
        self.assertIn(f"b'{needle}' in so", e15)
        self.assertIn("b'p2b_coop_df_kernelILi23ELi1E' in so", e15)
        self.assertIn("hasattr(e.Exl3Config, 'expert_bits_for_prefix')", e15)
        self.assertIn("hasattr(vllm_exl3_c, 'exl3_moe_kmix')", e15)
        self.assertIn("cp /opt/dsv41-patch/sitecustomize.py /usr/lib/python3.12/sitecustomize.py", e15)


if __name__ == "__main__":
    unittest.main()
