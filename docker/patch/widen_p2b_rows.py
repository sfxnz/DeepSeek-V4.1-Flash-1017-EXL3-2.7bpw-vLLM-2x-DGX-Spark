#!/usr/bin/env python3
"""Native p2b decode MoE for 9..24 rows: the dataflow kernel with a 144-pair sort cap (c >= 3 verify).

Before this patch the native path stopped at 8 rows: vllm_exl3 sends at most _NATIVE_MOE_CONTRACT_ROWS = 8
rows to p2b_fused_moe (the host checks 1..8), and the dataflow kernel (DSV41_P2B_COOP=2) takes
m * top_k <= P2B_SORT_CAP = 64 (row, expert) pairs. Four sessions at DSpark k = 3 verify 16 rows (96 pairs)
and adaptive depth up to k = 5 verifies 24 (144 pairs): those batches ran exllamav3 exl3_moe (exl3_moe_kmix
at K_gu != K_d), which decodes every active expert's full trellis per 16-row tile and runs ~40 us per active
expert whatever its K (kernel_study/p2b_kbits/prefill_kbits.py, m = 512), against ~24 us for the dataflow
kernel (bandwidth-bound, 89-94 % of the floor at m = 8, kb-final-time.log).

Kernel side (csrc/p2b_moe.cu, apply after widen_p2b_kbits.py):
- p2b_coop_dfw_kernel<BITS, CB> is p2b_coop_df_kernel's text (generated from it, see DFW_SUBS) with its
  sort / chunk arrays, unit count and control words sized by P2B_WIDE_CAP = 144 instead of 64, and its own
  shared arrays, so the e15 kernels' shared layout and machine code are untouched
  (kernel_study/p2b_rows/sass_identity_rows.sh: the 42 e15 p2b kernels byte-identical, 9 new). Chunks keep
  <= 8 rows (MMA rows 0..7): at m = 16 / 24 the census routing puts > 8 rows on 0.36 / 1.5 experts per
  layer, so splitting their B stream costs +0.7 / +2.3 % bytes (kernel_study/p2b_rows/README.txt).
- Same tiles (p2b_df_tile), epilogue, output sum and grid barriers: every row's output is bitwise the one
  the m <= 8 dataflow kernel gives that row (per-row MMA chain, fixed-order slot sum), and for one-hot
  routing weights the SORT=0 p2b's.
- Host: 1..24 rows (P2B_WIDE_ROWS). DSV41_P2B_COOP=2 with m * top_k in (64, 144] (MCG) launches the wide
  kernel; anything else is unchanged (m * top_k <= 64: the dataflow kernel; other modes: SORT=0/1/2, which
  have no row limit). LOG_ENGAGED is printed once per K by the first wide launch.
- vllm_exl3_c.P2B_WIDE_ROWS = 24 (bindings.cpp) tells the plugin the extension takes the wide rows.

Plugin side (src/vllm_exl3/exl3.py, apply after widen_p2b_kbits.py):
- _native_moe_dimensions_supported: when the row cap is the default 8-row contract, rows up to
  _native_moe_wide_rows(layer) go native: vllm_exl3_c.P2B_WIDE_ROWS for MCG layers under
  DSV41_P2B_COOP=2, 0 with DSV41_P2B_WIDE=0 (the e15 contract) or without the attribute (older .so).
  A row-cap override below 8 (VLLM_EXL3_NATIVE_MOE_MAX_ROWS, the measured per-K cap) keeps its cap.
The anchors of the runtime levers that rewrite exl3.py (moe_prep_fused.py, pfg8_loader_reindex.py) are
left verbatim. Idempotent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARK = "// --- widen_p2b_rows"
WIDE_ROWS = 24
WIDE_CAP = 6 * WIDE_ROWS  # (row, expert) pairs at top-6
LOG_ENGAGED = "dsv41: p2b coop dataflow wide kernel engaged (DSV41_P2B_COOP=2, m * top_k <= 144)"
# BITS of the wide dataflow instances (= the dataflow kernel's, widen_p2b_kbits).
WIDE_BITS = (2, 3, 4, 23, 24, 32, 34, 42, 43)

DF_KERNEL_START = ("template <int BITS, int CB>\n__global__ __launch_bounds__(256, BITS == 2 ? 3 : 2)"
                   "  // widen_p2b_kbits: K != 2 at 2 blocks/SM\nvoid p2b_coop_df_kernel(")
DF_BUILD_START = "__device__ __forceinline__ void p2b_df_build_k(const int32_t* __restrict__ ids, int pairs)\n"
KERNEL_ANCHOR = "template <int BITS, int CB>\nstatic void launch_moe_batched("

# p2b_df_build_k -> p2b_dfw_build: (old, new, count in the function).
BUILD_SUBS = (
    ("void p2b_df_build_k(", "void p2b_dfw_build(", 1),
    ("p2b_df_sort_smem_k<BITS>()", "p2b_dfw_sort_smem<BITS>()", 1),
    ("p2b_df_chunk_smem_k<BITS>()", "p2b_dfw_chunk_smem<BITS>()", 1),
    ("P2B_SORT_CAP", "P2B_WIDE_CAP", 4),
)
# p2b_coop_df_kernel -> p2b_coop_dfw_kernel: (old, new, count in the kernel). Nothing else differs.
DFW_SUBS = (
    ("void p2b_coop_df_kernel(", "void p2b_coop_dfw_kernel(", 1),
    ("p2b_df_chunk_smem_k<BITS>()", "p2b_dfw_chunk_smem<BITS>()", 1),
    ("p2b_df_build_k<BITS>(ids, m * experts)", "p2b_dfw_build<BITS>(ids, m * experts)", 1),
    ("p2b_df_sort_smem_k<BITS>()", "p2b_dfw_sort_smem<BITS>()", 4),
    ("p2b_df_red_k<BITS>()", "p2b_dfw_red<BITS>()", 2),
    ("p2b_df_ctl_words(inter)", "p2b_dfw_ctl_words(inter)", 1),
    ("P2B_DF_ACT", "P2B_DFW_ACT", 1),
    ("P2B_SORT_CAP", "P2B_WIDE_CAP", 6),
)

HEADER = r'''// --- widen_p2b_rows: the dataflow kernel for m * top_k in (P2B_SORT_CAP, P2B_WIDE_CAP] ---
// p2b_coop_dfw_kernel is p2b_coop_df_kernel with the sort / chunk arrays, the unit count and the control
// words sized by P2B_WIDE_CAP, and its own shared arrays (the dataflow kernels' layout stays e15's). Same
// tiles, epilogue, slot sum and barriers: every row gets the bits the m <= 8 kernel gives it.
constexpr int P2B_WIDE_ROWS = @ROWS@;
constexpr int P2B_WIDE_CAP = @CAP@;  // P2B_WIDE_ROWS x top-6 (row, expert) pairs
constexpr int P2B_DFW_ACT = P2B_DF_READY + P2B_WIDE_CAP;

__host__ __device__ __forceinline__ int p2b_dfw_ctl_words(int inter)
{
    return P2B_DFW_ACT + P2B_WIDE_CAP * (inter / 128);
}

// The wide kernel takes this call (DSV41_P2B_COOP=2, MCG): more pairs than the dataflow kernel's cap.
static inline bool p2b_dfw_rows_ok(int m, int e, int hidden, int inter)
{
    return m * e > P2B_SORT_CAP && m * e <= P2B_WIDE_CAP && p2b_dfw_ctl_words(inter) <= m * hidden;
}

template <int BITS>
__device__ __forceinline__ int* p2b_dfw_sort_smem()
{
    __shared__ int s[3 * P2B_WIDE_CAP];
    return s;
}

template <int BITS>
__device__ __forceinline__ int* p2b_dfw_chunk_smem()
{
    __shared__ int s[2 * P2B_WIDE_CAP + 1];
    return s;
}

template <int BITS>
__device__ __forceinline__ P2bCoopRed* p2b_dfw_red()
{
    __shared__ P2bCoopRed s[8];
    return s;
}

'''.replace("@ROWS@", str(WIDE_ROWS)).replace("@CAP@", str(WIDE_CAP))

LAUNCH_OLD = """                std::fprintf(stderr, "%s\\n", "dsv41: p2b coop dataflow kernel engaged (DSV41_P2B_COOP=2)");
            }
        }
    }
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 256, 0);
"""
LAUNCH_NEW = """                std::fprintf(stderr, "%s\\n", "dsv41: p2b coop dataflow kernel engaged (DSV41_P2B_COOP=2)");
            }
        }
        // widen_p2b_rows: more pairs than P2B_SORT_CAP (9..24 rows at top-6) on the wide dataflow kernel.
        if (p2b_coop_mode() == 2 && p2b_dfw_rows_ok(m, e, hidden, inter)) {
            kernel = (void*) p2b_coop_dfw_kernel<BITS, CB>;
            static bool logged_wide = false;
            if (!logged_wide) {
                logged_wide = true;
                std::fprintf(stderr, "%s\\n", "LOG_ENGAGED");
            }
        }
    }
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 256, 0);
""".replace("LOG_ENGAGED", LOG_ENGAGED)

ROWS_OLD = """    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 8 && x.size(1) > 0 && x.size(1) % 128 == 0,
                "fused MoE requires 1-8 input rows whose hidden width is a positive multiple of 128");
"""
ROWS_NEW = """    // widen_p2b_rows: 1..P2B_WIDE_ROWS rows (more than 8 run the wide dataflow kernel under DSV41_P2B_COOP=2,
    // else the SORT=0 p2b, which has no row limit).
    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= P2B_WIDE_ROWS && x.size(1) > 0 && x.size(1) % 128 == 0,
                "fused MoE requires 1-@ROWS@ input rows whose hidden width is a positive multiple of 128");
""".replace("@ROWS@", str(WIDE_ROWS))

DF_FLAG_OLD = ("    const bool df = mcg && p2b_coop_mode() == 2 && m * e <= P2B_SORT_CAP && "
               "p2b_df_ctl_words(inter) <= m * hidden;\n")
DF_FLAG_NEW = ("    const bool df = mcg && p2b_coop_mode() == 2 && ((m * e <= P2B_SORT_CAP && "
               "p2b_df_ctl_words(inter) <= m * hidden)\n"
               "                                               || p2b_dfw_rows_ok(m, e, hidden, inter));  // widen_p2b_rows\n")

BIND_OLD = '    m.attr("P2B_MOE_ABI_VERSION") = 2;\n'
BIND_NEW = BIND_OLD + f'    m.attr("P2B_WIDE_ROWS") = {WIDE_ROWS};  // widen_p2b_rows: rows the dataflow kernels take\n'


def _sub1(src: str, old: str, new: str, label: str) -> str:
    if src.count(old) != 1:
        raise SystemExit(f"widen_p2b_rows: {label}: expected 1 match, found {src.count(old)}")
    return src.replace(old, new, 1)


def _function(src: str, start: str) -> str:
    """The text from `start` through the function's closing brace line."""
    if src.count(start) != 1:
        raise SystemExit(f"widen_p2b_rows: {start.strip()[:60]}: expected 1 match, found {src.count(start)}")
    a = src.index(start)
    return src[a : src.index("\n}\n", a) + 3]


def _copy(text: str, subs, label: str) -> str:
    for old, new, n in subs:
        if text.count(old) != n:
            raise SystemExit(f"widen_p2b_rows: {label} {old}: expected {n}, found {text.count(old)}")
        text = text.replace(old, new)
    return text


def wide_kernel(src: str) -> str:
    """HEADER + p2b_dfw_build + p2b_coop_dfw_kernel, generated from the kbits dataflow text."""
    build = "template <int BITS>\n" + _copy(_function(src, DF_BUILD_START), BUILD_SUBS, "p2b_df_build_k")
    kernel = _copy(_function(src, DF_KERNEL_START), DFW_SUBS, "p2b_coop_df_kernel")
    return HEADER + build + "\n" + kernel + "\n"


def patch_cu(src: str) -> str:
    if MARK in src:
        return src
    if "// --- widen_p2b_kbits" not in src:
        raise SystemExit("widen_p2b_rows: apply widen_p2b_kbits.py first")
    out = _sub1(src, KERNEL_ANCHOR, wide_kernel(src) + KERNEL_ANCHOR, "launch_moe_batched anchor")
    out = _sub1(out, LAUNCH_OLD, LAUNCH_NEW, "dataflow launch selection")
    out = _sub1(out, ROWS_OLD, ROWS_NEW, "row check")
    out = _sub1(out, DF_FLAG_OLD, DF_FLAG_NEW, "dataflow host flag")
    if out.count("grid.sync()") != src.count("grid.sync()") + 2:
        raise SystemExit("widen_p2b_rows: expected exactly the wide kernel's 2 grid barriers")
    return out


def patch_bindings(src: str) -> str:
    if '"P2B_WIDE_ROWS"' in src:
        return src
    return _sub1(src, BIND_OLD, BIND_NEW, "bindings module")


# ----------------------------------------------------------------------------- plugin (exl3.py)

PY_EDITS = (
    (
        """    return _NATIVE_MOE_CONTRACT_ROWS


def _apply_native_fused_moe(
""",
        """    return _NATIVE_MOE_CONTRACT_ROWS


def _native_moe_wide_rows(layer: torch.nn.Module) -> int:
    \"\"\"widen_p2b_rows: decode rows the wide dataflow kernel takes for this layer (0 = none).

    vllm_exl3_c.P2B_WIDE_ROWS (24 = 144 (row, expert) pairs at top-6) for MCG layers under
    DSV41_P2B_COOP=2, unless DSV41_P2B_WIDE=0. The kernel is the dataflow kernel with a larger sort
    cap: the same tiles, so every row's output is the one an m <= 8 call gives that row.
    \"\"\"
    if os.environ.get("DSV41_P2B_WIDE", "1") == "0" or os.environ.get("DSV41_P2B_COOP", "0") != "2":
        return 0
    flags = getattr(layer, "_exl3_codebook_flags", (True, False, True, False, True, False))
    if not bool(flags[0]) or bool(flags[1]):
        return 0
    module = _load_native_exl3_ext()
    return int(getattr(module, "P2B_WIDE_ROWS", 0)) if module is not None else 0


def _apply_native_fused_moe(
""",
        "wide rows helper",
    ),
    (
        """    return rows <= _native_moe_max_rows(bits)
""",
        """    # widen_p2b_rows: with the default 8-row contract, rows up to the wide dataflow kernel's cap.
    cap = _native_moe_max_rows(bits)
    if rows > cap and cap == _NATIVE_MOE_CONTRACT_ROWS:
        cap = max(cap, _native_moe_wide_rows(layer))
    return rows <= cap
""",
        "native row cap",
    ),
)


def patch_py(src: str) -> str:
    if "def _native_moe_wide_rows(" in src:
        return src
    if "raw_expert_bits = kwargs.pop(\"expert_bits\"" not in src:
        raise SystemExit("widen_p2b_rows: apply widen_p2b_kbits.py to exl3.py first")
    out = src
    for old, new, label in PY_EDITS:
        out = _sub1(out, old, new, label)
    return out


def apply(root: Path) -> None:
    for path, fn in ((root / "csrc" / "p2b_moe.cu", patch_cu), (root / "csrc" / "bindings.cpp", patch_bindings),
                     (root / "src" / "vllm_exl3" / "exl3.py", patch_py)):
        path.write_text(fn(path.read_text()))
        print(f"patched {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    apply(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
