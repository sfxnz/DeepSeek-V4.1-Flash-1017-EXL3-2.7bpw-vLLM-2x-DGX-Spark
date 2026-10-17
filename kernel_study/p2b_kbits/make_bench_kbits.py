#!/usr/bin/env python3
"""K-bits microbench sources from the committed pin (CPU only).

Writes build/:
  chain_kbits.cu  the round-3 chain (kernel_study/p2b_coop/make_bench_r3.chain(): pin + Dockerfile.e14
                  p2b stages) + docker/patch/widen_p2b_kbits.py = what Dockerfile.e15 compiles
  bench_kbits.cu  chain_kbits with make_bench_r3's runtime variant switch (g_bench_variant =
                  DSV41_P2B_COOP value: 0 p2b SORT=0, 1 coop, 2 dataflow) and pybind module
The phase-stamped build of make_bench_r3 is not generated (its dataflow-tile stamp edits are K=2
text); the stream / bw / ld / flat probes are included.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "build"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R3 = _load("make_bench_r3", ROOT / "kernel_study/p2b_coop/make_bench_r3.py")
KBITS = _load("widen_p2b_kbits", ROOT / "docker/patch/widen_p2b_kbits.py")


def base_chain() -> str:
    """The canonical-e14 TU (pin + shapes .. dataflow)."""
    return R3.chain()


def chain() -> str:
    return KBITS.patch_cu(base_chain())


# Bench-only comparison arm: the dataflow kernel at 3 blocks/SM (80 registers) for every K, as before
# widen_p2b_kbits moved K != 2 to 2 blocks/SM (ab_lb.py).
LB_KBITS = ("__global__ __launch_bounds__(256, BITS == 2 ? 3 : 2)  // widen_p2b_kbits: K != 2 at 2 blocks/SM\n"
            "void p2b_coop_df_kernel(")
LB3 = "__global__ __launch_bounds__(256, 3)\nvoid p2b_coop_df_kernel("


def build() -> dict[str, str]:
    src = chain()
    bench = R3.bench(src, False)
    if bench.count(LB_KBITS) != 1:
        raise SystemExit("make_bench_kbits: dataflow launch bounds not found")
    return {"chain_r3.cu": base_chain(), "chain_kbits.cu": src, "bench_kbits.cu": bench,
            "bench_kbits_lb3.cu": bench.replace(LB_KBITS, LB3)}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    for name, text in build().items():
        path = OUT / name
        if path.exists() and path.read_text() == text:
            print(f"unchanged {path}")
            continue
        path.write_text(text)
        print(f"wrote {path} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
