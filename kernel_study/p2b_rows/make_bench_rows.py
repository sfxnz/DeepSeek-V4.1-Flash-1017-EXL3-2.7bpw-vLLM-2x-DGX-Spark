#!/usr/bin/env python3
"""Wide-rows microbench sources from the committed pin (CPU only).

Writes build/:
  chain_kbits.cu  the e15 TU (kernel_study/p2b_kbits/make_bench_kbits.chain(): pin + Dockerfile.e15 p2b stages)
  chain_rows.cu   chain_kbits + docker/patch/widen_p2b_rows.py = what Dockerfile.e16 compiles
  bench_rows.cu   chain_rows with make_bench_r3's runtime variant switch (g_bench_variant = DSV41_P2B_COOP
                  value: 0 p2b SORT=0, 1 coop, 2 dataflow incl. the wide kernel) and pybind module
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


KB = _load("make_bench_kbits", ROOT / "kernel_study/p2b_kbits/make_bench_kbits.py")
ROWS = _load("widen_p2b_rows", ROOT / "docker/patch/widen_p2b_rows.py")


def build() -> dict[str, str]:
    base = KB.chain()
    src = ROWS.patch_cu(base)
    return {"chain_kbits.cu": base, "chain_rows.cu": src, "bench_rows.cu": KB.R3.bench(src, False)}


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
