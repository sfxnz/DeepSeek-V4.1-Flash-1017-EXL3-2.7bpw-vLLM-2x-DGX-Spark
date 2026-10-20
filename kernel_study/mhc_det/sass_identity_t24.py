#!/usr/bin/env python3
"""CPU only (no GPU needed): NVRTC-compile two mhc_det.cu sources for sm_121a exactly as mhc_det_rt.Module does
and require every kernel of the old one byte-identical in the new one (machine code, .text sections); list
the new kernels (c4-kernels: mhc_det_gemm_t24).

  python3 kernel_study/mhc_det/sass_identity_t24.py OLD.cu NEW.cu [--opts ...]
  # e.g. in the serve image, no GPU:
  git show 20ee8ba:docker/patch/mhc_det.cu > /tmp/mhc_det.e15.cu
  docker run --rm --network none --memory 4g -v "$PWD":/repo -v /tmp/mhc_det.e15.cu:/old.cu:ro -w /repo \
    --entrypoint python3 dsv41-flash-exl3-sm121:e15-kbits kernel_study/mhc_det/sass_identity_t24.py /old.cu docker/patch/mhc_det.cu
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

from cuda.bindings import nvrtc

HERE = Path(__file__).resolve().parent


def cubin(src: str, name: str, arch: str, opts: list[str]) -> bytes:
    def ok(res, what):
        if res[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise SystemExit(f"{what}: {res[0]}")
        return res[1] if len(res) == 2 else res[1:]

    prog = ok(nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], []), "create")
    options = [f"--gpu-architecture={arch}".encode(), b"-std=c++17", b"-default-device"] + [o.encode() for o in opts]
    res = nvrtc.nvrtcCompileProgram(prog, len(options), options)
    size = ok(nvrtc.nvrtcGetProgramLogSize(prog), "log size")
    log = b" " * size
    nvrtc.nvrtcGetProgramLog(prog, log)
    if res[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise SystemExit(f"compile {name}: {res[0]}\n{log.decode(errors='replace')}")
    size = ok(nvrtc.nvrtcGetCUBINSize(prog), "cubin size")
    out = b" " * size
    ok(nvrtc.nvrtcGetCUBIN(prog, out), "cubin")
    nvrtc.nvrtcDestroyProgram(prog)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--arch", default="sm_121a")
    ap.add_argument("--opts", nargs="*", default=[])
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("text_identity", HERE.parent / "p2b_srcsort" / "text_identity.py")
    ti = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ti)
    secs = []
    with tempfile.TemporaryDirectory() as td:
        for tag, path in (("old", args.old), ("new", args.new)):
            out = Path(td) / f"{tag}.cubin"
            out.write_bytes(cubin(path.read_text(), "mhc_det.cu", args.arch, args.opts))
            secs.append({k[len(".text."):]: v for k, v in ti.sections(str(out)).items() if k.startswith(".text.")})
    old, new = secs
    bad = 0
    for name in sorted(old):
        same = old[name] == new.get(name)
        bad += not same
        print(f"{name}: {len(old[name])} vs {len(new.get(name, b''))} bytes {'IDENTICAL' if same else 'DIFFERENT'}")
    for name in sorted(set(new) - set(old)):
        print(f"new {name}: {len(new[name])} bytes")
    print("sass_identity_t24:", "OK" if not bad and old else "FAIL")
    return 0 if not bad and old else 1


if __name__ == "__main__":
    raise SystemExit(main())
