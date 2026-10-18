#!/usr/bin/env python3
"""Rebuild model.safetensors.index.json from shard headers in a pack dir.

Sept-era K=2 rebuild helper (called by tools/assemble_pack.sh). The 1017 pack's index is
written by `tools/build_pack_1017.py assemble`.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def tensor_names(path: Path) -> list[str]:
    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    return [k for k in hdr if k != "__metadata__"]


def rebuild(dst: Path) -> dict[str, str]:
    weight_map: dict[str, str] = {}
    for shard in sorted(dst.glob("model-*-of-*.safetensors")):
        if shard.name.endswith(".tmp"):
            continue
        for name in tensor_names(shard):
            weight_map[name] = shard.name
    idx = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    path = dst / "model.safetensors.index.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(idx, indent=2) + "\n")
    tmp.replace(path)
    return weight_map


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dst", type=Path, required=True, help="pack snapshot dir")
    args = ap.parse_args()
    wm = rebuild(args.dst)
    print(f"wrote {args.dst / 'model.safetensors.index.json'} tensors={len(wm)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
