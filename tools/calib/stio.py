"""Safetensors I/O for the layer-streaming harness.

- `Checkpoint`: name -> shard map (index json, or shard headers when the index is absent), bulk reads of
  many tensors with coalesced preads, and POSIX_FADV_DONTNEED afterwards so streaming 300 GB per pass
  does not pin the GB10's unified memory in page cache.
- `EngramTable`: row gather from the ~100 GB fp8 Engram tables by sorted pread (no full load).
- `Exl3Pack`: reads `.trellis/.suh/.svh` of routed experts from an EXL3 pack directory and reconstructs
  dense fp16 weights with exllamav3 (`reconstruct_had_slice`), K taken from the trellis shape.
"""

from __future__ import annotations

import json
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

DTYPES = {
    "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu,
    "I8": torch.int8, "U8": torch.uint8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
    "BOOL": torch.bool,
}


def read_header(path: str | os.PathLike) -> tuple[int, dict]:
    """(data_start, header dict without __metadata__)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


def _drop_cache(path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


class Checkpoint:
    """A safetensors checkpoint directory (HF snapshot or EXL3 pack)."""

    CHUNK = 32 << 20  # parallel pread piece size

    def __init__(self, root: str | os.PathLike, index: str | os.PathLike | None = None, drop_cache: bool = True,
                 io_threads: int = 8):
        self.root = Path(root)
        self.drop_cache = drop_cache
        self.io_pool = ThreadPoolExecutor(max_workers=io_threads)
        idx_path = Path(index) if index else self.root / "model.safetensors.index.json"
        if idx_path.exists():
            self.weight_map: dict[str, str] = json.loads(idx_path.read_text())["weight_map"]
        else:
            self.weight_map = {}
            for p in sorted(self.root.glob("*.safetensors")):
                for k in read_header(p)[1]:
                    self.weight_map[k] = p.name
        self._headers: dict[str, tuple[int, dict]] = {}
        self._lock = threading.Lock()

    def header(self, fname: str) -> tuple[int, dict]:
        with self._lock:
            if fname not in self._headers:
                self._headers[fname] = read_header(self.root / fname)
            return self._headers[fname]

    def names(self, prefix: str = "") -> list[str]:
        return [n for n in self.weight_map if n.startswith(prefix)]

    def meta(self, name: str) -> dict:
        return self.header(self.weight_map[name])[1][name]

    def read_cpu(self, names: list[str], max_gap: int = 1 << 20) -> dict[str, torch.Tensor]:
        """Read tensors to CPU with coalesced preads (ranges merged when gaps < max_gap)."""
        by_file: dict[str, list[str]] = {}
        for n in names:
            by_file.setdefault(self.weight_map[n], []).append(n)
        out: dict[str, torch.Tensor] = {}
        for fname, fnames in by_file.items():
            start, hdr = self.header(fname)
            items = sorted(((hdr[n]["data_offsets"][0], hdr[n]["data_offsets"][1], n) for n in fnames))
            runs, cur = [], None
            for a, b, n in items:
                if cur and a - cur[1] <= max_gap:
                    cur[1] = max(cur[1], b)
                    cur[2].append((a, b, n))
                else:
                    cur = [a, b, [(a, b, n)]]
                    runs.append(cur)
            path = self.root / fname
            fd = os.open(path, os.O_RDONLY)
            try:
                for a, b, members in runs:
                    buf = np.empty(b - a, dtype=np.uint8)
                    mv = memoryview(buf)

                    def piece(lo, hi):
                        got = lo
                        while got < hi:
                            r = os.preadv(fd, [mv[got:hi]], start + a + got)
                            if r <= 0:
                                raise IOError(f"short read {path}")
                            got += r

                    step = self.CHUNK
                    list(self.io_pool.map(lambda lo: piece(lo, min(b - a, lo + step)), range(0, b - a, step)))
                    t = torch.from_numpy(buf)
                    for ta, tb, n in members:
                        m = hdr[n]
                        dt = DTYPES[m["dtype"]]
                        raw = t[ta - a : tb - a]
                        if (ta - a) % dt.itemsize:  # misaligned for a view of this dtype
                            raw = raw.clone()
                        out[n] = raw.view(dt).reshape(m["shape"]) if raw.numel() else torch.empty(m["shape"], dtype=dt)
            finally:
                os.close(fd)
            if self.drop_cache:
                _drop_cache(path)
        return out

    def read(self, names: list[str], device="cuda") -> dict[str, torch.Tensor]:
        return {k: v.to(device, non_blocking=False) for k, v in self.read_cpu(names).items()}


class Prefetcher:
    """Reads the next group of tensors to CPU in a background thread."""

    def __init__(self, ckpt: Checkpoint):
        self.ckpt = ckpt
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending: dict = {}

    def submit(self, key, names: list[str]):
        if key not in self.pending:
            self.pending[key] = self.pool.submit(self.ckpt.read_cpu, names)

    def get(self, key, names: list[str]) -> dict[str, torch.Tensor]:
        self.submit(key, names)
        return self.pending.pop(key).result()


class EngramTable:
    """Row gather from `layers.L.engram.embed.{weight,scale}` (fp8 rows + E8M0 scales) on disk."""

    def __init__(self, ckpt: Checkpoint, layer: int, threads: int = 64):  # ~130K random rows/s on spark2 NVMe
        self.ckpt = ckpt
        self.parts = {}
        for kind in ("weight", "scale"):
            name = f"layers.{layer}.engram.embed.{kind}"
            fname = ckpt.weight_map[name]
            start, hdr = ckpt.header(fname)
            m = hdr[name]
            self.parts[kind] = (ckpt.root / fname, start + m["data_offsets"][0], m["shape"])
        self.pool = ThreadPoolExecutor(max_workers=threads)
        self.threads = threads

    def _gather(self, kind: str, rows: np.ndarray) -> np.ndarray:
        path, off, shape = self.parts[kind]
        width = int(shape[1])
        out = np.empty((len(rows), width), dtype=np.uint8)
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)

            def work(lo, hi):
                # rows are sorted; merge rows lying within one 64 KiB window into one pread
                i = lo
                while i < hi:
                    j = i + 1
                    while j < hi and (rows[j] - rows[i]) * width < 65536:
                        j += 1
                    a = int(rows[i]) * width
                    b = int(rows[j - 1]) * width + width
                    buf = os.pread(fd, b - a, off + a)
                    arr = np.frombuffer(buf, dtype=np.uint8)
                    for k in range(i, j):
                        s = int(rows[k]) * width - a
                        out[k] = arr[s : s + width]
                    i = j

            n = len(rows)
            step = max(1, (n + self.threads * 4 - 1) // (self.threads * 4))
            list(self.pool.map(lambda lo: work(lo, min(n, lo + step)), range(0, n, step)))
        finally:
            os.close(fd)
        return out

    def lookup(self, idx: torch.Tensor) -> torch.Tensor:
        """idx int64 [...]; returns bf16 [..., 256] dequantized exactly as ParallelEngramEmbedding."""
        flat = idx.reshape(-1)
        uniq, inv = torch.unique(flat, return_inverse=True)
        rows = uniq.cpu().numpy()
        w = torch.from_numpy(self._gather("weight", rows)).to(idx.device).view(torch.float8_e4m3fn)
        s = torch.from_numpy(self._gather("scale", rows)).to(idx.device).view(torch.float8_e8m0fnu)
        vals = w.float().unflatten(-1, (-1, 32)) * s.float().unsqueeze(-1)
        vals = vals.flatten(-2).to(torch.bfloat16)
        return vals[inv].view(*idx.shape, -1)

    def drop_cache(self):
        for path, _, _ in self.parts.values():
            _drop_cache(path)


class Exl3Pack:
    """Routed-expert EXL3 tensors from a pack dir; reconstructs [out, in] fp16 weights."""

    def __init__(self, root: str | os.PathLike):
        self.ckpt = Checkpoint(root)
        cfg = json.loads((Path(root) / "config.json").read_text())
        self.qcfg = cfg.get("quantization_config", {})
        self._ext = None

    def k_of(self, layer: int, expert: int, proj: str) -> int:
        return int(self.ckpt.meta(f"layers.{layer}.ffn.experts.{expert}.{proj}.trellis")["shape"][-1]) // 16

    def codebook(self, layer: int) -> str:
        base = f"layers.{layer}.ffn.experts.0.w1"
        if f"{base}.mcg" in self.ckpt.weight_map:
            return "mcg"
        if f"{base}.mul1" in self.ckpt.weight_map:
            return "mul1"
        return "3inst"

    def layer_names(self, layer: int, n_experts: int) -> list[str]:
        return [f"layers.{layer}.ffn.experts.{e}.{p}.{s}" for e in range(n_experts) for p in ("w1", "w2", "w3")
                for s in ("trellis", "suh", "svh")]

    def reconstruct(self, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, codebook: str) -> torch.Tensor:
        """-> fp16 [out, in] (exllamav3 reconstructs W^T as [in, out])."""
        if self._ext is None:
            from exllamav3.ext import exllamav3_ext as ext
            self._ext = ext
        K = trellis.shape[-1] // 16
        n_in, n_out = suh.numel(), svh.numel()
        o = torch.empty((n_in, n_out), dtype=torch.half, device=trellis.device)
        self._ext.reconstruct_had_slice(o, trellis, suh.half(), svh.half(), K, codebook == "mcg", codebook == "mul1", 0)
        return o.t()
