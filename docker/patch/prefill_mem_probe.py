"""Where does MemFree go outside the torch pool? In-worker probe (DSV41_PREFILL_MEM_PROBE=1, default 0).

perf2 integration4 boots PM-B4K and PM-A1: the node of rank 0 (spark1) loses 1.3-2.9 GiB of MemFree
in the first second of a long prompt's first chunks with torch's pool flat, and gets it back seconds
later, at the prompt end or at the next NVRM NV_ERR_NO_MEMORY; spark2 (rank 1, same steps) only
shows its page cache moving. No /proc/meminfo bucket and no process RSS holds it, and Worker_TP0
is the only CUDA process on spark1, so it is driver memory of that process. This probe reads, from
inside the worker, the stores such memory can sit in:
  - the CUDA default memory pool (stream-ordered cudaMallocAsync / cudaFreeAsync of a library),
    reserved and used bytes and its release threshold;
  - CUDA graph memory (reserved bytes);
  - the context's per-thread stack limit (local-memory reservation);
  - torch's pinned host cache;
  - Python's cyclic GC (library buffers freed only by a collection), as collections per generation;
  - frees the driver defers until the GPU catches up (processed by a device sync).
Per prefill step it prints a 'probe step' line: MemFree at the step's start and end, torch's pool
around the step, the MemFree loss the pool does not explain, cudaMemGetInfo free, and the values
above. On the first FLUSH_EVENTS steps that lose at least FLUSH_MIB more than the pool grew, it then
calls, in this order, and prints MemFree's change after each: torch.cuda.synchronize(),
gc.collect(), torch's pinned host cache release, cudaMemPoolTrimTo(default pool, 0),
cudaDeviceGraphMemTrim(device), cudaDeviceSetLimit(stack, its current value), and last a
SETTLE_S wait with this process blocked and the GPU idle after the sync (MemFree polled every
POLL_S). The call after which the memory comes back names the store; memory that comes back during
the wait, with nothing allocating, is released late by the driver (e.g. freed pages the RM
scrubs before they return; no public API reports scrub progress).

After each of the first RELEASE_SERIES releases of the hook (floor / grown / release), a daemon
thread samples MemFree and torch's reserved bytes every POLL_S for SERIES_S and prints a
'probe release' line: how fast the released pages come back to MemFree, while the next steps run.
DSV41_PREFILL_MEM_PROBE=2 keeps only the step lines (passive: no flush, no series, no sync).
Nothing here changes a computed value.

The CUDA runtime is the libcudart this process already mapped (ctypes over the same library, so
the same primary context as torch). Top-level imports are stdlib only.
"""

from __future__ import annotations

import ctypes
import gc
import os
import threading
import time

FLUSH_EVENTS = 6  # flush experiments per process
FLUSH_MIB = 512  # unexplained MemFree loss in one step that triggers one
SETTLE_S = 2.0  # the flush's last step: blocked wait, MemFree polled
POLL_S = 0.1
RELEASE_SERIES = 8  # hook releases followed by a MemFree time series
SERIES_S = 5.0
SERIES_AT = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)  # printed sample times (s)
LIMIT_STACK = 0  # cudaLimitStackSize
POOL_THRESHOLD, POOL_RESERVED, POOL_USED = 4, 5, 7  # cudaMemPoolAttr*
GRAPH_RESERVED = 2  # cudaGraphMemAttrReservedMemCurrent


def mode(env=None) -> str:
    """'' (off), 'full' (DSV41_PREFILL_MEM_PROBE=1) or 'passive' (=2: step lines only, no flush, no series)."""
    env = os.environ if env is None else env
    return {"1": "full", "2": "passive"}.get(env.get("DSV41_PREFILL_MEM_PROBE", "0"), "")


def enabled(env=None) -> bool:
    return bool(mode(env))


def memfree_kib(path: str = "/proc/meminfo") -> int:
    with open(path) as fh:
        for line in fh:
            if line.startswith("MemFree:"):
                return int(line.split()[1])
    raise ValueError(f"no MemFree in {path}")


class Cudart:
    """Reads and trims through the CUDA runtime already loaded in this process."""

    def __init__(self, lib) -> None:
        self.lib = lib

    @classmethod
    def load(cls, maps: str = "/proc/self/maps"):
        try:
            with open(maps) as fh:
                path = next((ln.split()[-1] for ln in fh if "/libcudart.so" in ln), None)
            return cls(ctypes.CDLL(path)) if path else None
        except OSError:
            return None

    def _ok(self, name: str, *args) -> bool:
        return getattr(self.lib, name)(*args) == 0

    def device(self) -> int:
        dev = ctypes.c_int(0)
        return dev.value if self._ok("cudaGetDevice", ctypes.byref(dev)) else 0

    def stack(self):
        value = ctypes.c_size_t(0)
        return value.value if self._ok("cudaDeviceGetLimit", ctypes.byref(value), LIMIT_STACK) else None

    def _pool(self):
        pool = ctypes.c_void_p(0)
        return pool if self._ok("cudaDeviceGetDefaultMemPool", ctypes.byref(pool), self.device()) else None

    def pool(self):
        """(reserved, used, release threshold) bytes of the default memory pool, or None."""
        pool = self._pool()
        if pool is None:
            return None
        out = []
        for attr in (POOL_RESERVED, POOL_USED, POOL_THRESHOLD):
            value = ctypes.c_uint64(0)
            if not self._ok("cudaMemPoolGetAttribute", pool, attr, ctypes.byref(value)):
                return None
            out.append(value.value)
        return tuple(out)

    def graph_reserved(self):
        value = ctypes.c_uint64(0)
        ok = self._ok("cudaDeviceGetGraphMemAttribute", self.device(), GRAPH_RESERVED, ctypes.byref(value))
        return value.value if ok else None

    def trim_pool(self) -> bool:
        pool = self._pool()
        return pool is not None and self._ok("cudaMemPoolTrimTo", pool, ctypes.c_size_t(0))

    def trim_graph(self) -> bool:
        return self._ok("cudaDeviceGraphMemTrim", self.device())

    def reset_stack(self) -> bool:
        value = self.stack()
        return value is not None and self._ok("cudaDeviceSetLimit", LIMIT_STACK, ctypes.c_size_t(value))


def pinned_bytes(torch):
    """torch's pinned host cache (reserved bytes), or None when this torch does not report it."""
    fn = getattr(torch.cuda, "host_memory_stats", None)
    if fn is None:
        return None
    try:
        stats = fn()
    except Exception:  # noqa: BLE001 - a probe never breaks a step
        return None
    value = stats.get("reserved_bytes.current")
    if value is None and isinstance(stats.get("reserved_bytes"), dict):
        value = stats["reserved_bytes"].get("current")
    return value


def host_empty_cache(torch):
    fn = getattr(getattr(torch, "_C", None), "_host_emptyCache", None)
    return fn


def _mib(value):
    return "-" if value is None else f"{value / 2**20:.0f}"


class Probe:
    """One per worker process; the hook calls start() before a prefill step and end() after it."""

    def __init__(self, torch, rt, say, memfree=None, clock=None, sleep=None, spawn=None, passive=False) -> None:
        self.torch, self.rt, self.say, self.passive = torch, rt, say, passive
        self.memfree = memfree if memfree is not None else (lambda: memfree_kib())
        self.clock = clock if clock is not None else time.monotonic
        self.sleep = sleep if sleep is not None else time.sleep
        self.spawn = spawn if spawn is not None else (
            lambda target, args: threading.Thread(target=target, args=args, daemon=True).start())
        self.steps = self.flushes = self.series = 0
        self.free0 = None

    def start(self) -> None:
        self.free0 = self.memfree()

    def end_free(self) -> int:
        """MemFree right after the step (read before the hook releases anything)."""
        return self.memfree()

    def stats(self) -> str:
        t, rt = self.torch, self.rt
        try:
            dev_free = f"{t.cuda.mem_get_info()[0] / 2**30:.2f} GiB"
        except Exception:  # noqa: BLE001
            dev_free = "-"
        pool = rt.pool() if rt else None
        pool_txt = "-" if pool is None else f"{_mib(pool[0])}/{_mib(pool[1])} MiB thr {_mib(pool[2])} MiB"
        graph = rt.graph_reserved() if rt else None
        stack = rt.stack() if rt else None
        gens = "/".join(str(s.get("collections", "?")) for s in gc.get_stats())
        return (f"dev_free {dev_free} mempool {pool_txt} graphmem {_mib(graph)} MiB "
                f"stack {'-' if stack is None else stack} B pinned {_mib(pinned_bytes(t))} MiB gc {gens}")

    def end(self, longest: int, res0: int, res1: int, free1: int) -> None:
        """Print the step line; run the flush experiment on an unexplained loss (first FLUSH_EVENTS)."""
        self.steps += 1
        free0 = self.free0 if self.free0 is not None else free1
        lost_mib = (free0 - free1) / 1024 - max(0, res1 - res0) / 2**20
        self.say(
            f"probe step {self.steps} longest {longest} memfree {free0 / 2**20:.2f} -> {free1 / 2**20:.2f} GiB "
            f"pool {res0 / 2**30:.2f} -> {res1 / 2**30:.2f} GiB unexplained {lost_mib:+.0f} MiB {self.stats()}"
        )
        if not self.passive and lost_mib >= FLUSH_MIB and self.flushes < FLUSH_EVENTS:
            self.flushes += 1
            self.say(f"probe flush {self.flushes} after step {self.steps}: {self.flush()}")

    def flush(self) -> str:
        t, rt = self.torch, self.rt
        calls = [
            ("sync", getattr(t.cuda, "synchronize", None)),
            ("gc", gc.collect),
            ("host cache", host_empty_cache(t)),
            ("mempool trim", rt.trim_pool if rt else None),
            ("graph mem trim", rt.trim_graph if rt else None),
            ("stack limit", rt.reset_stack if rt else None),
        ]
        free = self.memfree()
        parts = [f"MemFree {free / 2**20:.2f} GiB"]
        for name, fn in calls:
            if fn is None:
                parts.append(f"{name} n/a")
                continue
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - report and go on
                parts.append(f"{name} error {exc!r}")
                continue
            now = self.memfree()
            parts.append(f"{name} {(now - free) / 1024:+.0f} MiB")
            free = now
        parts.append(self.settle(free))
        return "; ".join(parts)

    def settle(self, base: int) -> str:
        """MemFree polled for SETTLE_S with nothing running: memory the driver returns late."""
        t0 = self.clock()
        best, best_t, now = base, 0.0, base
        while self.clock() - t0 < SETTLE_S:
            self.sleep(POLL_S)
            now = self.memfree()
            if now > best:
                best, best_t = now, self.clock() - t0
        return (f"settle {SETTLE_S:.0f} s {(now - base) / 1024:+.0f} MiB "
                f"(max {(best - base) / 1024:+.0f} MiB at {best_t:.1f} s)")

    def released(self, action: str, res_before: int, res_after: int) -> None:
        """The hook just released torch's cache: follow MemFree for SERIES_S in a daemon thread."""
        if self.passive or self.series >= RELEASE_SERIES:
            return
        self.series += 1
        self.spawn(self._series, (self.series, self.steps + 1, action, res_before - res_after))

    def _series(self, k: int, step: int, action: str, released: int) -> None:
        t0 = self.clock()
        samples = []
        while True:
            t = self.clock() - t0
            try:
                pool = self.torch.cuda.memory_reserved()
            except Exception:  # noqa: BLE001
                pool = None
            samples.append((t, self.memfree(), pool))
            if t >= SERIES_S:
                break
            self.sleep(POLL_S)
        shown, i = [], 0
        for at in SERIES_AT:
            while i + 1 < len(samples) and samples[i][0] < at - 1e-9:
                i += 1
            t, free, pool = samples[i]
            pool_txt = "-" if pool is None else f"{pool / 2**30:.2f}"
            shown.append(f"+{t:.2f}s free {free / 2**20:.2f} pool {pool_txt}")
        self.say(f"probe release {k} after step {step} ({action}, {released / 2**20:.0f} MiB released; GiB): "
                 + "; ".join(shown))
