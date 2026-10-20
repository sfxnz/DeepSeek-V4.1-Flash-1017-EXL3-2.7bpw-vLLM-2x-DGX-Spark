"""docker/patch/prefill_mem_probe.py (CPU only, no torch, no CUDA).

The CUDA runtime is a fake ctypes library whose functions write through the byref arguments the
probe passes (ctypes.byref(x)._obj is x); MemFree is scripted; the clock advances only by the
probe's own sleeps; torch is a namespace.
"""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docker/patch"))

import prefill_mem_probe as pmp  # noqa: E402

GIB, MIB, KIB_PER_GIB = 2**30, 2**20, 2**20


class FakeRt:
    """cudart functions by name; records calls."""

    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)
        self.pool_attrs = {pmp.POOL_RESERVED: 3 * GIB, pmp.POOL_USED: GIB, pmp.POOL_THRESHOLD: 0}

    def _ret(self, name):
        self.calls.append(name)
        return 1 if name in self.fail else 0

    def cudaGetDevice(self, dev):
        dev._obj.value = 0
        return self._ret("cudaGetDevice")

    def cudaDeviceGetLimit(self, value, limit):
        assert limit == pmp.LIMIT_STACK
        value._obj.value = 1024
        return self._ret("cudaDeviceGetLimit")

    def cudaDeviceSetLimit(self, limit, value):
        assert limit == pmp.LIMIT_STACK and value.value == 1024
        return self._ret("cudaDeviceSetLimit")

    def cudaDeviceGetDefaultMemPool(self, pool, dev):
        pool._obj.value = 0x1234
        return self._ret("cudaDeviceGetDefaultMemPool")

    def cudaMemPoolGetAttribute(self, pool, attr, value):
        assert pool.value == 0x1234
        value._obj.value = self.pool_attrs[attr]
        return self._ret("cudaMemPoolGetAttribute")

    def cudaDeviceGetGraphMemAttribute(self, dev, attr, value):
        assert attr == pmp.GRAPH_RESERVED
        value._obj.value = 64 * MIB
        return self._ret("cudaDeviceGetGraphMemAttribute")

    def cudaMemPoolTrimTo(self, pool, keep):
        assert pool.value == 0x1234 and keep.value == 0
        return self._ret("cudaMemPoolTrimTo")

    def cudaDeviceGraphMemTrim(self, dev):
        return self._ret("cudaDeviceGraphMemTrim")


def fake_torch(events=None, pinned=None, host_cache=True, reserved=None):
    events = [] if events is None else events
    cuda = types.SimpleNamespace(
        mem_get_info=lambda: (5 * GIB, 120 * GIB),
        synchronize=lambda: events.append("sync"),
        memory_reserved=reserved or (lambda: 100 * GIB),
    )
    if pinned is not None:
        cuda.host_memory_stats = lambda: pinned
    torch = types.SimpleNamespace(cuda=cuda, _C=types.SimpleNamespace())
    if host_cache:
        torch._C._host_emptyCache = lambda: events.append("host cache")
    return torch


class Clock:
    """Advances only when the probe sleeps."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def scripted(values):
    """MemFree (KiB) from a list; the last value repeats once the list is used up."""
    values = list(values)

    def memfree():
        return values.pop(0) if len(values) > 1 else values[0]

    return memfree


class ReadTests(unittest.TestCase):
    def test_memfree_and_gate(self) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False) as fh:
            fh.write("MemTotal: 127600752 kB\nMemFree: 4578932 kB\nMemAvailable: 19231004 kB\n")
        try:
            self.assertEqual(pmp.memfree_kib(fh.name), 4578932)
        finally:
            os.unlink(fh.name)
        self.assertTrue(pmp.enabled({"DSV41_PREFILL_MEM_PROBE": "1"}))
        self.assertEqual(pmp.mode({"DSV41_PREFILL_MEM_PROBE": "2"}), "passive")
        self.assertEqual(pmp.mode({"DSV41_PREFILL_MEM_PROBE": "1"}), "full")
        self.assertFalse(pmp.enabled({}))
        self.assertFalse(pmp.enabled({"DSV41_PREFILL_MEM_PROBE": "yes"}))

    def test_cudart_reads_and_trims(self) -> None:
        lib = FakeRt()
        rt = pmp.Cudart(lib)
        self.assertEqual(rt.stack(), 1024)
        self.assertEqual(rt.pool(), (3 * GIB, GIB, 0))
        self.assertEqual(rt.graph_reserved(), 64 * MIB)
        self.assertTrue(rt.trim_pool() and rt.trim_graph() and rt.reset_stack())
        self.assertIn("cudaMemPoolTrimTo", lib.calls)
        self.assertIn("cudaDeviceSetLimit", lib.calls)
        bad = pmp.Cudart(FakeRt(fail={"cudaMemPoolGetAttribute", "cudaDeviceGetLimit"}))
        self.assertIsNone(bad.pool())
        self.assertIsNone(bad.stack())
        self.assertFalse(bad.reset_stack())

    def test_load_finds_the_mapped_runtime_or_none(self) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False) as fh:
            fh.write("aaaa-bbbb r-xp 00000000 00:00 0 /usr/lib/libc.so.6\n")
        try:
            self.assertIsNone(pmp.Cudart.load(fh.name))
            with open(fh.name, "a") as again:
                again.write("cccc-dddd r-xp 00000000 00:00 0 /nonexistent/libcudart.so.13\n")
            self.assertIsNone(pmp.Cudart.load(fh.name))  # OSError from CDLL -> None
            with mock.patch.object(ctypes, "CDLL", side_effect=lambda path: ("lib", path)):
                self.assertEqual(pmp.Cudart.load(fh.name).lib, ("lib", "/nonexistent/libcudart.so.13"))
        finally:
            os.unlink(fh.name)

    def test_pinned_stats_flat_nested_missing(self) -> None:
        self.assertEqual(pmp.pinned_bytes(fake_torch(pinned={"reserved_bytes.current": 5 * MIB})), 5 * MIB)
        self.assertEqual(pmp.pinned_bytes(fake_torch(pinned={"reserved_bytes": {"current": 7}})), 7)
        self.assertIsNone(pmp.pinned_bytes(fake_torch()))


class ProbeTests(unittest.TestCase):
    def probe(self, frees, rt=None, events=None, host_cache=True, reserved=None):
        lines, clock, spawned = [], Clock(), []
        torch = fake_torch(events, pinned={"reserved_bytes.current": 96 * MIB}, host_cache=host_cache,
                           reserved=reserved)
        p = pmp.Probe(torch, rt, lines.append, memfree=frees if callable(frees) else scripted(frees),
                      clock=clock, sleep=clock.sleep, spawn=lambda target, args: spawned.append((target, args)))
        return p, lines, spawned

    def test_step_line_and_no_flush_when_the_pool_explains_the_loss(self) -> None:
        # MemFree falls 1 GiB while the pool grows 1 GiB: nothing unexplained
        p, lines, _ = self.probe([6 * KIB_PER_GIB, 5 * KIB_PER_GIB], pmp.Cudart(FakeRt()))
        p.start()
        p.end(8188, 100 * GIB, 101 * GIB, p.end_free())
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("probe step 1 longest 8188 memfree 6.00 -> 5.00 GiB pool 100.00 -> 101.00 GiB "
                      "unexplained +0 MiB dev_free 5.00 GiB mempool 3072/1024 MiB thr 0 MiB graphmem 64 MiB "
                      "stack 1024 B pinned 96 MiB gc ", lines[0])

    def test_unexplained_loss_runs_the_flush_in_order_then_settles(self) -> None:
        events = []
        g = KIB_PER_GIB
        # start 6, end 4 GiB with a flat pool; flush: base 4 GiB, then after each call; then the settle
        # polls: +256 MiB at 0.3 s, back down at the end
        frees = [6 * g, 4 * g, 4 * g, 4 * g + 1024, 4 * g + 1024 + 512 * 1024, 4 * g + 1024 + 512 * 1024,
                 5 * g + 1024 + 512 * 1024, 5 * g + 1024 + 512 * 1024, 6 * g,
                 6 * g + 100 * 1024, 6 * g + 200 * 1024, 6 * g + 256 * 1024, 6 * g + 128 * 1024]
        p, lines, _ = self.probe(frees, pmp.Cudart(FakeRt()), events)
        p.start()
        p.end(16376, 100 * GIB, 100 * GIB, p.end_free())
        self.assertEqual(len(lines), 2, lines)
        self.assertIn("unexplained +2048 MiB", lines[0])
        self.assertEqual(lines[1], "probe flush 1 after step 1: MemFree 4.00 GiB; sync +1 MiB; gc +512 MiB; "
                                   "host cache +0 MiB; mempool trim +1024 MiB; graph mem trim +0 MiB; "
                                   "stack limit +511 MiB; settle 2 s +128 MiB (max +256 MiB at 0.3 s)")
        self.assertEqual(events, ["sync", "host cache"])

    def test_flushes_are_capped_and_missing_calls_reported(self) -> None:
        mem = {"v": 6 * KIB_PER_GIB}
        p, lines, _ = self.probe(lambda: mem["v"], rt=None, host_cache=False)
        for _ in range(pmp.FLUSH_EVENTS + 1):
            mem["v"] = 6 * KIB_PER_GIB
            p.start()
            mem["v"] = 5 * KIB_PER_GIB
            p.end(8188, 100 * GIB, 100 * GIB, p.end_free())
        flushes = [ln for ln in lines if ln.startswith("probe flush")]
        self.assertEqual(len(flushes), pmp.FLUSH_EVENTS)
        self.assertIn("host cache n/a; mempool trim n/a; graph mem trim n/a; stack limit n/a; settle 2 s +0 MiB",
                      flushes[0])
        self.assertIn("mempool - graphmem - MiB stack - B", lines[0])

    def test_passive_mode_prints_steps_only(self) -> None:
        lines, spawned = [], []
        mem = {"v": 6 * KIB_PER_GIB}
        p = pmp.Probe(fake_torch(), None, lines.append, memfree=lambda: mem["v"], passive=True,
                      spawn=lambda target, args: spawned.append(target))
        p.start()
        mem["v"] = 3 * KIB_PER_GIB
        p.end(8188, 100 * GIB, 100 * GIB, p.end_free())
        p.released("floor", 103 * GIB, 101 * GIB)
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("unexplained +3072 MiB", lines[0])
        self.assertEqual(spawned, [])

    def test_release_series_follows_memfree_and_the_pool(self) -> None:
        g = KIB_PER_GIB
        clock_free = {"t": None}
        # released 1.5 GiB come back over ~1 s; the pool regrows 0.5 GiB at 0.5 s
        p, lines, spawned = self.probe([], reserved=None)
        clock = p.clock
        p.memfree = lambda: int((3 + min(1.5, 1.5 * clock() / 1.0)) * g)
        p.torch.cuda.memory_reserved = lambda: int((101.5 + (0.5 if clock() >= 0.5 else 0.0)) * GIB)
        for _ in range(pmp.RELEASE_SERIES + 1):
            p.released("floor", 103 * GIB, 101.5 * GIB)
        self.assertEqual(len(spawned), pmp.RELEASE_SERIES)
        target, args = spawned[0]
        target(*args)
        self.assertEqual(len(lines), 1, lines)
        line = lines[0]
        self.assertTrue(line.startswith("probe release 1 after step 1 (floor, 1536 MiB released; GiB): "
                                        "+0.00s free 3.00 pool 101.50; +0.10s free 3.15 pool 101.50"), line)
        self.assertIn("+0.50s free 3.75 pool 102.00", line)
        self.assertIn("+5.00s free 4.50 pool 102.00", line)
        self.assertEqual(line.count("; +"), len(pmp.SERIES_AT) - 1)
        del clock_free


class HookTests(unittest.TestCase):
    """The hook in prefill_empty_cache drives the probe around each prefill step."""

    def test_probe_lines_flush_and_release_series(self) -> None:
        sys.path.insert(0, str(ROOT / "tests"))
        import test_prefill_empty_cache as tpec

        torch_events, threads = [], []
        mem = {"v": 6 * KIB_PER_GIB}

        class SyncThread:
            def __init__(self, target, args, daemon):
                threads.append(daemon)
                self.target, self.args = target, args

            def start(self):
                self.target(*self.args)

        def memfree():
            return mem["v"]

        def inside_step(i):  # runs inside the stub step (between the hook's start and end reads)
            mem["v"] = 3 * KIB_PER_GIB if i == 1 else 5 * KIB_PER_GIB  # step 1 loses 3 GiB off-pool
            return None  # no pool change

        steps = tpec.prompt_steps("a", 20000)
        env = {"DSV41_PREFILL_EMPTY_CACHE_HOLD": "1", "DSV41_PREFILL_MEM_PROBE": "1"}
        with tpec.Harness(env, 2.5, inside_step) as h:  # below the 3 GiB floor: every long in-flight step releases
            h.torch.cuda.mem_get_info = lambda: (5 * GIB, 120 * GIB)
            h.torch.cuda.synchronize = lambda: torch_events.append("sync")
            with mock.patch.object(pmp, "memfree_kib", memfree), \
                    mock.patch.object(pmp.Cudart, "load", classmethod(lambda cls: None)), \
                    mock.patch.object(pmp.threading, "Thread", SyncThread), \
                    mock.patch.object(pmp, "SETTLE_S", 0.2), mock.patch.object(pmp, "SERIES_S", 0.2), \
                    mock.patch.dict(sys.modules, {"torch": h.torch}):
                h.run(steps)
        out = h.out.getvalue()
        steps_out = [ln for ln in out.splitlines() if "] probe step " in ln]
        self.assertEqual(len(steps_out), 3, out)
        self.assertIn("unexplained +3072 MiB", steps_out[0])
        self.assertEqual(out.count("] probe flush 1 after step 1: MemFree 3.00 GiB; sync +0 MiB"), 1, out)
        self.assertEqual(torch_events, ["sync"])
        self.assertIn("; memory probe (full))", out)
        series = [ln for ln in out.splitlines() if "] probe release " in ln]
        self.assertEqual(len(series), 3, out)  # chunks 1-2 below the floor, then the prompt end
        self.assertIn("probe release 1 after step 1 (floor, ", series[0])
        self.assertIn("probe release 3 after step 3 (release, ", series[2])
        self.assertEqual(threads, [True, True, True])  # daemon threads


if __name__ == "__main__":
    unittest.main()
