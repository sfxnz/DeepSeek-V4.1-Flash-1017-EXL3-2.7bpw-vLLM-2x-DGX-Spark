"""engram_poll.c + engram_native_poll.py on the host (numpy, gcc, ext4 /tmp; no torch).

- eng_gather_tables_bf16 returns the bytes of eng_gather_bf16 (per table) and of a numpy
  transcription of the stock chain: hot, cold and mixed page cache, unowned ids, padding
  heads, rows straddling 4 KiB pages (664-byte w offset), the last row before EOF, 0..64
  tokens, two tables in two files, POLL_US 0 (straight to the blocking reads).
- The page cache ends up holding the same pages as after the per-table path (mincore).
- Fallbacks to the per-table path: no RWF_NOWAIT (-EOPNOTSUPP), a failing call inside
  _NativeStager.gather (the same step is re-gathered per table, one LOG_DISARMED line),
  attach failing (LOG_DISARMED, nothing armed).
"""

from __future__ import annotations

import ast
import ctypes
import errno
import mmap
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker" / "patch"
sys.path.insert(0, str(PATCH))
import engram_native_poll as enp  # noqa: E402
import engram_native_stage as ens  # noqa: E402

DIM, SB = 256, 8
W_OFF = 664  # the real shard's w offset: 1 row in 16 straddles a 4 KiB page
NROWS = 6000


def e4m3fn_lut() -> np.ndarray:
    out = np.zeros(256, np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 3) & 0xF, b & 7
        out[b] = s * (m / 8.0) * 2.0**-6 if e == 0 else s * (1 + m / 8.0) * 2.0 ** (e - 7)
    bits = out.view(np.uint32)
    bits[0x7F], bits[0xFF] = 0x7FF00000, 0xFFF00000
    return out


LUT = e4m3fn_lut()


def bf16_bits(x: np.ndarray) -> np.ndarray:
    u = x.view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(x)
    r[nan] = ((u[nan] >> 16) | 0x40).astype(np.uint16)
    return r


def reference(w, s, ids: np.ndarray, local_heads: int, v0: int, v1: int) -> np.ndarray:
    """The stock stage's rows (pad heads with -1, owned mask, fp8 x ue8m0 -> bf16) [n, lh, DIM]."""
    n, hv = ids.shape
    if n == 0:
        return np.zeros((0, local_heads, DIM), np.uint16)
    local = np.full((n, local_heads), -1, np.int64)
    local[:, :hv] = ids
    flat = local.reshape(-1)
    owned = (flat >= v0) & (flat < v1)
    rows = np.where(owned, flat, 0)
    scale = (s[rows].astype(np.uint32) << 23).view(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        out = bf16_bits((LUT[w[rows]].reshape(len(rows), SB, -1) * scale[:, :, None]).reshape(len(rows), DIM))
    out[~owned] = 0
    return out.reshape(n, local_heads, DIM)


class Ptr:
    """A numpy buffer with torch's data_ptr()."""

    def __init__(self, a: np.ndarray):
        self.a = a

    def data_ptr(self) -> int:
        return self.a.ctypes.data


class PollTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("gcc") is None:
            raise unittest.SkipTest("gcc not on PATH")
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.nat_lib = ens.load(ens.build(ens.SRC, tmp / "cache"))
        cls.lib = enp.load(enp.build(tmp / "cache"))
        rng = np.random.default_rng(5)
        cls.fds, cls.w, cls.s, cls.s_off = [], [], [], []
        for t in range(2):
            w = rng.integers(0, 256, (NROWS, DIM), dtype=np.uint8)
            s = rng.integers(100, 140, (NROWS, SB), dtype=np.uint8)
            w[:2] = np.array([0x7F, 0xFF], np.uint8)[:, None]  # NaN rows
            s_off = W_OFF + NROWS * DIM + 40 + 8 * t
            path = tmp / f"t{t}.bin"
            with open(path, "wb") as fh:
                fh.write(b"\0" * W_OFF)
                fh.write(w.tobytes())
                fh.write(b"\0" * (40 + 8 * t))
                fh.write(s.tobytes())  # the s rows run to EOF
            cls.w.append(w)
            cls.s.append(s)
            cls.s_off.append(s_off)
        os.sync()
        for t in range(2):
            fd = os.open(tmp / f"t{t}.bin", os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)  # as DiskEngramTable._open
            cls.fds.append(fd)
        cls.libc = ctypes.CDLL(None, use_errno=True)
        cls.libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        cls.v = [(3, NROWS - 3), (0, NROWS)]  # table 0 leaves a few ids unowned at both ends
        cls.tables = [(cls.fds[t], W_OFF, cls.fds[t], cls.s_off[t], DIM, SB, cls.v[t][0], cls.v[t][1], t)
                      for t in range(2)]

    @classmethod
    def tearDownClass(cls) -> None:
        for fd in cls.fds:
            os.close(fd)
        cls._tmp.cleanup()

    def setUp(self) -> None:
        enp._STATE["armed"] = True

    def drop(self) -> None:
        for fd in self.fds:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)

    def resident(self) -> list:
        out = []
        for fd in self.fds:
            size = os.fstat(fd).st_size
            m = mmap.mmap(fd, size, access=mmap.ACCESS_COPY)
            try:
                buf = ctypes.c_char.from_buffer(m)
                vec = np.zeros((size + 4095) // 4096, np.uint8)
                rc = self.libc.mincore(ctypes.addressof(buf), size, vec.ctypes.data)
                del buf
                self.assertEqual(rc, 0)
                out.append(vec & 1)
            finally:
                m.close()
        return out

    def hashes(self, rng, n: int, hv: int = 12) -> np.ndarray:
        """[n, 2 tables, hv] int32 ids, some unowned, -1, and the last row before EOF."""
        ids = rng.integers(0, NROWS, (n, 2, hv), dtype=np.int64)
        if n:
            ids[0, 0, 0] = -1
            ids[-1, 0, -1] = 1  # unowned in table 0
            ids[n // 2, 1, 0] = NROWS - 1  # table 1: its s row ends the file
        return np.ascontiguousarray(ids.astype(np.int32))

    def per_table(self, ids: np.ndarray, n: int, lh: int) -> list:
        hv = ids.shape[2]
        out = []
        for w_fd, w_off, s_fd, s_off, d, sb, v0, v1, layer in self.tables:
            o = np.full((max(n, 1) * lh, DIM), 0xABCD, np.uint16)
            sw = np.zeros((max(n, 1) * lh, DIM), np.uint8)
            ss = np.zeros((max(n, 1) * lh, SB), np.uint8)
            st = (ctypes.c_int64 * ens.NSTATS)()
            rc = self.nat_lib.eng_gather_bf16(w_fd, w_off, s_fd, s_off, d, sb, v0, v1,
                                              ids.ctypes.data + layer * hv * 4, 2 * hv, n, hv, lh,
                                              LUT.ctypes.data, o.ctypes.data, sw.ctypes.data, ss.ctypes.data, 1, st)
            self.assertEqual(rc, 0)
            out.append(o[: n * lh].reshape(n, lh, DIM))
        return out

    def make(self, max_tokens: int = 64, lh: int = 12, hv: int = 12, tables=None, env=None):
        outs = [np.full((max_tokens, lh, DIM), 0xABCD, np.uint16) for _ in range(2)]
        poll = enp.NativePoll(tables or self.tables, 2, hv, lh, DIM, max_tokens, LUT.ctypes.data,
                              [o.ctypes.data for o in outs], env=env, lib=self.lib)
        return poll, outs

    def test_bytes_match_the_per_table_gather_and_the_stock_chain(self) -> None:
        rng = np.random.default_rng(1)
        for lh, hv in ((12, 12), (12, 11)):
            poll, outs = self.make(lh=lh, hv=hv)
            for n in (0, 1, 4, 16, 64):
                for cache in ("cold", "hot", "mixed"):
                    ids = self.hashes(rng, n, hv)
                    self.drop()
                    ref = self.per_table(ids, n, lh)  # also caches every row it reads
                    if cache == "cold":
                        self.drop()
                    elif cache == "mixed":
                        self.drop()
                        self.per_table(ids[: n // 2], n // 2, lh)
                    for o in outs:
                        o[:] = 0xABCD
                    st = poll.gather(ids.ctypes.data, n)
                    self.assertEqual(st[0], 2 * n * lh, (n, cache))
                    if cache == "hot":
                        self.assertEqual(st[2], 0, (n, cache))
                    elif cache == "cold" and n:
                        self.assertGreater(st[2], 0)
                    for t in range(2):
                        v0, v1 = self.v[t]
                        stock = reference(self.w[t], self.s[t], ids[:, t, :], lh, v0, v1)
                        self.assertTrue(np.array_equal(outs[t][:n], ref[t]), (lh, hv, n, cache, t))
                        self.assertTrue(np.array_equal(outs[t][:n], stock), (lh, hv, n, cache, t))

    def test_the_page_cache_ends_up_as_after_the_per_table_path(self) -> None:
        rng = np.random.default_rng(2)
        ids = self.hashes(rng, 16)
        self.drop()
        self.per_table(ids, 16, 12)
        want = self.resident()
        self.assertGreater(sum(int(x.sum()) for x in want), 0)
        self.drop()
        poll, _ = self.make()
        st = poll.gather(ids.ctypes.data, 16)
        self.assertGreater(st[2], 0)
        for got, ref in zip(self.resident(), want):
            self.assertTrue(np.array_equal(got, ref))

    def test_no_spin_goes_straight_to_the_blocking_reads(self) -> None:
        poll, outs = self.make(env={"DSV41_ENGRAM_NATIVE_POLL_US": "0"})
        self.assertEqual(poll.spin_ns, 0)
        rng = np.random.default_rng(4)
        ids = self.hashes(rng, 8)
        self.drop()
        ref = self.per_table(ids, 8, 12)
        self.drop()
        poll.gather(ids.ctypes.data, 8)
        for t in range(2):
            self.assertTrue(np.array_equal(outs[t][:8], ref[t]))

    def test_no_rwf_nowait_is_reported_for_the_fallback(self) -> None:
        fd = os.open("/proc/cpuinfo", os.O_RDONLY)  # RWF_NOWAIT: EOPNOTSUPP
        try:
            tables = [(fd, 0, fd, 0, DIM, SB, 0, NROWS, 0), (fd, 0, fd, 0, DIM, SB, 0, NROWS, 1)]
            poll, _ = self.make(tables=tables)
            ids = self.hashes(np.random.default_rng(3), 2)
            with self.assertRaises(OSError) as cm:
                poll.gather(ids.ctypes.data, 2)
            self.assertEqual(cm.exception.errno, errno.EOPNOTSUPP)
        finally:
            os.close(fd)
        poll, _ = self.make()
        with self.assertRaises(ValueError):
            poll.gather(ids.ctypes.data, 65)

    def _stand_in(self, n_max: int = 64):
        """The attributes _NativeStager.gather reads, numpy-backed."""
        bufs = [np.full((n_max, 12, DIM), 0xABCD, np.uint16) for _ in range(2)]
        nat = SimpleNamespace(
            lib=self.nat_lib, nl=2, heads_valid=12, local_heads=12, dim=DIM, max_tokens=n_max,
            tables=[t + (Ptr(b),) for t, b in zip(self.tables, bufs)],
            lut=Ptr(LUT), sw=Ptr(np.zeros((n_max * 12, DIM), np.uint8)), ss=Ptr(np.zeros((n_max * 12, SB), np.uint8)),
            stats=(ctypes.c_int64 * ens.NSTATS)(), census_every=0, acc=[0, 0, 0, 0, 0.0], poll=None,
        )
        nat._gather_poll = lambda stager, n: ens._NativeStager._gather_poll(nat, stager, n)
        return nat, bufs

    def test_attach_and_the_native_stage_gather(self) -> None:
        rng = np.random.default_rng(6)
        ids = self.hashes(rng, 16)
        stager = SimpleNamespace(hash_host=Ptr(ids))
        nat, bufs = self._stand_in()
        ens._NativeStager.gather(nat, stager, 16)  # poll off: the per-table path
        per_table = [b[:16].copy() for b in bufs]
        enp._LIB = self.lib
        with redirect_stdout(StringIO()) as log:
            nat.poll = enp.attach(nat)
        self.assertIsNotNone(nat.poll)
        self.assertIn(enp.LOG_ENGAGED, log.getvalue())
        for b in bufs:
            b[:] = 0xABCD
        self.drop()
        ens._NativeStager.gather(nat, stager, 16)
        self.assertIsNotNone(nat.poll)
        for b, ref in zip(bufs, per_table):
            self.assertTrue(np.array_equal(b[:16], ref))

    def test_a_failing_call_regathers_the_step_per_table(self) -> None:
        rng = np.random.default_rng(7)
        ids = self.hashes(rng, 16)
        stager = SimpleNamespace(hash_host=Ptr(ids))
        nat, bufs = self._stand_in()
        ens._NativeStager.gather(nat, stager, 16)
        ref = [b[:16].copy() for b in bufs]
        failing = SimpleNamespace(eng_gather_tables_bf16=lambda *a: -errno.EOPNOTSUPP)
        nat.poll, _ = self.make()
        nat.poll.lib = failing
        nat.poll.outs = (ctypes.c_void_p * 2)(*(b.ctypes.data for b in bufs))
        for b in bufs:
            b[:] = 0xABCD
        with redirect_stdout(StringIO()) as log:
            ens._NativeStager.gather(nat, stager, 16)
            ens._NativeStager.gather(nat, stager, 16)
        self.assertIsNone(nat.poll)
        self.assertEqual(log.getvalue().count(enp.LOG_DISARMED), 1)
        for b, r in zip(bufs, ref):
            self.assertTrue(np.array_equal(b[:16], r))
        with redirect_stdout(StringIO()) as log:
            self.assertIsNone(enp.attach(nat))  # disarmed for the process
        self.assertEqual(log.getvalue(), "")

    def test_attach_failure_keeps_the_per_table_path(self) -> None:
        nat, _ = self._stand_in()
        nat.tables[1] = nat.tables[1][:5] + (16,) + nat.tables[1][6:]  # sb differs between tables
        enp._LIB = self.lib
        with redirect_stdout(StringIO()) as log:
            self.assertIsNone(enp.attach(nat))
        self.assertIn(enp.LOG_DISARMED, log.getvalue())


class ModuleTests(unittest.TestCase):
    def test_env(self) -> None:
        self.assertFalse(enp.enabled({}))
        self.assertFalse(enp.enabled({"DSV41_ENGRAM_NATIVE_POLL": "0"}))
        self.assertTrue(enp.enabled({"DSV41_ENGRAM_NATIVE_POLL": "1"}))
        self.assertEqual(enp.spin_us({}), 2000)
        self.assertEqual(enp.spin_us({"DSV41_ENGRAM_NATIVE_POLL_US": "0"}), 0)
        with self.assertRaises(ValueError):
            enp.spin_us({"DSV41_ENGRAM_NATIVE_POLL_US": "-1"})

    def test_top_level_imports_are_stdlib_only(self) -> None:
        tree = ast.parse(Path(enp.__file__).read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {a.name for n in top if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in top if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"__future__", "ctypes", "hashlib", "os", "subprocess", "pathlib"})

    def test_markers_and_the_off_path(self) -> None:
        src = Path(enp.__file__).read_text()
        for marker in (enp.LOG_ENGAGED, enp.LOG_DISARMED):
            self.assertGreaterEqual(src.count(marker), 2, marker)
        stage = Path(ens.__file__).read_text()
        # the poll module is imported only behind the knob; the per-table loop is the off path
        self.assertIn('if (os.environ.get("DSV41_ENGRAM_NATIVE_POLL", "0") or "0") == "1":\n'
                      "            import engram_native_poll", stage)
        self.assertIn("in (self.tables if polled is None else ()):", stage)
        self.assertNotIn("engram_poll", ens.SRC.read_text())  # the default .so does not change
        self.assertIn('#include "engram_native.c"', enp.SRC.read_text())  # one dequant source


if __name__ == "__main__":
    unittest.main()
