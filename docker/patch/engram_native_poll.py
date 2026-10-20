"""Engram native stage: all tables in one pass, page-cache misses polled, not slept on.

DSV41_ENGRAM_NATIVE_POLL=1 (default 0; needs DSV41_ENGRAM_NATIVE_STAGE=1). The native
stage's eng_gather_bf16 runs table by table: an RWF_NOWAIT probe per row, WILLNEED for
every miss, then a blocking pread per missed row in row order. At c~4 a step has 334 rows
and 200-250 misses: 2.6 ms on spark1 (~11 us per miss) and 7.4 ms on spark2 (~37 us; B0c
census), and rank 0 waits ~7 ms for rank 1 at the first target all-reduce (decode-glue).

On this kernel the RWF_NOWAIT probe of a missing page starts that page's read, so the
WILLNEED pass submits nothing new, and the pread loop sleeps on each page still in flight
(paying the CPU idle-state exit on every wakeup). With the knob, _NativeStager.gather makes
one eng_gather_tables_bf16 call (engram_poll.c, its own .so that includes engram_native.c)
for all tables: the probe pass over every row of every table, no WILLNEED, then each miss
re-probed with a cpu yield between tries for up to DSV41_ENGRAM_NATIVE_POLL_US (default
2000), then WILLNEED for whatever is still missing and a blocking pread per row (today's
sequence, for a kernel whose probe does not start the read), the same dequant. Same buffered reads, so the same
bytes and the same pages end up in the page cache.

Not io_uring: Docker's default seccomp profile blocks it. Linux AIO + O_DIRECT is not used
either: the probe already starts every miss's buffered read, so an O_DIRECT read would be a
second device read. Fallback, always to today's per-table path: no RWF_NOWAIT or any failed
call prints one LOG_DISARMED line, and that step and every later one go per table. The
native stage's verify (its first DSV41_ENGRAM_NATIVE_VERIFY calls against the stock
gather, raw bf16 bits) covers this path too. Cost: ~0.4 MB of scratch; the stage thread
spins at most POLL_US per step instead of sleeping. Top-level imports are stdlib only.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dsv41: engram native poll armed"
LOG_DISARMED = "dsv41: engram native poll DISABLED ->"

SRC = Path(__file__).with_name("engram_poll.c")
NATIVE_SRC = Path(__file__).with_name("engram_native.c")  # included by SRC
CACHE_DIR = Path("/tmp/dsv41-native")
ABI_VERSION = 1
NSTATS = 5  # engram_native.c ENG_NSTATS
TAB = 9  # engram_poll.c ENG_TAB
DEFAULT_SPIN_US = 2000

_LIB = None
_STATE = {"armed": True}


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return (env.get("DSV41_ENGRAM_NATIVE_POLL", "0") or "0") == "1"


def spin_us(env=None) -> int:
    env = os.environ if env is None else env
    n = int(env.get("DSV41_ENGRAM_NATIVE_POLL_US", "") or DEFAULT_SPIN_US)
    if n < 0:
        raise ValueError(f"DSV41_ENGRAM_NATIVE_POLL_US={n} must be >= 0")
    return n


def so_name() -> str:
    h = hashlib.sha256(SRC.read_bytes() + b"\0" + NATIVE_SRC.read_bytes()).hexdigest()[:16]
    return f"engram_poll-{h}.so"


def build(cache_dir: Path = CACHE_DIR) -> Path:
    """Compile engram_poll.c (which includes engram_native.c) once per content hash."""
    cache_dir = Path(cache_dir)
    so = cache_dir / so_name()
    if so.is_file():
        return so
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = so.with_name(f".{so.name}.{os.getpid()}.tmp")
    try:
        res = subprocess.run(["gcc", "-O3", "-shared", "-fPIC", "-o", str(tmp), str(SRC)],
                             capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"gcc failed: {res.stderr.strip()[-400:]}")
        os.replace(tmp, so)
    finally:
        tmp.unlink(missing_ok=True)
    return so


def load(so: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(so))
    lib.eng_poll_abi_version.restype = ctypes.c_int
    if lib.eng_poll_abi_version() != ABI_VERSION:
        raise RuntimeError(f"{so}: ABI {lib.eng_poll_abi_version()} != {ABI_VERSION}")
    lib.eng_gather_tables_bf16.restype = ctypes.c_int
    lib.eng_gather_tables_bf16.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,  # n_tables tab hash tok_stride
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,  # n_tok heads_valid local_heads lut
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,  # outs sw ss spin stats
    ]
    return lib


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = load(build())
    return _LIB


def _disarm(exc: BaseException) -> None:
    if _STATE["armed"]:
        _STATE["armed"] = False
        print("dsv41: engram native poll DISABLED -> per-table native gather: %r" % (exc,), flush=True)


class NativePoll:
    """Table descriptors and scratch for one native stager.

    tables: [(w_fd, w_off, s_fd, s_off, dim, sb, vocab_start, vocab_end, layer)];
    outs: one [max_tokens, local_heads, dim] bf16 buffer address per table (rows_host).
    """

    def __init__(self, tables, nl: int, heads_valid: int, local_heads: int, dim: int, max_tokens: int,
                 lut_ptr: int, outs, env=None, lib=None):
        self.lib = _lib() if lib is None else lib
        if len({t[4] for t in tables}) != 1 or len({t[5] for t in tables}) != 1 or tables[0][4] != dim:
            raise ValueError("tables differ in row geometry")
        tab = []
        for w_fd, w_off, s_fd, s_off, d, sb, v0, v1, layer in tables:
            tab += [w_fd, w_off, s_fd, s_off, d, sb, v0, v1, layer * heads_valid]
        self.tab = (ctypes.c_int64 * len(tab))(*tab)
        self.n_tables = len(tables)
        rows = max_tokens * local_heads * self.n_tables
        self.sw = (ctypes.c_uint8 * (rows * dim))()
        self.ss = (ctypes.c_uint8 * (rows * tables[0][5]))()
        self.outs = (ctypes.c_void_p * self.n_tables)(*outs)
        self.stride, self.heads_valid, self.local_heads = nl * heads_valid, heads_valid, local_heads
        self.max_tokens, self.lut_ptr = max_tokens, lut_ptr
        self.spin_ns = spin_us(env) * 1000
        self.stats = (ctypes.c_int64 * NSTATS)()

    def gather(self, hash_ptr: int, n: int) -> list:
        """Every table's rows [:n] into its out buffer; engram_native.c stats. Raises on any error."""
        if not 0 <= n <= self.max_tokens:
            raise ValueError(f"n={n} outside 0..{self.max_tokens}")
        rc = self.lib.eng_gather_tables_bf16(
            self.n_tables, self.tab, hash_ptr, self.stride, n, self.heads_valid, self.local_heads,
            self.lut_ptr, self.outs, self.sw, self.ss, self.spin_ns, self.stats,
        )
        if rc:
            raise OSError(-rc, f"eng_gather_tables_bf16: {os.strerror(-rc)}")
        return list(self.stats)


def attach(nat):
    """NativePoll for an engram_native_stage._NativeStager, or None (one LOG_DISARMED line)."""
    if not _STATE["armed"]:
        return None
    try:
        poll = NativePoll([t[:9] for t in nat.tables], nat.nl, nat.heads_valid, nat.local_heads, nat.dim,
                          nat.max_tokens, nat.lut.data_ptr(), [t[9].data_ptr() for t in nat.tables])
    except Exception as exc:  # noqa: BLE001 - the per-table path stays
        _disarm(exc)
        return None
    print("dsv41: engram native poll armed (%d tables, %d rows max, spin %d us)"
          % (poll.n_tables, poll.max_tokens * poll.local_heads * poll.n_tables, poll.spin_ns // 1000), flush=True)
    return poll


def drop(exc: BaseException) -> None:
    """A failed call: one LOG_DISARMED line; the caller gathers this step per table."""
    _disarm(exc)
