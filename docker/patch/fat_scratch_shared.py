"""One shared fat-expert scratch set for vllm_exl3's prefill MoE (DSV41_FAT_SCRATCH_SHARED=1).

vllm_exl3.exl3._fat_scratch (image e15-kbits; the same function as tests/fixtures/exl3_e14.pin.py)
caches one scratch set per (device, bucketed_cap, intermediate, hidden, k_words) in
_FAT_SCRATCH_CACHE and never frees it. bucketed_cap rounds the expert's row count up to 256. Fat
experts have 257..2048 rows (VLLM_EXL3_FAT_THRESHOLD 256; a chunk with an expert above 2048 rows is
re-sliced into 2048-token pieces), so there are 7 buckets, 512..2048. k_words follows K_gu, which is
2 or 3 on 2.7bpw-mcg-hq. Per rank (hidden 5120, intermediate 1152) one set is 56.25 MiB of weight
scratch (w13, w2, w_gate, w_up), plus packed13 (1.41 MiB x K_gu), plus 67 KiB per row (h, h13,
gate_up, act, act_h, h2, down, g_tmp, u_tmp): 92.6 MiB at 512 rows, 194.5 MiB at 2048.
- perf2 baseline 16K prefill (B0c traces; perf2/prefill-mem/tools/fat_keys.py reads the fat
  experts' row counts and K from the kernel grids): the first chunk hit all 14 (bucket, K) keys on
  both ranks, 2,009 MiB held per rank for the life of the process.
- prefill-moe's GPU bench measured the same rule per call: single census calls held 202 / 328 /
  205 / 637 MiB, which is 2 / 3 / 2 / 5 of these sets (stock_bytes below, to 0.5 MiB).

apply_exl3_batched_fat is the only caller. It runs the fat experts one at a time on the current
stream. It takes [:n_rows] prefix views of the row buffers, and it rewrites w13 / w2 / w_gate / w_up
/ packed13 / svh13 for every expert before reading them. So one set sized for the largest bucket
(exl3.TEMP_ROWS_FUSED = 2048 rows) can serve every key. Each key still gets its own dict with the
stock shapes: the row buffers are [:cap] prefix views of the shared buffers, and packed13 is a
contiguous view of a flat int16 buffer sized for K_gu 4 (or the widest k_words seen). The kernels get
the same operands with the same shapes, strides and base alignment; only the addresses differ from
the stock run, so the output is bit-identical. The shared set is 195.9 MiB per rank. That saves
~1.8 GiB per rank once a serve has hit all 14 keys (estimate from the shapes; GPU evidence pending).
With DSV41_MOE_PREFILL_TC=1 the fat path runs only in that lever's stock reference calls, which
clear _FAT_SCRATCH_CACHE afterwards; the shared set and its views live in that dict, so the clear
frees them too.

Runtime patch, installed by sitecustomize after load_general_plugins: it replaces exl3._fat_scratch.
It refuses (LOG_DISARMED) unless the image's _fat_scratch builds exactly EXPECTED_STOCK (field ->
torch.empty call, plus the bucket and key expressions, compared as ast.unparse text).
DSV41_FAT_SCRATCH_SHARED=0 keeps the stock cache. Top-level imports are stdlib only.
"""

from __future__ import annotations

import os

MARK = "dsv41: fat scratch shared"
LOG_ENGAGED = "dsv41: fat scratch shared engaged"
LOG_DISARMED = "dsv41: fat scratch shared DISABLED"
BASE_TAG = "dsv41-fat-scratch-shared"

# The stock function's dict, field -> ast.unparse of its value (pinned e14 == e15 image source).
EXPECTED_STOCK = {
    "packed13": "torch.empty((in_tiles, 2 * out_tiles, k_words), dtype=torch.int16, device=device)",
    "svh13": "torch.empty(2 * intermediate, dtype=torch.float16, device=device)",
    "w13": "torch.empty((hidden, 2 * intermediate), dtype=torch.float16, device=device)",
    "w2": "torch.empty((intermediate, hidden), dtype=torch.float16, device=device)",
    "h": "torch.empty((bucketed_cap, hidden), dtype=torch.float16, device=device)",
    "h13": "torch.empty((bucketed_cap, hidden), dtype=torch.float16, device=device)",
    "gate_up": "torch.empty((bucketed_cap, 2 * intermediate), dtype=torch.float32, device=device)",
    "act": "torch.empty((bucketed_cap, intermediate), dtype=torch.float32, device=device)",
    "act_h": "torch.empty((bucketed_cap, intermediate), dtype=torch.float16, device=device)",
    "h2": "torch.empty((bucketed_cap, intermediate), dtype=torch.float16, device=device)",
    "down": "torch.empty((bucketed_cap, hidden), dtype=torch.float32, device=device)",
    "w_gate": "torch.empty((hidden, intermediate), dtype=torch.float16, device=device)",
    "w_up": "torch.empty((hidden, intermediate), dtype=torch.float16, device=device)",
    "g_tmp": "torch.empty((bucketed_cap, intermediate), dtype=torch.float32, device=device)",
    "u_tmp": "torch.empty((bucketed_cap, intermediate), dtype=torch.float32, device=device)",
}
EXPECTED_ASSIGNS = (
    "bucketed_cap = max(256, (int(capacity) + 255) // 256 * 256)",
    "key = (str(device), bucketed_cap, intermediate, hidden, k_words)",
)

# Shapes in terms of R (rows), H (hidden), I (intermediate); packed13 is handled separately.
ROW_FIELDS = {
    "h": ("H", "float16"),
    "h13": ("H", "float16"),
    "gate_up": ("2I", "float32"),
    "act": ("I", "float32"),
    "act_h": ("I", "float16"),
    "h2": ("I", "float16"),
    "down": ("H", "float32"),
    "g_tmp": ("I", "float32"),
    "u_tmp": ("I", "float32"),
}
FIXED_FIELDS = {
    "svh13": (("2I",), "float16"),
    "w13": (("H", "2I"), "float16"),
    "w2": (("I", "H"), "float16"),
    "w_gate": (("H", "I"), "float16"),
    "w_up": (("H", "I"), "float16"),
}
PACKED_FIELD = "packed13"
PACKED_K_WORDS = 64  # the flat packed13 buffer starts at K = 4 width (16 words per K): K 2/3/4 never regrow it
ITEMSIZE = {"float16": 2, "float32": 4, "int16": 2}


def enabled() -> bool:
    return os.environ.get("DSV41_FAT_SCRATCH_SHARED", "0") == "1"


def bucket(capacity: int) -> int:
    """The stock bucketed_cap."""
    return max(256, ((int(capacity) + 255) // 256) * 256)


def _dim(sym: str, hidden: int, inter: int) -> int:
    return {"H": hidden, "I": inter, "2I": 2 * inter}[sym]


def set_bytes(rows: int, hidden: int, inter: int, packed_numel: int) -> int:
    """Bytes of one scratch set with `rows` row capacity and a packed13 of `packed_numel` int16."""
    total = packed_numel * ITEMSIZE["int16"]
    for sym, dt in ROW_FIELDS.values():
        total += rows * _dim(sym, hidden, inter) * ITEMSIZE[dt]
    for dims, dt in FIXED_FIELDS.values():
        n = 1
        for sym in dims:
            n *= _dim(sym, hidden, inter)
        total += n * ITEMSIZE[dt]
    return total


def stock_bytes(cap: int, hidden: int, inter: int, in_tiles: int, out_tiles: int, k_words: int) -> int:
    """Bytes the stock _fat_scratch allocates for one key."""
    return set_bytes(cap, hidden, inter, in_tiles * 2 * out_tiles * k_words)


def stock_mismatch(src: str) -> str | None:
    """None when `src` (the _fat_scratch source) builds exactly EXPECTED_STOCK, else why not."""
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(src))
    fields: dict[str, str] | None = None
    assigns = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [getattr(t, "id", None) for t in node.targets]
        if targets == ["scratch"] and isinstance(node.value, ast.Dict):
            fields = {}
            for k, v in zip(node.value.keys, node.value.values):
                if not isinstance(k, ast.Constant):
                    return "non-literal scratch key"
                fields[k.value] = ast.unparse(v)
        else:
            assigns.append(ast.unparse(node))
    if fields is None:
        return "no `scratch = {...}` dict"
    if fields != EXPECTED_STOCK:
        diff = sorted(set(fields.items()) ^ set(EXPECTED_STOCK.items()))
        return f"scratch fields differ: {diff[:3]}"
    for want in EXPECTED_ASSIGNS:
        if want not in assigns:
            return f"missing `{want}`"
    return None


def _alloc_base(torch, device, rows: int, hidden: int, inter: int, packed_numel: int) -> dict:
    def dt(name):
        return getattr(torch, name)

    base = {"rows": rows, "packed": torch.empty(packed_numel, dtype=torch.int16, device=device)}
    for name, (sym, dtype) in ROW_FIELDS.items():
        base[name] = torch.empty((rows, _dim(sym, hidden, inter)), dtype=dt(dtype), device=device)
    for name, (dims, dtype) in FIXED_FIELDS.items():
        shape = tuple(_dim(s, hidden, inter) for s in dims)
        base[name] = torch.empty(shape if len(shape) > 1 else shape[0], dtype=dt(dtype), device=device)
    return base


def make_fat_scratch(exl3, torch):
    """The replacement for exl3._fat_scratch (same signature, same per-key dict and shapes)."""
    cache = exl3._FAT_SCRATCH_CACHE
    shared_rows = int(getattr(exl3, "TEMP_ROWS_FUSED", 2048))
    seen: dict[tuple, int] = {}  # stock key -> stock bytes; outlives cache clears, for the census lines

    def _fat_scratch(device, capacity, gate):
        hidden = int(getattr(gate, "in_features", 4096))
        intermediate = int(getattr(gate, "out_features", 2048))
        in_tiles, out_tiles, k_words = map(int, gate.trellis.shape)
        cap = bucket(capacity)
        dev = str(device)
        key = (dev, cap, intermediate, hidden, k_words)
        views = cache.get(key)
        if views is not None:
            return views
        base_key = (BASE_TAG, dev, intermediate, hidden)
        base = cache.get(base_key)
        packed_numel = in_tiles * 2 * out_tiles * k_words
        if base is None or base["rows"] < cap or base["packed"].numel() < packed_numel:
            rows = max(cap, shared_rows, base["rows"] if base is not None else 0)
            pk = max(packed_numel, in_tiles * 2 * out_tiles * PACKED_K_WORDS,
                     base["packed"].numel() if base is not None else 0)
            stale = [k for k in cache if isinstance(k, tuple) and len(k) == 5 and k[0] == dev
                     and k[2:4] == (intermediate, hidden)]
            for k in stale:
                del cache[k]  # views of the old base (or stock sets made before the install)
            cache.pop(base_key, None)
            base = None
            base = _alloc_base(torch, device, rows, hidden, intermediate, pk)
            cache[base_key] = base
            print(
                f"{MARK}: one set of {rows} rows, {set_bytes(rows, hidden, intermediate, pk) / 2**20:.1f} MiB "
                f"(hidden {hidden}, intermediate {intermediate}, packed13 {pk} int16)",
                flush=True,
            )
        views = {name: base[name][:cap] for name in ROW_FIELDS}
        views.update({name: base[name] for name in FIXED_FIELDS})
        views[PACKED_FIELD] = base["packed"][:packed_numel].view(in_tiles, 2 * out_tiles, k_words)
        cache[key] = views
        if key not in seen:
            seen[key] = stock_bytes(cap, hidden, intermediate, in_tiles, out_tiles, k_words)
            shared = set_bytes(base["rows"], hidden, intermediate, base["packed"].numel())
            print(
                f"{MARK}: stock key {len(seen)} (bucket {cap} rows, K {k_words // 16}): the stock cache would "
                f"hold {sum(seen.values()) / 2**20:.1f} MiB in {len(seen)} sets, shared {shared / 2**20:.1f} MiB",
                flush=True,
            )
        return views

    _fat_scratch._dsv41_fat_scratch_shared = True
    return _fat_scratch


def install(exl3=None) -> str:
    """Replace exl3._fat_scratch. Returns 'engaged', 'already installed' or 'disabled: <why>'."""
    import inspect

    import torch

    if exl3 is None:
        import vllm_exl3.exl3 as exl3

    cur = getattr(exl3, "_fat_scratch", None)
    if getattr(cur, "_dsv41_fat_scratch_shared", False):
        return "already installed"
    why = None
    if cur is None or not isinstance(getattr(exl3, "_FAT_SCRATCH_CACHE", None), dict):
        why = "vllm_exl3.exl3 has no _fat_scratch / _FAT_SCRATCH_CACHE"
    else:
        try:
            why = stock_mismatch(inspect.getsource(cur))
        except (OSError, SyntaxError, TypeError) as exc:
            why = f"cannot read _fat_scratch source ({exc!r})"
    if why is not None:
        print(f"dsv41: fat scratch shared DISABLED: {why}; the stock per-bucket cache stays", flush=True)
        return f"disabled: {why}"
    exl3._fat_scratch = make_fat_scratch(exl3, torch)
    print(
        f"dsv41: fat scratch shared engaged (one scratch set of {int(getattr(exl3, 'TEMP_ROWS_FUSED', 2048))} "
        "rows for every row bucket and K)",
        flush=True,
    )
    return "engaged"
