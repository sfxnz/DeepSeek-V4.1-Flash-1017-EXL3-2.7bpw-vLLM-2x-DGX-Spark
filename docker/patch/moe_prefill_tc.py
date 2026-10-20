"""Routed-MoE prefill on tensor cores, EXL3 trellis decoded once per row chunk (DSV41_MOE_PREFILL_TC=1, default off).

vllm_exl3's apply_exl3_fused_moe serves every routed-expert call above the native p2b decode cap (8 rows).
For an 8192-token chunk it costs ~110-115 ms per layer per rank on the 1017 pack (kernel_study/p2b_kbits/
prefill_kbits.py, kb-final-prefill.log), flat in K: exllamav3 exl3_moe decodes each 16x16 trellis tile once
per 16 routed rows, and experts with > 256 rows are reconstructed to fp16 (35 MB each) for cuBLAS.

With the lever, a call with >= DSV41_MOE_PREFILL_TC_MIN_ROWS token rows (default 9: every call the
native p2b decode kernels do not take; kernel_study/moe_prefill pm-time-small: faster than stock at every m
from 16 up) runs moe_prefill_tc.cuh instead: rows sorted by expert on the device (stable argsort, no host sync), a one-block
work list, then two persistent kernels whose warps decode each gate/up/down tile once per chunk of 64
(gate/up) or 128 (down) rows of one expert and feed it to every row of the chunk with mma.sync, fp32
accumulate. The decoded fp16 weights are ExLlamaV3's own (dq_dispatch); the input/output rotations are
ExLlamaV3's had_r_128 math; the activation is the fat path's (clamp, sigmoid(g) * g * u in fp32, fp16).
The K sum of each output runs in ascending order inside one CTA, and the six experts of a token meet in
fp32 atomic adds into the output, as today. Bits differ from the stock path (its exl3_moe keeps split-K
partial sums in fp16, its fat path is cuBLAS), so the gate is an error bound, not equality:

- self-check, first eager call of the process: synthetic rows on that layer's weights (hot experts that
  span several chunks, a one-row chunk, an invalid id) through both paths; max |d| / max |stock| <=
  TOL_MAX and rms(d) / rms(stock) <= TOL_RMS, else the lever disarms (LOG_DISARMED) and the stock path
  serves everything. Pass prints LOG_ENGAGED with the measured errors.
- the first DSV41_MOE_PREFILL_TC_VERIFY (default 2) real eager calls run both paths and compare the same
  way; a failure disarms and returns the stock output.
Never during CUDA graph capture (prefill runs eager; captured decode batches are native).

exllamav3 exl3_moe's scratch (concurrency x 2048 rows x (2 hidden + 2 intermediate) fp16, 294 MiB per rank,
one buffer shared by all layers) is no longer allocated at load: build_exl3_fused_state is rewritten (anchor
TEMPS_OLD, else it stays eager and LOG_DISARMED says so) to leave it unset, and the first stock call that
needs it (a disarmed lever, a call under DSV41_MOE_PREFILL_TC_MIN_ROWS, a captured batch the native kernels do
not take) allocates it once with LOG_TEMPS. The self-check / verify references borrow it and free it again.

The kernel is built ahead of time by docker/Dockerfile and docker/Dockerfile.e16 (prebuilt/<source hash>/), else
compiled at first use with torch.utils.cpp_extension against the installed ExLlamaV3's headers (~1-2 min per rank).
Top-level imports are stdlib only; decode_levers.install() calls install() when the env is on.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import os
import threading
from pathlib import Path

# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dsv41: moe prefill tc self-check ok"
LOG_DISARMED = "dsv41: moe prefill tc DISABLED ->"

ENV = "DSV41_MOE_PREFILL_TC"
DEFAULT_MIN_ROWS = 9
LOG_TEMPS = "dsv41: moe prefill tc: exl3_moe scratch allocated at the first stock call"
DEFAULT_VERIFY = 2
TOL_MAX = 2e-2
TOL_RMS = 5e-3

HERE = Path(__file__).resolve().parent
SOURCES = ("moe_prefill_tc.cu", "moe_prefill_tc.cuh")
PREBUILT_ROOT = Path("/opt/dsv41-moe-prefill-tc")
MODULE_PREFIX = "dsv41_moe_prefill_tc_"

_STATE = {"armed": True, "bypass": False, "checked": False, "verify_left": DEFAULT_VERIFY,
          "min_rows": DEFAULT_MIN_ROWS, "reasons": set(), "lazy_temps": False, "temps_logged": False}

# build_exl3_fused_state's exl3_moe scratch allocation (vllm_exl3 d3cfd39 + widen_p2b_kbits, e15-kbits), and
# its lazy replacement: the key is kept on the layer, the tensors come from ensure_temps().
TEMPS_OLD = """    key = (str(device), hidden, intermediate, concurrency)
    temps = _FUSED_TEMP_CACHE.get(key)
    if temps is None:
        temps = (
            torch.empty((concurrency, TEMP_ROWS_FUSED, hidden), dtype=torch.float16, device=device),
            torch.empty((concurrency, TEMP_ROWS_FUSED, hidden), dtype=torch.float16, device=device),
            torch.empty((concurrency, TEMP_ROWS_FUSED, intermediate), dtype=torch.float16, device=device),
            torch.empty((concurrency, TEMP_ROWS_FUSED, intermediate), dtype=torch.float16, device=device),
        )
        _FUSED_TEMP_CACHE[key] = temps
    layer._exl3_fused_temps = temps
"""
TEMPS_NEW = """    key = (str(device), hidden, intermediate, concurrency)
    # dsv41 moe_prefill_tc: exl3_moe scratch allocated at the first stock call (moe_prefill_tc.ensure_temps).
    layer._exl3_fused_temps = _FUSED_TEMP_CACHE.get(key)
    layer._dsv41_temps_key = key
"""


def patch_build_source(src: str) -> str:
    """vllm_exl3 build_exl3_fused_state without the eager exl3_moe scratch."""
    if src.count(TEMPS_OLD) != 1:
        raise ValueError("build_exl3_fused_state scratch block not found verbatim")
    return src.replace(TEMPS_OLD, TEMPS_NEW)


def temps_bytes(key) -> int:
    _dev, hidden, intermediate, concurrency = key
    return 2 * concurrency * 2048 * (2 * hidden + 2 * intermediate)


def ensure_temps(exl3, layer, rows: int, borrow: bool = False) -> bool:
    """Give the layer exl3_moe's scratch before a stock call; True when this call allocated it."""
    if getattr(layer, "_exl3_fused_temps", None) is not None:
        return False
    key = getattr(layer, "_dsv41_temps_key", None)
    if key is None:
        return False  # not a lazily built layer (native-only build, or an older layer object)
    cache = exl3._FUSED_TEMP_CACHE
    made = key not in cache
    if made:
        import torch

        dev, hidden, intermediate, concurrency = key
        rows_cap = exl3.TEMP_ROWS_FUSED
        cache[key] = tuple(
            torch.empty((concurrency, rows_cap, n), dtype=torch.float16, device=torch.device(dev))
            for n in (hidden, hidden, intermediate, intermediate)
        )
        if not borrow and not _STATE["temps_logged"]:
            _STATE["temps_logged"] = True
            print(f"{LOG_TEMPS} ({temps_bytes(key) / 2**20:.0f} MiB per rank; a {rows}-row call)", flush=True)
    layer._exl3_fused_temps = cache[key]
    return made


def release_temps(exl3, layer) -> None:
    """Undo a borrow: the scratch goes back to the allocator."""
    key = getattr(layer, "_dsv41_temps_key", None)
    if key is not None:
        exl3._FUSED_TEMP_CACHE.pop(key, None)
        layer._exl3_fused_temps = None
_ext_lock = threading.Lock()
_ext_mod = None


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return (env.get("DSV41_MOE_PREFILL_TC", "0") or "0") == "1"


def min_rows(env=None) -> int:
    env = os.environ if env is None else env
    return max(9, int(env.get("DSV41_MOE_PREFILL_TC_MIN_ROWS", "") or DEFAULT_MIN_ROWS))


def verify_calls(env=None) -> int:
    env = os.environ if env is None else env
    return max(0, int(env.get("DSV41_MOE_PREFILL_TC_VERIFY", "") or DEFAULT_VERIFY))


def source_tag(root: Path = HERE) -> str:
    h = hashlib.sha1()
    for name in SOURCES:
        h.update((root / name).read_bytes())
    return h.hexdigest()[:12]


def _disarm(reason: str) -> None:
    if _STATE["armed"]:
        print(f"dsv41: moe prefill tc DISABLED -> {reason}", flush=True)  # LOG_DISARMED
    _STATE["armed"] = False


def _note(reason: str) -> None:
    """A call the kernel does not take (shape, codebook): one line per reason; the stock path serves it."""
    if reason not in _STATE["reasons"]:
        _STATE["reasons"].add(reason)
        print(f"dsv41: moe prefill tc DISABLED -> this call: {reason}", flush=True)  # LOG_DISARMED


def exllamav3_include_dirs() -> list[str]:
    spec = importlib.util.find_spec("exllamav3")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("exllamav3 is not installed (its extension headers are needed)")
    ext = Path(list(spec.submodule_search_locations)[0]) / "exllamav3_ext"
    if not (ext / "quant" / "exl3_dq.cuh").is_file():
        raise RuntimeError(f"{ext} has no quant/exl3_dq.cuh")
    return [str(ext), str(ext / "quant")]


def cuda_arch_flags(arch: str | None = None) -> list[str]:
    """-gencode for `arch` (e.g. "121a"), else for the current device."""
    if arch is None:
        import torch

        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}" + ("a" if major >= 9 else "")
    return [f"-gencode=arch=compute_{arch},code=sm_{arch}"]


def build(build_dir: Path, name: str, arch: str | None = None, verbose: bool = False):
    """Compile the extension into build_dir (torch.utils.cpp_extension, ninja)."""
    from torch.utils.cpp_extension import load

    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name=name,
        sources=[str(HERE / "moe_prefill_tc.cu")],
        extra_include_paths=[str(HERE)] + exllamav3_include_dirs(),
        build_directory=str(build_dir),
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", "-diag-suppress=20012"] + cuda_arch_flags(arch),
        verbose=verbose,
    )


def _load_prebuilt(tag: str):
    name = MODULE_PREFIX + tag
    for so in sorted((PREBUILT_ROOT / tag).glob(f"{name}*.so")):
        loader = importlib.machinery.ExtensionFileLoader(name, str(so))
        spec = importlib.util.spec_from_file_location(name, str(so), loader=loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        return mod
    return None


def ext():
    """The compiled module: prebuilt for these sources if the image has it, else built once per process."""
    global _ext_mod
    with _ext_lock:
        if _ext_mod is None:
            import torch  # noqa: F401  (the extension links against libtorch)

            tag = source_tag()
            mod = _load_prebuilt(tag)
            if mod is None:
                print(f"dsv41: moe prefill tc: no prebuilt kernel for {tag}, compiling (1-2 min)", flush=True)
                mod = build(Path(f"/tmp/dsv41-moe-prefill-tc-{tag}"), MODULE_PREFIX + tag)
            if int(getattr(mod, "ABI", 0)) != 1:
                raise RuntimeError(f"unexpected kernel ABI {getattr(mod, 'ABI', None)}")
            _ext_mod = mod
        return _ext_mod


def unsupported(layer, x2d, inners) -> str | None:
    """Why this call cannot take the kernel, or None."""
    import torch

    if x2d.dim() != 2 or not x2d.is_cuda or x2d.dtype not in (torch.bfloat16, torch.float16):
        return f"x {tuple(x2d.shape)} {x2d.dtype}"
    hidden = int(x2d.shape[1])
    inter = int(getattr(layer, "_exl3_intermediate_local", -1))
    if hidden % 128 or inter <= 0 or inter % 128 or hidden != int(getattr(layer, "_exl3_hidden_size", hidden)):
        return f"hidden {hidden} / intermediate {inter} not multiples of 128"
    flags = tuple(getattr(layer, "_exl3_codebook_flags", ()))
    if flags != (True, False, True, False, True, False):
        return f"codebook flags {flags} (MCG only)"
    k, k_d = int(getattr(layer, "_exl3_k", -1)), int(getattr(layer, "_exl3_k_d", -1))
    if k not in (2, 3, 4) or k_d not in (2, 3, 4):
        return f"K_gu/K_d {k}/{k_d}"
    ptrs = getattr(layer, "_exl3_ptrs", None)
    if not isinstance(ptrs, dict) or any(n not in ptrs for n in PTR_NAMES):
        return "no pointer tables"
    if not 1 <= len(inners) <= 1024:
        return f"{len(inners)} local experts"
    return None


PTR_NAMES = ("gate_trellis", "gate_suh", "gate_svh", "up_trellis", "up_suh", "up_svh",
             "down_trellis", "down_suh", "down_svh")


def run(mod, x2d, ids, weights, layer, inners, expert_map, limit, map_topk_to_local):
    """fp32 [T, hidden] routed output (the stock apply_exl3_fused_moe contract)."""
    import torch

    tokens, hidden = x2d.shape
    topk = int(ids.shape[-1])
    n_exp = len(inners)
    dev = x2d.device
    inter = int(layer._exl3_intermediate_local)
    local = map_topk_to_local(ids, n_exp, expert_map).reshape(-1)  # invalid -> n_exp, sorts last
    order = torch.argsort(local, stable=True)
    counts = torch.zeros(n_exp + 1, dtype=torch.long, device=dev)
    counts.scatter_add_(0, local, torch.ones_like(local))
    token_sorted = torch.div(order, topk, rounding_mode="floor").to(torch.int32)
    weight_sorted = weights.reshape(-1).to(torch.float16)[order]  # the stock path's fp16 routing weights
    slots = tokens * topk
    out = torch.zeros(tokens, hidden, dtype=torch.float32, device=dev)
    a2 = torch.empty(slots, inter, dtype=torch.float16, device=dev)
    items1 = torch.empty(4 * (-(-slots // int(mod.MT1)) + n_exp), dtype=torch.int32, device=dev)
    items2 = torch.empty(4 * (-(-slots // int(mod.MT2)) + n_exp), dtype=torch.int32, device=dev)
    ctrl = torch.empty(4, dtype=torch.int32, device=dev)
    p = layer._exl3_ptrs
    mod.moe_prefill_tc(
        x2d.contiguous(), out, token_sorted, weight_sorted.contiguous(), counts,
        *(p[n] for n in PTR_NAMES), a2, items1, items2, ctrl,
        int(layer._exl3_k), int(layer._exl3_k_d), float(limit) if (limit is not None and limit > 0) else 0.0,
    )
    return out


def compare(out, ref) -> tuple[bool, float, float]:
    """(ok, max |d| / max |ref|, rms(d) / rms(ref)) in fp64 over the finite entries; the non-finite
    entries must sit at the same positions in both."""
    import torch

    o, r = out.double(), ref.double()
    fo, fr = torch.isfinite(o), torch.isfinite(r)
    if not torch.equal(fo, fr):
        return False, float("inf"), float("inf")
    o, r = o[fr], r[fr]
    if r.numel() == 0:
        return True, 0.0, 0.0
    d = o - r
    mx = float(d.abs().max() / r.abs().max().clamp_min(1e-30))
    rms = float(d.pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-30))
    return (mx <= TOL_MAX and rms <= TOL_RMS), mx, rms


def synthetic_case(n_exp: int, hidden: int, dtype, device, topk: int = 6, rows: int = 256):
    """Rows for the self-check: expert 0 in 150 rows (3 gate/up chunks, 2 down chunks), expert 1 in 129
    (a one-row down chunk), the rest spread; one invalid id. Distinct experts per row."""
    import torch

    g = torch.Generator(device="cpu")
    g.manual_seed(1017)
    ids = torch.empty(rows, topk, dtype=torch.long)
    for t in range(rows):
        picks = []
        if t < 150:
            picks.append(0)
        if t < 129:
            picks.append(1 if n_exp > 1 else 0)
        while len(picks) < topk:
            e = int(torch.randint(0, n_exp, (1,), generator=g))
            if e not in picks or n_exp < topk:
                picks.append(e)
        ids[t] = torch.tensor(picks[:topk])
    ids[rows - 1, topk - 1] = -1
    x = (torch.randn(rows, hidden, generator=g) * 0.5).to(dtype)
    w = torch.rand(rows, topk, generator=g) + 0.05
    w = w / w.sum(dim=1, keepdim=True) * 2.5
    return x.to(device), ids.to(device), w.to(device)


def install_lazy_temps(exl3) -> str:
    """Rewrite build_exl3_fused_state (must run before the weights load) so exl3_moe's scratch is lazy."""
    import inspect

    fn = exl3.build_exl3_fused_state
    if getattr(fn, "_dsv41_lazy_temps", False):
        return "lazy"
    try:
        new_src = patch_build_source(inspect.getsource(fn))
    except (OSError, ValueError) as exc:
        print(f"dsv41: moe prefill tc DISABLED -> scratch: exl3_moe scratch stays eager ({exc})", flush=True)
        return "eager"
    exec(compile(new_src, f"{exl3.__file__} [dsv41 moe_prefill_tc]", "exec"), exl3.__dict__)
    exl3.__dict__["build_exl3_fused_state"]._dsv41_lazy_temps = True
    _STATE["lazy_temps"] = True
    return "lazy"


def install() -> str:
    import torch
    import vllm_exl3.exl3 as exl3

    orig = exl3.apply_exl3_fused_moe
    if getattr(orig, "_dsv41_moe_prefill_tc", False):
        return "already installed"
    _STATE["min_rows"] = min_rows()
    _STATE["verify_left"] = verify_calls()
    map_local = exl3.map_topk_to_local
    scratch = install_lazy_temps(exl3)

    def fallback(*args):
        """A stock call: exl3_moe's scratch exists from here on (allocated once, LOG_TEMPS)."""
        ensure_temps(exl3, args[3], int(args[0].shape[0]))
        return orig(*args)

    def stock(*args):
        """The stock path for a self-check / verify reference. It borrows exl3_moe's scratch when nothing holds
        it yet and drops its fat-expert scratch (persistent, ~0.2 GB per row bucket) afterwards."""
        layer = args[3]
        made = ensure_temps(exl3, layer, int(args[0].shape[0]), borrow=True)
        _STATE["bypass"] = True
        try:
            return orig(*args)
        finally:
            _STATE["bypass"] = False
            getattr(exl3, "_FAT_SCRATCH_CACHE", {}).clear()
            if made:
                release_temps(exl3, layer)

    def self_check(mod, layer, inners, x2d, limit) -> bool:
        x, ids, w = synthetic_case(len(inners), int(x2d.shape[1]), x2d.dtype, x2d.device)
        out = run(mod, x, ids, w, layer, inners, None, limit, map_local)
        ref = stock(x, ids, w, layer, inners, None, limit)
        ok, mx, rms = compare(out, ref)
        if not ok:
            _disarm(f"self-check vs stock failed (max_rel {mx:.3g}, rms_rel {rms:.3g})")
            return False
        print(f"dsv41: moe prefill tc self-check ok (synthetic 256 rows vs stock: max_rel {mx:.3g}, rms_rel {rms:.3g}; "
              f"K_gu/K_d {layer._exl3_k}/{layer._exl3_k_d}; min rows {_STATE['min_rows']}, "
              f"verify {_STATE['verify_left']})", flush=True)
        return True

    def apply_exl3_fused_moe(x2d, ids, weights, layer, inners, expert_map, limit=None):
        args = (x2d, ids, weights, layer, inners, expert_map, limit)
        if _STATE["bypass"]:
            return orig(*args)  # inside a reference call (stock's own 2048-row re-slices)
        if (not _STATE["armed"] or int(x2d.shape[0]) < _STATE["min_rows"]
                or torch.cuda.is_current_stream_capturing()):
            return fallback(*args)
        reason = unsupported(layer, x2d, inners)
        if reason is not None:
            _note(reason)
            return fallback(*args)
        try:
            mod = ext()
            if not _STATE["checked"]:
                _STATE["checked"] = True
                if not self_check(mod, layer, inners, x2d, limit):
                    return fallback(*args)
            out = run(mod, x2d, ids, weights, layer, inners, expert_map, limit, map_local)
        except Exception as exc:  # noqa: BLE001 - the stock path keeps serving
            _disarm(f"{type(exc).__name__}: {exc}"[:300])
            return fallback(*args)
        if _STATE["verify_left"] > 0:
            _STATE["verify_left"] -= 1
            ref = stock(*args)
            ok, mx, rms = compare(out, ref)
            if not ok:
                _disarm(f"verify of a {int(x2d.shape[0])}-row call failed (max_rel {mx:.3g}, rms_rel {rms:.3g})")
                return ref
        layer._exl3_last_apply = "prefill_tc"
        return out

    apply_exl3_fused_moe._dsv41_moe_prefill_tc = True
    apply_exl3_fused_moe.__wrapped__ = orig
    exl3.apply_exl3_fused_moe = apply_exl3_fused_moe
    return (f"vllm_exl3 apply_exl3_fused_moe wrapped (min rows {_STATE['min_rows']}, "
            f"verify {_STATE['verify_left']}; self-check at the first eager call; exl3_moe scratch {scratch})")


def main(argv: list[str] | None = None) -> int:
    """Image build (docker/Dockerfile, docker/Dockerfile.e16): compile into <prebuild>/<source tag>/ without a GPU."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--prebuild", type=Path, required=True)
    ap.add_argument("--arch", default="121a")
    args = ap.parse_args(argv)
    tag = source_tag()
    build(args.prebuild / tag, MODULE_PREFIX + tag, arch=args.arch, verbose=True)
    print(f"moe_prefill_tc prebuilt {args.prebuild / tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
