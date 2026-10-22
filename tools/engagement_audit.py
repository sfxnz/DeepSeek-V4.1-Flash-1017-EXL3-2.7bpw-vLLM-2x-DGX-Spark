#!/usr/bin/env python3
"""Post-ready engagement audit: did the default-on patches engage on every rank?

run.sh saves `docker logs` of the head (and the worker over ssh) after
/v1/models is up and runs:

    env <FORWARD_ENVS values> python3 tools/engagement_audit.py --mode warn \
        head=.run-state/audit-head-<ts>.log worker=.run-state/audit-worker-<ts>.log

The marker strings are the LOG_ENGAGED / LOG_DISARMED constants next to each
patch in docker/patch (read with ast, nothing is imported). A marker is
expected only when its env gate is on. Any disarm line is reported.
--mode warn prints WARNING lines and exits 0; --mode strict exits 1.

API_GATEWAY=1: run.sh also passes gateway=.run-state/audit-gateway-<ts>.log,
the gateway container's log. Only gateway/dsv41_gateway.py's LOG_ENGAGED is
expected there, and the patch markers are not.

K-aware (1017 mixed-K packs): with --pack-config <snapshot>/config.json and
DSV41_P2B_COOP=2, every routed-expert (K_gu, K_d) of the pack (expert_bits,
else layer_bits/bits) must be one the image's p2b dataflow kernel takes:
DATAFLOW_K_BY_PATCH below (widen_p2b_kbits.py present: K 2/3/4, mixed), or
module constants DATAFLOW_K (tuple of K) / DATAFLOW_MIXED_K (bool) in a
docker/patch file; otherwise only K = 2 is covered (widen_p2b_dataflow.py). An uncovered pair is a
problem even without a log line (exl3_moe takes such layers silently). If a
patch declares LOG_ENGAGED_K (format keys {kgu} {kd}), its line is expected
on every rank for every covered pair. A dataflow "lever is OFF" line stays a
disarm whatever its K: on an image that covers the pack's K it is genuine.
--coverage-only prints just the pack coverage problems (tools/disarm_scan.sh).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

PATCH_DIR = Path(__file__).resolve().parents[1] / "docker" / "patch"

# patch file -> env values that must all hold for LOG_ENGAGED to be expected.
PATCHES = {
    "lmhead_mxfp8.py": {"DSV41_LMHEAD_MXFP8": "1"},
    "engram_prefetch_v3.py": {"DSV41_ENGRAM_PREFETCH": "1"},
    "engram_gather_v2.py": {"DSV41_ENGRAM_GATHER_V2": "1"},
    "engram_native_stage.py": {"DSV41_ENGRAM_NATIVE_STAGE": "1"},
    "engram_early_hash.py": {"DSV41_ENGRAM_NATIVE_STAGE": "1", "DSV41_ENGRAM_EARLY_HASH": "1"},
    "engram_native_poll.py": {"DSV41_ENGRAM_NATIVE_STAGE": "1", "DSV41_ENGRAM_NATIVE_POLL": "1"},
    "engram_prefill_ahead.py": {"DSV41_ENGRAM_PREFILL_AHEAD": "1"},
    "moe_prep_fused.py": {"DSV41_MOE_PREP_FUSED": "1"},
    "moe_prefill_tc.py": {"DSV41_MOE_PREFILL_TC": "1"},  # opt-in lever; self-check line expected when on
    "candidate_mask_bounded.py": {"DSV41_CANDIDATE_MASK_BOUNDED": "1"},
    "indexer_wp_gemv.py": {"DSV41_INDEXER_WP_GEMV": "1"},
    "swa_meta_fused.py": {"DSV41_SWA_META_FUSED": "1"},
    "swa_meta_batched.py": {"DSV41_SWA_META_FUSED": "1", "DSV41_SWA_META_BATCHED": "1"},
    "kv_packed_groups.py": {"DSV41_KV_PACKED_GROUPS": "1"},  # sitecustomize rewrite, every process
    "indexer_expanded_lazy.py": {"DSV41_INDEXER_EXPANDED_TABLE_LAZY": "1"},  # sitecustomize rewrite, every process
    "attn_t2r_dedup.py": {"DSV41_ATTN_T2R_DEDUP": "1"},
    "engram_wkv_tp.py": {"DSV41_ENGRAM_WKV_TP": "1"},
    "router_gemv.py": {"DSV41_ROUTER_GEMV": "1"},
    "main_proj_overlap.py": {"DSV41_MAIN_PROJ_OVERLAP": "1"},
    "fix_o_proj_woa_fp8.py": {},  # baked into the image at build time
    "drop_page_cache.py": {"DSV41_DROP_PAGE_CACHE": "1"},
    "prefill_empty_cache.py": {"DSV41_PREFILL_EMPTY_CACHE_TOKENS": "8192"},
    "fat_scratch_shared.py": {"DSV41_FAT_SCRATCH_SHARED": "1"},
    "model_loop_release.py": {"DSV41_MODEL_LOOP_RELEASE": "1"},  # sitecustomize rewrite, every process
    "attn_out_defer.py": {"DSV41_ATTN_OUT_DEFER": "1"},
    "engram_post_inplace.py": {"DSV41_ENGRAM_POST_INPLACE": "1"},
    "nccl_eager_twin.py": {"DSV41_NCCL_EAGER_TWIN": "1"},
    "pm_qos.py": {"DSV41_PM_QOS_US": "20"},  # any other value: disarm markers only
    "ar_l2_prefetch.py": {"DSV41_AR_L2_PREFETCH": "1"},
    "sitecustomize.py": None,  # disarm markers only
    "decode_levers.py": None,  # disarm markers only
    "dense_mxfp8_deepgemm.py": None,  # disarm markers only (opt-in lever)
    "widen_p2b_dataflow.py": {"DSV41_P2B_COOP": "2"},  # printed by the patched vllm_exl3_c
    "dense_gemv.py": {"DSV41_DENSE_GEMV": "1"},  # opt-in lever; armed line expected when on
    "mhc_det.py": {"DSV41_MHC_DET_SPLITS": "16"},
    "mhc_det_overlap.py": {"DSV41_MHC_DET_SPLITS": "16", "DSV41_MHC_DET_OVERLAP": "1"},
    "chat_encoder_1017.py": {},  # sitecustomize rewrite, every process
    "image_16bit_gray.py": {"DSV41_IMAGE_16BIT_GRAY": "1"},  # sitecustomize rewrite, every process
    "dspark_adaptive.py": {"DSV41_DSPARK_ADAPTIVE": "1"},  # opt-in; armed line per rank when on
    "decoder_replay.py": {"DSV41_DECODER_REPLAY_ROWS": "2670"},  # other W: disarm markers only
    "prefix_end_fallback.py": {"DSV41_PREFIX_END_FALLBACK": "1"},
    "prefix_reply_retain.py": {"DSV41_PREFIX_REPLY_RETAIN": "1"},
    "sched_prefill_cap.py": {"DSV41_SCHED_PREFILL_CAP": "2048"},  # every process; other caps: disarm markers only
}
# K coverage of the p2b dataflow lever (see the module docstring).
DATAFLOW_GATE = {"DSV41_P2B_COOP": "2"}
DATAFLOW_K_DEFAULT = (2,)  # widen_p2b_dataflow.py: K = 2 MCG only
# Patch file present in docker/patch -> (K it adds to the dataflow kernel, K_gu != K_d support).
# widen_p2b_kbits.py (image e15-kbits): every MCG K in {2, 3, 4}, uniform or per projection.
DATAFLOW_K_BY_PATCH = {"widen_p2b_kbits.py": ((2, 3, 4), True)}
BACKBONE_LAYERS = 40
# vLLM's own line (vllm/compilation/breakable_cudagraph.py:290 in the pinned image).
BREAKABLE_CUDAGRAPH = "Breakable CUDA graph enabled"
BREAKABLE_GATE = {"VLLM_USE_BREAKABLE_CUDAGRAPH": "1", "ENFORCE_EAGER": "0"}
# The API gateway runs in its own container without the image's sitecustomize: its startup line
# (deepseek-recipe imported, tokenizer loaded) is expected in its log only.
GATEWAY_SRC = Path(__file__).resolve().parents[1] / "gateway" / "dsv41_gateway.py"
GATEWAY_RANK = "gateway"
GATEWAY_GATE = {"API_GATEWAY": "1"}


def markers(path: Path) -> dict[str, tuple[str, ...]]:
    """Module-level LOG_ENGAGED / LOG_DISARMED string (or tuple) constants."""
    out: dict[str, tuple[str, ...]] = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", "")
            if name in ("LOG_ENGAGED", "LOG_DISARMED"):
                value = ast.literal_eval(node.value)
                out[name] = (value,) if isinstance(value, str) else tuple(value)
    return out


def constants(path: Path, names) -> dict:
    """Module-level literal constants of a patch file (ast, nothing imported)."""
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", "")
            if name in names:
                out[name] = ast.literal_eval(node.value)
    return out


def dataflow_support(patch_dir: Path | None = None) -> dict:
    """{"k": set of K, "mixed": bool, "engaged_k": [templates]} declared by the patch set."""
    patch_dir = PATCH_DIR if patch_dir is None else patch_dir
    ks, mixed, engaged = set(DATAFLOW_K_DEFAULT), False, []
    for name, (k, mx) in DATAFLOW_K_BY_PATCH.items():
        if (patch_dir / name).is_file():
            ks.update(k)
            mixed = mixed or mx
    for path in sorted(patch_dir.glob("*.py")):
        c = constants(path, ("DATAFLOW_K", "DATAFLOW_MIXED_K", "LOG_ENGAGED_K"))
        ks.update(int(k) for k in c.get("DATAFLOW_K", ()))
        mixed = mixed or bool(c.get("DATAFLOW_MIXED_K", False))
        if "LOG_ENGAGED_K" in c:
            engaged.append(c["LOG_ENGAGED_K"])
    return {"k": ks, "mixed": mixed, "engaged_k": engaged}


def pack_k_layers(config: dict) -> dict[tuple[int, int], list[int]]:
    """(K_gu, K_d) -> backbone layers, from quantization_config expert_bits / layer_bits / bits."""
    q = config.get("quantization_config") or {}
    bits = int(q.get("bits", 2))
    lb = {int(k): int(v) for k, v in (q.get("layer_bits") or {}).items()}
    eb = {int(k): tuple(int(x) for x in v) for k, v in (q.get("expert_bits") or {}).items()}
    out: dict[tuple[int, int], list[int]] = {}
    for layer in range(BACKBONE_LAYERS):
        pair = eb.get(layer) or (lb.get(layer, bits),) * 2
        out.setdefault(pair, []).append(layer)
    return out


def _span(layers: list[int]) -> str:
    runs, start = [], None
    for i, x in enumerate(layers):
        if start is None:
            start = x
        if i + 1 == len(layers) or layers[i + 1] != x + 1:
            runs.append(f"{start}" if start == x else f"{start}-{x}")
            start = None
    return ",".join(runs)


def coverage(config: dict | None, env, patch_dir: Path | None = None) -> tuple[list[str], list[tuple[str, str]]]:
    """(problems, [(label, expected engaged-per-K marker)]) for the pack's routed-expert K."""
    if config is None or not all(env.get(k, "") == v for k, v in DATAFLOW_GATE.items()):
        return [], []
    sup = dataflow_support(patch_dir)
    problems, expected = [], []
    for (kgu, kd), layers in sorted(pack_k_layers(config).items()):
        ok = kgu in sup["k"] and kd in sup["k"] and (kgu == kd or sup["mixed"])
        if not ok:
            problems.append(f"pack: K_gu/K_d={kgu}/{kd} (layers {_span(layers)}) has no p2b dataflow kernel in this "
                            f"patch set (DATAFLOW_K={sorted(sup['k'])}, mixed={sup['mixed']}); those layers run the "
                            f"SORT=0 / exl3_moe fallback with DSV41_P2B_COOP=2")
        else:
            expected += [("p2b dataflow K", t.format(kgu=kgu, kd=kd)) for t in sup["engaged_k"]]
    return problems, expected


def expectations(env) -> tuple[list[tuple[str, str]], list[str]]:
    """([(label, engaged marker)] expected under env, [disarm markers])."""
    expected: list[tuple[str, str]] = []
    disarm: list[str] = []
    for name, gate in PATCHES.items():
        found = markers(PATCH_DIR / name)
        disarm.extend(found.get("LOG_DISARMED", ()))
        if gate is not None and all(env.get(k, "") == v for k, v in gate.items()):
            expected.extend((name, m) for m in found["LOG_ENGAGED"])
    if all(env.get(k, "") == v for k, v in BREAKABLE_GATE.items()):
        expected.append(("vllm", BREAKABLE_CUDAGRAPH))
    return expected, disarm


def audit(logs: dict[str, str], env, pack_config: dict | None = None) -> list[str]:
    """Problems found in {rank: log text}; empty means every rank engaged."""
    expected, disarm = expectations(env)
    problems, per_k = coverage(pack_config, env)
    expected = expected + per_k
    if all(env.get(k, "") == v for k, v in GATEWAY_GATE.items()):
        for marker in markers(GATEWAY_SRC)["LOG_ENGAGED"]:
            if marker not in logs.get(GATEWAY_RANK, ""):
                problems.append(f"{GATEWAY_RANK}: missing {marker!r} (gateway/{GATEWAY_SRC.name})")
    for rank, text in logs.items():
        if rank == GATEWAY_RANK:
            continue
        for label, marker in expected:
            if marker not in text:
                problems.append(f"{rank}: missing {marker!r} ({label})")
        seen = set()
        for line in text.splitlines():
            if line not in seen and any(m in line for m in disarm):
                seen.add(line)
                problems.append(f"{rank}: {line.strip()[:240]}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("warn", "strict", "off"), default="warn")
    ap.add_argument("--pack-config", default=None, help="served snapshot's config.json (K-aware coverage)")
    ap.add_argument("--coverage-only", action="store_true", help="print pack K coverage problems only; exit 1 if any")
    ap.add_argument("logs", nargs="*", metavar="RANK=LOGFILE")
    args = ap.parse_args(argv)
    if args.mode == "off":
        return 0
    pack = None
    if args.pack_config:
        try:
            pack = json.loads(Path(args.pack_config).read_text())
        except (OSError, ValueError) as exc:
            print(f"WARNING audit pack: cannot read {args.pack_config}: {exc}", file=sys.stderr)
            if args.coverage_only:
                return 1
    if args.coverage_only:
        problems, _ = coverage(pack, os.environ)
        for p in problems:
            print(p)
        return 1 if problems else 0
    if not args.logs:
        ap.error("need RANK=LOGFILE arguments")
    logs = {}
    for item in args.logs:
        rank, _, path = item.partition("=")
        try:
            logs[rank] = Path(path).read_text(errors="replace")
        except OSError as exc:
            logs[rank] = ""
            print(f"WARNING audit {rank}: cannot read {path}: {exc}", file=sys.stderr)
    problems = audit(logs, os.environ, pack)
    for p in problems:
        print(f"WARNING audit {p}", file=sys.stderr)
    if not problems:
        print(f"==> audit ok: every expected patch engaged on {', '.join(logs)}")
    return 1 if problems and args.mode == "strict" else 0


if __name__ == "__main__":
    sys.exit(main())
