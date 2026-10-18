#!/usr/bin/env python3
"""Build the DeepSeek-V4.1-Flash-1017 mixed-K EXL3 pack, resumable across both Sparks.

Routed experts (backbone layers 0-39) are re-encoded from the 1017 MXFP4 source
with exllamav3 1.5.1 (image dsv41-quant151): MCG codebook, integer K per
(layer, projection group) from an allocation JSON

    {"<layer 0..39>": [K_gu, K_d], ...}     K_gu: w1 and w3, K_d: w2, K in {2, 3, 4}

Encoder (--hessian-mode):
  real      upstream quantize_exl3_batch with a real activation Hessian per
            expert and input: LDLQ over tail-biting Viterbi tiles, global scale
            search, refit_scales in the Hessian metric. Default.
  identity  the served 2.0 pack's encoder (tools/requant_full.py): uncalibrated
            all-tiles Viterbi, then the H = I scale refit kept only when it
            lowers relerr. No Hessians needed.
Seeds are crc32(tensor stem): output does not depend on node, order or batch.

Hessian format (dsv41-hessian-v1, written by the calibration harness):
  <hessians>/layer_LL/meta.json      {"format": "dsv41-hessian-v1", "layer": L,
                                      "complete": true, ...}   (other keys free-form)
  <hessians>/layer_LL/expert_EEE.safetensors
      "w13_in" F32 [5120, 5120]  input of w1/w3 (the MoE input x)
      "w2_in"  F32 [2304, 2304]  input of w2 (act = silu(w1 x) * w3 x, full
                                 2304 intermediate: the pack stores experts
                                 unsplit; TP slicing happens at load)
  Each H is the final, already mixed mean  sum_t g_t a_t a_t^T / sum_t g_t
  (routed-weight weighted on the expert's tokens, plus whatever all-token
  prior the harness mixes in for cold experts). The builder adds exllamav3's
  usual 2.5% mean-diagonal damping and nothing else. A layer is encoded only
  once its meta.json says complete (the harness can stream layers ahead of
  the builder). ~48.4 GB per layer, ~1.9 TB for 40 layers: stream, do not keep.

Everything that is not a routed expert comes from the 1017 source: the expert
shards' attention / shared-expert / norm / gate tensors byte-for-byte, the
lm_head in MXFP8 (tools/quantize_lmhead_mxfp8.py on source shard 43), and at
assembly every other shard (vision + aligner 1, embed 2, DSpark/MTP 44-46 with
source MXFP4 drafter experts, Engram 47-48) hard-linked from the source blobs.

Work units: one per routed-expert shard (model-00003..00042 = layers 0..39)
plus the lm_head shard (00043). Shared state lives on spark1 under OUT/state
exactly as in requant_full.py (atomic mkdir claims, spark1 ascending, spark2
descending over ssh), plus state/alloc.json and state/build.json pinning the
allocation and encoder settings for every worker.

Subcommands:
  plan        sizes, per-node disk, per-rank GiB, ETA (host, no torch). Also `run --dry-run`.
  run         worker loop (in the dsv41-quant151 container, GPU in-process)
  encode      one unit without claiming (--limit N: first N experts, nothing written)
  status      progress / ETA (host)
  launch-cmd  print the docker command for a worker on a node
  assemble    gather units, link source shards, write config/index/manifest (host)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from pack_meta import (  # noqa: E402
    BACKBONE_LAYERS,
    PACK_REPO,
    SERVE_REVISION,
    SOURCE_SNAPSHOT_REL,
    apply_pack_config,
    build_quantization_config,
    is_routed_expert_tensor,
    is_routed_expert_weight,
    layer_bpw,
    routed_bpw,
    validate_expert_bits,
)
from requant_full import (  # noqa: E402
    LocalStore,
    Log,
    SshStore,
    _gpu_lock,
    _natural,
    default_hub,
    find_src,
    init_state,
    load_done,
    ordered,
    pick_unit,
    read_header,
    sha256_file,
    ssh_argv,
    status_report,
    summarize,
    tensor_seed,
    utcnow,
)

OUT = Path("/home/sfxnz/projects/data/dsv41-1017/pack-build")
PACK_DIR = "models--" + PACK_REPO.replace("/", "--")
IMAGE = "dsv41-quant151"
CONTAINER_PREFIX = "dsv41-1017-build"
STATE_HOST = "spark1"
CODEBOOK = "mcg"
TILE_CHUNK = 4608
HESSIAN_FORMAT = "dsv41-hessian-v1"
HESSIAN_KEYS = {"gu": "w13_in", "d": "w2_in"}
GROUPS = {"gu": ("w1", "w3"), "d": ("w2",)}
EXPERT_SUFFIXES = (".trellis", ".suh", ".svh", "." + CODEBOOK)
LMHEAD_SRC_KEY = "head.weight"
LMHEAD_KEYS = ("lm_head.weight", "lm_head.weight_scale")
# Gate limits per K, fixed before the run: broken-encode catchers, not quality targets.
# relerr = ||W - Q||_F / ||W||_F against the dequantized MXFP4 source; proxy =
# tr(E^T H E) / tr(W^T H W) with the supplied (undamped) Hessian. Basis (GPU smoke,
# spark2 2026-10-17, 1017 layer 0 experts 0-7, data/dsv41-1017/recipe-tooling/gpu-smoke):
# H = I relerr K3 0.132 / K4 0.069 (Sept K2 pack 0.262); real-H mode on SYNTHETIC
# Hessians relerr K2 0.293-0.297 / K3 0.148-0.150 / K4 0.075-0.077. relerr limits are
# ~1.25x (mean) and ~1.5x (max) the real-H values; proxy limits are ~1.3x the H = I
# relerr^2 (an LDLQ encode should do better in its own metric). Re-baseline on the first
# real-Hessian layer; override with --gates.
GATES = {
    "relerr_mean": {2: 0.37, 3: 0.19, 4: 0.097},
    "relerr_max": {2: 0.45, 3: 0.23, 4: 0.12},
    "proxy_mean": {2: 0.09, 3: 0.023, 4: 0.0062},
}
# Sept 2.0 pack, both ranks: "Model loading took 75.58 GiB" (results/2026-09-27-viterbi-adopt/V-1/load-times.txt).
SEPT_RANK_GIB = 75.58
DTYPE_BYTES = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "U32": 4, "F16": 2, "BF16": 2, "I16": 2,
               "U16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}
_EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$")
GIB = 1 << 30


# ---------------------------------------------------------------- pure logic (no torch)


def load_alloc(path) -> dict[str, list[int]]:
    return validate_expert_bits(json.loads(Path(path).read_text()))


def parse_expert(name: str) -> tuple[int, int, str]:
    """'layers.L.ffn.experts.E.wN.weight' -> (L, E, 'wN')."""
    m = _EXPERT_RE.match(name)
    if m is None:
        raise ValueError(f"not a routed expert weight: {name}")
    return int(m.group(1)), int(m.group(2)), m.group(3)


def group_of(proj: str) -> str:
    return "d" if proj == "w2" else "gu"


def expert_k(name: str, alloc: dict) -> int:
    layer, _, proj = parse_expert(name)
    k_gu, k_d = alloc[str(layer)]
    return k_d if proj == "w2" else k_gu


def plan_units(weight_map: dict[str, str]) -> dict[str, dict]:
    """unit ('00003') -> {"file", "kind": experts|lmhead, "layer"} from the source index."""
    files: dict[str, set[int]] = {}
    for name, fname in weight_map.items():
        if is_routed_expert_weight(name):
            files.setdefault(fname, set()).add(parse_expert(name)[0])
    units: dict[str, dict] = {}
    for fname, layers in files.items():
        if len(layers) != 1:
            raise ValueError(f"{fname} holds routed experts of layers {sorted(layers)}")
        units[_unit_of(fname)] = {"file": fname, "kind": "experts", "layer": layers.pop()}
    head = weight_map.get(LMHEAD_SRC_KEY)
    if head is None:
        raise ValueError(f"source index has no {LMHEAD_SRC_KEY}")
    if _unit_of(head) in units:
        raise ValueError(f"{head} holds both experts and {LMHEAD_SRC_KEY}")
    units[_unit_of(head)] = {"file": head, "kind": "lmhead", "layer": None}
    layers = sorted(u["layer"] for u in units.values() if u["kind"] == "experts")
    if layers != list(range(BACKBONE_LAYERS)):
        raise ValueError(f"routed-expert layers {layers[:3]}..{layers[-3:]} != 0..{BACKBONE_LAYERS - 1}")
    return dict(sorted(units.items()))


def _unit_of(fname: str) -> str:
    m = re.match(r"^model-(\d{5})-of-\d{5}\.safetensors$", fname)
    if m is None:
        raise ValueError(f"unexpected shard name {fname}")
    return m.group(1)


def expert_entries(name: str, src_meta: dict, k: int) -> dict[str, tuple[str, list[int]]]:
    """Output EXL3 tensors (dtype, shape) replacing one MXFP4 expert weight [out, in/2] I8."""
    if src_meta["dtype"] not in ("I8", "U8") or len(src_meta["shape"]) != 2:
        raise ValueError(f"{name}: unexpected source {src_meta['dtype']} {src_meta['shape']}")
    out_f, in_f = src_meta["shape"][0], src_meta["shape"][1] * 2
    if in_f % 16 or out_f % 16:
        raise ValueError(f"{name}: {in_f}x{out_f} not a multiple of 16")
    stem = name[: -len(".weight")]
    return {
        stem + ".trellis": ("I16", [in_f // 16, out_f // 16, 16 * k]),
        stem + ".suh": ("F16", [in_f]),
        stem + ".svh": ("F16", [out_f]),
        stem + "." + CODEBOOK: ("I32", []),
    }


def expected_layout(src_hdr: dict, alloc: dict) -> tuple[dict, dict]:
    """(copied, encoded): {name: (dtype, shape)} of an output expert shard."""
    copied, encoded = {}, {}
    for name, meta in src_hdr.items():
        if name == "__metadata__":
            continue
        if is_routed_expert_weight(name):
            encoded.update(expert_entries(name, meta, expert_k(name, alloc)))
        elif not is_routed_expert_tensor(name):
            copied[name] = (meta["dtype"], list(meta["shape"]))
    return copied, encoded


def tensor_bytes(dtype: str, shape) -> int:
    return DTYPE_BYTES[dtype] * math.prod(shape)


def shard_bytes(entries: dict[str, tuple[str, list[int]]]) -> int:
    """safetensors file size for these tensors (header estimated to within a few bytes per key)."""
    data = sum(tensor_bytes(d, s) for d, s in entries.values())
    hdr = {k: {"dtype": d, "shape": s, "data_offsets": [data, data]} for k, (d, s) in entries.items()}
    raw = len(json.dumps(hdr, separators=(",", ":")).encode())
    return 8 + raw + (-raw) % 8 + data


def lmhead_layout(src_hdr: dict) -> dict:
    out = {}
    for name, meta in src_hdr.items():
        if name == "__metadata__":
            continue
        if name == LMHEAD_SRC_KEY:
            n, k = meta["shape"]
            out[LMHEAD_KEYS[0]] = ("F8_E4M3", [n, k])
            out[LMHEAD_KEYS[1]] = ("U8", [n, k // 32])
        else:
            out[name] = (meta["dtype"], list(meta["shape"]))
    return out


def rank_gib(src_hdrs: dict[str, dict], alloc: dict, tp: int = 2) -> dict:
    """Weights per TP rank: routed experts from the layout, the rest calibrated on the Sept pack.

    w1/w3 are column-split (trellis and svh / tp, suh replicated), w2 row-split
    (trellis and suh / tp, svh replicated). Non-routed GiB per rank = the Sept
    2.0 pack's measured 75.58 GiB minus this same model's K=2 routed bytes.
    """
    def routed(kmap) -> float:
        total = 0.0
        for hdr in src_hdrs.values():
            for name, meta in hdr.items():
                if name == "__metadata__" or not is_routed_expert_weight(name):
                    continue
                k = expert_k(name, kmap)
                ent = expert_entries(name, meta, k)
                stem = name[: -len(".weight")]
                tr = tensor_bytes(*ent[stem + ".trellis"]) / tp
                su, sv = tensor_bytes(*ent[stem + ".suh"]), tensor_bytes(*ent[stem + ".svh"])
                total += tr + (su + sv / tp if parse_expert(name)[2] != "w2" else su / tp + sv) + 4
        return total / GIB

    k2 = {str(n): [2, 2] for n in range(BACKBONE_LAYERS)}
    routed_k2, routed_new = routed(k2), routed(alloc)
    nonrouted = SEPT_RANK_GIB - routed_k2
    return {"routed_gib": routed_new, "routed_k2_gib": routed_k2, "nonrouted_gib": nonrouted,
            "total_gib": nonrouted + routed_new, "basis": f"Sept 2.0 pack load {SEPT_RANK_GIB} GiB/rank"}


def node_split(units: dict[str, dict], cost: dict[str, float]) -> dict[str, list[str]]:
    """Expected unit split: spark1 ascending, spark2 descending, equal speed, meeting by cost."""
    asc = ordered(units, "asc")
    total, run, cut = sum(cost[u] for u in asc), 0.0, len(asc)
    for i, u in enumerate(asc):
        if run + cost[u] / 2 > total / 2:
            cut = i
            break
        run += cost[u]
    return {"spark1": asc[:cut], "spark2": asc[cut:]}


def build_plan(src_hdrs: dict[str, dict], weight_map: dict, alloc: dict, sec_per_tensor: float,
               hessian_mode: str) -> dict:
    units = plan_units(weight_map)
    rows, cost = {}, {}
    for u, info in units.items():
        hdr = src_hdrs[info["file"]]
        if info["kind"] == "lmhead":
            entries = lmhead_layout(hdr)
            n = 1
        else:
            copied, encoded = expected_layout(hdr, alloc)
            entries = {**copied, **encoded}
            n = sum(1 for k in hdr if is_routed_expert_weight(k))
        src_size = sum(tensor_bytes(m["dtype"], m["shape"]) for k, m in hdr.items() if k != "__metadata__")
        rows[u] = {"file": info["file"], "kind": info["kind"], "layer": info["layer"],
                   "k": alloc[str(info["layer"])] if info["kind"] == "experts" else None,
                   "tensors": n, "bytes": shard_bytes(entries), "src_bytes": src_size}
        cost[u] = n * sec_per_tensor if info["kind"] == "experts" else 180.0
    split = node_split(units, cost)
    unit_files = {r["file"] for r in rows.values()}
    linked = {f: sum(tensor_bytes(m["dtype"], m["shape"]) for k, m in h.items() if k != "__metadata__")
              for f, h in src_hdrs.items() if f not in unit_files}
    new_bytes = sum(r["bytes"] for r in rows.values())
    hess_layer = 0
    if hessian_mode == "real":
        any_hdr = src_hdrs[units[ordered(units, "asc")[0]]["file"]]
        dims = {}
        for name, meta in any_hdr.items():
            if name != "__metadata__" and is_routed_expert_weight(name):
                dims[parse_expert(name)[2]] = meta["shape"][1] * 2
        n_exp = len({parse_expert(n)[1] for n in any_hdr if n != "__metadata__" and is_routed_expert_weight(n)})
        hess_layer = n_exp * 4 * (dims["w1"] ** 2 + dims["w2"] ** 2)
    nodes = {}
    for node, us in split.items():
        nodes[node] = {"units": us, "encode_hours": sum(cost[u] for u in us) / 3600,
                       "encoded_bytes": sum(rows[u]["bytes"] for u in us)}
    return {"units": rows, "nodes": nodes, "pack_new_bytes": new_bytes,
            "linked_source_bytes": sum(linked.values()), "linked_files": sorted(linked),
            "source_bytes": sum(r["src_bytes"] for r in rows.values()) + sum(linked.values()),
            "hessian_bytes_per_layer": hess_layer, "routed_bpw": routed_bpw(alloc),
            "rank": rank_gib(src_hdrs, alloc), "sec_per_tensor": sec_per_tensor,
            "hessian_mode": hessian_mode}


def format_plan(plan: dict, free_bytes: int | None = None) -> str:
    gb = 1e9
    lines = [f"routed bpw (trellis, param-weighted): {plan['routed_bpw']:.4f}   encoder: {plan['hessian_mode']}"]
    lines.append("unit  file                                 kind     layer K_gu/K_d tensors   out GB   src GB")
    for u, r in plan["units"].items():
        k = "/".join(map(str, r["k"])) if r["k"] else "-"
        lines.append(f"{u}  {r['file']}  {r['kind']:<7}  {str(r['layer']):>5}  {k:>7}  {r['tensors']:>6}  "
                     f"{r['bytes'] / gb:7.3f}  {r['src_bytes'] / gb:7.3f}")
    lines.append(f"pack: {plan['pack_new_bytes'] / gb:.1f} GB new files + {plan['linked_source_bytes'] / gb:.1f} GB "
                 f"hard-linked from the source ({', '.join(f[6:11] for f in plan['linked_files'])}); "
                 f"source snapshot {plan['source_bytes'] / gb:.1f} GB must be on each node")
    for node, n in plan["nodes"].items():
        lines.append(f"{node}: {len(n['units'])} units ({n['units'][0] if n['units'] else '-'}.."
                     f"{n['units'][-1] if n['units'] else '-'}), encode ~{n['encode_hours']:.1f} h at "
                     f"{plan['sec_per_tensor']:.2f} s/tensor, {n['encoded_bytes'] / gb:.1f} GB encoded here; "
                     f"after assemble every node holds {plan['pack_new_bytes'] / gb:.1f} GB of new files "
                     f"(+ source {plan['source_bytes'] / gb:.1f} GB, links free on the same filesystem)")
    if plan["hessian_bytes_per_layer"]:
        hb = plan["hessian_bytes_per_layer"]
        lines.append(f"hessians: {hb / gb:.1f} GB per layer, {hb * BACKBONE_LAYERS / gb:.0f} GB for all "
                     f"{BACKBONE_LAYERS} layers (stream: keep only the layers not yet encoded)")
    r = plan["rank"]
    lines.append(f"per rank (TP=2): routed {r['routed_gib']:.2f} GiB (K=2 would be {r['routed_k2_gib']:.2f}) + "
                 f"non-routed {r['nonrouted_gib']:.2f} GiB = {r['total_gib']:.2f} GiB weights ({r['basis']})")
    if free_bytes is not None:
        lines.append(f"this node: {free_bytes / gb:.0f} GB free under the output dir")
    return "\n".join(lines)


def gate_unit(rows: list[dict], alloc_k: list[int], gates: dict, hessian: bool) -> dict:
    """Per projection group: finite, mean/max relerr and (real H) mean proxy within the K's limits."""
    out = {"ok": bool(rows), "groups": {}}
    for g, k in zip(("gu", "d"), alloc_k):
        sub = [r for r in rows if r["group"] == g]
        rel = [r["relerr"] for r in sub]
        prx = [r["proxy"] for r in sub if r.get("proxy") is not None]
        finite = all(math.isfinite(x) for x in rel + prx)
        st = {"k": k, "n": len(sub), "finite": finite,
              "relerr_mean": sum(rel) / len(rel) if rel else None, "relerr_max": max(rel) if rel else None,
              "relerr_mean_limit": gates["relerr_mean"][k], "relerr_max_limit": gates["relerr_max"][k]}
        ok = bool(sub) and finite and st["relerr_mean"] <= st["relerr_mean_limit"] \
            and st["relerr_max"] <= st["relerr_max_limit"]
        if hessian:
            st["proxy_mean"] = sum(prx) / len(prx) if prx else None
            st["proxy_mean_limit"] = gates["proxy_mean"][k]
            ok = ok and len(prx) == len(sub) and st["proxy_mean"] <= st["proxy_mean_limit"]
        st["ok"] = ok
        out["groups"][g] = st
        out["ok"] = out["ok"] and ok
    return out


def layout_problems(hdr: dict, expected: dict) -> list[str]:
    """Keys, dtypes and shapes of a written header vs {name: (dtype, shape)}."""
    have = {k: v for k, v in hdr.items() if k != "__metadata__"}
    probs = []
    missing, extra = sorted(set(expected) - set(have)), sorted(set(have) - set(expected))
    if missing:
        probs.append(f"missing {len(missing)} keys, e.g. {missing[:3]}")
    if extra:
        probs.append(f"extra {len(extra)} keys, e.g. {extra[:3]}")
    for k in sorted(set(have) & set(expected)):
        d, s = expected[k]
        if have[k]["dtype"] != d or list(have[k]["shape"]) != list(s):
            probs.append(f"{k}: {have[k]['dtype']} {have[k]['shape']} != {d} {s}")
    return probs


def copied_bytes_problems(path: Path, src: Path, names) -> list[str]:
    """Tensors in `names` must be byte-equal (and same dtype/shape) between two safetensors files."""
    raw_n, hn = read_header(path)
    raw_s, hs = read_header(src)
    base_n, base_s = 8 + len(raw_n), 8 + len(raw_s)
    probs = []
    with open(path, "rb") as fn, open(src, "rb") as fs:
        for k in sorted(names):
            a, b = hn[k], hs[k]
            if (a["dtype"], a["shape"]) != (b["dtype"], b["shape"]):
                probs.append(f"{k}: {a['dtype']} {a['shape']} vs source {b['dtype']} {b['shape']}")
                continue
            fn.seek(base_n + a["data_offsets"][0])
            fs.seek(base_s + b["data_offsets"][0])
            if fn.read(a["data_offsets"][1] - a["data_offsets"][0]) != fs.read(b["data_offsets"][1] - b["data_offsets"][0]):
                probs.append(f"{k}: bytes differ from the 1017 source")
    return probs


def hessian_ready(root: Path | None, layer: int) -> bool:
    if root is None:
        return True
    meta = Path(root) / f"layer_{layer:02d}" / "meta.json"
    try:
        m = json.loads(meta.read_text())
    except (OSError, ValueError):
        return False
    return m.get("format") == HESSIAN_FORMAT and m.get("layer") == layer and m.get("complete") is True


def parse_layer_list(spec: str) -> set[int]:
    """'1,3,5-9' -> {1, 3, 5, 6, 7, 8, 9}."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def unit_selected(info: dict, only_layers: set[int] | None, lmhead: bool, limit: int | None) -> bool:
    """run --layers/--lmhead/--limit: expert units of only_layers (all if None); the lm_head unit unless
    --layers is given without --lmhead; never the lm_head under --limit (a test run records nothing)."""
    if info["kind"] == "lmhead":
        return not limit and (only_layers is None or lmhead)
    return only_layers is None or info["layer"] in only_layers


def pin_state(store: LocalStore, rel: str, value) -> None:
    """Write state/<rel> once; later workers must agree with it."""
    known = store.read(rel)
    if known is None:
        store.write(rel, json.dumps(value, indent=1, sort_keys=True) + "\n")
    elif json.loads(known) != value:
        raise SystemExit(f"state/{rel} disagrees with this worker's settings: {known.strip()[:300]}")


def build_manifest(units: dict, done: dict, alloc: dict, build: dict) -> dict:
    exp = [u for u in sorted(done) if done[u].get("kind") == "experts"]
    rel = {g: [done[u]["gate"]["groups"][g]["relerr_mean"] for u in exp] for g in ("gu", "d")}
    return {
        "revision": build.get("revision"), "build": build, "alloc": alloc, "routed_bpw": routed_bpw(alloc),
        "units_total": len(units), "units_done": len(done), "complete": set(done) == set(units),
        "relerr_mean_by_group": {g: (sum(v) / len(v) if v else None) for g, v in rel.items()},
        "units": {u: {k: v for k, v in done[u].items() if k != "rows"} for u in sorted(done)},
        "updated": utcnow(),
    }


def refresh_manifest(store: LocalStore, units: dict, alloc: dict, build: dict) -> dict:
    man = build_manifest(units, load_done(store), alloc, build)
    store.write("manifest.json", json.dumps(man, indent=1) + "\n")
    if man["complete"] and store.read("DONE") is None:
        store.write("DONE", json.dumps({"revision": build.get("revision"), "units_total": man["units_total"],
                                        "routed_bpw": man["routed_bpw"], "completed": utcnow()}, indent=1) + "\n")
    return man


def build_index(snap: Path) -> dict:
    wm, total = {}, 0
    for f in sorted(snap.glob("model-*-of-*.safetensors")):
        for k, m in read_header(f)[1].items():
            if k == "__metadata__":
                continue
            if k in wm:
                raise SystemExit(f"{k} in both {wm[k]} and {f.name}")
            wm[k] = f.name
            total += m["data_offsets"][1] - m["data_offsets"][0]
    return {"metadata": {"total_size": total}, "weight_map": dict(sorted(wm.items()))}


def expected_pack_names(src_map: dict) -> set[str]:
    """Tensor names the assembled pack must hold, derived from the source index."""
    out = set()
    for name in src_map:
        if is_routed_expert_weight(name):
            stem = name[: -len(".weight")]
            out.update(stem + s for s in EXPERT_SUFFIXES)
        elif is_routed_expert_tensor(name):
            continue
        elif name == LMHEAD_SRC_KEY:
            out.update(LMHEAD_KEYS)
        else:
            out.add(name)
    return out


# ---------------------------------------------------------------- GPU / torch (container)


def _drop_cache(path) -> None:
    """POSIX_FADV_DONTNEED on a file just read (the calibration harness' page-cache mechanism)."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def load_hessian(root: Path, layer: int, expert: int, group: str, dim: int):
    """One expert's H for one input (CPU fp32), checked: shape, finite, symmetric, PSD-ish diagonal."""
    import torch
    from safetensors import safe_open

    path = Path(root) / f"layer_{layer:02d}" / f"expert_{expert:03d}.safetensors"
    key = HESSIAN_KEYS[group]
    with safe_open(str(path), framework="pt") as fh:
        # clone: get_tensor is backed by the file's mmap, so every later read of h (the checks below,
        # the encoder's copy) would fault the dropped pages straight back into the page cache
        h = fh.get_tensor(key).clone()
    _drop_cache(path)  # 48 GB of Hessians per layer: the GB10 does not reclaim page cache for CUDA allocations
    if h.dtype != torch.float32 or tuple(h.shape) != (dim, dim):
        raise ValueError(f"{path}:{key}: {h.dtype} {tuple(h.shape)}, want float32 ({dim}, {dim})")
    if not torch.isfinite(h).all():
        raise ValueError(f"{path}:{key}: non-finite values")
    scale = h.abs().max().item()
    if scale == 0 or (h - h.T).abs().max().item() > 1e-4 * scale:
        raise ValueError(f"{path}:{key}: zero or not symmetric")
    if (h.diagonal() < 0).any():
        raise ValueError(f"{path}:{key}: negative diagonal")
    return h


def h_data(h, device: str, key: str) -> dict:
    """exllamav3 Hessian record for an already-normalized H (count 1: finalize divides by it)."""
    import torch

    dev = torch.device(device)
    # Always a copy: finalize_capture_H scales, damps and rotates H in place, and the caller
    # keeps the original for the proxy metric.
    return {"H": h.to(dev, dtype=torch.float32, copy=True).contiguous(), "device": dev, "finalized": False, "count": 1,
            "first_key": key, "num_total": 1, "inf_nan": torch.zeros(2, dtype=torch.long, device=dev),
            "L": None, "q_fallback": False}


def proxy_error(w, q, h) -> float:
    """tr(E^T H E) / tr(W^T H W), E = W - Q, all (in, out); H (in, in)."""
    e = w - q
    num = (e * (h @ e)).sum().item()
    den = (w * (h @ w)).sum().item()
    return num / max(den, 1e-30)


def encode_identity(w, stem: str, k: int, device: str, h_cache: dict, chunk: int, recon) -> tuple[dict, dict]:
    """The served 2.0 pack's encoder at K: all-tiles Viterbi, then the H = I refit when it helps."""
    import torch

    from quantize_experts_exl3 import _quantize_fast
    from requant_probe import refit_identity, relerr

    torch.manual_seed(tensor_seed(stem))
    enc = _quantize_fast([w.clone()], k, device, h_cache, codebook=CODEBOOK, tile_chunk=chunk)[0]
    trellis, su, sv = enc["trellis"], enc["suh"].to(w.device), enc["svh"].to(w.device)
    q = recon(trellis, su, sv, w.shape, k)
    e_vit = relerr(q, w)
    _, r, c = refit_identity(w, q.float())
    su_f, sv_f = (su.float() * r).half(), (sv.float() * c).half()
    e_fit = relerr(recon(trellis, su_f, sv_f, w.shape, k), w)
    keep = math.isfinite(e_fit) and e_fit <= e_vit
    if keep:
        su, sv = su_f, sv_f
    return ({"trellis": trellis, "suh": su.cpu().contiguous(), "svh": sv.cpu().contiguous(), CODEBOOK: enc[CODEBOOK]},
            {"viterbi_relerr": e_vit, "refit": keep})


def encode_hessian(ws: list, hs: list, stems: list[str], k: int, device: str) -> list[tuple[dict, dict]]:
    """quantize_exl3_batch over same-shape tensors; hs[i] is one H per tensor (objects may repeat:
    w1 and w3 of an expert share their input Hessian and its sign flips)."""
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3_batch

    from quantize_experts_exl3 import quant_args_for

    records, by_id = [], {}
    for h, stem in zip(hs, stems):
        if id(h) not in by_id:
            by_id[id(h)] = h_data(h, device, stem)
        records.append(by_id[id(h)])
    qas = []
    for stem in stems:
        qa = quant_args_for(k, device, CODEBOOK)
        qa["seed"] = tensor_seed(stem)
        qas.append(qa)
    res = quantize_exl3_batch(list(ws), records, qas)
    out = []
    for (proxy_internal, t), qa in zip(res, qas):
        enc = {"trellis": t["trellis"].cpu(), "suh": t["suh"].cpu().contiguous(),
               "svh": t["svh"].cpu().contiguous(), CODEBOOK: t[CODEBOOK].cpu()}
        out.append((enc, {"proxy_ldlq": float(proxy_internal), "q_fallback": bool(qa.get("q_fallback")),
                          "g_scale": float(qa.get("g_scale", float("nan")))}))
    return out


def encode_experts_unit(u: str, info: dict, srcs: list[Path], alloc: dict, out: Path, *, mode: str,
                        hessians: Path | None, gates: dict, batch: int = 8, device: str = "cuda:0",
                        tile_chunk: int = TILE_CHUNK, limit: int | None = None, heartbeat=None,
                        log=print) -> dict:
    """Encode one routed-expert shard; write OUT/shards/<file> and OUT/units/<u>.json."""
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    from safetensors import safe_open
    from safetensors.torch import save_file

    from quantize_experts_exl3 import _dequant_t, _load_index
    from requant_probe import package_version, relerr

    fname, layer = info["file"], info["layer"]
    k_gu, k_d = alloc[str(layer)]
    src_file = find_src(srcs, fname)
    if src_file is None:
        raise FileNotFoundError(f"no source {fname} under {srcs}")
    src_hdr = read_header(src_file)[1]
    names = sorted(n for n, f in _load_index(Path(srcs[0]))["weight_map"].items() if f == fname)
    if set(names) != {k for k in src_hdr if k != "__metadata__"}:
        raise SystemExit(f"{fname}: index and header disagree")
    copied, encoded = expected_layout(src_hdr, alloc)
    experts = sorted({parse_expert(n)[1] for n in names if is_routed_expert_weight(n)})
    if limit:
        experts = experts[:limit]
    dev = torch.device(device)

    def recon(trellis, suh, svh, shape, k):
        o = torch.empty(shape, dtype=torch.half, device=dev)
        ext.reconstruct_had_slice(o, trellis.to(dev), suh.to(dev).half(), svh.to(dev).half(), k, True, False, 0)
        return o

    tensors: dict = {}
    rows: list[dict] = []
    h_cache: dict = {}
    n_total = len(experts) * 3
    t0, started = time.time(), utcnow()
    if heartbeat is not None:
        heartbeat({"time": started, "tensors_done": 0, "tensors": n_total, "sec_per_tensor": None})
    with safe_open(str(src_file), framework="pt") as fh:
        for n in copied:
            tensors[n] = fh.get_tensor(n)
        for b0 in range(0, len(experts), batch):
            chunk = experts[b0:b0 + batch]
            for group, projs in GROUPS.items():
                k = k_gu if group == "gu" else k_d
                stems = [f"layers.{layer}.ffn.experts.{e}.{p}" for e in chunk for p in projs]
                ta = time.time()
                ws = [_dequant_t(fh.get_tensor(s + ".weight"), fh.get_tensor(s + ".scale"), device) for s in stems]
                hs = None
                if mode == "real":
                    per_expert = {e: load_hessian(hessians, layer, e, group, ws[i * len(projs)].shape[0])
                                  for i, e in enumerate(chunk)}
                    hs = [per_expert[e] for e in chunk for _ in projs]
                    encs = encode_hessian(ws, hs, stems, k, device)
                else:
                    encs = [encode_identity(w.to(dev), s, k, device, h_cache, tile_chunk, recon)
                            for w, s in zip(ws, stems)]
                t_enc = (time.time() - ta) / len(stems)
                for i, (stem, (enc, extra)) in enumerate(zip(stems, encs)):
                    w = ws[i].to(dev)
                    q = recon(enc["trellis"], enc["suh"], enc["svh"], w.shape, k).float()
                    row = {"tensor": stem, "group": group, "k": k, "relerr": relerr(q, w), **extra}
                    if hs is not None:
                        row["proxy"] = proxy_error(w, q, hs[i].to(dev))
                    row["t_enc"] = t_enc
                    rows.append(row)
                    for suf, val in enc.items():
                        tensors[f"{stem}.{suf}"] = val
                    del w, q
                del ws, encs, hs
            kdone = len(rows)
            spt = (time.time() - t0) / kdone
            log(f"unit {u} L{layer} K{k_gu}/{k_d} {kdone}/{n_total} {spt:.3f} s/tensor "
                f"relerr gu {_mean(rows, 'gu', 'relerr'):.4f} d {_mean(rows, 'd', 'relerr'):.4f}"
                + (f" proxy gu {_mean(rows, 'gu', 'proxy'):.5f} d {_mean(rows, 'd', 'proxy'):.5f}"
                   if mode == "real" else "")
                + f" unit eta {(n_total - kdone) * spt / 60:.1f} min")
            if heartbeat is not None:
                heartbeat({"time": utcnow(), "tensors_done": kdone, "tensors": n_total, "sec_per_tensor": spt})
    encode_s = time.time() - t0
    gate = gate_unit(rows, [k_gu, k_d], gates, mode == "real")
    rec = {"unit": u, "file": fname, "kind": "experts", "layer": layer, "k": [k_gu, k_d], "mode": mode,
           "tensors": len(rows), "gate": gate, "stats": {g: {"relerr": summarize(r["relerr"] for r in rows if r["group"] == g),
                                                              "proxy": summarize(r["proxy"] for r in rows if r["group"] == g and "proxy" in r)}
                                                         for g in GROUPS},
           "started": started, "encode_s": encode_s, "sec_per_tensor": encode_s / max(len(rows), 1),
           "batch": batch, "tile_chunk": tile_chunk, "exllamav3": package_version("exllamav3"),
           "torch": torch.__version__, "src": str(src_file)}
    if mode == "real":  # record which Hessian set encoded this unit
        meta = Path(hessians) / f"layer_{layer:02d}" / "meta.json"
        rec["hessians"] = {"root": str(hessians), "meta": json.loads(meta.read_text()) if meta.is_file() else None}
    return _finish_unit(u, fname, src_file, tensors, {**copied, **encoded}, list(copied), rows, rec, out,
                        partial=bool(limit), expert_keys=list(encoded))


def _mean(rows, group, key) -> float:
    xs = [r[key] for r in rows if r["group"] == group and r.get(key) is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def encode_lmhead_unit(u: str, info: dict, srcs: list[Path], out: Path, log=print) -> dict:
    """Source shard 43 with head.weight -> lm_head MXFP8 pair (tools/quantize_lmhead_mxfp8.py)."""
    import torch
    from safetensors import safe_open

    from quantize_lmhead_mxfp8 import rewrite_shard

    fname = info["file"]
    src_file = find_src(srcs, fname)
    if src_file is None:
        raise FileNotFoundError(f"no source {fname} under {srcs}")
    expected = lmhead_layout(read_header(src_file)[1])
    t0, started = time.time(), utcnow()
    (out / "shards").mkdir(parents=True, exist_ok=True)
    tmp = out / "shards" / (fname + ".tmp")
    shutil.copyfile(src_file, tmp)
    nv, ns = rewrite_shard(tmp)
    with safe_open(str(tmp), framework="pt") as fq, safe_open(str(src_file), framework="pt") as fs:
        v, s = fq.get_tensor(LMHEAD_KEYS[0]), fq.get_tensor(LMHEAD_KEYS[1])
        w = fs.get_tensor(LMHEAD_SRC_KEY)
        num = den = 0.0
        for a in range(0, w.shape[0], 8192):
            ref = w[a:a + 8192].double()
            deq = v[a:a + 8192].double().view(-1, w.shape[1] // 32, 32) * torch.exp2(s[a:a + 8192].double() - 127).unsqueeze(-1)
            num += (deq.view(-1, w.shape[1]) - ref).square().sum().item()
            den += ref.square().sum().item()
    rel = math.sqrt(num / max(den, 1e-300))
    log(f"unit {u} lm_head mxfp8: {nv:,} e4m3, {ns:,} e8m0, relerr {rel:.4f}")
    rows = [{"tensor": "lm_head.weight", "group": "lmhead", "relerr": rel}]
    gate = {"ok": math.isfinite(rel) and rel < 0.05, "relerr": rel, "limit": 0.05}
    rec = {"unit": u, "file": fname, "kind": "lmhead", "tensors": 1, "gate": gate, "started": started,
           "encode_s": time.time() - t0, "sec_per_tensor": time.time() - t0, "src": str(src_file)}
    return _finish_unit(u, fname, src_file, None, expected, [k for k in expected if k not in LMHEAD_KEYS],
                        rows, rec, out, partial=False, expert_keys=[], written_tmp=tmp)


def _finish_unit(u, fname, src_file, tensors, expected, copied_names, rows, rec, out: Path, *, partial: bool,
                 expert_keys, written_tmp: Path | None = None) -> dict:
    """Gate, save, verify (layout, source bytes, reload), then move into OUT/shards."""
    (out / "shards").mkdir(parents=True, exist_ok=True)
    (out / "units").mkdir(parents=True, exist_ok=True)
    if partial or not rec["gate"]["ok"]:
        rec["written"] = False
        tag = "partial" if partial else "failed"
        (out / "units" / f"{u}.{tag}.json").write_text(json.dumps(rec | {"rows": rows}, indent=1) + "\n")
        if written_tmp is not None:
            written_tmp.unlink(missing_ok=True)
        if not rec["gate"]["ok"]:
            raise RuntimeError(f"unit {u} gate failed: {json.dumps(rec['gate'])[:600]}")
        return rec
    final = out / "shards" / fname
    tmp = written_tmp or final.with_name(final.name + ".tmp")
    if tensors is not None:
        from safetensors.torch import save_file

        save_file(tensors, str(tmp))
    probs = verify_shard(tmp, src_file, expected, copied_names, tensors, expert_keys)
    if probs:
        rec["problems"] = probs[:50]
        (out / "units" / f"{u}.failed.json").write_text(json.dumps(rec | {"rows": rows}, indent=1) + "\n")
        raise RuntimeError(f"unit {u} verification failed: {probs[:5]}")
    os.chmod(tmp, 0o644)
    os.replace(tmp, final)
    rec.update(written=True, size=final.stat().st_size, sha256=sha256_file(final), finished=utcnow())
    rec["wall_s"] = time.time() - _ts(rec["started"])
    tmpj = out / "units" / f".{u}.json.tmp"
    tmpj.write_text(json.dumps(rec | {"rows": rows}, indent=1) + "\n")
    os.replace(tmpj, out / "units" / f"{u}.json")
    return rec | {"rows": rows}


def _ts(s: str) -> float:
    from requant_full import parse_utc

    return parse_utc(s)


def verify_shard(path: Path, src: Path, expected: dict, copied_names, tensors, expert_keys) -> list[str]:
    """Layout vs the plan, copied tensors byte-equal to the 1017 source, encoded tensors reload equal."""
    probs = layout_problems(read_header(path)[1], expected)
    if probs:
        return probs
    probs += copied_bytes_problems(path, src, copied_names)
    if tensors is not None and expert_keys:
        import torch
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as fh:
            for key in expert_keys:
                if not torch.equal(fh.get_tensor(key), tensors[key]):
                    probs.append(f"{key}: reload differs from the encoded tensor")
    return probs


def reuse_local(out: Path, u: str, fname: str) -> dict | None:
    j, f = out / "units" / f"{u}.json", out / "shards" / fname
    if not (j.is_file() and f.is_file()):
        return None
    rec = json.loads(j.read_text())
    if not (rec.get("written") and rec.get("gate", {}).get("ok")):
        return None
    if rec.get("size") != f.stat().st_size or rec.get("sha256") != sha256_file(f):
        return None
    return rec


# ---------------------------------------------------------------- commands


def make_store(args) -> LocalStore:
    root = Path(args.out) / "state"
    if getattr(args, "state_host", None):
        return SshStore(ssh_argv(args.state_host, Path(args.ssh_dir).expanduser()), root)
    return LocalStore(root)


def _source_index(srcs) -> dict:
    return json.loads((Path(srcs[0]) / "model.safetensors.index.json").read_text())["weight_map"]


def _gates(args) -> dict:
    g = json.loads(json.dumps(GATES))
    if getattr(args, "gates", None):
        for key, per_k in json.loads(Path(args.gates).read_text()).items():
            g[key].update({int(k): float(v) for k, v in per_k.items()})
    return {key: {int(k): v for k, v in per_k.items()} for key, per_k in g.items()}


def build_settings(args, alloc) -> dict:
    return {"revision": args.revision, "hessian_mode": args.hessian_mode, "codebook": CODEBOOK,
            "batch": args.batch, "tile_chunk": args.tile_chunk, "gates": {k: {str(x): y for x, y in v.items()}
                                                                         for k, v in _gates(args).items()},
            "routed_bpw": routed_bpw(alloc)}


def cmd_plan(args) -> int:
    srcs = [Path(s) for s in args.src]
    wm = _source_index(srcs)
    hdrs = {}
    for f in sorted(set(wm.values())):
        p = find_src(srcs, f)
        if p is not None:
            hdrs[f] = read_header(p)[1]
        elif args.headers and (Path(args.headers) / f"{f}.json").is_file():
            hdrs[f] = json.loads((Path(args.headers) / f"{f}.json").read_text())
        else:
            raise SystemExit(f"no header for {f}: not in {args.src} and not in --headers")
    alloc = load_alloc(args.alloc)
    plan = build_plan(hdrs, wm, alloc, args.sec_per_tensor, args.hessian_mode)
    free = None
    p = Path(args.out)
    while not p.exists() and p != p.parent:
        p = p.parent
    if p.exists():
        free = shutil.disk_usage(p).free
    if args.json:
        print(json.dumps(plan, indent=1))
    else:
        print(format_plan(plan, free))
    return 0


def cmd_run(args) -> int:
    import fcntl

    if args.dry_run:
        return cmd_plan(args)
    out = Path(args.out)
    for d in ("shards", "units", "logs"):
        (out / d).mkdir(parents=True, exist_ok=True)
    log = Log(out / "logs" / f"run-{args.node}.log")
    node_lock = open(out / f".run-{args.node}.lock", "a")
    try:
        fcntl.flock(node_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log(f"another {args.node} worker holds {out}/.run-{args.node}.lock")
        return 1
    if not args.state_host:
        (out / "state").mkdir(exist_ok=True)
    store = make_store(args)
    srcs = [Path(s) for s in args.src]
    units = plan_units(_source_index(srcs))
    alloc = load_alloc(args.alloc)
    build = build_settings(args, alloc)
    init_state(store, units)
    pin_state(store, "alloc.json", alloc)
    pin_state(store, "build.json", build)
    hess = Path(args.hessians) if args.hessian_mode == "real" and args.hessians else None
    if args.hessian_mode == "real" and hess is None:
        raise SystemExit("--hessian-mode real needs --hessians")
    gates = _gates(args)
    gpu_lock = _gpu_lock(args.gpu_lock, log)  # noqa: F841 (held until exit)
    log(f"worker {args.node} order={args.order} units={len(units)} mode={args.hessian_mode} "
        f"bpw={build['routed_bpw']:.4f} state={args.state_host or 'local'}:{out / 'state'}")

    only = parse_layer_list(args.layers) if args.layers else None
    tried: set[str] = set()  # --limit: partial units are encoded once per process, never recorded as done

    def can_run(x):
        info = units[x]
        if x in tried or not unit_selected(info, only, args.lmhead, args.limit):
            return False
        if find_src(srcs, info["file"]) is None:
            return False
        return info["kind"] != "experts" or hessian_ready(hess, info["layer"])

    failures: dict[str, int] = {}
    while True:
        if (out / "STOP").exists() or (out / f"STOP.{args.node}").exists():
            log("STOP file present; exiting after the last finished unit")
            break
        u, resumed = pick_unit(units, args.order, store, args.node, can_run=can_run)
        if u is None:
            if args.wait_hessians and any(f"{x}.json" not in store.ls("done") for x in units):
                time.sleep(60)
                continue
            log("no claimable unit left for this node")
            break
        info = units[u]
        log(f"unit {u} {info['file']} {info['kind']} {'resumed' if resumed else 'claimed'}")
        rec = reuse_local(out, u, info["file"])
        if rec is not None:
            log(f"unit {u} verified output already on this node, reused (sha256 {rec['sha256'][:16]})")
        else:
            hb = (lambda h, u=u: store.write(f"claims/{u}/heartbeat.json", json.dumps(h | {"node": args.node}) + "\n"))
            try:
                if info["kind"] == "lmhead":
                    rec = encode_lmhead_unit(u, info, srcs, out, log=log)
                else:
                    rec = encode_experts_unit(u, info, srcs, alloc, out, mode=args.hessian_mode, hessians=hess,
                                              gates=gates, batch=args.batch, tile_chunk=args.tile_chunk,
                                              limit=args.limit, heartbeat=hb, log=log)
            except Exception:
                failures[u] = failures.get(u, 0) + 1
                log(f"unit {u} FAILED ({failures[u]}x):\n{traceback.format_exc()}")
                if failures[u] >= 2:
                    log(f"unit {u} failed twice; stopping (claim kept for resume)")
                    return 2
                continue
        if args.limit:
            tried.add(u)
            log(f"unit {u} --limit {args.limit}: partial record units/{u}.partial.json, not recorded as done "
                f"(gate ok={rec['gate']['ok']})")
            continue
        rec.update(node=args.node, host=socket.gethostname(), recorded=utcnow())
        store.write(f"done/{u}.json", json.dumps(rec, indent=1) + "\n")
        man = refresh_manifest(store, units, alloc, build)
        log(f"unit {u} DONE sha256 {rec['sha256'][:16]} {rec['sec_per_tensor']:.3f} s/tensor; "
            f"{man['units_done']}/{man['units_total']} units done")
    man = refresh_manifest(store, units, alloc, build)
    log(f"exit: {man['units_done']}/{man['units_total']} done; DONE={'yes' if store.read('DONE') else 'no'}")
    return 0


def cmd_encode(args) -> int:
    srcs = [Path(s) for s in args.src]
    units = plan_units(_source_index(srcs))
    info = units[args.unit]
    if info["kind"] == "lmhead":
        rec = encode_lmhead_unit(args.unit, info, srcs, Path(args.out), log=Log(None))
    else:
        rec = encode_experts_unit(args.unit, info, srcs, load_alloc(args.alloc), Path(args.out),
                                  mode=args.hessian_mode, hessians=Path(args.hessians) if args.hessians else None,
                                  gates=_gates(args), batch=args.batch, tile_chunk=args.tile_chunk,
                                  limit=args.limit, log=Log(None))
    print(json.dumps({k: v for k, v in rec.items() if k != "rows"}, indent=1))
    return 0


def cmd_status(args) -> int:
    rep = status_report(make_store(args))
    print(json.dumps(rep, indent=1) if args.json else
          f"units {rep['units_done']}/{rep['units_total']} done {rep['done_by_node']} DONE={rep['DONE']}  "
          f"in flight {sorted(rep['inflight'])}  remaining {rep['remaining_tensors']} tensors  ETA {rep['eta_utc'] or 'n/a'}")
    return 0


def launch_argv(node: str, order: str, *, out: Path = OUT, alloc: str, hessians: str | None,
                hessian_mode: str = "real", hf_cache: str = "/home/sfxnz/.cache/huggingface",
                state_host: str | None = None, ssh_dir: str = "/home/sfxnz/.ssh", gpu_lock: str | None = None,
                image: str = IMAGE, memory: str = "32g") -> list[str]:
    """docker run for one worker (--rm, memory-capped). Code runs from the frozen copy in OUT/code."""
    o = str(out)
    argv = ["docker", "run", "-d", "--rm", "--name", f"{CONTAINER_PREFIX}-{node}", "--gpus", "all",
            "--network", "host" if state_host else "none", "--memory", memory, "--memory-swap", memory,
            "--user", "1000:1000", "-e", "HOME=/tmp", "-e", "PYTHONPATH=/usr/local/lib/python3.12/dist-packages",
            "-e", "PYTHONUNBUFFERED=1",
            "-v", f"{hf_cache}:/cache/huggingface:ro", "-v", f"{o}:{o}", "-v", f"{o}/code:/repo:ro",
            "-v", f"{alloc}:{alloc}:ro"]
    if hessians:
        argv += ["-v", f"{hessians}:{hessians}:ro"]
    if state_host:
        argv += ["-v", "/usr/bin/ssh:/usr/bin/ssh:ro", "-v", f"{ssh_dir}:{ssh_dir}:ro"]
    if gpu_lock:
        argv += ["-v", f"{gpu_lock}:{gpu_lock}"]
    argv += ["--entrypoint", "python3", image, "-S", "/repo/tools/build_pack_1017.py", "run",
             "--node", node, "--order", order, "--out", o, "--alloc", alloc, "--hessian-mode", hessian_mode]
    if hessians:
        argv += ["--hessians", hessians, "--wait-hessians"]
    if state_host:
        argv += ["--state-host", state_host, "--ssh-dir", ssh_dir]
    if gpu_lock:
        argv += ["--gpu-lock", gpu_lock]
    return argv


def cmd_launch_cmd(args) -> int:
    print(shlex.join(launch_argv(args.node, args.order, out=Path(args.out), alloc=args.alloc,
                                 hessians=args.hessians, hessian_mode=args.hessian_mode,
                                 state_host=args.state_host, gpu_lock=args.gpu_lock)))
    return 0


def link_file(src: Path, dst: Path, allow_copy: bool) -> str:
    """Hard link the real blob (not the Hub symlink) into the pack snapshot."""
    real = src.resolve()
    try:
        os.link(real, dst)
        return "link"
    except OSError as exc:
        if not allow_copy:
            raise SystemExit(f"cannot hard-link {real} -> {dst} ({exc}); --allow-copy copies instead") from exc
        shutil.copy2(real, dst)
        return "copy"


def cmd_assemble(args) -> int:
    from quantize_experts_exl3 import copy_sidecars

    out = Path(args.out)
    store = make_store(args)
    if store.read("DONE") is None:
        raise SystemExit("state/DONE missing: the build is not complete")
    units = json.loads(store.read("units.json"))
    alloc = json.loads(store.read("alloc.json"))
    build = json.loads(store.read("build.json"))
    man = refresh_manifest(store, units, alloc, build)
    if not man["complete"]:
        raise SystemExit(f"only {man['units_done']}/{man['units_total']} units have done records")
    done = load_done(store)
    shards = out / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    for u, rec in sorted(done.items()):
        f = shards / rec["file"]
        if not f.is_file():
            if rec["node"] == args.node:
                raise SystemExit(f"{f} missing on its own node")
            subprocess.run(["rsync", "-a", "--partial", f"{rec['node']}:{f}", str(f)], check=True)
        if f.stat().st_size != rec["size"] or sha256_file(f) != rec["sha256"]:
            raise SystemExit(f"{f}: size/sha256 differ from the done record")
        print(f"ok {rec['file']} ({rec['node']})", flush=True)
    src = Path(args.src[0])
    src_map = _source_index([src])
    snap = Path(args.hub) / PACK_DIR / "snapshots" / build["revision"]
    snap.mkdir(parents=True, exist_ok=True)
    unit_files = {rec["file"] for rec in done.values()}
    for f in sorted(set(src_map.values())):
        dst = snap / f
        if dst.exists():
            continue
        if f in unit_files:
            os.link(shards / f, dst)
        else:
            print(f"{link_file(src / f, dst, args.allow_copy)} {f} (1017 source)", flush=True)
    copy_sidecars(src, snap)
    cfg = json.loads((src / "config.json").read_text())
    quant = build_quantization_config(bits=2, codebook=CODEBOOK, expert_bits=alloc)
    (snap / "config.json").write_text(json.dumps(apply_pack_config(cfg, quant), indent=2) + "\n")
    idx = build_index(snap)
    if set(idx["weight_map"]) != expected_pack_names(src_map):
        miss = sorted(expected_pack_names(src_map) - set(idx["weight_map"]))
        extra = sorted(set(idx["weight_map"]) - expected_pack_names(src_map))
        raise SystemExit(f"pack tensors != source-derived names: missing {miss[:3]} extra {extra[:3]}")
    (snap / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2) + "\n")
    for rec in done.values():
        if (snap / rec["file"]).stat().st_ino != (shards / rec["file"]).stat().st_ino:
            raise SystemExit(f"{rec['file']} is not the verified unit output")
    manifest = man | {"rows": {u: done[u].get("rows", []) for u in sorted(done)}}
    (snap / "build-1017-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"assembled {snap}: {len(idx['weight_map'])} tensors, {len(unit_files)} built shards, "
          f"routed bpw {routed_bpw(alloc):.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    hub = default_hub()

    def common(p, *, src=False, alloc=False, enc=False) -> None:
        p.add_argument("--out", default=str(OUT))
        p.add_argument("--state-host", default=None, help="node holding OUT/state (default: this node)")
        p.add_argument("--ssh-dir", default="~/.ssh")
        if src:
            p.add_argument("--src", action="append", default=None, help="1017 source snapshot dir(s), first match wins")
        if alloc:
            p.add_argument("--alloc", required=True, help='allocation JSON {"<layer>": [K_gu, K_d]}')
        if enc:
            p.add_argument("--hessian-mode", choices=("real", "identity"), default="real")
            p.add_argument("--hessians", default=None, help="dsv41-hessian-v1 root (layer_LL/expert_EEE.safetensors)")
            p.add_argument("--batch", type=int, default=8, help="experts per quantize_exl3_batch call")
            p.add_argument("--tile-chunk", type=int, default=TILE_CHUNK)
            p.add_argument("--gates", default=None, help='JSON {"relerr_mean": {"3": 0.2}, ...} overriding GATES')
            p.add_argument("--revision", default=SERVE_REVISION)

    def plan_args(p) -> None:
        p.add_argument("--headers", default=None, help="dir of <shard>.json headers for shards not on disk yet")
        p.add_argument("--sec-per-tensor", type=float, default=1.92, help="ETA basis (1.92 = Sept H=I Viterbi)")
        p.add_argument("--json", action="store_true")

    p = sub.add_parser("plan")
    common(p, src=True, alloc=True, enc=True)
    plan_args(p)
    p = sub.add_parser("run")
    common(p, src=True, alloc=True, enc=True)
    plan_args(p)
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit (no GPU, no state)")
    p.add_argument("--node", required=False)
    p.add_argument("--order", choices=("asc", "desc"))
    p.add_argument("--gpu-lock", default=None)
    p.add_argument("--wait-hessians", action="store_true", help="poll while layers wait for their Hessians")
    p.add_argument("--layers", default=None, help="only the expert units of these layers, e.g. 1,3,5-9 (phase C)")
    p.add_argument("--lmhead", action="store_true", help="with --layers: also the lm_head unit")
    p.add_argument("--limit", type=int, default=None,
                   help="test: first N experts per unit, partial records only (nothing recorded as done)")
    p = sub.add_parser("encode")
    common(p, src=True, alloc=True, enc=True)
    p.add_argument("--unit", required=True)
    p.add_argument("--limit", type=int, default=None, help="first N experts only (nothing written to shards/)")
    p = sub.add_parser("status")
    common(p)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("launch-cmd")
    common(p, alloc=True, enc=True)
    p.add_argument("--node", required=True)
    p.add_argument("--order", choices=("asc", "desc"), required=True)
    p.add_argument("--gpu-lock", default=None)
    p = sub.add_parser("assemble")
    common(p, src=True)
    p.add_argument("--node", required=True)
    p.add_argument("--hub", default=str(hub))
    p.add_argument("--allow-copy", action="store_true", help="copy when a hard link is impossible")
    args = ap.parse_args(argv)
    if getattr(args, "src", "unset") is None:
        args.src = [str(hub / SOURCE_SNAPSHOT_REL)]
    if args.cmd == "run" and not args.dry_run and not (args.node and args.order):
        ap.error("run needs --node and --order (or --dry-run)")
    return {"plan": cmd_plan, "run": cmd_run, "encode": cmd_encode, "status": cmd_status,
            "launch-cmd": cmd_launch_cmd, "assemble": cmd_assemble}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
