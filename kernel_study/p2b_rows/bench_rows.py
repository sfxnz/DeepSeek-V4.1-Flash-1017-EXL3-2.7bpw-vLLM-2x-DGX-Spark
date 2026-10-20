#!/usr/bin/env python3
"""Wide-rows routed-MoE microbench (widen_p2b_rows, image e16-rows): the dataflow kernel at m = 9..24 rows
against the e15 fallback (vllm_exl3 exl3_moe / exl3_moe_kmix), per (K_gu, K_d), on real layer-20 experts
(kernel_study/p2b_kbits/encode_layer.py output under /data/tensors).

GPU only (spark2 via kernel_study/p2b_kbits/spark2.sh), in a serve image: JIT-builds build/bench_rows.cu (the
e16 p2b TU + make_bench_r3's runtime variant switch); the fallback and native arms call the installed plugin.

  python3 kernel_study/p2b_rows/bench_rows.py --pairs 2:2,2:3,3:3,3:4 --mode check,capture,stress,time \
      --out /data/runs/rows-<name>.json

Routing (source c4): m rows = S sequences x consecutive decode tokens of the layer's s10 census (S = 1 for
m <= 4, 2 for m <= 8, 3 for m <= 12, else 4: the c-session verify batch, e.g. m = 16 = 4 x (k = 3 + 1));
c4_all draws the same composition from every routed layer's census, layer-stratified (a step runs each layer
once). Activations and routing weights as kernel_study/p2b_coop/bench_r3.

Gates (fixed before any result):
  check    v2 = DSV41_P2B_COOP=2 (the wide kernel at m * 6 > 64): (a) batch invariance: v2 on m rows equals,
           bitwise, v2 run on the same rows in slices of 8 and of 4 (the m <= 8 kernels); (b) one-hot routing
           weights: bitwise v0 (SORT=0 p2b) at the same m; (c) real weights: max|v2 - v0| / max|v0| <= 1e-3
           (fp32 slot-sum order), repeat bitwise, finite. Informational: the fallback's max relative distance
           to v2, and with --ref the fp64-reference error of v2 and of the fallback.
  capture  CUDA graph of v2 per m and of the installed plugin's native path (arm native, e16 only):
           replays after new ids / weights bitwise equal to eager.
  stress   --stress-calls fresh routings per m: v2 eager == v2 in slices of 8 == v2 graph replay, bitwise
           (every 4th call after an L2 flush).
  time     every arm as a CUDA-graph replay (the serve replays graphs), routing copied in before each
           replay outside the timed window, a GPU sleep hiding the launch; cold (L2 flush) and warm;
           median / p10 / p90 us, GB/s over the unique experts' bytes, paired per-call savings against the
           fallback x 40 layers. Arms: p2b (v0), df (v2), fallback (plugin exl3_moe / exl3_moe_kmix, the
           e15 path above 8 rows), native (plugin _apply_native_fused_moe with the stock prep ops; e16).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kernel_study/p2b_kbits"))
sys.path.insert(0, str(HERE))

import bench_kbits as KB  # noqa: E402  (imports bench_r3, which pops DSV41_P2B_* and imports torch)
import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

R = KB.R
HIDDEN, INTER, EXPERTS, TOPK = KB.HIDDEN, KB.INTER, KB.EXPERTS, KB.TOPK
SWIGLU_LIMIT = KB.SWIGLU_LIMIT
log = R.log
ARMS = {"p2b": 0, "df": 2, "fallback": 10, "native": 11}


def build_ext():
    import make_bench_rows

    make_bench_rows.main()
    return load(
        name="p2b_bench_rows",
        sources=[str(make_bench_rows.OUT / "bench_rows.cu")],
        extra_include_paths=[R.EXL, os.path.join(R.EXL, "quant")],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


SEQS = 0  # --seqs: sequences per batch (0 = seqs_for(m))


def seqs_for(m: int) -> int:
    if SEQS:
        return min(SEQS, m)
    return 1 if m <= 4 else 2 if m <= 8 else 3 if m <= 12 else 4


class Routes(R.Routings):
    """bench_r3's census routings plus the c-session composition (S sequences x consecutive tokens)."""

    def _c4(self, caps, m: int) -> list[list[int]]:
        s = seqs_for(m)
        sizes = [m // s + (i < m % s) for i in range(s)]
        picks = self.rng.sample(caps, s) if s <= len(caps) else [self.rng.choice(caps) for _ in range(s)]
        rows: list[list[int]] = []
        for cap, n in zip(picks, sizes):
            rows += self._window(cap, n)
        return rows

    def draw(self, source: str, m: int) -> list[list[int]]:
        if source == "c4":
            return self._c4(self.caps, m)
        if source == "c4_all":
            layers = list(self.by_layer)
            self.last_layer = layers[self.next_layer % len(layers)]
            self.next_layer += 1
            return self._c4(self.by_layer[self.last_layer], m)
        return super().draw(source, m)


class Bench(KB.Bench):
    """Bench-module variants 0/1/2 plus the installed plugin's fallback (10) and native (11) paths."""

    def __init__(self, ext, layer, dev, seed, plugin) -> None:
        super().__init__(ext, layer, dev, seed)
        self.P = plugin
        self.lay = self.inners = None
        if plugin is not None:
            L = layer
            mark = torch.tensor(R.MCG_MARK, dtype=torch.int32, device=dev)
            mk = plugin.make_linear_exl3
            self.inners = [{"gate": mk(L.w13[e, 0], L.w13_suh[e, 0], L.w13_svh[e, 0], mark, None),
                            "up": mk(L.w13[e, 1], L.w13_suh[e, 1], L.w13_svh[e, 1], mark, None),
                            "down": mk(L.w2[e], L.w2_suh[e], L.w2_svh[e], mark, None)} for e in range(EXPERTS)]
            self.lay = SimpleNamespace(w13_trellis=L.w13, _exl3_hidden_size=HIDDEN, _exl3_intermediate_local=INTER,
                                       _exl3_bits=L.kgu, _exl3_bits_d=L.kd,
                                       _exl3_codebook_flags=(True, False, True, False, True, False), expert_map=None)
            plugin.build_exl3_fused_state(self.lay, self.inners)

    def plugin_call(self, v: int, x, ids, rw):
        P = self.P
        ids_l, w = ids.long(), rw.float()
        if v == 10:
            old = os.environ.get("VLLM_EXL3_MOE_KERNEL")
            os.environ["VLLM_EXL3_MOE_KERNEL"] = "exllamav3"  # the e15 path above 8 rows
            try:
                return P.apply_exl3_fused_moe(x, ids_l, w, self.lay, self.inners, None, SWIGLU_LIMIT)
            finally:
                if old is None:
                    os.environ.pop("VLLM_EXL3_MOE_KERNEL", None)
                else:
                    os.environ["VLLM_EXL3_MOE_KERNEL"] = old
        out = P._apply_native_fused_moe(x, ids_l, w, self.lay, self.inners, None, SWIGLU_LIMIT)
        if out is None:
            raise RuntimeError(f"plugin native path refused m={x.shape[0]}")
        return out

    def call(self, v: int, x, ids, rw, out):
        if v >= 10:
            out.copy_(self.plugin_call(v, x, ids, rw).to(out.dtype))
            return out
        return super().call(v, x, ids, rw, out)


def sliced(b: Bench, v: int, x, ids, rw, step: int):
    """v on the rows in slices of `step` (the <= 8-row calls the e15 kernels take)."""
    return torch.cat([b.once(v, x[i : i + step].contiguous(), ids[i : i + step].contiguous(), rw[i : i + step].contiguous())
                      for i in range(0, x.shape[0], step)])


def check(b: Bench, routes: Routes, ms, n: int, ref) -> dict:
    res = {}
    for m in ms:
        r = {"invariant8": True, "invariant4": True, "onehot_bitexact": True, "repeat_bitexact": True, "finite": True,
             "full_max_rel_vs_v0": 0.0, "fallback_max_rel_vs_v2": None, "ref_max_rel_v2": None, "ref_max_rel_fallback": None}
        for _ in range(n):
            rows = routes.draw("c4", m)
            x, ids, rw = b.x(m), b.ids(rows), b.rw(m)
            y = b.once(2, x, ids, rw)
            r["repeat_bitexact"] &= R.bitwise(y, b.once(2, x, ids, rw))
            r["invariant8"] &= R.bitwise(y, sliced(b, 2, x, ids, rw, 8))
            r["invariant4"] &= R.bitwise(y, sliced(b, 2, x, ids, rw, 4))
            base = b.once(0, x, ids, rw)
            r["finite"] &= bool(torch.isfinite(y).all()) and bool(torch.isfinite(base).all())
            d = (y.float() - base.float()).abs().max() / base.float().abs().max().clamp_min(1e-30)
            r["full_max_rel_vs_v0"] = max(r["full_max_rel_vs_v0"], float(d))
            for slot in range(TOPK):
                oh = torch.zeros_like(rw)
                oh[:, slot] = 1.0
                r["onehot_bitexact"] &= R.bitwise(b.once(2, x, ids, oh), b.once(0, x, ids, oh))
            if b.P is not None:
                fb = b.plugin_call(10, x, ids, rw).float()
                rel = float((fb - y.float()).abs().max() / y.float().abs().max().clamp_min(1e-30))
                r["fallback_max_rel_vs_v2"] = max(r["fallback_max_rel_vs_v2"] or 0.0, rel)
                if ref is not None:
                    exact = ref(x, rows, rw)
                    r["ref_max_rel_v2"] = max(r["ref_max_rel_v2"] or 0.0, R.ref_err(y, exact)[0])
                    r["ref_max_rel_fallback"] = max(r["ref_max_rel_fallback"] or 0.0, R.ref_err(fb, exact)[0])
        r["routings"] = n
        r["pass"] = bool(r["invariant8"] and r["invariant4"] and r["onehot_bitexact"] and r["repeat_bitexact"]
                         and r["finite"] and r["full_max_rel_vs_v0"] <= R.REL_TOL)
        res[f"m{m}"] = r
        log(f"[check] m={m}: " + json.dumps(r))
    return res


def capture_native(b: Bench, routes: Routes, ms, replays: int) -> dict:
    """The installed plugin's native path (e16: the wide kernel above 8 rows) in a CUDA graph."""
    res = {}
    for m in ms:
        x, ids, rw = b.x(m), b.ids(routes.draw("c4", m)), b.rw(m)
        ids_l, w = ids.long(), rw.float()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                b.P._apply_native_fused_moe(x, ids_l, w, b.lay, b.inners, None, SWIGLU_LIMIT)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gout = b.P._apply_native_fused_moe(x, ids_l, w, b.lay, b.inners, None, SWIGLU_LIMIT)
        ok = True
        for _ in range(replays):
            ids_l.copy_(b.ids(routes.draw("c4", m)).long())
            w.copy_(b.rw(m).float())
            g.replay()
            torch.cuda.synchronize()
            eager = b.P._apply_native_fused_moe(x, ids_l, w, b.lay, b.inners, None, SWIGLU_LIMIT)
            ok &= R.bitwise(gout.clone(), eager)
            # and the plugin's native output is the bench module's v2 on the same inputs (installed == bench)
            ok &= R.bitwise(eager.half(), b.once(2, x, ids_l.int(), w.half()))
        res[f"native/m{m}"] = {"replays": replays, "replay_equals_eager_and_bench": ok}
        log(f"[capture] native m={m}: replay_equals_eager_and_bench={ok} ({replays} replays)")
        del g
    return res


def stress(b: Bench, routes: Routes, ms, calls: int, flush) -> dict:
    res = {}
    for m in ms:
        x, rw = b.x(m), b.rw(m)
        ids = b.ids(routes.draw("c4", m))
        gout, out = torch.empty_like(x), torch.empty_like(x)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                b.call(2, x, ids, rw, gout)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        b.ext.set_variant(2)
        with torch.cuda.graph(g):
            b.raw(x, gout, ids, rw)
        eager_bad = slice_bad = graph_bad = 0
        for i in range(calls):
            ids.copy_(b.ids(routes.draw("c4", m)))
            if i % 4 == 0:
                flush()
            b.call(2, x, ids, rw, out)
            g.replay()
            torch.cuda.synchronize()
            ref = sliced(b, 2, x, ids, rw, 8)
            slice_bad += not R.bitwise(out, ref)
            graph_bad += not R.bitwise(gout, out)
            eager_bad += not R.bitwise(out, b.once(2, x, ids, rw))
        res[f"m{m}"] = {"calls": calls, "eager_mismatch": eager_bad, "slice_mismatch": slice_bad, "graph_mismatch": graph_bad}
        log(f"[stress] m={m}: {calls} calls, eager {eager_bad}, slice {slice_bad}, graph {graph_bad} mismatches")
        del g
    return res


def time_graphs(b: Bench, routes: Routes, arms, ms, source: str, iters: int, warmup: int, flush, sleep_cycles: int) -> dict:
    res = {}
    for m in ms:
        rows = [routes.draw(source, m) for _ in range(iters + warmup)]
        uniq = [R.n_unique(r) for r in rows]
        x, rw = b.x(m), b.rw(m)
        ids = b.ids(rows[0])
        ids_l, w = ids.long(), rw.float()
        out = torch.empty_like(x)
        graphs = {}
        for arm in arms:
            v = ARMS[arm]

            def fn(v=v):
                if v == 10:
                    return b.P.apply_exl3_fused_moe(x, ids_l, w, b.lay, b.inners, None, SWIGLU_LIMIT)
                if v == 11:
                    return b.P._apply_native_fused_moe(x, ids_l, w, b.lay, b.inners, None, SWIGLU_LIMIT)
                b.ext.set_variant(v)
                b.raw(x, out, ids, rw)
                return out

            if ARMS[arm] == 10:
                os.environ["VLLM_EXL3_MOE_KERNEL"] = "exllamav3"
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            os.environ.pop("VLLM_EXL3_MOE_KERNEL", None)
            graphs[arm] = g
        entry: dict = {"unique_mean": statistics.mean(uniq[warmup:]), "routing": source, "seqs": seqs_for(m)}
        for kind in ("cold", "warm"):
            t = {arm: [] for arm in arms}
            for i in range(iters + warmup):
                ids.copy_(b.ids(rows[i]))
                ids_l.copy_(ids)
                order = arms[i % len(arms):] + arms[: i % len(arms)]
                for arm in order:
                    if kind == "cold":
                        flush()
                    else:
                        graphs[arm].replay()
                    torch.cuda._sleep(sleep_cycles)  # the replay is enqueued behind it: launch hidden
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[arm].replay()
                    e1.record()
                    torch.cuda.synchronize()
                    if i >= warmup:
                        t[arm].append(e0.elapsed_time(e1) * 1000.0)
            nbytes = [u * R.EXPERT_BYTES for u in uniq[warmup:]]
            entry[kind] = {arm: R.stats(v, nbytes) for arm, v in t.items()}
            if "fallback" in arms:
                entry[kind]["paired_vs_fallback"] = {arm: R.paired(t["fallback"], t[arm]) for arm in arms if arm != "fallback"}
            log(f"[time] m={m} {source} {kind}: " + "  ".join(
                f"{arm} {entry[kind][arm]['median_us']:.1f}us [{entry[kind][arm]['p10_us']:.1f},{entry[kind][arm]['p90_us']:.1f}] "
                f"{entry[kind][arm]['gbps_median']:.0f}GB/s" for arm in arms)
                + (" | saving vs fallback " + "  ".join(
                    f"{a} {p['mean_us']:.1f}+-{p['ci95_us']:.1f}us = {p['ms_per_step_40']:.2f} ms/step"
                    for a, p in entry[kind]["paired_vs_fallback"].items()) if "fallback" in arms else ""))
        res[f"m{m}"] = entry
        del graphs
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="2:2,2:3,3:3,3:4", help="K_gu:K_d list (the 1017 2.70 allocation: 7/8/24/1 layers)")
    ap.add_argument("--mode", default="check,capture,stress,time")
    ap.add_argument("--ms", default="1,4,8,9,10,11,12,16,20,24")
    ap.add_argument("--time-ms", default="4,8,12,16,24")
    ap.add_argument("--arms", default="p2b,df,fallback,native")
    ap.add_argument("--source", default="c4_all", help="time routing: c4 (layer 20) or c4_all (every layer)")
    ap.add_argument("--seqs", type=int, default=0, help="sequences per batch for c4 / c4_all (0: 1/2/3/4 by m)")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--tp-rank", type=int, default=0)
    ap.add_argument("--check-routings", type=int, default=8)
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--replays", type=int, default=32)
    ap.add_argument("--stress-calls", type=int, default=500)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--sleep-cycles", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    global SEQS
    SEQS = args.seqs

    modes = set(args.mode.split(","))
    ms = [int(m) for m in args.ms.split(",")]
    time_ms = [int(m) for m in args.time_ms.split(",")]
    pairs = [tuple(int(k) for k in p.split(":")) for p in args.pairs.split(",")]
    dev = "cuda"
    props = torch.cuda.get_device_properties(0)
    t0 = time.time()
    ext = build_ext()
    log(f"extension ready in {time.time() - t0:.1f} s")
    try:
        import vllm_exl3.exl3 as P
        import vllm_exl3_c

        wide = int(getattr(vllm_exl3_c, "P2B_WIDE_ROWS", 0))
    except Exception as exc:  # noqa: BLE001
        P, wide = None, 0
        log(f"plugin not importable ({exc!r}): fallback / native arms off")
    # The installed vllm_exl3_c reads DSV41_P2B_COOP once, at its first call: the native arm is the serve's.
    os.environ["DSV41_P2B_COOP"] = "2"
    arms = [a for a in args.arms.split(",") if a in ARMS and (a in ("p2b", "df") or P is not None)]
    if "native" in arms and not wide:
        arms.remove("native")
        log("installed vllm_exl3_c has no P2B_WIDE_ROWS (e15): native arm off")
    flush = R.Flusher(dev)
    result: dict = {"device": {"name": props.name, "sms": props.multi_processor_count, "l2_bytes": props.L2_cache_size},
                    "args": vars(args) | {"out": str(args.out)}, "arms": arms, "installed_wide_rows": wide,
                    "tolerance": {"invariant": "bitwise", "onehot": "bitwise vs v0", "full_max_rel": R.REL_TOL,
                                  "capture": "bitwise", "stress": "bitwise"}, "pairs": {}}
    ok = True
    for kgu, kd in pairs:
        torch.manual_seed(args.seed)
        rng = random.Random(args.seed)
        t0 = time.time()
        layer = KB.Layer(args.layer, args.tp_rank, kgu, kd, dev)
        b = Bench(ext, layer, dev, args.seed, P)
        log(f"=== K_gu={kgu} K_d={kd}: loaded layer {args.layer} rank {args.tp_rank} in {time.time() - t0:.1f} s")
        routes = Routes(args.layer, rng)
        R.EXPERT_BYTES = KB.expert_bytes(kgu, kd)
        entry: dict = {"expert_bytes": R.EXPERT_BYTES}
        launch = {}
        for m in (8, 16):
            x = b.x(m)
            b.call(2, x, b.ids(routes.draw("c4", m)), b.rw(m), torch.empty_like(x))
            launch[f"v2/m{m}"] = {"grid": ext.last_grid(), "blocks_per_sm": ext.last_grid() / props.multi_processor_count}
        entry["launch"] = launch
        log(f"[setup] K_gu={kgu} K_d={kd} launch {json.dumps(launch)}")
        if "check" in modes:
            ref = KB.Reference(layer) if args.ref else None
            entry["check"] = check(b, routes, ms, args.check_routings, ref)
        if "capture" in modes:
            entry["capture"] = KB.capture(b, routes, [2], ms, args.replays)
            if "native" in arms:
                entry["capture"] |= capture_native(b, routes, ms, args.replays)
        if "stress" in modes:
            entry["stress"] = stress(b, routes, [m for m in ms if m > 8], args.stress_calls, flush)
        if "time" in modes:
            entry["time"] = time_graphs(b, routes, arms, time_ms, args.source, args.iters, args.warmup, flush,
                                        args.sleep_cycles)
        p_ok = all(r["pass"] for r in entry.get("check", {}).values()) and all(
            r.get("replay_equals_eager", r.get("replay_equals_eager_and_bench")) for r in entry.get("capture", {}).values()) and all(
            r["eager_mismatch"] == 0 and r["slice_mismatch"] == 0 and r["graph_mismatch"] == 0
            for r in entry.get("stress", {}).values())
        entry["pass"] = p_ok
        ok &= p_ok
        result["pairs"][f"{kgu}:{kd}"] = entry
        log(f"=== K_gu={kgu} K_d={kd}: pass={p_ok}")
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result | {"pass": ok, "partial": True}, indent=1) + "\n")
        del b, layer
        torch.cuda.empty_cache()
    result["pass"] = ok
    if "time" in modes:
        result["projection"] = project(result)
        log(f"[projection] {json.dumps(result['projection'])}")
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    log(f"pass={ok}")
    return 0 if ok else 1


def project(result: dict) -> dict:
    """ms/step per arm and m: the per-call cold medians weighted by the 2.70 allocation (layers per (K_gu, K_d))."""
    alloc = json.loads((ROOT / "tools/calib/alloc/expert_bits_2.70.json").read_text())
    counts: dict[str, int] = {}
    for pair in alloc.values():
        key = f"{pair[0]}:{pair[1]}"
        counts[key] = counts.get(key, 0) + 1
    out: dict = {"layers": counts}
    timed = {k: v["time"] for k, v in result["pairs"].items() if "time" in v}
    if set(counts) - set(timed):
        out["missing_pairs"] = sorted(set(counts) - set(timed))
        return out
    for mkey in next(iter(timed.values())):
        for arm in result["arms"]:
            out.setdefault(mkey, {})[arm] = sum(n * timed[p][mkey]["cold"][arm]["median_us"] for p, n in counts.items()) / 1000.0
    return out


if __name__ == "__main__":
    raise SystemExit(main())
