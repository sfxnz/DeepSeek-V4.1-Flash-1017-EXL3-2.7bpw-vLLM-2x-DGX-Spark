"""Measurement hooks for the native harness: routing statistics and per-expert Hessian capture."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from native import Hooks


class RoutingStats(Hooks):
    """Per (layer, label, expert): token picks, routed mass (normalized weight x route_scale), and output
    energy sum ||weighted expert output||^2; per (layer, expert) the largest single-token energy (to tell
    a real super-expert from one massive-activation token such as BOS). label = domain_id*2 + is_image.
    Also per-(layer, label) token counts. Saved as .npz."""

    def __init__(self, out_path):
        self.out_path = Path(out_path)

    def begin(self, hz, rows):
        c = hz.cfg
        self.domains = hz.domains
        nl = 2 * len(self.domains)
        self.E = c.n_routed_experts
        dev = hz.dev
        shape = (c.n_layers, nl, self.E)
        self.counts = torch.zeros(shape, dtype=torch.float64, device=dev)
        self.mass = torch.zeros(shape, dtype=torch.float64, device=dev)
        self.energy = torch.zeros(shape, dtype=torch.float64, device=dev)
        self.energy_max = torch.zeros(c.n_layers, self.E, dtype=torch.float64, device=dev)  # largest single token
        self.tokens = torch.zeros(c.n_layers, nl, dtype=torch.float64, device=dev)
        self.nl = nl

    def routing(self, layer, indices, weights, labels):
        k = indices.size(1)
        flat = (labels.unsqueeze(1).expand(-1, k) * self.E + indices).flatten()
        self.counts[layer].view(-1).index_add_(0, flat, torch.ones_like(flat, dtype=torch.float64))
        self.mass[layer].view(-1).index_add_(0, flat, weights.flatten().double())
        self.tokens[layer].index_add_(0, labels, torch.ones_like(labels, dtype=torch.float64))

    def wants_expert(self, layer):
        return True

    def expert(self, layer, e, tok, w, x, act, out, labels):
        en = out.float().square().sum(-1).double()
        self.energy[layer, :, e].index_add_(0, labels, en)
        self.energy_max[layer, e] = torch.maximum(self.energy_max[layer, e], en.max())

    def finish(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.out_path, counts=self.counts.cpu().numpy(), mass=self.mass.cpu().numpy(),
                            energy=self.energy.cpu().numpy(), energy_max=self.energy_max.cpu().numpy(),
                            tokens=self.tokens.cpu().numpy(),
                            domains=np.array(self.domains), label_doc=np.array("label = domain_id*2 + is_image"))


class HessianCapture(Hooks):
    """Per-expert Hessian inputs from the native stream, handed to a sink as each expert finishes.

    Definitions for expert e of layer L, routed tokens t with routing weight w_t (incl. route_scale),
    x_t = MoE input (bf16, BEFORE the fp8 act quant, i.e. what an EXL3 expert sees in serving),
    a_t = silu(clamp(x W1^T)) * clamp(x W3^T) (unweighted, fp32 before rounding):
        H_gu = sum_t w_t^p x_t x_t^T   [dim, dim]      shared by w1 and w3
        H_d  = sum_t w_t^p a_t a_t^T   [inter, inter]  input of w2
    Priors for cold/rare experts: H_gu_all = sum over ALL native tokens of x x^T (per layer), and
    per-expert prior activations a on P fixed random native tokens (expert applied unrouted).

    fmt="x" (default) stores the factors, which is smaller than H whenever an expert has < ~10K tokens:
        e{E}.safetensors: x [n, dim] bf16, act [n, inter] bf16, w [n] fp32, prior_act [P, inter] bf16
        -> consumer: H_gu = (x * w^p).T @ x,  H_d = (act * w^p).T @ act,  H_d_prior = prior_act.T @ prior_act
    fmt="h" stores H_gu, H_d, H_d_prior fp32 directly (~142 MB per expert, ~55 GB per layer).
    Every layer also gets prior_gu.safetensors (H_gu_all fp32, n_all) and done.json (written last).
    Sink protocol: sink(layer, name, tensors: dict[str, cpu Tensor] | None, meta: dict); name None = layer done."""

    def __init__(self, layers, sink, wpow: float = 1.0, prior_tokens: int = 512, seed: int = 0, fmt: str = "x"):
        assert fmt in ("x", "h")
        self.layers = set(layers)
        self.sink = sink
        self.wpow = wpow
        self.P = prior_tokens
        self.seed = seed
        self.fmt = fmt
        self.prior = {}
        self.prior_out = {}

    def begin(self, hz, rows):
        self.E = hz.cfg.n_routed_experts

    def wants_expert(self, layer):
        return layer in self.layers

    def layer_tokens(self, layer, x, labels):
        H = torch.zeros(x.size(1), x.size(1), dtype=torch.float32, device=x.device)
        for a in range(0, x.size(0), 16384):
            xa = x[a : a + 16384].float()
            H.addmm_(xa.t(), xa)
        self.sink(layer, "prior_gu", {"H_gu_all": H.cpu()}, {"n_all": int(x.size(0))})
        g = torch.Generator(device="cpu").manual_seed(self.seed * 1000 + layer)
        P = min(self.P, x.size(0))
        self.prior[layer] = torch.randperm(x.size(0), generator=g)[:P].sort().values.to(x.device)
        self.done_experts = set()

    def prior_tokens(self, layer):
        return self.prior.get(layer)

    def expert_prior(self, layer, e, act):
        if self.fmt == "x":
            self.prior_out[e] = {"prior_act": act.to(torch.bfloat16).cpu()}
        else:
            a = act.float()
            self.prior_out[e] = {"H_d_prior": (a.t() @ a).cpu()}

    def expert(self, layer, e, tok, w, x, act, out, labels):
        if self.fmt == "x":
            t = {"x": x.to(torch.bfloat16).cpu(), "act": act.to(torch.bfloat16).cpu(), "w": w.float().cpu()}
        else:
            ww = w.float().pow(self.wpow).unsqueeze(-1)
            xf, a = x.float(), act.float()
            t = {"H_gu": ((xf * ww).t() @ xf).cpu(), "H_d": ((a * ww).t() @ a).cpu()}
        t.update(self.prior_out.pop(e, {}))
        self.sink(layer, f"e{e:03d}", t, {"n": int(tok.numel()), "wsum": float(w.double().sum()),
                                           "w2sum": float(w.double().square().sum()), "wpow": self.wpow, "fmt": self.fmt})
        self.done_experts.add(e)

    def end_layer(self, layer):
        if layer not in self.layers:
            return
        for e in range(self.E):  # experts with no routed token in this panel: prior only
            if e not in self.done_experts:
                self.sink(layer, f"e{e:03d}", self.prior_out.pop(e, {}),
                          {"n": 0, "wsum": 0.0, "w2sum": 0.0, "wpow": self.wpow, "fmt": self.fmt})
        self.prior_out.clear()
        self.sink(layer, None, None, {"done": True, "prior_tokens": int(self.prior[layer].numel()), "fmt": self.fmt})

    def finish(self):
        if hasattr(self.sink, "close"):
            self.sink.close()


class DiskSink:
    """Writes <out>/L{layer:02d}/{name}.safetensors (meta in the file metadata) with a few writer threads;
    done.json is written after all of that layer's files. max_pending_gb throttles the producer until a
    consumer has deleted enough (it may delete whole L* dirs after reading done.json)."""

    def __init__(self, out_dir, max_pending_gb: float | None = None, threads: int = 4):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.max_pending = max_pending_gb
        self.pool = ThreadPoolExecutor(max_workers=threads)
        self.futs: dict = {}
        self.meta: dict = {}

    def _backlog_gb(self) -> float:
        tot = 0
        for root, _, files in os.walk(self.out):
            for f in files:
                try:
                    tot += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return tot / 2**30

    def _write(self, d: Path, name: str, tensors: dict, meta: dict):
        from safetensors.torch import save_file
        tmp = d / f".{name}.tmp"
        save_file({k: v.contiguous() for k, v in tensors.items()} or {"empty": torch.zeros(0)}, str(tmp),
                  metadata={k: json.dumps(v) for k, v in meta.items()})
        tmp.rename(d / f"{name}.safetensors")

    def __call__(self, layer, name, tensors, meta):
        d = self.out / f"L{layer:02d}"
        d.mkdir(exist_ok=True)
        futs = self.futs.setdefault(layer, [])
        if name is None:
            for f in futs:
                f.result()
            (d / "done.json").write_text(json.dumps({**meta, "experts": self.meta.pop(layer, {})}))
            self.futs.pop(layer)
            return
        if self.max_pending is not None:
            while self._backlog_gb() > self.max_pending:
                time.sleep(10)
        while sum(not f.done() for f in futs) > 16:  # bound queued host memory
            time.sleep(0.05)
        self.meta.setdefault(layer, {})[name] = meta
        futs.append(self.pool.submit(self._write, d, name, tensors, meta))

    def close(self):
        for fl in self.futs.values():
            for f in fl:
                f.result()
        self.pool.shutdown()


# ------------------------------------------------------------------------- dsv41-hessian-v1 (decision B-3)

HESSIAN_FORMAT = "dsv41-hessian-v1"


def drop_page_cache(path) -> None:
    """fdatasync + POSIX_FADV_DONTNEED (the harness' page-cache mechanism, stio._drop_cache): the GB10 does
    not reclaim page cache for CUDA allocations, and a layer of Hessians is ~48 GB of writes."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fdatasync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def save_safetensors(path: Path, tensors: dict, meta: dict | None = None) -> None:
    """Atomic (tmp + rename) safetensors write, then flushed out of the page cache. The metadata goes in
    ONE key "meta" (sorted JSON): safetensors serializes its metadata map in a per-process random order,
    so several keys would make byte-identical tensors give different files."""
    from safetensors.torch import save_file
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(tmp),
              metadata={"meta": json.dumps(meta or {}, sort_keys=True)})
    drop_page_cache(tmp)
    os.replace(tmp, path)


def read_meta(fh) -> dict:
    """Metadata of a file written by save_safetensors (or the older one-key-per-field layout)."""
    md = fh.metadata() or {}
    if "meta" in md:
        return json.loads(md["meta"])
    return {k: json.loads(v) for k, v in md.items()}


def _gram(x: torch.Tensor, g: torch.Tensor | None = None, mask: torch.Tensor | None = None,
          chunk: int = 8192) -> torch.Tensor:
    """sum_t g_t x_t x_t^T over rows with mask_t (all if None) in fp64: fp32 GEMM per chunk of tokens,
    fp64 accumulation across chunks."""
    d = x.size(1)
    acc = torch.zeros(d, d, dtype=torch.float64, device=x.device)
    for a in range(0, x.size(0), chunk):
        xa = x[a : a + chunk]
        ga = None if g is None else g[a : a + chunk]
        if mask is not None:
            m = mask[a : a + chunk]
            xa = xa[m]
            ga = None if ga is None else ga[m]
        xa = xa.float()
        xw = xa if ga is None else xa * ga.float().unsqueeze(-1)
        acc += (xw.t() @ xa).double()
    return acc


class PinnedPool:
    """Reusable pinned host buffers per shape. A plain .cpu() of a 105 MB tensor measured 2 ms to 330 ms on
    the GB10 depending on the allocation (hess/logs/bench_sym.log, prof_hook.log); a copy into pinned memory
    is a plain DMA. take() blocks until a writer gave a buffer back."""

    def __init__(self, per_shape: int):
        import threading
        self.n = per_shape
        self.free: dict = {}
        self.cv = threading.Condition()

    def take(self, shape, dtype=torch.float32) -> torch.Tensor:
        key = (tuple(shape), dtype)
        with self.cv:
            if key not in self.free:
                self.free[key] = [torch.empty(shape, dtype=dtype, pin_memory=True) for _ in range(self.n)]
            while not self.free[key]:
                self.cv.wait()
            return self.free[key].pop()

    def give(self, t: torch.Tensor):
        with self.cv:
            self.free[(tuple(t.shape), t.dtype)].append(t)
            self.cv.notify_all()


def _finish_h(h: torch.Tensor, pool: PinnedPool | None = None) -> torch.Tensor:
    """fp64 -> exactly symmetric fp32 on the CPU (rounded to fp32 first, then (a + a^T) / 2: fp32
    a+b == b+a keeps it exactly symmetric), into a pinned buffer from `pool` when given."""
    h = h.float()
    h = (h + h.t()) * 0.5
    if pool is None or not h.is_cuda:
        return h.cpu().contiguous()
    buf = pool.take(h.shape)
    buf.copy_(h)
    return buf


class _Acc:
    """Per-layer accumulators of one token set (all tokens, or a train fold)."""

    def __init__(self, mask):
        self.mask = mask           # bool [T] over native tokens, None = all
        self.H_all = None          # fp64 mean x x^T over the set's tokens
        self.n_all = 0
        self.m0 = None
        self.mean_wp = None
        self.prior_H = {}          # e -> fp64 mean a a^T over the prior tokens (pending experts)


class HessianV1(Hooks):
    """Per-expert Hessians in the builder's dsv41-hessian-v1 format, decision B-3 exactly.

    For expert e of layer L, routed tokens t with routing weight w_t (incl. route_scale), p = wpow (2):
        S_e = sum_t w_t^p x_t x_t^T,   W_e = sum_t w_t^p,   x = MoE input (bf16, before the fp8 act quant)
        H_all = (1/T) sum_{all native tokens} x x^T,   m0 = m0_mult (256) * mean(w^p) over all routed picks
        w13_in = (S_e + m0 H_all) / (W_e + m0)
        w2_in  = (S^act_e + m0 P_e) / (W_e + m0),  S^act_e = sum_t w_t^p a_t a_t^T, a = silu(gate) * up of the
                 native expert (unweighted), P_e = mean a a^T of expert e applied unrouted to `prior_tokens`
                 (512) random native tokens of the layer (seed*1000 + layer)
    An expert without routed tokens gets w13_in = H_all, w2_in = P_e. Written per expert as soon as it
    finishes: <out>/layer_LL/expert_EEE.safetensors {w13_in F32 [dim,dim], w2_in F32 [inter,inter]}
    (metadata n, wsum, wpsum), then <out>/layer_LL/meta.json {"format", "layer", "complete": true, ...}
    last. Sums in fp64 (fp32 GEMM per 8192-token chunk), stored fp32, exactly symmetric.

    experts: None = all; a set of ids; or "strat16" = 16 experts chosen per layer from this layer's routed
    mass (ranks round(i*(E-1)/15), i = 0..15, mass-descending; tercile label hot/mid/cold in meta).
    dump_factors > 0: also <factors>/layer_LL/expert_EEE.safetensors with up to that many routed tokens
    (seeded uniform subsample): x bf16, w fp32, row int32 (calib row index), tok int32.
    holdout_mod > 0: a second Hessian set under train_out from the train fold only (rows with
    index % holdout_mod != 0: S_e, W_e, H_all, m0 and the prior tokens); the prior tokens are then drawn
    from the train fold for both sets."""

    def __init__(self, layers, out, experts=None, wpow: float = 2.0, m0_mult: float = 256.0,
                 prior_tokens: int = 512, seed: int = 0, dump_factors: int = 0, factors_out=None,
                 holdout_mod: int = 0, train_out=None, threads: int = 3, max_pending: int = 6, info=None, log=print):
        self.layers = set(layers)
        self.out = Path(out)
        self.experts = experts
        self.p = float(wpow)
        self.m0_mult = float(m0_mult)
        self.P = prior_tokens
        self.seed = seed
        self.dump = dump_factors
        self.factors_out = Path(factors_out) if factors_out else None
        self.holdout = holdout_mod
        self.train_out = Path(train_out) if train_out else None
        assert not holdout_mod or self.train_out is not None
        assert not dump_factors or self.factors_out is not None
        self.pool = ThreadPoolExecutor(max_workers=threads)
        self.max_pending = max_pending
        self.pinned = PinnedPool(max_pending + threads + 1)  # ~1 GB pinned: > every file in flight
        self.futs: list = []
        self.info = info or {}
        self.log = log
        self.layer_meta: dict = {}
        self.t_hess = 0.0

    # -------------------------------------------------------------- setup
    def begin(self, hz, rows):
        self.E = hz.cfg.n_routed_experts
        self.dev = hz.dev
        nat = [r for r in rows if r.variant is None]
        self.row_of_tok = torch.cat([torch.full((r.n,), i, dtype=torch.int32) for i, r in enumerate(nat)]).to(self.dev)
        self.T = int(self.row_of_tok.numel())
        self.train_mask = (self.row_of_tok % self.holdout != 0) if self.holdout else None

    def wants_expert(self, layer):
        return layer in self.layers

    def _selected(self, e):
        return self.sel is None or e in self.sel

    # -------------------------------------------------------------- per layer
    def routing(self, layer, indices, weights, labels):
        if layer not in self.layers:
            return
        t0 = time.time()
        self.cur = layer
        wp = weights.double().pow(self.p)
        self.accs = {"full": _Acc(None)}
        if self.holdout:
            self.accs["train"] = _Acc(self.train_mask)
        for a in self.accs.values():
            sub = wp if a.mask is None else wp[a.mask]
            a.mean_wp = float(sub.mean())
            a.m0 = self.m0_mult * a.mean_wp
        flat = indices.flatten()
        self.counts = torch.bincount(flat, minlength=self.E).tolist()
        mass = torch.zeros(self.E, dtype=torch.float64, device=self.dev).index_add_(0, flat, weights.flatten().double())
        if self.experts == "strat16":
            order = torch.argsort(mass, descending=True).tolist()
            ranks = [round(i * (self.E - 1) / 15) for i in range(16)]
            self.sel = {order[r]: {"rank": r, "tier": "hot" if r < self.E / 3 else ("mid" if r < 2 * self.E / 3 else "cold")}
                        for r in ranks}
        elif self.experts is None:
            self.sel = None
        else:
            self.sel = {int(e): {} for e in self.experts}
        self.mass = mass.tolist()
        self.emitted = set()
        self.t_hess += time.time() - t0
        self.bytes = 0
        for d in self._layer_dirs(layer):
            d.mkdir(parents=True, exist_ok=True)
            (d / "meta.json").unlink(missing_ok=True)  # a re-run of a layer is incomplete until it finishes

    def _layer_dirs(self, layer):
        ds = [self.out / f"layer_{layer:02d}"]
        if self.holdout:
            ds.append(self.train_out / f"layer_{layer:02d}")
        if self.dump:
            ds.append(self.factors_out / f"layer_{layer:02d}")
        return ds

    def layer_tokens(self, layer, x, labels):
        t0 = time.time()
        for a in self.accs.values():
            a.n_all = int(x.size(0)) if a.mask is None else int(a.mask.sum())
            a.H_all = _gram(x, mask=a.mask) / a.n_all
        self.t_hess += time.time() - t0

    def prior_tokens(self, layer):
        g = torch.Generator(device="cpu").manual_seed(self.seed * 1000 + layer)
        pool = torch.arange(self.T, device=self.dev)
        if self.holdout:
            pool = pool[self.train_mask]
        P = min(self.P, int(pool.numel()))
        idx = torch.randperm(int(pool.numel()), generator=g)[:P].sort().values.to(self.dev)
        self.prior_idx = pool[idx]
        return self.prior_idx

    def expert_prior(self, layer, e, act):
        if not self._selected(e):
            return
        t0 = time.time()
        for k in list(self.pending_zero()):
            self._emit(layer, k, None)
        ph = _gram(act) / act.size(0)
        for a in self.accs.values():
            a.prior_H[e] = ph  # prior tokens are shared by both sets (drawn from the train fold if holdout)
        self.t_hess += time.time() - t0

    def pending_zero(self):
        """Experts whose prior is in but that had no routed token (the harness skips expert() for them)."""
        a = self.accs["full"]
        return [k for k in a.prior_H if k not in self.emitted and self.counts[k] == 0]

    def expert(self, layer, e, tok, w, x, act, out, labels):
        if not self._selected(e):
            return
        self._emit(layer, e, (tok, w, x, act))

    def _emit(self, layer, e, data):
        t0 = time.time()
        self.emitted.add(e)
        for name, a in self.accs.items():
            if data is None:
                n, S_x, S_a, W, wsum = 0, None, None, 0.0, 0.0
            else:
                tok, w, x, act = data
                if a.mask is not None:
                    m = a.mask[tok]
                    tok, w, x, act = tok[m], w[m], x[m], act[m]
                n = int(tok.numel())
                g = w.double().pow(self.p)
                W, wsum = float(g.sum()), float(w.double().sum())
                S_x = _gram(x, g) if n else None
                S_a = _gram(act, g) if n else None
            Pe = a.prior_H.pop(e)
            den = W + a.m0
            h13 = (a.m0 * a.H_all if S_x is None else S_x + a.m0 * a.H_all) / den
            h2 = (a.m0 * Pe if S_a is None else S_a + a.m0 * Pe) / den
            root = self.out if name == "full" else self.train_out
            meta = {"n": n, "wsum": wsum, "wpsum": W, "m0": a.m0, "prior_weight": a.m0 / den}
            self._submit(root / f"layer_{layer:02d}" / f"expert_{e:03d}.safetensors",
                         {"w13_in": _finish_h(h13, self.pinned), "w2_in": _finish_h(h2, self.pinned)}, meta,
                         pinned=True)
            if name == "full":
                self.layer_meta.setdefault(layer, {})[e] = meta
        if self.dump and data is not None:
            tok, w, x, act = data
            n = int(tok.numel())
            sel = torch.arange(n, device=self.dev)
            if n > self.dump:
                g = torch.Generator(device="cpu").manual_seed(self.seed * 1_000_003 + layer * 1009 + e)
                sel = torch.randperm(n, generator=g)[: self.dump].sort().values.to(self.dev)
            self._submit(self.factors_out / f"layer_{layer:02d}" / f"expert_{e:03d}.safetensors",
                         {"x": x[sel].to(torch.bfloat16).cpu(), "w": w[sel].float().cpu(),
                          "row": self.row_of_tok[tok[sel]].cpu(), "tok": tok[sel].int().cpu()},
                         {"n_routed": n, "n_kept": int(sel.numel())})
        self.t_hess += time.time() - t0

    def _write(self, path, tensors, meta, pinned):
        try:
            save_safetensors(path, tensors, meta)
        finally:
            if pinned:
                for t in tensors.values():
                    if t.is_pinned():  # CPU-only runs (tests) never take from the pool
                        self.pinned.give(t)

    def _submit(self, path, tensors, meta, pinned=False):
        self.bytes += sum(v.numel() * v.element_size() for v in tensors.values())
        while sum(not f.done() for f in self.futs) >= self.max_pending:
            time.sleep(0.02)
        for f in self.futs:
            if f.done():
                f.result()  # re-raise a writer error here
        self.futs = [f for f in self.futs if not f.done()]
        self.futs.append(self.pool.submit(self._write, path, tensors, meta, pinned))

    def end_layer(self, layer):
        if layer not in self.layers:
            return
        t0 = time.time()
        for e in sorted(self.accs["full"].prior_H):
            if e not in self.emitted:
                self._emit(layer, e, None)
        for f in self.futs:
            f.result()
        self.futs = []
        written = sorted(self.layer_meta.get(layer, {}))
        want = sorted(range(self.E)) if self.sel is None else sorted(self.sel)
        if written != want:
            raise RuntimeError(f"layer {layer}: wrote {len(written)} experts, expected {len(want)}")
        a = self.accs["full"]
        meta = {"format": HESSIAN_FORMAT, "layer": layer, "complete": True, "wpow": self.p, "m0_mult": self.m0_mult,
                "m0": a.m0, "mean_wp": a.mean_wp, "prior_tokens": int(self.prior_idx.numel()), "seed": self.seed,
                "tokens": self.T, "n_all": a.n_all, "experts_written": len(written),
                "subset": self.sel is not None, "experts": {str(e): self.layer_meta[layer][e] for e in written},
                "routed_counts": self.counts, "routed_mass": self.mass,
                "definition": "w13_in=(S_e+m0*H_all)/(W_e+m0), w2_in=(S^act_e+m0*P_e)/(W_e+m0), p=wpow (decision B-3)",
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **self.info}
        if self.sel is not None:
            meta["selection"] = {str(e): v for e, v in sorted(self.sel.items())}
        dirs = self._layer_dirs(layer)
        for d in dirs[1:]:
            sub = dict(meta)
            if self.holdout and d.parent == self.train_out:
                t = self.accs["train"]
                sub.update(fold=f"train: row % {self.holdout} != 0", m0=t.m0, mean_wp=t.mean_wp, n_all=t.n_all)
            (d / "meta.json").write_text(json.dumps(sub, indent=1) + "\n")
        (dirs[0] / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")  # last: the builder's ready flag
        self.t_hess += time.time() - t0
        self.log(f"[hessian L{layer:02d}] {len(written)} experts, {self.bytes / 1e9:.1f} GB, hook {self.t_hess:.1f}s")
        self.layer_meta.pop(layer, None)
        self.accs = None

    def finish(self):
        for f in self.futs:
            f.result()
        self.pool.shutdown()
