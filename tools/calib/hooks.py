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
