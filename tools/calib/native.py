"""Layer-streaming NATIVE forward of DeepSeek-V4.1-Flash-1017 on one GPU.

Faithful re-statement of the official `inference/model.py` (prefill path, start_pos=0, world_size=1):
MXFP4 routed experts dequantized exactly per expert on the fly, MXFP8 dense linears with the reference
`act_quant` (UE8M0 block-32) on their inputs, fp8/fp4 KV quant-dequant, two-level indexer with candidate
blocks, Hyper-Connections with Sinkhorn, Engram (rows gathered from disk), vision tower + aligner with
`bias_vl` routing for image-span tokens and Engram masking. TileLang kernels are replaced by
`refkernel` (torch) equivalents.

Execution model: every sequence ("row") keeps its hyper-connection stream h [s, 4, dim] bf16 and its
pre_mix on the GPU. Layers run one at a time over ALL rows: attention per row, then one MoE over all
rows' tokens (each expert dequantized once per layer and applied to every token routed to it). Only one
layer's weights are resident (MXFP4 packed); the next layer is prefetched to CPU in a thread.

Fan-out variants: a `Variant` row is a clone of a native row spawned at its first modified layer. At
modified layers its routed experts use perturbed weights or EXL3-reconstructed weights from a pack
("swap"). Native and variant streams share every weight load and expert dequant, but each stream runs
its own GEMMs with identical shapes, so an unperturbed variant is bit-identical to native (verified:
r=0 -> KL exactly 0). KL(native || variant) of the final logits is computed in-stream at the head,
exactly over the full vocab, per domain.

Noise formulation (sensitivity): for every routed expert of the chosen (layer, group) the weights are
replaced by W + r * rms(W) * G, G iid N(0,1) (per tensor, seeded per variant/layer/expert), for
group GU on w1 and w3, for group D on w2. This is weight-space noise of relative Frobenius error r, the
error model of EXL3: after the randomized Hadamard rotation the trellis quantization error is close to
iid per tensor, and its output error scales with ||x|| (not with ||Wx||) as real quantization error does.
Hessian-aware encodes shape the error away from the activation subspace, so map K -> r by matching
the measured `routed_out_rel_mse` of real encodes (swap variants report the same metric) rather than
by weight relerr. Measured caveat: the native model has a chaotic KL floor. fp8 act-quant roundings and
near-tie top-6 routing flips make any perturbation in early layers give KL ~0.005-0.01 on a 2K-token
panel, even at r = 0.001, and the official model vs this harness differ by KL 0.016 for the same reason.
So fit KL(r) = floor_l + s_l * r^2 with r spanning the real K range (and a tiny-r point for floor_l),
on panels large enough to average the floor.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refkernel as rk  # noqa: E402
from stio import Checkpoint, EngramTable, Exl3Pack, Prefetcher  # noqa: E402

TEXT = -1
GU, D = "GU", "D"


# ----------------------------------------------------------------------------------------- config

@dataclass
class Cfg:
    vocab_size: int = 129280
    dim: int = 5120
    moe_inter_dim: int = 2304
    n_layers: int = 40
    n_heads: int = 64
    n_routed_experts: int = 384
    n_activated_experts: int = 6
    route_scale: float = 1.5
    gate_temp: float = 1.0
    swiglu_limit: float = 10.0
    q_lora_rank: int = 1280
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: tuple = ()
    kv_source_layers: tuple = ()
    index_source_layers: tuple = ()
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer: int = -1
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    engram_layer_ids: tuple = ()
    dspark_target_layer_ids: tuple = ()
    image_token_id: int = 129264
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path, **overrides) -> "Cfg":
        raw = json.loads(Path(path).read_text())
        raw.update(overrides)
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**kw, raw=raw)


# ------------------------------------------------------------------------------------ primitives

def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (w * x).to(dtype)


def lin8(x: torch.Tensor, wd: torch.Tensor) -> torch.Tensor:
    """Reference `linear()` for an fp8 weight: UE8M0 block-32 fp8 act quant, bf16 out. wd = dequantized bf16."""
    return torch.matmul(rk.act_quant_dequant(x, 32), wd.t())


@torch.no_grad()
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_(x: torch.Tensor, freqs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """In-place rotary on x [s, d] or [s, h, d] (adjacent pairs as complex), like model.apply_rotary_emb."""
    y = x
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs = freqs.conj()
    f = freqs.view(xc.size(0), xc.size(-1)) if xc.ndim == 2 else freqs.view(xc.size(0), 1, xc.size(-1))
    y.copy_(torch.view_as_real(xc * f).flatten(-2))
    return y


def window_idxs(s: int, win: int, device) -> torch.Tensor:
    end = torch.arange(s, device=device).unsqueeze(1)
    idxs = (end - win + 1).clamp(0) + torch.arange(min(s, win), device=device)
    return torch.where(idxs > end, -1, idxs).int()


def select_candidate_blocks(logits, compress_lens, topk_blocks: int, block_size: int) -> torch.Tensor:
    """Reference select_candidate_blocks, returning the BLOCK-level keep mask [..., n_blocks]."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)


def hc_mixes(x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, cfg: Cfg):
    xf = x.flatten(-2).float()
    rsq = torch.rsqrt(xf.square().mean(-1, keepdim=True) + cfg.norm_eps)
    mixes = F.linear(xf, fn) * rsq
    return rk.hc_split_sinkhorn(mixes, scale, base, cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps)


def hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    return torch.sum(pre.unsqueeze(-1) * x.float(), dim=-2).to(x.dtype)


def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3)
    return y.type_as(x)


def chunked(n: int, c: int):
    for a in range(0, n, c):
        yield a, min(n, a + c)


# ---------------------------------------------------------------------------------------- rows

@dataclass
class Variant:
    """A perturbed copy of every native row. mods: {layer: ("noise", group, r) | ("swap",)}."""
    name: str
    mods: dict
    seed: int = 0

    @property
    def start_layer(self) -> int:
        return min(self.mods)


@dataclass
class Row:
    id: str
    tokens: torch.Tensor                     # int64 [s] (cpu)
    domain: str = "default"
    token_types: torch.Tensor | None = None  # int64 [s], TEXT=-1 else image span type
    images: list | None = None               # image_processor.ImageInput list
    score_start: int = 1                     # first target position scored (NLL/KL/dumps)
    score_mask: torch.Tensor | None = None   # bool [s]: if set, score exactly the targets t with mask[t]
    variant: Variant | None = None
    twin: int = -1                           # index of the native row a variant copies
    # streamed state
    h: torch.Tensor | None = None
    pre_mix: torch.Tensor | None = None
    shared: dict = field(default_factory=dict)
    hashes: torch.Tensor | None = None
    main_hidden: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.tokens.numel())

    def image_mask(self, device):
        return None if self.token_types is None else (self.token_types >= 0).to(device)


def spawn(row: Row, variant: Variant, twin_idx: int) -> Row:
    r = Row(id=f"{row.id}@{variant.name}", tokens=row.tokens, domain=row.domain, token_types=row.token_types,
            images=None, score_start=row.score_start, score_mask=row.score_mask, variant=variant, twin=twin_idx)
    r.h = row.h.clone()
    r.pre_mix = row.pre_mix.clone()
    r.shared = dict(row.shared)  # tensors are replaced, never mutated in place, so sharing is safe
    r.hashes = row.hashes
    return r


# ----------------------------------------------------------------------------------------- hooks

class Hooks:
    """Override what you need. All tensors live on the GPU; token indices are into the layer's flat
    concatenation of NATIVE rows' tokens (variant rows never reach the stats hooks)."""

    def begin(self, harness: "Harness", rows: list[Row]):
        pass

    def routing(self, layer: int, indices: torch.Tensor, weights: torch.Tensor, labels: torch.Tensor):
        """indices/weights [T, topk] (weights include route_scale), labels [T] int (domain*2 + is_image)."""

    def expert(self, layer: int, e: int, tok: torch.Tensor, w: torch.Tensor, x: torch.Tensor,
               act: torch.Tensor, out: torch.Tensor, labels: torch.Tensor):
        """tok [n] native token indices routed to e; w [n] routing weights; x [n, dim] bf16 expert input
        (before fp8 act quant); act [n, inter] fp32 silu(gate)*up (unweighted, after clamps);
        out [n, dim] bf16 weighted expert output; labels [n]."""

    def layer_tokens(self, layer: int, x: torch.Tensor, labels: torch.Tensor):
        """All native tokens' MoE input [T, dim] bf16 for this layer (for priors)."""

    def wants_expert(self, layer: int) -> bool:
        return False

    def prior_tokens(self, layer: int) -> torch.Tensor | None:
        """Native token indices every expert is also evaluated on (unrouted, weight 1) -> expert_prior."""
        return None

    def expert_prior(self, layer: int, e: int, act: torch.Tensor):
        """act [P, inter] fp32 of expert e on the prior tokens."""

    def end_layer(self, layer: int):
        pass

    def finish(self):
        pass


# --------------------------------------------------------------------------------------- harness

class Harness:
    def __init__(self, ckpt_dir, cfg_path=None, device="cuda", index=None, tokenizer_dir=None,
                 engram=True, cfg_overrides=None, attn_chunk=256, moe_chunk=65536, log=print):
        self.dev = torch.device(device)
        self.ckpt = Checkpoint(ckpt_dir, index=index)
        ckpt_dir = Path(ckpt_dir)
        cfg_path = cfg_path or ckpt_dir / "inference" / "config.json"
        self.cfg = Cfg.load(cfg_path, **(cfg_overrides or {}))
        self.ref_dir = ckpt_dir / "inference"
        self.tokenizer_dir = tokenizer_dir or ckpt_dir
        self.use_engram = engram
        self.attn_chunk = attn_chunk
        self.moe_chunk = moe_chunk
        self.log = log
        self.pref = Prefetcher(self.ckpt)
        self.freqs = {}
        self.engram_tables = {}
        self.pack: Exl3Pack | None = None
        self.timing: dict[str, float] = {}
        self._tok = None
        self._hasher = None

    # ---------------------------------------------------------------- reference helpers
    def ref_import(self):
        if str(self.ref_dir) not in sys.path:
            sys.path.insert(0, str(self.ref_dir))
        enc = self.ref_dir.parent / "encoding"
        if str(enc) not in sys.path:
            sys.path.insert(0, str(enc))

    @property
    def tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(str(self.tokenizer_dir))
        return self._tok

    def model_args(self, **kw):
        """A reference ModelArgs-like namespace (for engram/vision/image_processor helpers)."""
        from types import SimpleNamespace
        raw = dict(self.cfg.raw)
        raw.setdefault("vision_max_wh_ratio", None)
        raw.update(kw)
        ns = SimpleNamespace(**raw)
        ns.vision_enabled = raw.get("vision_n_layers", 0) > 0
        return ns

    def _tick(self, key, t0):
        torch.cuda.synchronize(self.dev)
        self.timing[key] = self.timing.get(key, 0.0) + time.time() - t0

    def layer_freqs(self, layer: int, s: int) -> torch.Tensor:
        c = self.cfg
        key = layer
        have = self.freqs.get(key)
        if have is None or have.size(0) < s:
            n = max(s, 4096)
            if c.compress_ratios[layer]:
                f = precompute_freqs_cis(c.rope_head_dim, n, c.original_seq_len, c.compress_rope_theta,
                                         c.rope_factor, c.beta_fast, c.beta_slow)
            else:
                f = precompute_freqs_cis(c.rope_head_dim, n, 0, c.rope_theta, c.rope_factor, c.beta_fast, c.beta_slow)
            self.freqs[key] = have = f.to(self.dev)
        return have

    # ---------------------------------------------------------------- weights
    def layer_names(self, layer: int) -> list[str]:
        p = f"layers.{layer}."
        return [n for n in self.ckpt.names(p) if ".engram.embed." not in n]

    def load_layer(self, layer: int) -> dict:
        t0 = time.time()
        names = self.layer_names(layer)
        cpu = self.pref.get(layer, names)
        if layer + 1 < self.cfg.n_layers:
            self.pref.submit(layer + 1, self.layer_names(layer + 1))
        self._tick("load_io_wait", t0)
        t0 = time.time()
        p = f"layers.{layer}."
        W = {}
        for k, v in cpu.items():
            W[k[len(p):]] = v.to(self.dev)
        del cpu
        # dense fp8 linears -> exact bf16 once per layer (wo_a included: convert.py dequantizes it to bf16)
        for k in list(W):  # routed experts are I8 (packed MXFP4) and stay packed
            if k.endswith(".weight") and W[k].dtype == torch.float8_e4m3fn:
                W[k] = rk.dequant_fp8_weight(W[k], W.pop(k[:-len("weight")] + "scale"))
        self._tick("load_h2d", t0)
        return W

    def expert_weights(self, W: dict, e: int, dtype=torch.bfloat16):
        q = f"ffn.experts.{e}."
        w13 = torch.cat([rk.dequant_fp4_weight(W[q + "w1.weight"], W[q + "w1.scale"], dtype),
                         rk.dequant_fp4_weight(W[q + "w3.weight"], W[q + "w3.scale"], dtype)])
        w2 = rk.dequant_fp4_weight(W[q + "w2.weight"], W[q + "w2.scale"], dtype)
        return w13, w2

    # ---------------------------------------------------------------- embed / vision / engram
    def embed_rows(self, rows: list[Row]):
        c = self.cfg
        E = self.ckpt.read(["embed.weight", "image_start", "image_end", "image_newline"], self.dev)
        vis = None
        if any(r.images for r in rows):
            vis = self.load_vision()
        self.ref_import()
        from image_processor import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START
        for r in rows:
            h = F.embedding(r.tokens.to(self.dev), E["embed.weight"])
            for img in r.images or ():
                types = img.types.to(self.dev)
                span = h[img.start : img.start + types.numel()]
                span[types == IMAGE_START] = E["image_start"]
                span[types == IMAGE_END] = E["image_end"]
                span[types == IMAGE_NEW_LINE] = E["image_newline"]
                with torch.no_grad(), torch.device(self.dev):  # vision.py builds RoPE tables on the default device
                    emb = vis[1](vis[0](img.patches.to(self.dev), img.n_vit_h, img.n_vit_w), img.n_vit_h, img.n_vit_w)
                span[types == IMAGE] = emb.to(h.dtype)
            r.h = h.unsqueeze(1).repeat(1, c.hc_mult, 1)
            r.pre_mix = torch.zeros(r.n, c.hc_mult, dtype=torch.float32, device=self.dev)
            r.pre_mix[:, 0] = 1.0
        del E, vis
        torch.cuda.empty_cache()

    def load_vision(self):
        self.ref_import()
        from vision import Aligner, ViT
        args = self.model_args()
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device(self.dev):
                vit, aligner = ViT(args), Aligner(args)
        finally:
            torch.set_default_dtype(prev)
        sd = self.ckpt.read(self.ckpt.names("vision.") + self.ckpt.names("aligner."), self.dev)
        vit.load_state_dict({k[len("vision."):]: v for k, v in sd.items() if k.startswith("vision.")})
        aligner.load_state_dict({k[len("aligner."):]: v for k, v in sd.items() if k.startswith("aligner.")})
        return vit.eval(), aligner.eval()

    def compute_hashes(self, rows: list[Row]):
        if not self.cfg.engram_layer_ids or not self.use_engram:
            return
        self.ref_import()
        from engram import EngramLayout, NgramHashState
        if self._hasher is None:
            args = self.model_args(max_batch_size=1, max_seq_len=max(r.n for r in rows))
            self._hasher = NgramHashState(args, EngramLayout.from_args(args), self.tokenizer).to(self.dev)
        need = max(r.n for r in rows)
        if self._hasher.cache.size(1) < need:
            self._hasher.cache = torch.empty(1, need, dtype=torch.int64, device=self.dev)
        for r in rows:
            mask = None if r.token_types is None else (r.token_types < 0).to(self.dev)[None]
            r.hashes = self._hasher(r.tokens.to(self.dev)[None], 0, mask)[0]  # [s, n_engram_layers, cols]

    def engram(self, layer: int, W: dict, r: Row):
        c = self.cfg
        li = list(c.engram_layer_ids).index(layer)
        if layer not in self.engram_tables:
            self.engram_tables[layer] = EngramTable(self.ckpt, layer)
        tbl = self.engram_tables[layer]
        mask = None if r.token_types is None else (r.token_types < 0).to(self.dev)
        weight = W["engram.q_weight"].float() * W["engram.k_weight"].float()
        out = torch.empty_like(r.h)
        for a, b in chunked(r.n, 4096):
            emb = tbl.lookup(r.hashes[a:b, li, :])
            kv = lin8(emb.flatten(-2), W["engram.wkv.weight"])
            key, value = kv.split([c.hc_mult * c.dim, c.dim], dim=-1)
            key = key.float().unflatten(-1, (c.hc_mult, c.dim))
            h = r.h[a:b].float()
            rstd = torch.rsqrt(h.square().mean(-1) + c.norm_eps) * torch.rsqrt(key.square().mean(-1) + c.norm_eps)
            dot = (h * weight * key).sum(-1) * rstd * c.dim ** -0.5
            gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
            if mask is not None:
                gate = gate.masked_fill(~mask[a:b].unsqueeze(-1), 0)
            out[a:b] = (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(r.h.dtype)
        r.h = out

    # ---------------------------------------------------------------- attention
    def compressor(self, W, x, ratio):
        c = self.cfg
        s = x.size(0)
        if ratio == 1:
            return rmsnorm(F.linear(x, W["attn.compressor.wkv.weight"]), W["attn.compressor.norm.weight"], c.norm_eps)
        if s < ratio:
            return None
        xf = x.float()
        kv = F.linear(xf, W["attn.compressor.wkv.weight"].float())
        score = F.linear(xf, W["attn.compressor.wgate.weight"].float())
        cut = s - s % ratio
        kv = kv[:cut].unflatten(0, (-1, ratio))
        score = score[:cut].unflatten(0, (-1, ratio))
        kv = (kv * score.softmax(dim=1)).sum(dim=1)
        return rmsnorm(kv.to(x.dtype), W["attn.compressor.norm.weight"], c.norm_eps)

    def indexer(self, layer, W, r: Row, x, qr, latent, offset):
        c = self.cfg
        s = x.size(0)
        ratio, rd = c.compress_ratios[layer], c.rope_head_dim
        freqs = self.layer_freqs(layer, s)
        st = r.shared
        if layer in c.kv_source_layers and latent is not None:
            k = rmsnorm(F.linear(latent, W["attn.indexer.wk.weight"]), W["attn.indexer.k_norm.weight"], c.norm_eps)
            rope_(k[..., -rd:], freqs[: s - s % ratio : ratio])
            rk.fp4_act_quant(k, 32, True)
            st["index_k"] = k
        q = lin8(qr, W["attn.indexer.wq_b.weight"]).unflatten(-1, (c.index_n_heads, c.index_head_dim))
        rope_(q[..., -rd:], freqs[:s])
        rk.fp4_act_quant(q, 32, True)
        clen_total = s // ratio
        index_k = st["index_k"][:clen_total]
        weights = F.linear(x, W["attn.indexer.weights_proj.weight"]) * (c.index_head_dim ** -0.5 * c.index_n_heads ** -0.5)
        topk = min(c.index_topk, clen_total)
        is_cand_src = layer == c.candidate_source_layer
        uses_cand = 0 <= c.candidate_source_layer < layer
        out = torch.empty(s, topk, dtype=torch.int32, device=self.dev)
        cand_store = [] if is_cand_src else None
        pos_t = torch.arange(clen_total, device=self.dev)
        step = max(1, min(s, (1 << 28) // max(1, c.index_n_heads * clen_total)))
        for a, b in chunked(s, step):
            sc = torch.matmul(q[a:b].flatten(0, 1), index_k.t()).view(b - a, c.index_n_heads, clen_total)
            sc = (sc.relu_() * weights[a:b].unsqueeze(-1)).sum(dim=1)
            clens = (torch.arange(a + 1, b + 1, device=self.dev) // ratio).unsqueeze(-1)
            sc.masked_fill_(pos_t >= clens, -torch.inf)
            if is_cand_src:
                cand_store.append(select_candidate_blocks(sc, clens, c.candidate_topk_blocks, c.candidate_block_size))
            elif uses_cand:
                keep = st["cand"][a:b].repeat_interleave(c.candidate_block_size, dim=-1)[..., :clen_total]
                sc = sc.masked_fill(~keep, -torch.inf)
            idx = sc.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
            out[a:b] = torch.where(idx < clens, idx + offset, -1).int()
        if is_cand_src:
            st["cand"] = torch.cat(cand_store)
        return out

    def attention(self, layer: int, W: dict, r: Row, x: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        s = x.size(0)
        ratio, rd = c.compress_ratios[layer], c.rope_head_dim
        freqs = self.layer_freqs(layer, s)
        fc = freqs[:s]
        qr = rmsnorm(lin8(x, W["attn.wq_a.weight"]), W["attn.q_norm.weight"], c.norm_eps)
        q = lin8(qr, W["attn.wq_b.weight"]).unflatten(-1, (c.n_heads, c.head_dim))
        rope_(q[..., -rd:], fc)
        kv = rmsnorm(lin8(x, W["attn.wkv.weight"]), W["attn.kv_norm.weight"], c.norm_eps)
        rope_(kv[..., -rd:], fc)
        rk.act_quant(kv, 32, "ue8m0", torch.float8_e8m0fnu, True)
        idx = window_idxs(s, c.window_size, self.dev)
        if ratio:
            st = r.shared
            clen = s // ratio
            latent = None
            if layer in c.kv_source_layers:
                latent = self.compressor(W, x, ratio)
                st["ckv"] = torch.zeros(clen, c.head_dim, dtype=x.dtype, device=self.dev)
            if layer in c.index_source_layers:
                cidx = (self.indexer(layer, W, r, x, qr, latent, s) if clen > 0
                        else torch.empty(s, 0, dtype=torch.int32, device=self.dev))
                st["topk"] = cidx
            else:
                cidx = st["topk"]
            if latent is not None:
                rope_(latent[..., -rd:], freqs[: s - s % ratio : ratio])
                rk.fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
                st["ckv"] = latent
            kv = torch.cat([kv, st["ckv"][:clen]], dim=0)
            idx = torch.cat([idx, cidx], dim=-1)
        o = rk.sparse_attn(q[None], kv[None], W["attn.attn_sink"], idx[None], c.head_dim ** -0.5, self.attn_chunk)[0]
        rope_(o[..., -rd:], fc, inverse=True)
        o = o.view(s, c.o_groups, -1)
        rk.act_quant(o, 32, "ue8m0", torch.float8_e8m0fnu, True)
        wo_a = W["attn.wo_a.weight"].view(c.o_groups, c.o_lora_rank, -1)
        o = torch.einsum("sgd,grd->sgr", o, wo_a)
        return lin8(o.flatten(1), W["attn.wo_b.weight"])

    # ---------------------------------------------------------------- MoE
    def gate(self, W, x, image_mask):
        c = self.cfg
        scores = F.linear(x.float(), W["ffn.gate.weight"].float()) / c.gate_temp
        scores = F.softplus(scores).sqrt()
        bias = W["ffn.gate.bias"]
        if image_mask is not None and "ffn.gate.bias_vl" in W:
            bias = torch.where(image_mask.unsqueeze(-1), W["ffn.gate.bias_vl"], bias)
        indices = (scores + bias).topk(c.n_activated_experts, dim=-1)[1]
        weights = scores.gather(1, indices)
        weights /= weights.sum(dim=-1, keepdim=True) + 1e-20
        weights *= c.route_scale
        return weights, indices

    def _variant_weights(self, layer, e, var: Variant, w13, w2, mod):
        """Perturbed or swapped (w13, w2, kind) for one expert of one variant."""
        if mod[0] == "swap":
            pk = self.pack
            names = [f"layers.{layer}.ffn.experts.{e}.{p}.{s}" for p in ("w1", "w3", "w2") for s in ("trellis", "suh", "svh")]
            T = pk.ckpt.read(names, self.dev)
            cb = pk.codebook(layer)
            rec = [pk.reconstruct(T[f"layers.{layer}.ffn.experts.{e}.{p}.trellis"],
                                  T[f"layers.{layer}.ffn.experts.{e}.{p}.suh"],
                                  T[f"layers.{layer}.ffn.experts.{e}.{p}.svh"], cb) for p in ("w1", "w3", "w2")]
            return torch.cat(rec[:2]), rec[2], "swap"
        _, group, rel = mod
        seed = int.from_bytes(hashlib.blake2b(f"{var.seed}/{var.name}/{layer}/{e}".encode(), digest_size=8).digest(), "little") >> 1
        g = torch.Generator(device=self.dev)
        if group in (GU, "ALL"):
            g.manual_seed(seed)
            half = w13.size(0) // 2
            n1 = torch.randn(w13.shape, generator=g, device=self.dev, dtype=torch.float32)
            w1, w3 = w13[:half].float(), w13[half:].float()
            s1 = w1.square().mean().sqrt() * rel
            s3 = w3.square().mean().sqrt() * rel
            w13 = torch.cat([w1 + n1[:half] * s1, w3 + n1[half:] * s3]).to(torch.bfloat16)
        if group in (D, "ALL"):
            g.manual_seed(seed ^ 0x5DEECE66D)
            w2f = w2.float()
            w2 = (w2f + torch.randn(w2.shape, generator=g, device=self.dev, dtype=torch.float32)
                  * (w2f.square().mean().sqrt() * rel)).to(torch.bfloat16)
        return w13, w2, "noise"

    def _expert_fwd(self, x_in, xq, w, w13, w2, kind):
        """One expert on n tokens. Native/noise: reference fp8 act quant + bf16 GEMMs. Swap: EXL3 serving
        numerics (fp16 activations, no act quant, fp16 weights)."""
        c = self.cfg
        inter = c.moe_inter_dim
        if kind == "swap":
            gu = torch.mm(x_in.half(), w13.t(), out_dtype=torch.float32).to(torch.bfloat16).float()
        else:
            gu = torch.matmul(xq, w13.t()).float()
        g, u = gu[:, :inter], gu[:, inter:]
        if c.swiglu_limit > 0:
            u = torch.clamp(u, min=-c.swiglu_limit, max=c.swiglu_limit)
            g = torch.clamp(g, max=c.swiglu_limit)
        act = F.silu(g) * u
        a = (w.unsqueeze(-1) * act).to(torch.bfloat16)
        if kind == "swap":
            out = torch.mm(a.half(), w2.t(), out_dtype=torch.float32).to(torch.bfloat16)
        else:
            out = torch.matmul(rk.act_quant_dequant(a, 32), w2.t())
        return out, act

    def moe(self, layer, W, x, image_mask, streams, labels, hooks: Hooks):
        """MoE over all rows. `streams` = [(a, b, variant, mod)] contiguous token ranges of x: the native
        rows first (variant None), then each variant's clone rows. Every stream is routed and computed in
        its own GEMMs, so an unperturbed variant is bit-identical to the native stream (GEMM shapes, and
        therefore cuBLAS kernels and summation orders, match). Each expert is dequantized once per layer.
        Returns y [T, dim] bf16 and, when a variant is modified here, the routed-only fp32 part."""
        c = self.cfg
        k_act = c.n_activated_experts
        y = torch.zeros(x.size(0), c.dim, dtype=torch.float32, device=self.dev)
        want = hooks.wants_expert(layer)
        S = []
        for si, (a, b, var, mod) in enumerate(streams):
            xs = x[a:b]
            weights, indices = self.gate(W, xs, None if image_mask is None else image_mask[a:b])
            if si == 0:
                hooks.routing(layer, indices, weights, labels)
                if want:
                    hooks.layer_tokens(layer, xs, labels)
            flat = indices.flatten()
            order = torch.argsort(flat, stable=True)  # tokens of each expert in ascending order (torch.where)
            S.append(dict(a=a, b=b, var=var, mod=mod, w=weights, tok=order // k_act, slot=order % k_act,
                          counts=torch.bincount(flat, minlength=c.n_routed_experts).tolist(), cur=0,
                          xq=rk.act_quant_dequant(xs, 32)))
        prior = hooks.prior_tokens(layer) if want else None
        if prior is not None:
            px = x[streams[0][0] : streams[0][1]][prior]
            pxq, pw = rk.act_quant_dequant(px, 32), torch.ones(px.size(0), device=self.dev)
        for e in range(c.n_routed_experts):
            if prior is None and not any(st["counts"][e] for st in S):
                continue
            w13, w2 = self.expert_weights(W, e)
            if prior is not None:
                hooks.expert_prior(layer, e, self._expert_fwd(px, pxq, pw, w13, w2, "native")[1])
            for si, st in enumerate(S):
                n = st["counts"][e]
                if n == 0:
                    continue
                tok = st["tok"][st["cur"] : st["cur"] + n]
                slot = st["slot"][st["cur"] : st["cur"] + n]
                st["cur"] += n
                w = st["w"][tok, slot]
                if st["mod"] is None:
                    ww13, ww2, kind = w13, w2, "native"
                else:
                    ww13, ww2, kind = self._variant_weights(layer, e, st["var"], w13, w2, st["mod"])
                xin = x[st["a"] : st["b"]][tok]
                out, act = self._expert_fwd(xin, st["xq"][tok], w, ww13, ww2, kind)
                y[st["a"] : st["b"]].index_add_(0, tok, out.float())
                if want and si == 0:
                    hooks.expert(layer, e, tok, w, xin, act, out, labels[tok])
        routed = y.clone() if any(st["mod"] is not None for st in S) else None
        sw13 = torch.cat([W["ffn.shared_experts.w1.weight"], W["ffn.shared_experts.w3.weight"]])
        for st in S:  # per stream, so unperturbed streams stay bit-identical
            for a, b in chunked(st["b"] - st["a"], self.moe_chunk):
                sh, _ = self._expert_fwd(None, st["xq"][a:b], torch.ones(b - a, device=self.dev), sw13,
                                         W["ffn.shared_experts.w2.weight"], "native")
                y[st["a"] + a : st["a"] + b] += sh.float()
        return y.to(x.dtype), routed

    # ---------------------------------------------------------------- driver
    def start(self, rows: list[Row], hooks: Hooks):
        """Embed (+ vision) and hash all native rows; fixes the domain ids used for hook labels."""
        t0 = time.time()
        self.embed_rows(rows)
        self.compute_hashes(rows)
        self._tick("embed_vision_hash", t0)
        self.dom_ids = {}
        for r in rows:
            self.dom_ids.setdefault(r.domain, len(self.dom_ids))
        self.domains = list(self.dom_ids)
        hooks.begin(self, rows)

    def layer_forward(self, layer: int, W: dict, rows: list[Row], hooks: Hooks, keep_main_hidden: bool = False):
        """One decoder layer (Engram if any, then the Block) over all rows, in place on each row's state."""
        c = self.cfg
        t0 = time.time()
        if layer in c.engram_layer_ids and self.use_engram:
            for r in rows:
                self.engram(layer, W, r)
            self.engram_tables[layer].drop_cache()
        self._tick("engram", t0)
        if keep_main_hidden and layer in c.dspark_target_layer_ids:
            for r in rows:  # drafter input (model.py: h.mean(dim=2) entering layers 37-39)
                r.main_hidden[layer] = r.h.float().mean(dim=1).to(torch.bfloat16)
        t0 = time.time()
        xs, posts, combs, new_pre = [], [], [], []
        for r in rows:
            residual = r.h
            attn_pre, attn_post, attn_comb = hc_mixes(residual, W["hc_attn_fn"], W["hc_attn_scale"], W["hc_attn_base"], c)
            x = rmsnorm(hc_pre(residual, r.pre_mix), W["attn_norm.weight"], c.norm_eps)
            x = self.attention(layer, W, r, x)
            h = torch.empty_like(residual)
            for a, b in chunked(r.n, 4096):
                h[a:b] = hc_post(x[a:b], residual[a:b], attn_post[a:b], attn_comb[a:b])
            r.h = h
            ffn_pre, ffn_post, ffn_comb = hc_mixes(h, W["hc_ffn_fn"], W["hc_ffn_scale"], W["hc_ffn_base"], c)
            xs.append(rmsnorm(hc_pre(h, attn_pre), W["ffn_norm.weight"], c.norm_eps))
            posts.append(ffn_post)
            combs.append(ffn_comb)
            new_pre.append(ffn_pre)
        self._tick("attn", t0)
        t0 = time.time()
        x_all = torch.cat(xs)
        del xs
        masks = [r.image_mask(self.dev) if r.token_types is not None
                 else torch.zeros(r.n, dtype=torch.bool, device=self.dev) for r in rows]
        image_mask = torch.cat(masks) if any(r.token_types is not None for r in rows) else None
        # streams: native rows first, then each variant's clone rows (contiguous, same order and lengths)
        streams, labels, off, cur = [], [], 0, None
        for r, im in zip(rows, masks):
            name = None if r.variant is None else r.variant.name
            if cur is None or cur[0] != name:
                mod = None if r.variant is None else r.variant.mods.get(layer)
                cur = [name, off, off, r.variant, mod]
                streams.append(cur)
            off += r.n
            cur[2] = off
            if r.variant is None:
                labels.append(self.dom_ids[r.domain] * 2 + im.long())
        streams = [(a, b, v, m) for _, a, b, v, m in streams]
        y, routed = self.moe(layer, W, x_all, image_mask, streams, torch.cat(labels), hooks)
        del x_all
        self._tick("moe", t0)
        t0 = time.time()
        offs, off = [], 0
        for r in rows:
            offs.append(off)
            off += r.n
        if routed is not None:
            self.variant_out_err(layer, rows, offs, routed)
        for r, o, post, comb, pre in zip(rows, offs, posts, combs, new_pre):
            h = torch.empty_like(r.h)
            for a, b in chunked(r.n, 4096):
                h[a:b] = hc_post(y[o + a : o + b], r.h[a:b], post[a:b], comb[a:b])
            r.h, r.pre_mix = h, pre
        del y, routed
        hooks.end_layer(layer)
        self._tick("hc_post", t0)

    def run(self, rows: list[Row], variants: list[Variant] = (), hooks: Hooks | None = None,
            head_cb=None, pack_dir=None):
        """Full forward over native `rows` plus variant rows spawned from them. Returns all rows
        (natives first). head_cb(harness, rows) runs after the last layer (see Head)."""
        c = self.cfg
        hooks = hooks or Hooks()
        if pack_dir and self.pack is None:
            self.pack = Exl3Pack(pack_dir)
        rows = list(rows)
        n_native = len(rows)
        self.pref.submit(0, self.layer_names(0))
        self.start(rows, hooks)
        for layer in range(c.n_layers):
            tl = time.time()
            for v in variants:
                if v.start_layer == layer:
                    rows.extend(spawn(rows[i], v, i) for i in range(n_native))
            W = self.load_layer(layer)
            self.layer_forward(layer, W, rows, hooks, keep_main_hidden=bool(variants))
            del W
            self.log(f"[layer {layer:2d}] {time.time() - tl:6.2f}s rows={len(rows)} tokens={sum(r.n for r in rows)} "
                     f"peak={torch.cuda.max_memory_allocated(self.dev) / 2**30:.1f}GiB")
        if head_cb is not None:
            t0 = time.time()
            head_cb(self, rows)
            self._tick("head", t0)
        hooks.finish()
        return rows

    def variant_out_err(self, layer, rows, offs, routed):
        """Routed-output error of each variant at its first modified layer vs its native twin (inputs are
        identical there): aggregate ||y_v - y_n||^2 / ||y_n||^2 and the per-token mean of that ratio."""
        self.out_err = getattr(self, "out_err", {})
        for r, o in zip(rows, offs):
            v = r.variant
            if v is None or v.start_layer != layer:
                continue
            o2 = offs[r.twin]
            yn = routed[o2 : o2 + r.n]
            yv = routed[o : o + r.n]
            d2 = (yv - yn).square().sum(-1)
            n2 = yn.square().sum(-1)
            acc = self.out_err.setdefault(v.name, [0.0, 0.0, 0.0, 0])
            acc[0] += d2.sum().item()
            acc[1] += n2.sum().item()
            acc[2] += (d2 / n2.clamp_min(1e-30)).sum().item()  # per-token ratio: not dominated by massive tokens
            acc[3] += r.n

    # ---------------------------------------------------------------- head
    def final_hidden(self, r: Row, norm_w) -> torch.Tensor:
        return rmsnorm(hc_pre(r.h, r.pre_mix), norm_w, self.cfg.norm_eps)


class Head:
    """Final norm + fp32 logits (bf16 x bf16 GEMM with fp32 output = exact products, fp32 sums), per row:
    NLL of the actual next token on scored positions, optional top-k dump with logsumexp, and for
    variant rows the exact full-vocab KL(native || variant) vs their twin, per domain."""

    def __init__(self, topk_dump: int = 0, chunk: int = 2048, keep_last_logits: bool = False):
        self.topk = topk_dump
        self.chunk = chunk
        self.keep_last = keep_last_logits
        self.nll = {}       # row id -> (sum, n)
        self.per_tok = {}   # row id -> tensor of per-target NLL
        self.dump = {}      # row id -> dict(ids, logprobs, lse, nll)
        self.kl = {}        # variant -> domain -> [sum_kl, n, sum_nll]
        self.drafter = {}   # variant -> layer -> [sum_sq_err, sum_sq_ref]
        self.last_logits = {}

    def targets(self, r: Row):
        """Predicting positions p (target p+1): from score_start, or where score_mask[p+1]; never image targets."""
        s = r.n
        if r.score_mask is not None:
            pos = r.score_mask[1:].nonzero().squeeze(-1)
        else:
            pos = torch.arange(max(0, r.score_start - 1), s - 1)
        if r.token_types is not None and pos.numel():
            pos = pos[r.token_types[pos + 1] < 0]
        return pos

    def __call__(self, hz: Harness, rows: list[Row]):
        dev = hz.dev
        Wt = hz.ckpt.read(["head.weight", "norm.weight"], dev)
        Wh, nw = Wt["head.weight"], Wt["norm.weight"]
        xs = {}
        for i, r in enumerate(rows):
            if r.variant is None:
                xs[i] = hz.final_hidden(r, nw)
        for i, r in enumerate(rows):
            x = xs[i] if r.variant is None else hz.final_hidden(r, nw)
            if self.keep_last:
                self.last_logits[r.id] = torch.mm(x[-1:], Wh.t(), out_dtype=torch.float32)[0].cpu()
            pos = self.targets(r).to(dev)
            tg = r.tokens.to(dev)
            nll_all, dumps = [], []
            for a, b in chunked(pos.numel(), self.chunk):
                p = pos[a:b]
                lg = torch.mm(x[p], Wh.t(), out_dtype=torch.float32)
                lse = torch.logsumexp(lg, dim=-1)
                lp = lg - lse.unsqueeze(-1)
                nll = -lp.gather(1, tg[p + 1].unsqueeze(-1)).squeeze(-1)
                nll_all.append(nll)
                if r.variant is None and self.topk:
                    tv, ti = lg.topk(self.topk, dim=-1)
                    dumps.append((ti.int().cpu(), (tv - lse.unsqueeze(-1)).cpu(), lse.cpu(), nll.cpu()))
                if r.variant is not None:
                    xn = xs[r.twin][p]
                    lgn = torch.mm(xn, Wh.t(), out_dtype=torch.float32)
                    lpn = lgn - torch.logsumexp(lgn, dim=-1, keepdim=True)  # same formula as lp: identical logits -> KL exactly 0
                    kl = (lpn.exp() * (lpn - lp)).sum(-1)
                    d = self.kl.setdefault(r.variant.name, {}).setdefault(r.domain, [0.0, 0, 0.0])
                    d[0] += kl.sum().item()
                    d[1] += kl.numel()
                    d[2] += nll.sum().item()
                del lg, lp
            nll_all = torch.cat(nll_all) if nll_all else torch.empty(0, device=dev)
            self.nll[r.id] = (nll_all.sum().item(), int(nll_all.numel()))
            self.per_tok[r.id] = nll_all.cpu()
            if dumps:
                self.dump[r.id] = {
                    "pos": pos.cpu().int(),
                    "ids": torch.cat([d[0] for d in dumps]),
                    "logprobs": torch.cat([d[1] for d in dumps]),
                    "lse": torch.cat([d[2] for d in dumps]),
                    "nll": torch.cat([d[3] for d in dumps]),
                }
            if r.variant is not None:
                tw = rows[r.twin]
                for L, hv in r.main_hidden.items():
                    hn = tw.main_hidden[L].float()
                    acc = self.drafter.setdefault(r.variant.name, {}).setdefault(L, [0.0, 0.0])
                    acc[0] += (hv.float() - hn).square().sum().item()
                    acc[1] += hn.square().sum().item()
