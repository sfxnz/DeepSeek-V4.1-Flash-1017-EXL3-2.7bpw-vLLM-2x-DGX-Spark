"""Lockstep check: official inference/model.py (Block, Engram, Transformer embed/vision/head; TileLang
kernels when importable, else refkernel) vs native.Harness on the real 1017 weights, layer by layer.

One prompt = chat template + one image (examples/images/carrots.jpeg, bias_vl routing + Engram mask)
+ a ~500-token passage. Per layer it records
  free: rel. Frobenius diff of the hc stream after the layer, each side on its own previous output
  tf:   teacher-forced per-layer error (harness layer fed the reference input)
and finally KL(ref || harness) of the full-vocab logits at every position and top-1 agreement.

Needs GPU + weights; skipped otherwise. Env: CALIB_LOCKSTEP_LAYERS (default 40), CALIB_REF_KERNEL
(tilelang | torch, default tilelang), CALIB_CFG_OVERRIDE (JSON applied to BOTH sides, e.g.
{"index_topk": 64, "candidate_topk_blocks": 16}), CALIB_OUT (JSON result path).
"""

import contextlib
import importlib.util
import json
import os
import sys
import time
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SNAP = Path(os.environ.get("CALIB_SNAP", Path.home() / ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"))
PASSAGES = Path(__file__).resolve().parents[3] / "tests/quality/nll_passages.jsonl"
N_LAYERS = int(os.environ.get("CALIB_LOCKSTEP_LAYERS", "40"))
OVR = json.loads(os.environ.get("CALIB_CFG_OVERRIDE", "{}"))
HAVE = torch.cuda.is_available() and (SNAP / "model-00003-of-00048.safetensors").exists()


@contextlib.contextmanager
def default_dtype(dt):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dt)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def import_reference():
    """Import the official model.py with `kernel` bound to TileLang (or refkernel)."""
    ref = SNAP / "inference"
    sys.path.insert(0, str(ref))
    sys.path.insert(0, str(SNAP / "encoding"))
    which = os.environ.get("CALIB_REF_KERNEL", "tilelang")
    kmod = None
    if which == "tilelang":
        try:
            spec = importlib.util.spec_from_file_location("kernel", ref / "kernel.py")
            kmod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(kmod)
            # TileLang sparse_attn needs 141 KB smem at h=64, d=512 (> sm_121 limit): torch version there
            import refkernel
            kmod.sparse_attn = refkernel.sparse_attn
            which = "tilelang (sparse_attn: refkernel)"
        except Exception as e:
            print("tilelang import failed, using refkernel:", e)
            kmod, which = None, "torch"
    if kmod is None:
        import refkernel as kmod
    sys.modules["kernel"] = kmod
    spec = importlib.util.spec_from_file_location("ref_model", ref / "model.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model, which


def to_reference_sd(raw: dict, prefix: str) -> dict:
    """HF checkpoint tensors of one layer -> state dict of model.Block (convert.py semantics)."""
    sd = {}
    for k, v in raw.items():
        name = k[len(prefix):]
        if name.startswith("engram."):
            continue
        sd[name] = v
    w, s = sd["attn.wo_a.weight"], sd.pop("attn.wo_a.scale")
    w = w.unflatten(0, (-1, 32)).unflatten(-1, (-1, 32)).float() * s[:, None, :, None].float()
    sd["attn.wo_a.weight"] = w.flatten(2, 3).flatten(0, 1).bfloat16()
    for k in list(sd):
        if ".experts." in k and "shared" not in k and k.endswith(".weight"):
            sd[k] = sd[k].view(torch.float4_e2m1fn_x2)
    return sd


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


@unittest.skipUnless(HAVE, "needs CUDA and the 1017 weights")
class Lockstep(unittest.TestCase):
    def test_lockstep(self):
        from native import Harness, Hooks, Row, hc_pre, rmsnorm
        dev = torch.device("cuda")
        model, which = import_reference()
        hz = Harness(SNAP, cfg_overrides=OVR, log=print)
        hz.ref_import()
        from encoding import encode_messages
        from image_processor import prepare_vl_inputs

        passage = json.loads(PASSAGES.read_text().splitlines()[0])["text"]
        img = str(SNAP / "inference/examples/images/carrots.jpeg")
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}},
                                             {"type": "text", "text": "What is in this picture? Then continue this text: " + passage}]}]
        prompt, media = encode_messages(msgs, thinking_mode="chat", return_multi_modal_data=True)
        ids, types, images = prepare_vl_inputs(prompt, media["images"], hz.tokenizer, hz.model_args())
        s = len(ids)
        print(f"seq {s} tokens, {sum(t >= 0 for t in types)} image-span tokens, ref kernels={which}")
        tokens = torch.tensor(ids)
        ttypes = torch.tensor(types)

        # ---------------- reference side: Transformer shell (embed/vision/head/hash), Blocks per layer
        cfg = json.loads((SNAP / "inference/config.json").read_text())
        cfg.update(OVR)
        cfg.update(max_batch_size=1, max_seq_len=((s + 255) // 256) * 256, dspark_block_size=0, n_layers=0)
        args = model.ModelArgs(**{k: v for k, v in cfg.items() if k in model.ModelArgs.__dataclass_fields__})
        with default_dtype(torch.bfloat16), torch.device(dev):
            shell = model.Transformer(args, hz.tokenizer)
        names = ["embed.weight", "image_start", "image_end", "image_newline", "head.weight", "norm.weight"]
        names += hz.ckpt.names("vision.") + hz.ckpt.names("aligner.")
        sd = hz.ckpt.read(names, dev)
        missing, unexpected = shell.load_state_dict(sd, strict=False)
        self.assertFalse(unexpected, unexpected)
        args.n_layers = 40  # Blocks and DSpark logic read n_layers
        del sd

        tok_d = tokens.to(dev)[None]
        image_mask = (ttypes >= 0).to(dev)[None]
        with torch.inference_mode(), default_dtype(torch.bfloat16), torch.device(dev):
            hashes = shell.engram_hash(tok_d, 0, ~image_mask)
            h_ref = shell.embed(tok_d)
            shell.merge_image_embeddings([images], h_ref)
            h_ref = h_ref.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
            pre_ref = model.make_identity_pre_mix(h_ref, args.hc_mult)

        # ---------------- harness side: free-running row + teacher-forced row
        row = Row(id="free", tokens=tokens, token_types=ttypes, images=images, score_start=1)
        tf = Row(id="tf", tokens=tokens, token_types=ttypes, images=images, score_start=1)
        with torch.inference_mode():
            hz.start([row, tf], Hooks())
        res = {"seq_len": s, "ref_kernels": which, "override": OVR, "layers": []}
        self.assertLess(rel(row.h, h_ref[0]), 1e-6, "embedding/vision merge mismatch")
        res["embed_rel"] = rel(row.h, h_ref[0])

        for layer in range(N_LAYERS):
            t0 = time.time()
            p = f"layers.{layer}."
            raw = hz.ckpt.read([n for n in hz.ckpt.names(p) if ".engram.embed." not in n], dev)
            h_start, pre_start = h_ref, pre_ref  # reference state entering this layer (before Engram)
            # generate.py runs the reference with default dtype bf16 and default device cuda
            with torch.inference_mode(), default_dtype(torch.bfloat16), torch.device(dev):
                blk = model.Block(layer, args, None)
                blk.load_state_dict(to_reference_sd(raw, p), strict=True)
                h_in = h_start
                if layer in args.engram_layer_ids:
                    h_in = self.ref_engram(model, args, hz, layer, raw, h_start, hashes, ~image_mask)
                h_ref, pre_ref = blk(h_in, 0, pre_start, image_mask)
            del blk
            W = hz.load_layer(layer)
            with torch.inference_mode():
                tf.h, tf.pre_mix = h_start[0].clone(), pre_start[0].clone()  # teacher forcing
                hz.layer_forward(layer, W, [row, tf], Hooks())
            del W, raw
            e = {"layer": layer, "free": rel(row.h, h_ref[0]), "tf": rel(tf.h, h_ref[0]),
                 "free_premix": rel(row.pre_mix, pre_ref[0]), "s": round(time.time() - t0, 1)}
            res["layers"].append(e)
            print(json.dumps(e), flush=True)

        if N_LAYERS == 40:
            with torch.inference_mode():
                Wt = hz.ckpt.read(["head.weight", "norm.weight"], dev)
                with default_dtype(torch.bfloat16), torch.device(dev):
                    x = model.Block.hc_pre(None, h_ref, pre_ref)
                    lg_ref = shell.head(shell.norm(x), full_logits=True)[0]
                xh = rmsnorm(hc_pre(row.h, row.pre_mix), Wt["norm.weight"], hz.cfg.norm_eps)
                lg = torch.mm(xh, Wt["head.weight"].t(), out_dtype=torch.float32)
                text = (ttypes[1:] < 0).to(dev)  # positions predicting a text token
                lp_r, lp_h = lg_ref.log_softmax(-1), lg.log_softmax(-1)
                kl = (lp_r.exp() * (lp_r - lp_h)).sum(-1)[:-1][text]
                top1 = (lg_ref.argmax(-1) == lg.argmax(-1)).float()[:-1][text]
                tgt = tokens.to(dev)[1:]
                nll_r = -lp_r[:-1].gather(1, tgt[:, None])[:, 0][text]
                nll_h = -lp_h[:-1].gather(1, tgt[:, None])[:, 0][text]
            res["final"] = {"kl_mean": kl.mean().item(), "kl_max": kl.max().item(), "top1_agree": top1.mean().item(),
                            "nll_ref": nll_r.mean().item(), "nll_harness": nll_h.mean().item(), "n": int(kl.numel())}
            print(json.dumps(res["final"]))
        if os.environ.get("CALIB_OUT"):
            Path(os.environ["CALIB_OUT"]).write_text(json.dumps(res, indent=1))
        self.assertLess(max(e["tf"] for e in res["layers"]), 0.02)
        if "final" in res:
            self.assertGreater(res["final"]["top1_agree"], 0.97)

    def ref_engram(self, model, args, hz, layer, raw, h, hashes, mask):
        """Official Engram module with its table restricted to the rows this prompt touches."""
        from engram import EngramLayout
        layout = EngramLayout.from_args(args)
        li = list(args.engram_layer_ids).index(layer)
        ids = hashes[0, :, li, :]
        uniq, inv = torch.unique(ids, return_inverse=True)
        from stio import EngramTable
        tbl = EngramTable(hz.ckpt, layer)
        rows = uniq.cpu().numpy()
        w = torch.from_numpy(tbl._gather("weight", rows)).cuda().view(torch.float8_e4m3fn)
        s = torch.from_numpy(tbl._gather("scale", rows)).cuda().view(torch.float8_e8m0fnu)
        with torch.device("meta"):
            eng = model.Engram(args, layer, layout)
        eng.embed = model.ParallelEngramEmbedding(len(rows), layout.head_dim).to_empty(device="cuda")
        eng = eng.to_empty(device="cuda")
        p = f"layers.{layer}.engram."
        eng.load_state_dict({"embed.weight": w, "embed.scale": s, "wkv.weight": raw[p + "wkv.weight"],
                             "wkv.scale": raw[p + "wkv.scale"], "q_weight": raw[p + "q_weight"],
                             "k_weight": raw[p + "k_weight"]}, strict=True)
        eng.wkv.weight.scale = eng.wkv.scale  # Linear attaches it in __init__; to_empty made new Parameters
        return eng(h, inv.view(1, *ids.shape), mask)


if __name__ == "__main__":
    unittest.main()
