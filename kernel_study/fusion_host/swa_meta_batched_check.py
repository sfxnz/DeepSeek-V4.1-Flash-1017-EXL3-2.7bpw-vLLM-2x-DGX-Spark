#!/usr/bin/env python3
"""DSV41_SWA_META_BATCHED: the batched SWA decode-row kernel vs the image's stock ops, bit for bit.

Runs in the serving image (patch dir at /opt/dsv41-patch, or --patch-dir). On a GPU it
uses CUDA; without one it needs TRITON_INTERPRET=1 and CUDA_VISIBLE_DEVICES set to empty
(both in the environment before python starts: with no active driver vllm.triton_utils
swaps in placeholders unless CUDA_VISIBLE_DEVICES is set) and runs the same Triton kernels
on CPU tensors.

1. kernels: random causal decode batches (requests 1..4, DSpark k=3 verify rows, a padded
   request, padded tokens, sequences shorter than the window, random block tables per
   group), G groups. Reference per group = the stock ops of
   DeepseekSparseSWAMetadataBuilder.build's causal decode branch (validity = slot >= 0,
   lens tail zeroed, _compute_swa_indices_and_lens_kernel). The batched kernel over
   poisoned buffers must equal it in every written element: decode index rows, the whole
   lens buffer and the validity row of every group. swa_meta_fused's per-group kernel is
   checked against the same reference.
2. verify path: SwaBatched.verify on a real batch (must pass and count down), then a
   sabotaged launch (one group's rows wrong) must disarm and leave the fused values.
Prints one JSON line; exit 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import types

WINDOW, BLOCK = 128, 64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-dir", default="/opt/dsv41-patch")
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256, help="builder buffer rows (8192 in the serve)")
    ap.add_argument("--cases", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1017)
    a = ap.parse_args()
    sys.path.insert(0, a.patch_dir)

    import torch

    cuda = torch.cuda.is_available()
    if not cuda and os.environ.get("TRITON_INTERPRET") != "1":
        print(json.dumps({"ok": False, "error": "no GPU: run with TRITON_INTERPRET=1"}))
        return 1
    dev = torch.device("cuda" if cuda else "cpu")
    import swa_meta_batched as smb
    import swa_meta_fused as smf
    from vllm.triton_utils import tl, triton
    from vllm.v1.attention.backends.mla import sparse_swa

    stock_kernel = sparse_swa._compute_swa_indices_and_lens_kernel  # the raw @triton.jit function
    batched = smb._build_kernel(tl, triton)
    fused_kernel = smf._build_kernel(tl, triton)
    rng = random.Random(a.seed)
    G, T = a.groups, a.max_tokens

    def builder():
        return types.SimpleNamespace(
            window_size=WINDOW, block_size=BLOCK,
            decode_swa_indices=torch.full((T, 1, WINDOW), 99, dtype=torch.int32, device=dev),
            decode_swa_lens=torch.full((T,), 99, dtype=torch.int32, device=dev),
            is_valid_token=torch.zeros(T, dtype=torch.bool, device=dev),
        )

    def batch():
        num_reqs = rng.randint(1, 4)
        ragged = rng.random() < 0.3
        qlens = [rng.randint(1, 4) if ragged else 4 for _ in range(num_reqs)]
        seq = [rng.choice([q, rng.randint(q, WINDOW - 1), rng.randint(WINDOW, 4000)]) for q in qlens]
        pad_reqs = 1 if rng.random() < 0.3 else 0  # a padded request: seq_len 0, no tokens
        qsl = [0]
        for q in qlens:
            qsl.append(qsl[-1] + q)
        nd = qsl[-1]
        pad_tokens = rng.choice([0, 0, 1, 2])
        n_slots = nd + pad_tokens
        qsl += [nd] * pad_reqs
        seq += [0] * pad_reqs
        t2r = [r for r, q in enumerate(qlens) for _ in range(q)] + [0] * pad_tokens
        max_blocks = (4000 + 4) // BLOCK + 2
        tables, slots = [], []
        for _ in range(G):
            bt = torch.tensor([[rng.randint(1, 5000) for _ in range(max_blocks)] for _ in range(num_reqs + pad_reqs)],
                              dtype=torch.int32, device=dev)
            row = []
            for r, q in enumerate(qlens):
                for j in range(q):
                    pos = seq[r] - q + j
                    row.append(int(bt[r, pos // BLOCK]) * BLOCK + pos % BLOCK)
            row += [-1] * pad_tokens
            if rng.random() < 0.2 and nd:  # an invalid (padded) slot inside the decode rows
                row[rng.randrange(nd)] = -1
            tables.append(bt)
            slots.append(row)
        slot_mappings = torch.tensor(slots, dtype=torch.int64, device=dev)
        i32 = dict(dtype=torch.int32, device=dev)
        return (torch.tensor(qsl, **i32), torch.tensor(seq, **i32), torch.tensor(t2r + [0] * (T - len(t2r)), **i32),
                tables, slot_mappings, nd, n_slots)

    def stock(b, qsl, seq_lens, t2r, slots, bt, nd):
        valid = b.is_valid_token[: slots.shape[0]]
        valid.copy_(slots >= 0)
        b.decode_swa_lens[nd:] = 0
        stock_kernel[(nd,)](
            b.decode_swa_indices, b.decode_swa_indices.stride(0), b.decode_swa_lens, WINDOW,
            b.decode_swa_indices.shape[-1], b.decode_swa_lens, b.decode_swa_lens,
            qsl, seq_lens, t2r, b.is_valid_token, bt, bt.stride(0), BLOCK, 0,
            HAS_IMAGE=False, TRITON_BLOCK_SIZE=1024,
        )

    def snap(b, nd, n_slots):
        return (b.decode_swa_indices[:nd].clone(), b.decode_swa_lens.clone(), b.is_valid_token[:n_slots].clone())

    def same(x, y):
        return all(torch.equal(p, q) for p, q in zip(x, y))

    fused = smf.SwaFused(torch, triton, fused_kernel, None)
    results = {"device": str(dev), "groups": G, "cases": 0, "mismatch": []}
    for case in range(a.cases):
        qsl, seq_lens, t2r, tables, slot_mappings, nd, n_slots = batch()
        builders = [builder() for _ in range(G)]
        ref = []
        for g, b in enumerate(builders):
            stock(b, qsl, seq_lens, t2r, slot_mappings[g], tables[g], nd)
            ref.append(snap(b, nd, n_slots))
        # per-group fused kernel
        for g, b in enumerate(builders):
            b.decode_swa_indices.fill_(-7), b.decode_swa_lens.fill_(-7), b.is_valid_token.fill_(True)
            fused._launch(b, b.decode_swa_indices, qsl, seq_lens, t2r, slot_mappings[g],
                          b.is_valid_token[:n_slots], tables[g], nd)
            if not same(snap(b, nd, n_slots), ref[g]):
                results["mismatch"].append(f"fused case {case} group {g}")
        # batched kernel
        for b in builders:
            b.decode_swa_indices.fill_(-7), b.decode_swa_lens.fill_(-7), b.is_valid_token.fill_(True)
        plan = smb.Plan(list(enumerate(builders)), smb.static_pointers(builders), tables, slot_mappings)
        helper = smb.SwaBatched(torch, triton, batched, fused)
        helper.launch(plan, qsl, seq_lens, t2r, nd)
        for g, b in enumerate(builders):
            if not same(snap(b, nd, n_slots), ref[g]):
                results["mismatch"].append(f"batched case {case} group {g} (nd {nd}, slots {n_slots})")
        results["cases"] += 1

    # verify path, then sabotage
    qsl, seq_lens, t2r, tables, slot_mappings, nd, n_slots = batch()
    while not bool((slot_mappings[0][:nd] >= 0).any()):
        qsl, seq_lens, t2r, tables, slot_mappings, nd, n_slots = batch()
    builders = [builder() for _ in range(G)]
    plan = smb.Plan(list(enumerate(builders)), smb.static_pointers(builders), tables, slot_mappings)
    helper = smb.SwaBatched(torch, triton, batched, fused)
    smb._STATE.update(armed=True, verify_left=2, engaged=False)
    helper.verify(plan, qsl, seq_lens, t2r, nd)
    results["verify_counted_down"] = smb._STATE["verify_left"] == 1 and smb._STATE["engaged"]
    real_launch = helper.launch

    def sabotaged(*args):
        real_launch(*args)
        builders[-1].decode_swa_lens[0] += 1

    helper.launch = sabotaged
    try:
        helper.verify(plan, qsl, seq_lens, t2r, nd)
        results["sabotage_disarmed"] = False
    except RuntimeError:
        want = []
        for g, b in enumerate(builders):
            c = builder()
            stock(c, qsl, seq_lens, t2r, slot_mappings[g], tables[g], nd)
            want.append(snap(c, nd, n_slots))
        results["sabotage_disarmed"] = (not smb._STATE["armed"]) and all(
            same(snap(b, nd, n_slots), w) for b, w in zip(builders, want))
    ok = not results["mismatch"] and results["verify_counted_down"] and results["sabotage_disarmed"]
    results["ok"] = ok
    print(json.dumps(results))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
