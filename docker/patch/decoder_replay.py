"""Decoder-side SWA bounded replay (CED prefill skip) on e15 (DSV41_DECODER_REPLAY_ROWS).

Backport of vLLM 0.32's decoder-side bounded replay (#58132 / #59532 / #59894,
models/deepseek_v41/{nvidia/model.py, decoder_replay_layers.py,
nvidia/model_state.py}) onto the 0909 image's models/deepseek_v4_1, with an
exact default window.

What runs. cut = max(kv_source_layer_ids) = 20. In an eager prefill step the
layers before the cut run on every row. Layer 20 then writes its sliding-window
KV, its ratio-1 compressed KV and its index K for every row (write_kv ->
forward_kv). Layer 20's attention / wo / MoE and layers 21-39 run on the kept
rows only, as a sub-batch with its own attention metadata and forward context.
Their outputs are scattered back; a trimmed row carries its stream at the cut
(run_tail). Decode rows of other requests are always kept. FULL and PIECEWISE
graph steps never trim.

Window (end-anchored, rows >= prompt_len - W of each request, whichever chunk
holds them; a chunk wholly before that runs no decoder layer at all).
Past the cut a row reaches older rows only through the sliding window of the
same layer (127 rows back); the compressed KV, index K and candidates of layers
21-39 come from layer 20's input, written for every row. Let S = N - W. Rows
below S hold no KV in layers 21-39; the first rows of the window see a window
truncated at S (clamp below), so H_{21+k} (input of layer 21+k) can differ
from a full forward only on rows [S, S + 127k). What decoding needs:
  - the last row's output H_40(N-1),
  - each decoder layer's window KV on rows >= N-127 (the next decode step),
  - the DSpark taps H_37..H_39 (inputs of layers 37-39) on rows >= N-128: the
    drafter's three SWA layers read context KV [N-128, N) (their context KV
    is main_proj of those taps, row by row; dspark.py precompute_and_store).
H_39 rows >= N-128 needs S + 18*127 <= N-128, i.e. W >= 128 + 18*127 = 2414;
H_40(N-1) needs S + 19*127 <= N-1, the same bound. 2414 is the smallest exact
window (TensorFold C1's per-layer plan gives the same rows); the drafter's
128-row context needs no more. Any exact W leaves valid_tail = W - 18*127
prompt rows exact in every decoder layer (window KV and taps): 384 at the
default 2670. W < 2414 (128 = upstream / DeepSeek) is approximate: every
prefill chunk keeps its own last W rows with the window clamped at
chunk_end - W, as upstream does, so a later chunk never reads unwritten KV.

Why 2670. A prefix hit at a 128-aligned P reads the window KV [P-128, P) and
the DSpark peek block [P, P+64), so after a trimmed prompt of length N it is
exact only if P-128 >= N - valid_tail. With e15's default SWA retention
(--prefix-cache-retention-interval 0; kv-pack's 4096 adds a checkpoint every
4096 tokens) the prompt end keeps only the replay boundary P = (N-1)//128*128.
A later prompt that shares D tokens hits P when the peek block is shared,
D >= P+64 (a resend: P+64 <= N-1), at most 256 rows back (N - P = 128).
prefix_end_fallback.py also keeps P-128, which a prompt sharing D >= P-64
tokens hits; its window [P-256, P-128) reaches N-P+256 rows back, 384 for N a
multiple of 128. Dense retention (interval None) lands these hits at the same
two P. 2670 = 2286 + 384 keeps every one. 2606 keeps the resends and the
follow-ups that diverge at or after N (they need P-128 only when N-P <= 64),
not a new final question that diverges in the last 64 tokens past P with
N-P > 64. Boot I1: N = 32591, P = 32512, the same document with another
question shared 32574 tokens (62 past P); its fallback window starts at 32256,
335 rows back. Divergence before P-64 lands below the frontier: see Prefix
caching.

Junction anchors (exact W only). When e15 flags a shared-prefix junction J
(Request.shared_prefix_boundary: the full-attention groups matched J tokens an
SWA group did not, e.g. a second request that shares only a long system prompt
with a cached one) and the prefill trims, the request also keeps the run
[J - 2542, J) (from the hit on; 2542 = 18*127 + 2*128), its window clamped at
the run start. The SWA KV of every decoder layer and the DSpark taps are then
exact on [J-256, J): the window of a hit at J and of its fallback J-128. The
scheduler exempts those blocks and the peek block [J, J+64), whose content a
hit drops, from the frontier, so a third request hits J as with stock e15. A
chunk keeps the span of its kept rows (one segment per request: the attention
builders size per-request state by max_num_seqs), so when the run ending at J
and the end window share a chunk the gap between them runs too. The scheduler
hands J to the workers on NewRequestData (ANCHOR_ATTR, pickled to every rank);
warmup requests built on a worker carry none.

Prefix caching. Every SWA KV-cache group of the stock grouping holds decoder
or DSpark layers (group 0 = layers 0/15/30), so the SWA blocks of trimmed or
inexact rows must never serve a hit. The scheduler side records each
request's frontier at admission (allocate_slots of a non-running request):
prompt_len - VALID_TAIL when prompt_len - hit > W, else 0, and masks every SWA
group's block whose first row is below it (outside a junction anchor's blocks)
out of the prefix cache (BlockPool.cache_full_blocks block_mask). The mask is ANDed with e15's own
(SlidingWindowManager.reachable_block_mask), so the retention checkpoints of
--prefix-cache-retention-interval and the shared-prefix junction tails below
the frontier are never cached either. Full-attention groups (compressed KV,
index K, compressor rings) are written for every row and stay cached. e15
hits an SWA spec group at a 128-aligned P only where every SWA group holds the
2 window blocks [P-128, P) and the peek block [P, P+64) of matched tokens, so
a hit lands only where the window KV is valid. A later prompt that diverges
from a trimmed one (length N) at D gets an SWA hit from it only at a boundary
frontier + 128 <= P <= D - 64: at 2670 only P or P-128 near the end, so only
when D >= P-64 (64 tokens before the replay boundary or later):
it then recomputes from P, otherwise from the start (or from a boundary an
untrimmed request cached, or a junction anchor), never from a stale
checkpoint. A second request that shares only a long system prompt with a
trimmed one misses (as with stock e15) and anchors the junction; the third
hits it. Under kv-pack's 4096 interval the checkpoints below the frontier are
lost. Prompt-logprob and pooling requests keep every row and are never masked.

Gates (worker; LOG_DISARMED names the first that fails): one DeepseekV4Model
holding all 40 layers, no SP / PCP / DCP / DP / ubatching / EPLB / LoRA /
routed-expert return / fast MoE cold start / adaptive verification, Engram
layers before the cut, layers past the cut neither KV sources nor Engram,
the SM120 FlashInfer attention class past the cut (its SWA index rows are
what the clamp rewrites), fp8_ds_mla SWA cache, no drafter or DSpark with a
window <= the target's, 0 or >= 128 rows. Install is atomic: the scheduler
side and the worker side arm together in every process or not at all.

Top-level imports are stdlib only. sitecustomize calls install().
"""

from __future__ import annotations

import os

# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dsv41: decoder replay armed"
LOG_DISARMED = "dsv41: decoder replay DISABLED ->"

ENV = "DSV41_DECODER_REPLAY_ROWS"
DEFAULT_ROWS = 2670
# Request attribute (scheduler process), set at admission: (first prompt row whose SWA
# blocks may be cached, junction anchor J or 0).
ATTR = "_dsv41_replay_frontier"
# NewRequestData attribute: the request's junction anchor J (0 = none), scheduler -> workers.
ANCHOR_ATTR = "_dsv41_replay_anchor"
# Rows per dummy-Q chunk in forward_kv (the e15 insert op also writes a padded Q).
INSERT_CHUNK = 1024
# First replay builds that check the SWA rows are the stock causal window layout.
VERIFY_BUILDS = 4
CENSUS_FROM = 64

_STATE = {"armed": False, "verify_left": VERIFY_BUILDS, "steps": 0, "rows": 0, "kept": 0, "anchors": 0,
          "admitted_anchors": 0}


# ---------------------------------------------------------------- pure helpers


def rows_from_env(env=None) -> int:
    """W from the env: unset/empty = DEFAULT_ROWS, 0 = off. Rejects negatives and junk."""
    env = os.environ if env is None else env
    raw = (env.get(ENV, "") or "").strip()
    if not raw:
        return DEFAULT_ROWS
    rows = int(raw)
    if rows < 0:
        raise ValueError(f"{ENV}={rows} is negative")
    return rows


def geometry(kv_source_layer_ids, num_hidden_layers: int, sliding_window: int) -> tuple[int, int, int]:
    """(cut, layers after the cut, window)."""
    cut = max(int(i) for i in kv_source_layer_ids)
    return cut, int(num_hidden_layers) - cut - 1, int(sliding_window)


def exact_min_rows(window: int, after: int) -> int:
    """Smallest end-anchored W that keeps H_end(N-1), the decode window KV and the taps exact."""
    return window + (after - 1) * (window - 1)


def mode_for(rows: int, window: int, after: int) -> str | None:
    """"exact" (end-anchored), "approx" (per chunk, upstream), None (off)."""
    if rows == 0:
        return None
    if rows < window:
        raise ValueError(f"{ENV}={rows} is below the {window}-row window (0 = off)")
    return "exact" if rows >= exact_min_rows(window, after) else "approx"


def valid_tail(rows: int, window: int, after: int) -> int:
    """Prompt-end rows whose SWA KV (every decoder layer) and taps a hit may reuse.

    Exact: the rows no truncation reaches, W - (after-1)*(window-1). Approximate:
    the decode window the request itself continued from.
    """
    if rows >= exact_min_rows(window, after):
        return rows - (after - 1) * (window - 1)
    return window


def anchor_rows(window: int, after: int) -> int:
    """Rows of a junction run [J - this, J): exact on [J - 2*window, J), the windows of a hit
    at J and at J - window (the prefix-end fallback)."""
    return (after - 1) * (window - 1) + 2 * window


def trims(prompt_len: int, hit: int, rows: int) -> bool:
    """Whether a prefill from ``hit`` to ``prompt_len`` can skip decoder rows."""
    return rows > 0 and prompt_len - hit > rows


def frontier(prompt_len: int, hit: int, keeps_rows: bool, rows: int, tail: int) -> int:
    """First prompt row whose SWA blocks may enter the prefix cache (0 = every row)."""
    if keeps_rows or not trims(prompt_len, hit, rows):
        return 0
    return max(0, prompt_len - tail)


def admission(prompt_len: int, hit: int, keeps_rows: bool, junction: int, rows: int, tail: int,
              anchored: bool) -> tuple[int, int]:
    """(frontier, junction anchor) of a request at admission. The anchor is e15's junction
    when the prefill trims, anchors are on (exact W) and the junction lies past the hit."""
    front = frontier(prompt_len, hit, keeps_rows, rows, tail)
    return front, (junction if front and anchored and hit < junction < prompt_len else 0)


def anchor_of(request) -> int:
    """The junction anchor admission recorded on a scheduler Request (0 = none)."""
    plan = getattr(request, ATTR, None)
    return int(plan[1]) if plan else 0


def block_mask(num_cached_blocks: int, num_full_blocks: int, block_size: int, front: int,
               anchor: int = 0, before: int = 0) -> list[bool] | None:
    """Mask over blocks[num_cached_blocks:num_full_blocks]: False where a row is below ``front``,
    except, with a junction anchor, blocks inside [anchor - before, anchor) and the peek block
    starting at the anchor (a hit at the anchor matches it and drops it)."""
    if front <= 0 or num_full_blocks <= num_cached_blocks:
        return None

    def keep(b: int) -> bool:
        lo = b * block_size
        if lo >= front:
            return True
        return bool(anchor) and anchor - before <= lo and (lo + block_size <= anchor or lo == anchor)

    mask = [keep(b) for b in range(num_cached_blocks, num_full_blocks)]
    return None if all(mask) else mask


def and_masks(a: list[bool] | None, b: list[bool] | None) -> list[bool] | None:
    if a is None:
        return b
    if b is None:
        return a
    assert len(a) == len(b)
    return [x and y for x, y in zip(a, b)]


def kept_runs(rows: int, prompt_len: int, hit: int, anchor: int = 0, anchor_len: int = 0) -> list:
    """Exact mode, a prefill that trims: its decoder-kept position runs [(first, end, clamp)]
    in order (end None = open). The end window [prompt_len - W, ...) and, with a junction
    anchor J past the hit, [max(hit, J - anchor_len), J), merged when they meet. A run's
    window is clamped at its first row, or not at all (0) when it starts at the hit."""
    s = prompt_len - rows
    runs = []
    if anchor_len and anchor > hit:
        a = max(hit, anchor - anchor_len)
        if anchor >= s:
            s = min(s, a)
        else:
            runs.append((a, anchor, a if a > hit else 0))
    runs.append((s, None, s if s > hit else 0))
    return runs


def plan_batch(mode, rows, window, prompt_len, hit, keeps, prefilling, start, qlen, anchor=None, anchor_len=0):
    """Per batch request (numpy arrays): (kept rows, window clamp start, needs replay,
    first kept row as an offset into the chunk).

    The kept rows of a request's chunk [start, start+qlen) are [start+lo, start+lo+kept).
    exact: the runs of kept_runs (the end window, and a junction run when the request
    has an anchor); a chunk keeps the span from its first to its last kept row, under the
    clamp of the run holding the first. The step needs the replay path if a chunk drops
    rows or a row reads below its clamp.
    approx: each chunk longer than W keeps its last W rows, clamped at chunk_end - W
    (upstream).
    """
    import numpy as np

    qlen = np.asarray(qlen, dtype=np.int64)
    n = qlen.shape[0]
    kept = qlen.copy()
    clamp = np.zeros(n, dtype=np.int64)
    needs = np.zeros(n, dtype=bool)
    lo = np.zeros(n, dtype=np.int64)
    if not rows or mode is None:
        return kept, clamp, needs, lo
    prompt_len, hit, start = (np.asarray(x, dtype=np.int64) for x in (prompt_len, hit, start))
    keeps, prefilling = np.asarray(keeps, dtype=bool), np.asarray(prefilling, dtype=bool)
    anchor = np.zeros(n, dtype=np.int64) if anchor is None else np.asarray(anchor, dtype=np.int64)
    for i in range(n):
        q, s = int(qlen[i]), int(start[i])
        if not prefilling[i] or keeps[i] or not trims(int(prompt_len[i]), int(hit[i]), rows):
            continue
        if mode != "exact":
            if q > rows:
                lo[i], kept[i], clamp[i], needs[i] = q - rows, rows, s + q - rows, True
            continue
        runs = kept_runs(rows, int(prompt_len[i]), int(hit[i]), int(anchor[i]), anchor_len)
        segs = [(max(a, s), s + q if b is None else min(b, s + q), c) for a, b, c in runs]
        segs = [g for g in segs if g[0] < g[1]]
        if not segs:
            lo[i], kept[i], clamp[i], needs[i] = q, 0, runs[-1][2], True
            continue
        first, c = segs[0][0], segs[0][2]
        lo[i], kept[i], clamp[i] = first - s, segs[-1][1] - first, c
        needs[i] = kept[i] < q or (c > 0 and first < c + window - 1)
    return kept, clamp, needs, lo


def step_kind(kept, qlen) -> str:
    """run_tail's branch for a step on the replay path, from the plan alone (so every TP
    rank takes the same one): "skip" (no row kept: nothing runs past write_kv), "clamp"
    (every row kept: the tail runs on the whole batch) or "trim"."""
    import numpy as np

    total = int(np.asarray(kept, dtype=np.int64).sum())
    if total == 0:
        return "skip"
    return "clamp" if total == int(np.asarray(qlen, dtype=np.int64).sum()) else "trim"


def kept_rows(query_start_loc, kept, lo=None):
    """Batch rows of each request's kept rows [lo[i], lo[i] + kept[i]) of its chunk (lo None:
    the last kept[i] rows), in batch order (int64)."""
    import numpy as np

    kept = np.asarray(kept, dtype=np.int64)
    qsl = np.asarray(query_start_loc, dtype=np.int64)[: kept.shape[0] + 1]
    first_row = qsl[1:] - kept if lo is None else qsl[:-1] + np.asarray(lo, dtype=np.int64)
    total = int(kept.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    first = np.cumsum(kept) - kept
    return np.repeat(first_row, kept) + (np.arange(total) - np.repeat(first, kept))


def clamp_rows_reference(indices, lens, positions, clamp, window: int):
    """numpy reference of clamp_swa_rows: the stock causal row [max(pos-w+1,0), pos]
    becomes [max(that, clamp), pos], shifted to the row start, -1 after."""
    import numpy as np

    out_i, out_l = indices.copy(), lens.copy()
    width = indices.shape[-1]
    for t in range(lens.shape[0]):
        if lens[t] <= 0:
            continue
        start = max(int(positions[t]) - (window - 1), 0)
        shift = max(int(clamp[t]) - start, 0)
        n = max(int(lens[t]) - shift, 0)
        row = np.full(width, -1, dtype=indices.dtype)
        row[:n] = indices[t, shift : shift + n]
        out_i[t], out_l[t] = row, n
    return out_i, out_l


# --------------------------------------------------------------- forward rewrite

LOOP_OLD = """        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
"""
LOOP_NEW = """        # dsv41 decoder_replay: the loop stops at the cut when this step replays.
        _dsv41_stop = _dsv41_decoder_replay.stop_layer(self)
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, _dsv41_stop),
            start=self.start_layer,
        ):
"""
TAIL_OLD = """        if layer is not None:
            # Reuse if the last layer was captured as an aux hidden state
"""
TAIL_NEW = """        if _dsv41_stop < self.end_layer:
            # dsv41 decoder_replay: the cut layer writes every row's KV, the rest
            # of it and the layers after it run on the kept rows.
            hidden_states, pre_mix, _dsv41_aux = _dsv41_decoder_replay.run_tail(
                self, hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual
            )
            aux_hidden_states.extend(_dsv41_aux)
        elif layer is not None:
            # Reuse if the last layer was captured as an aux hidden state
"""


def patch_forward_source(src: str) -> str:
    """DeepseekV4Model.forward with the layer loop split at the cut (verbatim anchors)."""
    if "super()" in src or "__class__" in src:
        raise ValueError("DeepseekV4Model.forward needs its class scope")
    for old, new, what in ((LOOP_OLD, LOOP_NEW, "layer loop"), (TAIL_OLD, TAIL_NEW, "final mhc post")):
        if src.count(old) != 1:
            raise ValueError(f"{what} of DeepseekV4Model.forward not found verbatim")
        src = src.replace(old, new)
    return src


def original_function(fn, filenames, name: str):
    """The function compiled from one of ``filenames`` named ``name`` under any wrapper chain
    (``__wrapped__`` or a closure cell, e.g. the step census and ar_l2_prefetch wrappers)."""
    todo, seen = [fn], set()
    while todo:
        f = todo.pop()
        if f is None or id(f) in seen:
            continue
        seen.add(id(f))
        code = getattr(f, "__code__", None)
        if code is not None and code.co_filename in filenames and code.co_name == name:
            return f
        todo.append(getattr(f, "__wrapped__", None))
        for cell in getattr(f, "__closure__", None) or ():
            try:
                inner = cell.cell_contents
            except ValueError:  # empty cell
                continue
            if callable(inner):
                todo.append(inner)
    return None


# ----------------------------------------------------------- scheduler guard


class CacheGuard:
    """Scheduler side: frontiers and junction anchors at admission, SWA block masks at caching.

    ``before`` = rows below a junction anchor that its run makes exact (2 * window), 0 when
    anchors are off (approximate W)."""

    def __init__(self, rows: int, tail: int, swa_groups, before: int = 0) -> None:
        self.rows, self.tail, self.swa_groups, self.before = rows, tail, frozenset(swa_groups), before

    @staticmethod
    def _keeps(request) -> bool:
        params = getattr(request, "sampling_params", None)
        return params is None or getattr(params, "prompt_logprobs", None) is not None

    def frontier(self, request, hit: int) -> int:
        return frontier(int(request.num_prompt_tokens), int(hit), self._keeps(request), self.rows, self.tail)

    def admit(self, request, hit: int) -> tuple[int, int]:
        """(frontier, anchor) from e15's Request.shared_prefix_boundary, set just before."""
        junction = int(getattr(request, "shared_prefix_boundary", 0) or 0)
        return admission(int(request.num_prompt_tokens), int(hit), self._keeps(request), junction,
                         self.rows, self.tail, bool(self.before))

    def mask(self, request, num_cached_blocks, num_full_blocks, block_size, group_id, given):
        if group_id not in self.swa_groups:
            return given
        plan = getattr(request, ATTR, None)
        if plan is None:  # never admitted through allocate_slots: assume no hit, no anchor
            plan = (self.frontier(request, 0), 0)
        front, anchor = plan
        return and_masks(given, block_mask(num_cached_blocks, num_full_blocks, block_size, front, anchor, self.before))


def scheduler_guard(vllm_config, kv_cache_config, rows: int) -> CacheGuard | None:
    from vllm.v1.kv_cache_interface import SlidingWindowSpec, iter_layer_specs

    mc = vllm_config.model_config
    cfg = getattr(mc, "hf_text_config", None) or mc.hf_config
    ids = getattr(cfg, "kv_source_layer_ids", None)
    if not ids or not getattr(cfg, "sliding_window", None):
        return None
    _, after, window = geometry(ids, cfg.num_hidden_layers, cfg.sliding_window)
    try:
        mode = mode_for(rows, window, after)
    except ValueError:  # below the window: the workers disarm with the reason, nothing trims
        return None
    if mode is None:
        return None
    groups = [
        i
        for i, g in enumerate(kv_cache_config.kv_cache_groups)
        if any(isinstance(s, SlidingWindowSpec) for s in iter_layer_specs(g.kv_cache_spec))
    ]
    return CacheGuard(rows, valid_tail(rows, window, after), groups, 2 * window if mode == "exact" else 0)


# --------------------------------------------------------------- worker side


def _model_module():
    import importlib

    return importlib.import_module("vllm.models.deepseek_v4_1.nvidia.model")


def find_core(model):
    """The one DeepseekV4Model inside the served model (the drafter is not one)."""
    cls = _model_module().DeepseekV4Model
    cores = [m for m in model.modules() if type(m) is cls]
    return cores[0] if len(cores) == 1 else None


def unsupported_reason(vllm_config, core, rows: int) -> str | None:
    """Why this rank must not replay, or None."""
    from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import DeepseekV4FlashInferSM120Attention

    if core is None:
        return "no single DeepseekV4Model in the served model"
    cfg = core.config
    pc = vllm_config.parallel_config
    cut, after, window = geometry(cfg.kv_source_layer_ids, cfg.num_hidden_layers, cfg.sliding_window)
    try:
        mode_for(rows, window, after)
    except ValueError as exc:
        return str(exc)
    spec = vllm_config.speculative_config
    checks = (
        (core.start_layer != 0 or core.end_layer != cfg.num_hidden_layers or pc.pipeline_parallel_size != 1,
         "the rank must hold every layer (no pipeline parallel)"),
        (cut >= core.end_layer - 1, f"no layer after the last KV source {cut}"),
        (bool(core.use_sequence_parallel) or getattr(pc, "prefill_context_parallel_size", 1) != 1
         or getattr(pc, "decode_context_parallel_size", 1) != 1 or pc.data_parallel_size != 1
         or bool(getattr(pc, "use_ubatching", False)) or bool(getattr(pc, "enable_eplb", False)),
         "sequence / context / data parallel, ubatching or EPLB"),
        (any(int(i) >= cut for i in (getattr(cfg, "engram_layer_ids", None) or ())),
         "an Engram layer at or after the cut"),
        (vllm_config.lora_config is not None, "LoRA"),
        (bool(getattr(vllm_config.model_config, "enable_return_routed_experts", False)), "routed-expert return"),
        (bool(getattr(vllm_config.compilation_config, "fast_moe_cold_start", False)),
         "fast_moe_cold_start (MoE layer names index the forward context)"),
        (spec is not None and spec.method != "dspark", f"drafter {getattr(spec, 'method', None)} (only DSpark)"),
        (spec is not None and bool(getattr(spec, "enable_adaptive_verification", False)), "adaptive verification"),
    )
    for bad, reason in checks:
        if bad:
            return reason
    if spec is not None:
        dcfg = spec.draft_model_config.hf_config
        dwin = getattr(dcfg, "sliding_window", None)
        if dwin is None or int(dwin) > window:
            return f"the drafter window {dwin} exceeds the target's {window}"
    head = core.layers[cut]
    if not head.attn.is_kv_source or head.attn.compress_ratio != 1 or head.attn.compressor is None:
        return f"layer {cut} is not a ratio-1 KV source"
    for idx in range(cut, core.end_layer):
        layer = core.layers[idx]
        if idx > cut and (layer.attn.is_kv_source or layer.engram is not None):
            return f"layer {idx} after the cut is a KV source or an Engram layer"
        if not isinstance(layer.attn, DeepseekV4FlashInferSM120Attention):
            return f"layer {idx} runs {type(layer.attn).__name__} (the clamp is wired for the SM120 class)"
        if layer.attn.kv_cache_dtype != "fp8_ds_mla":
            return f"layer {idx} SWA cache {layer.attn.kv_cache_dtype} (needs fp8_ds_mla)"
    return None


def clamp_swa_rows(torch, indices, lens, positions, clamp, window: int, verify: bool) -> None:
    """In place: each stock causal SWA row [max(pos-w+1,0), pos] starts at max(that, clamp).

    The row's slot ids are ascending positions, so dropping the first ``shift``
    entries is the clamped row; no block-table lookup. ``verify`` first checks
    the rows have the stock layout (len = pos + 1 - start) and raises otherwise.
    """
    n = lens.shape[0]
    if n == 0:
        return
    width = indices.shape[-1]
    flat = indices.view(n, width)
    pos = positions.to(torch.int64)
    start = (pos - (window - 1)).clamp_min(0)
    if verify:
        valid = lens > 0
        if not torch.equal(lens[valid].to(torch.int64), (pos + 1 - start)[valid]):
            raise RuntimeError("SWA index rows are not the stock causal window layout")
    shift = torch.where(lens > 0, (clamp.to(torch.int64) - start).clamp_min(0), 0)
    newlen = (lens.to(torch.int64) - shift).clamp_min(0)
    col = torch.arange(width, device=flat.device, dtype=torch.int64)
    src = (col[None, :] + shift[:, None]).clamp_max(width - 1)
    rows = flat.gather(1, src)
    flat.copy_(torch.where(col[None, :] < newlen[:, None], rows, torch.full_like(rows, -1)))
    lens.copy_(newlen.to(lens.dtype))


class ReplayBatch:
    """One eager step's kept rows (None for "skip") and step_kind; ``build(positions)``
    makes the sub-batch's forward context. It is built at the cut, after write_kv: it
    reuses the step's metadata builders, whose buffers layers before the cut have
    consumed."""

    __slots__ = ("attn_metadata", "rows", "kind", "build")

    def __init__(self, attn_metadata, rows, kind, build) -> None:
        self.attn_metadata, self.rows, self.kind, self.build = attn_metadata, rows, kind, build


class ReplayRunner:
    """Worker side: per-step plan (model state) and the replay tail (model forward)."""

    def __init__(self, vllm_config, core, rows: int, max_num_reqs: int, device) -> None:
        import numpy as np

        cfg = core.config
        self.vllm_config, self.core, self.rows, self.device = vllm_config, core, rows, device
        self.cut, self.after, self.window = geometry(cfg.kv_source_layer_ids, cfg.num_hidden_layers, cfg.sliding_window)
        self.mode = mode_for(rows, self.window, self.after)
        self.end = core.end_layer
        self.batch: ReplayBatch | None = None
        self.prompt_len = np.zeros(max_num_reqs, dtype=np.int64)
        self.hit = np.zeros(max_num_reqs, dtype=np.int64)
        self.keeps = np.ones(max_num_reqs, dtype=bool)
        self.anchor = np.zeros(max_num_reqs, dtype=np.int64)
        self.anchor_len = anchor_rows(self.window, self.after) if self.mode == "exact" else 0
        tail = [core.layers[i].attn for i in range(self.cut, self.end)]
        self.cut_swa = tail[0].swa_cache_layer.prefix
        self.clamp_swa = {a.swa_cache_layer.prefix for a in tail[1:]}
        self.prefixes = {a.swa_cache_layer.prefix for a in tail}
        for a in tail:
            if a.compressed_cache_prefix is not None:
                self.prefixes.add(a.compressed_cache_prefix)
            if a.indexer is not None:
                self.prefixes.add(a.indexer.k_cache.prefix)
        self._groups = None  # (attn groups per KV group, needed group ids, cut-SWA copy or None)
        self.disarmed: str | None = None

    def describe(self) -> str:
        return (
            f"W={self.rows} {self.mode} (exact from {exact_min_rows(self.window, self.after)}), "
            f"cut={self.cut}, layers {self.cut}-{self.end - 1} on kept rows in eager prefill steps, "
            f"valid tail {valid_tail(self.rows, self.window, self.after)} rows, "
            f"junction runs {self.anchor_len} rows"
        )

    def add_request(self, req_index: int, new_req_data) -> None:
        self.prompt_len[req_index] = int(new_req_data.prompt_len)
        self.hit[req_index] = int(new_req_data.num_computed_tokens)
        params = new_req_data.sampling_params
        self.keeps[req_index] = params is None or params.prompt_logprobs is not None
        # The scheduler's anchor (scheduler_guard admission); warmup requests built here have none.
        anchor = int(getattr(new_req_data, ANCHOR_ATTR, 0) or 0) if self.anchor_len else 0
        self.anchor[req_index] = anchor
        if anchor and not _STATE["anchors"]:
            print(
                "dsv41: decoder replay junction anchor: first at J=%d (prompt %d, hit %d): rows [%d, %d) run "
                "layers %d-%d" % (anchor, self.prompt_len[req_index], self.hit[req_index],
                                  max(self.hit[req_index], anchor - self.anchor_len), anchor, self.cut, self.end - 1),
                flush=True,
            )
        _STATE["anchors"] += bool(anchor)

    def _disarm(self, reason: str) -> None:
        self.disarmed = reason
        print("dsv41: decoder replay DISABLED -> %s; every step runs every layer" % reason, flush=True)

    def _sub_groups(self, attn_groups):
        """Attention groups the replay layers read, checked once (both ranks agree)."""
        if self._groups is not None:
            return self._groups
        from dataclasses import replace

        from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadataBuilder

        groups, needed, cut_copy = [], set(), None
        for gid, gs in enumerate(attn_groups):
            mine = [g for g in gs if not self.prefixes.isdisjoint(g.layer_names)]
            groups.append(mine)
            if mine:
                needed.add(gid)
            for g in mine:
                for b in g.metadata_builders:
                    if isinstance(b, DeepseekSparseSWAMetadataBuilder) and (
                        getattr(b, "max_image_tokens", 0) or b.window_size != self.window
                    ):
                        raise RuntimeError(f"SWA builder of group {gid} is not the plain {self.window}-row window")
                if self.cut_swa in g.layer_names and not self.clamp_swa.isdisjoint(g.layer_names):
                    # The cut layer reads unclamped rows: one builder of its own.
                    cut_copy = (gid, replace(g, metadata_builders=[]))
                    cut_copy[1].create_metadata_builders(
                        self.vllm_config,
                        self.device,
                        kernel_block_size=getattr(g.metadata_builders[0], "kernel_block_size", None),
                    )
        self._groups = (groups, needed, cut_copy)
        return self._groups

    def plan_step(self, input_batch):
        """(query_start_loc, kept, clamp, step_kind, first kept row offsets) of an eager prefill
        step that takes the replay path, else None. Reads only the scheduler's batch (host
        arrays) and what add_request stored from NewRequestData, so every TP rank gets the
        same answer."""
        import numpy as np

        num_reqs = input_batch.num_reqs
        qsl = np.asarray(input_batch.query_start_loc_np[: num_reqs + 1], dtype=np.int64)
        idx = input_batch.idx_mapping_np[:num_reqs]
        kept, clamp, needs, lo = plan_batch(
            self.mode, self.rows, self.window,
            self.prompt_len[idx], self.hit[idx], self.keeps[idx],
            input_batch.is_prefilling_np[:num_reqs],
            input_batch.num_computed_tokens_np[:num_reqs],
            np.diff(qsl),
            anchor=self.anchor[idx], anchor_len=self.anchor_len,
        )
        num_tokens = int(qsl[-1])
        # run_tail maps the forward's rows 1:1 onto the step's tokens (eager steps are unpadded).
        if not needs.any() or int(getattr(input_batch, "num_tokens_after_padding", num_tokens)) != num_tokens:
            return None
        return qsl, kept, clamp, step_kind(kept, np.diff(qsl)), lo

    def prepare(self, input_batch, attn_metadata, block_tables, slot_mappings, attn_groups, kv_cache_config) -> None:
        if self.disarmed is not None:
            return
        try:
            self._sub_groups(attn_groups)
        except Exception as exc:  # noqa: BLE001 - deterministic (config), both ranks agree
            self._disarm(repr(exc))
            return
        plan = self.plan_step(input_batch)
        if plan is None:
            return
        import numpy as np
        import torch

        qsl, kept, clamp, kind, lo = plan
        num_tokens = int(qsl[-1])
        rows_np = kept_rows(qsl, kept, lo)
        _STATE["steps"] += 1
        _STATE["rows"] += num_tokens
        _STATE["kept"] += int(rows_np.size)
        if _STATE["steps"] == 1:
            print(
                "dsv41: decoder replay engaged: first trimmed step %d tokens -> %d kept rows (%s)"
                % (num_tokens, rows_np.size, self.describe()),
                flush=True,
            )
        elif _STATE["steps"] >= CENSUS_FROM and _STATE["steps"] & (_STATE["steps"] - 1) == 0:
            print(
                "[replay-census] trimmed steps=%d rows=%d kept=%d (%.1f%%)"
                % (_STATE["steps"], _STATE["rows"], _STATE["kept"], 100.0 * _STATE["kept"] / max(_STATE["rows"], 1)),
                flush=True,
            )
        rows = None if kind == "skip" else torch.from_numpy(rows_np).to(self.device, non_blocking=True)

        # Rows of each request's chunk after its kept segment (a run ending at a junction).
        after = (np.diff(qsl) - lo - kept).astype(np.int64)

        def build(positions):
            return self._build(input_batch, block_tables, slot_mappings, attn_groups, kv_cache_config,
                               kept, clamp, after, rows, positions)

        self.batch = ReplayBatch(attn_metadata, rows, kind, build)

    def _build(self, input_batch, block_tables, slot_mappings, attn_groups, kv_cache_config,
               kept, clamp, after, rows, positions):
        import numpy as np
        import torch
        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor, create_forward_context, get_forward_context
        # attn_utils' own build_attn_metadata, not the name model_states.default calls (kv-pack's
        # swa_meta_batched wraps that one); the base ModelSpecificAttnMetadata adds no kwargs.
        from vllm.v1.worker.gpu.attn_utils import build_attn_metadata, build_slot_mappings_by_layer
        from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata

        groups, needed, cut_copy = self._sub_groups(attn_groups)
        sel = np.nonzero(kept > 0)[0]
        kept_sel = kept[sel]
        qsl_np = np.zeros(sel.size + 1, dtype=np.int32)
        np.cumsum(kept_sel, out=qsl_np[1:])
        qsl_cpu = torch.from_numpy(qsl_np)
        sel_cpu = torch.from_numpy(sel.astype(np.int64))
        sel_gpu = sel_cpu.to(self.device, non_blocking=True)
        seq_cpu = input_batch.seq_lens_cpu_upper_bound[sel_cpu]
        seq_gpu = input_batch.seq_lens.index_select(0, sel_gpu)
        short = after[sel]
        if short.any():  # a segment that ends before its chunk does (a run ending at a junction)
            short_cpu = torch.from_numpy(short.astype(np.int64))
            seq_cpu = seq_cpu - short_cpu.to(seq_cpu.dtype)
            seq_gpu = seq_gpu - short_cpu.to(self.device, non_blocking=True).to(seq_gpu.dtype)
        slots = slot_mappings.index_select(1, rows)
        tables = tuple(t.index_select(0, sel_gpu) if g in needed else t for g, t in enumerate(block_tables))
        common = dict(
            num_reqs=int(sel.size),
            num_tokens=int(rows.shape[0]),
            query_start_loc_gpu=qsl_cpu.to(self.device, non_blocking=True),
            query_start_loc_cpu=qsl_cpu,
            max_query_len=int(kept_sel.max()),
            seq_lens=seq_gpu,
            max_seq_len=int(seq_cpu.max()),
            block_tables=tables,
            slot_mappings=slots,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_cpu,
            positions=positions,
            is_prefilling=torch.from_numpy(np.ascontiguousarray(input_batch.is_prefilling_np[sel])),
            # Batching that shares one plan across a call's SWA groups stays off this
            # build (swa_meta_batched.call_eligible: model_specific_attn_metadata is None).
            model_specific_attn_metadata=ModelSpecificAttnMetadata(),
        )
        meta = build_attn_metadata(attn_groups=groups, **common)
        tok_clamp_np = np.repeat(clamp[sel], kept_sel).astype(np.int32)
        if tok_clamp_np.any():
            tok_clamp = torch.from_numpy(tok_clamp_np).to(self.device, non_blocking=True)
            verify = _STATE["verify_left"] > 0
            done = set()
            for prefix in sorted(self.clamp_swa):
                m = meta.get(prefix)
                if m is None or id(m) in done:
                    continue
                done.add(id(m))
                nd, np_ = m.num_decode_tokens, m.num_prefill_tokens
                if nd:
                    clamp_swa_rows(torch, m.decode_swa_indices, m.decode_swa_lens, positions[:nd],
                                   tok_clamp[:nd], self.window, verify)
                if np_:
                    clamp_swa_rows(torch, m.prefill_swa_indices, m.prefill_swa_lens, positions[nd : nd + np_],
                                   tok_clamp[nd : nd + np_], self.window, verify)
            if verify:
                _STATE["verify_left"] -= 1
            if id(meta.get(self.cut_swa)) in done:
                if cut_copy is None:
                    raise RuntimeError(f"{self.cut_swa} shares clamped SWA metadata without a builder of its own")
                gid, copy = cut_copy
                only = [[copy] if g == gid else [] for g in range(len(groups))]
                meta[self.cut_swa] = build_attn_metadata(attn_groups=only, **common)[self.cut_swa]
        outer = get_forward_context()
        return create_forward_context(
            meta,
            self.vllm_config,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=int(rows.shape[0])),
            slot_mapping=build_slot_mappings_by_layer(slots, kv_cache_config),
            additional_kwargs=dict(outer.additional_kwargs),
            skip_compiled=outer.skip_compiled,
        )


def stop_layer(core) -> int:
    """Where DeepseekV4Model.forward's own loop stops this forward."""
    runner = getattr(core, "_dsv41_replay", None)
    if runner is None or runner.batch is None:
        return core.end_layer
    from vllm.forward_context import get_forward_context

    if get_forward_context().attn_metadata is not runner.batch.attn_metadata:
        runner.batch = None  # not this step's batch (a dummy run): run every layer
        return core.end_layer
    return runner.cut


def run_tail(core, hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual):
    """The cut layer's KV for every row, then layers cut.. on the kept rows.

    Returns the full-row (post-final-mHC hidden states, last pre-mix, aux hidden
    states of layers past the cut). A trimmed row carries its stream at the cut
    (write_kv's mHC post), the cut's pre-mix and that stream's mean as every aux
    state: finite, in-distribution rows for the final collapse / norm, the
    logits and the DSpark taps, which run on every row of the step. Their
    results for trimmed rows are discarded: a trimmed row is at most the last
    row of a chunk that does not end its prompt (its sample is dropped), and
    the drafter context KV it writes sits below the request's frontier. With
    no row kept ("skip": every request is a prefill chunk wholly before its
    window) nothing runs past write_kv: no decoder layer, metadata build or
    collective, so no op launches on 0 rows. step_kind comes from the plan
    alone, so both TP ranks take the same branch.
    """
    runner = core._dsv41_replay
    batch, runner.batch = runner.batch, None
    cut, end = runner.cut, core.end_layer
    rows = batch.rows
    if batch.kind == "clamp":  # every row runs the tail, in batch order
        stream = None
        h, r, pm, rm, pre, pos, ids = hidden_states, residual, post_mix, res_mix, pre_mix, positions, input_ids
    else:
        stream = core.layers[cut].write_kv(hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual)
        if batch.kind == "skip":
            filler = stream.mean(dim=1)
            return stream, pre_mix, [filler for i in range(cut + 1, end + 1) if i in core.aux_hidden_state_layers]
        h, r, pm, rm, pre, pos = (t.index_select(0, rows) for t in (hidden_states, residual, post_mix, res_mix,
                                                                     pre_mix, positions))
        ids = None if input_ids is None else input_ids.index_select(0, rows)
    from vllm.forward_context import override_forward_context

    m = _model_module()
    context = batch.build(pos)
    aux, recon = [], None
    with override_forward_context(context):
        for idx in range(cut, end):
            h, r, pm, rm, pre = core.layers[idx](h, pos, ids, pre, pm, rm, r, None, None)
            if idx + 1 in core.aux_hidden_state_layers:
                recon = m.mhc_post_tilelang(h, r, pm, rm)
                aux.append(recon.mean(dim=1))
        h = recon if end in core.aux_hidden_state_layers else m.mhc_post_tilelang(h, r, pm, rm)
    if stream is None:
        return h, pre, aux
    filler = stream.mean(dim=1) if aux else None
    return (stream.index_copy_(0, rows, h), pre_mix.index_copy(0, rows, pre),
            [filler.index_copy(0, rows, a) for a in aux])


def insert_swa_kv(torch, attn, kv, positions, slot_mapping) -> None:
    """SWA KV insert of _fused_qnorm_rope_kv_insert's fp8_ds_mla path with a dummy Q
    (e15 has no KV-only op; dspark._insert_context_kv does the same), chunked so
    the dummy and the op's padded Q stay small."""
    cache = attn.swa_cache_layer.kv_cache
    if cache.dtype != torch.uint8:
        raise RuntimeError(f"SWA cache dtype {cache.dtype}: only the fp8_ds_mla layout is wired")
    cache_2d = cache.view(cache.shape[0], -1)
    n = kv.shape[0]
    dummy = torch.zeros((min(n, INSERT_CHUNK), attn.n_local_heads, attn.head_dim), dtype=kv.dtype, device=kv.device)
    for i in range(0, n, INSERT_CHUNK):
        j = min(n, i + INSERT_CHUNK)
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            dummy[: j - i],
            kv[i:j],
            cache_2d,
            slot_mapping[i:j],
            positions[i:j],
            attn.rotary_emb.cos_sin_cache,
            attn.padded_heads,
            attn.eps,
            attn.swa_cache_layer.block_size,
            False,
        )


def _forward_kv(self, positions, hidden_states) -> None:
    """DeepseekV4Attention: write only the KV forward writes (SWA KV, compressed KV,
    index K); no wq_b, indexer top-k or attention (0.32rc1 attention.forward_kv)."""
    import torch
    from vllm.forward_context import get_forward_context

    attn_metadata = get_forward_context().attn_metadata
    qr_kv, kv_score, _ = self._run_parallel_input_projections(hidden_states)
    _, _, kv = self._split_qkv_and_norm(qr_kv)
    if isinstance(attn_metadata, dict):
        insert_swa_kv(torch, self, kv, positions, attn_metadata[self.swa_cache_layer.prefix].slot_mapping)
    if self.compressor is not None:
        latent = self.compressor(kv_score, positions)
        self.compressor.insert_cache(latent, positions, self.rotary_emb)
        if self.indexer is not None and self.indexer.owns_k:
            self.indexer._produce_k(latent, positions, self.indexer_rotary_emb)


def _write_kv(self, x, positions, input_ids, pre_mix, post_mix, res_mix, residual):
    """DeepseekV4DecoderLayer: forward's mHC post + attention pre, then forward_kv only
    (0.32rc1 DeepseekV4DecoderLayer.write_kv on the e15 mHC calls). Returns the
    stream at this layer's input (the mHC post), a fresh tensor."""
    assert self.engram is None and not self.use_sequence_parallel
    m = _model_module()
    residual = m.mhc_post_tilelang(x, residual, post_mix, res_mix)
    _, _, x, _ = m.mhc_pre_delayed_tilelang(
        residual,
        self.hc_attn_fn,
        self.hc_attn_scale,
        self.hc_attn_base,
        self.rms_norm_eps,
        self.hc_eps,
        self.hc_eps,
        self.hc_post_alpha,
        self.hc_sinkhorn_iters,
        pre_mix=pre_mix,
        norm_weight=self.attn_norm.weight,
        norm_eps=self.attn_norm.variance_epsilon,
    )
    self.attn.forward_kv(positions, x)
    return residual


# ------------------------------------------------------------------- install


def install(env=None) -> str:
    """Arm both sides (every process) or nothing. "off" when the env is 0."""
    import functools
    import inspect
    import linecache
    import sys
    import textwrap

    rows = rows_from_env(env)
    if rows == 0:
        return "off"
    from vllm.config.compilation import CUDAGraphMode
    from vllm.models.deepseek_v4_1 import attention as attn_mod
    from vllm.models.deepseek_v4_1.nvidia import model_state as ms_mod
    from vllm.v1.core import block_pool as bp_mod
    from vllm.v1.core import kv_cache_manager as kvm_mod
    from vllm.v1.core.sched import output as out_mod
    from vllm.v1.core.sched import scheduler as sched_mod
    from vllm.v1.request import RequestStatus

    m = _model_module()
    model_cls = m.DeepseekV4Model
    # The image's forward, under any wrapper another patch put on it; its code object is
    # replaced in place, so those wrappers keep calling it.
    filename = f"{m.__file__} [dsv41 decoder_replay]"
    orig = original_function(model_cls.forward, (m.__file__, filename), "forward")
    if orig is None:
        raise ValueError("DeepseekV4Model.forward from the image's model.py not found")
    if getattr(orig, "_dsv41_replay", False):
        return "already installed"

    # -- everything is built first; nothing is assigned until the end.
    src = textwrap.dedent(patch_forward_source(inspect.getsource(orig)))
    local: dict = {}
    exec(compile(src, filename, "exec", dont_inherit=True), m.__dict__, local)
    code = local["forward"].__code__
    if code.co_freevars or orig.__code__.co_freevars:
        raise ValueError("DeepseekV4Model.forward has free variables")

    alloc = kvm_mod.KVCacheManager.allocate_slots
    alloc_sig = inspect.signature(alloc)
    cache_full = bp_mod.BlockPool.cache_full_blocks
    for fn, names in (
        (alloc, ("request", "num_new_computed_tokens", "num_external_computed_tokens")),
        (cache_full, ("request", "num_cached_blocks", "num_full_blocks", "block_size", "kv_cache_group_id",
                      "block_mask")),
    ):
        missing = [n for n in names if n not in inspect.signature(fn).parameters]
        if missing:
            raise ValueError(f"{fn.__qualname__} lacks {missing}")
    ms_cls = ms_mod.DeepseekV41ModelState
    ms_init, ms_add, ms_attn = ms_cls.__init__, ms_cls.add_request, ms_cls.prepare_attn
    if "for_capture" not in inspect.signature(ms_attn).parameters:
        raise ValueError("DeepseekV41ModelState.prepare_attn lacks for_capture")
    sched_init = sched_mod.Scheduler.__init__
    # NewRequestData carries the junction anchor to the workers as an instance attribute: the
    # executor pickles the scheduler output (shm and ZMQ), so a plain dataclass keeps it.
    nrd_cls = out_mod.NewRequestData
    raw_from = nrd_cls.__dict__.get("from_request")
    params = getattr(nrd_cls, "__dataclass_params__", None)
    if not isinstance(raw_from, classmethod) or params is None or params.frozen or "__slots__" in nrd_cls.__dict__:
        raise ValueError("NewRequestData is not a plain dataclass with a from_request classmethod")
    from_stock = raw_from.__func__

    @functools.wraps(sched_init)
    def scheduler_init(self, *args, **kwargs):
        sched_init(self, *args, **kwargs)
        guard = scheduler_guard(self.vllm_config, self.kv_cache_config, rows)
        kvm = getattr(self, "kv_cache_manager", None)
        if guard is None or kvm is None:
            return
        kvm._dsv41_replay = guard
        kvm.block_pool._dsv41_replay = guard
        print(
            "dsv41: decoder replay prefix-cache guard armed (%d SWA groups; SWA blocks below "
            "prompt_len - %d stay out of the prefix cache when a prefill skips rows; junction anchors "
            "keep [J-%d, J+peek), 0 = none)" % (len(guard.swa_groups), guard.tail, guard.before),
            flush=True,
        )

    @functools.wraps(alloc)
    def allocate_slots(self, request, *args, **kwargs):
        guard = getattr(self, "_dsv41_replay", None)
        if guard is not None and request.status != RequestStatus.RUNNING:
            bound = alloc_sig.bind(self, request, *args, **kwargs)
            bound.apply_defaults()
            a = bound.arguments
            hit = request.num_computed_tokens + a["num_new_computed_tokens"] + a["num_external_computed_tokens"]
            plan = guard.admit(request, hit)
            if plan[1] and not _STATE["admitted_anchors"]:
                print("dsv41: decoder replay junction anchor admitted: first at J=%d (prompt %d, hit %d): SWA "
                      "blocks [J-%d, J+peek) stay cacheable" % (plan[1], request.num_prompt_tokens, hit, guard.before),
                      flush=True)
            _STATE["admitted_anchors"] += bool(plan[1])
            setattr(request, ATTR, plan)
        return alloc(self, request, *args, **kwargs)

    @functools.wraps(from_stock)
    def from_request(cls, request, *args, **kwargs):
        data = from_stock(cls, request, *args, **kwargs)
        setattr(data, ANCHOR_ATTR, anchor_of(request))
        return data

    @functools.wraps(cache_full)
    def cache_full_blocks(self, request, blocks, num_cached_blocks, num_full_blocks, block_size,
                          kv_cache_group_id, block_mask=None):
        guard = getattr(self, "_dsv41_replay", None)
        if guard is not None:
            block_mask = guard.mask(request, num_cached_blocks, num_full_blocks, block_size, kv_cache_group_id,
                                    block_mask)
        return cache_full(self, request, blocks, num_cached_blocks, num_full_blocks, block_size,
                          kv_cache_group_id, block_mask=block_mask)

    @functools.wraps(ms_init)
    def model_state_init(self, vllm_config, model, encoder_cache, device):
        ms_init(self, vllm_config, model, encoder_cache, device)
        self._dsv41_replay = None
        core = find_core(model)
        try:
            reason = unsupported_reason(vllm_config, core, rows)
        except Exception as exc:  # noqa: BLE001
            reason = repr(exc)
        if reason is not None:
            print("dsv41: decoder replay DISABLED -> %s; every step runs every layer" % reason, flush=True)
            return
        runner = ReplayRunner(vllm_config, core, rows, self.max_num_reqs, device)
        core._dsv41_replay = runner
        self._dsv41_replay = runner
        _STATE["armed"] = True
        print("dsv41: decoder replay armed (%s)" % runner.describe(), flush=True)

    @functools.wraps(ms_add)
    def add_request(self, req_index, new_req_data):
        ms_add(self, req_index, new_req_data)
        runner = getattr(self, "_dsv41_replay", None)
        if runner is not None:
            runner.add_request(req_index, new_req_data)

    @functools.wraps(ms_attn)
    def prepare_attn(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups,
                     kv_cache_config, for_capture=False):
        attn_metadata = ms_attn(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups,
                                kv_cache_config, for_capture=for_capture)
        runner = getattr(self, "_dsv41_replay", None)
        if runner is not None:
            runner.batch = None
            if cudagraph_mode == CUDAGraphMode.NONE and not for_capture and input_batch.has_prefill:
                runner.prepare(input_batch, attn_metadata, block_tables, slot_mappings, attn_groups,
                               kv_cache_config)
        return attn_metadata

    # -- assign (cannot fail from here on)
    m._dsv41_decoder_replay = sys.modules[__name__]
    attn_mod.DeepseekV4Attention.forward_kv = _forward_kv
    m.DeepseekV4DecoderLayer.write_kv = _write_kv
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    orig.__code__ = code
    orig._dsv41_replay = True
    sched_mod.Scheduler.__init__ = scheduler_init
    kvm_mod.KVCacheManager.allocate_slots = allocate_slots
    bp_mod.BlockPool.cache_full_blocks = cache_full_blocks
    nrd_cls.from_request = classmethod(from_request)
    ms_cls.__init__ = model_state_init
    ms_cls.add_request = add_request
    ms_cls.prepare_attn = prepare_attn
    return f"W={rows}: DeepseekV4Model.forward split at the cut, scheduler SWA prefix-cache guard, junction anchors"
