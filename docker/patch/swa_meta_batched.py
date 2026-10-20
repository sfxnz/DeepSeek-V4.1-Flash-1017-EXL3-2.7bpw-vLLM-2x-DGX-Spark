"""All causal SWA groups' decode window rows in one launch per step (DSV41_SWA_META_BATCHED=1).

Needs DSV41_SWA_META_FUSED=1 (swa_meta_fused.py); this is the e15 counterpart of
vLLM PR #60996's second commit ("Build sliding-window window metadata once per
step, not per group", deepseek_v41/swa_metadata.py), which makes the packed KV
grouping (kv_packed_groups.py: 43 SWA groups of one layer instead of 15 of
three) affordable. A target decode step builds one DeepseekSparseSWAMetadata
per SWA KV-cache group; their window rows differ only in the group's paged
block table and slot-mapping row. Query lens, seq lens and the token -> request
map are shared by every group of one build_attn_metadata call.

How: build_attn_metadata in vllm.v1.worker.gpu.model_states.default (the target
model state's call; the DSpark draft's non-causal builds go through the
speculator's own import and stay per group) is wrapped. Per call it lists the
causal SWA groups (SlidingWindowMLASpec, one DeepseekSparseSWAMetadataBuilder
each, same window / block size / buffer shapes) with their block tables and
slot-mapping rows. swa_meta_fused's SwaFused.decode (the fused path of a causal
pure-decode build) asks rows_written() first. The first group of the call to
get there (the leader) launches _swa_decode_batched_kernel over (tokens, groups):
for each group it writes exactly what that group's own fused launch would write
into that group's own persistent buffers (validity from its slot row, the lens
tail, the window indices through its block table), using the leader's token ->
request map (a function of the shared query_start_loc). Later groups of the
same call (same query_start_loc / seq_lens objects, decode token count, slot
count, block table and slot row) skip their launch. Every builder keeps its own
buffers, so captured graph addresses and values are the stock ones. The kernel
body is swa_meta_fused._swa_decode_fused_kernel's, line for line (a test pins
it); only the five per-group pointers are loaded from arrays. Any change to the
fused kernel's arithmetic (e.g. a replay-window clamp) must be mirrored here.

Batching is off for a call that passes model_specific_attn_metadata (a model
state that sends per-request replay metadata takes the per-group path), a
non-bool-True causal flag, fewer than two groups, or while a CUDA graph is being
captured, and until swa_meta_fused's own verification is done. Safety: the first
DSV41_SWA_META_BATCHED_VERIFY batched launches outside graph capture (default 8,
minimum 1; only launches with a valid decode token count down) first run every
group's fused launch into its buffers, snapshot them, poison them, run the
batched launch and compare every written element. A mismatch or a launch error
puts the fused values back, prints one LOG_DISARMED line and disarms for good
(every later build is per group). LOG_ENGAGED prints with the first passing
check (a boot-time dummy batch).

Top-level imports are stdlib only. decode_levers.install() calls install()
after swa_meta_fused's.
"""

from __future__ import annotations

import os

# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dsv41: swa metadata batched self-check bit-exact"
LOG_DISARMED = "dsv41: swa metadata batched DISABLED ->"

DEFAULT_VERIFY = 8
CHUNK = 1024  # the fused kernel's tail chunk
_STATE = {"armed": True, "verify_left": DEFAULT_VERIFY, "engaged": False}
# The wrapped build_attn_metadata call in flight: its plan and, once launched, its batch key.
_CALL = {"plan": None, "batch": None}


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return (env.get("DSV41_SWA_META_BATCHED", "0") or "0") == "1" and (
        env.get("DSV41_SWA_META_FUSED", "0") or "0"
    ) == "1"


def verify_calls(env=None) -> int:
    env = os.environ if env is None else env
    return max(1, int(env.get("DSV41_SWA_META_BATCHED_VERIFY", "") or DEFAULT_VERIFY))


def _disarm(exc: BaseException) -> None:
    if _STATE["armed"]:
        _STATE["armed"] = False
        print("dsv41: swa metadata batched DISABLED -> per-group builds: %r" % (exc,), flush=True)


def call_eligible(kwargs) -> bool:
    """Whether a build_attn_metadata call may batch its SWA groups (before looking at the groups)."""
    return (
        _STATE["armed"]
        and kwargs.get("causal", True) is True
        and kwargs.get("model_specific_attn_metadata") is None
        and all(kwargs.get(k) is not None for k in ("attn_groups", "kv_cache_config", "block_tables", "slot_mappings"))
    )


def swa_groups(attn_groups, kv_cache_config, swa_spec_cls, builder_cls) -> list:
    """[(group id, builder)] of the call's SWA KV-cache groups; [] when one of them cannot batch."""
    out = []
    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if not isinstance(getattr(spec, "first_spec", spec), swa_spec_cls):
            continue
        groups = attn_groups[gid] if gid < len(attn_groups) else []
        if not groups:
            continue  # nothing is built for this group in this call
        if len(groups) != 1:
            return []
        builder = groups[0].get_metadata_builder(0)
        if not isinstance(builder, builder_cls):
            return []
        out.append((gid, builder))
    return out


def shapes_agree(builders) -> bool:
    """One kernel needs one window, block size, index width, row stride and lens length."""
    keys = {
        (
            b.window_size,
            b.block_size,
            b.decode_swa_indices.shape[-1],
            b.decode_swa_indices.stride(0),
            b.decode_swa_lens.shape[0],
            str(b.decode_swa_indices.dtype),
            str(b.decode_swa_lens.dtype),
            str(b.is_valid_token.dtype),
        )
        for b in builders
    }
    return len(keys) == 1


class Plan:
    """The causal SWA groups of one build_attn_metadata call."""

    def __init__(self, members, static_ptrs, block_tables, slot_mappings):
        self.members = members  # [(gid, builder)]
        self.index = {id(b): i for i, (_, b) in enumerate(members)}
        self.static_ptrs = static_ptrs  # per group: (indices, lens, valid) addresses, fixed per builder
        self.block_tables = [block_tables[gid] for gid, _ in members]
        self.slot_mappings = slot_mappings  # [num_kv_cache_groups, num_tokens_padded]
        self.n_slots = int(slot_mappings.shape[1])
        self.slot_row_bytes = int(slot_mappings.stride(0)) * slot_mappings.element_size()
        self.slot_base = slot_mappings.data_ptr()

    def slot_ptr(self, i: int) -> int:
        return self.slot_base + self.members[i][0] * self.slot_row_bytes

    def slot_row(self, i: int):
        return self.slot_mappings[self.members[i][0]]

    def key(self) -> tuple:
        """What the pointer arrays depend on beyond the builders: tables, slot rows, table stride."""
        return (
            tuple(t.data_ptr() for t in self.block_tables),
            self.slot_base,
            self.slot_row_bytes,
            int(self.block_tables[0].stride(0)),
        )

    def pointers(self) -> list:
        """Rows (indices, lens, valid, block table, slot row) of addresses, one column per group."""
        bts = [t.data_ptr() for t in self.block_tables]
        slots = [self.slot_ptr(i) for i in range(len(self.members))]
        ind, lens, valid = (list(c) for c in zip(*self.static_ptrs))
        return [ind, lens, valid, bts, slots]


def static_pointers(builders) -> list:
    return [(b.decode_swa_indices.data_ptr(), b.decode_swa_lens.data_ptr(), b.is_valid_token.data_ptr())
            for b in builders]


def _build_kernel(tl, triton):
    @triton.jit(do_not_specialize=["num_decode", "num_slots", "max_tokens", "swa_stride", "bt_stride"])
    def _swa_decode_batched_kernel(
        swa_indices_ptrs, swa_stride, swa_lens_ptrs, window_size, index_width,
        query_start_loc, seq_lens, token_to_req, slot_mapping_ptrs, is_valid_ptrs,
        block_table_ptrs, bt_stride, block_size,
        num_decode, num_slots, max_tokens,
        TRITON_BLOCK_SIZE: tl.constexpr, CHUNK: tl.constexpr,
    ):
        # Group g = program_id(1): its five buffers, one int64 address each.
        group = tl.program_id(1)
        swa_indices = tl.cast(tl.load(swa_indices_ptrs + group), tl.pointer_type(tl.int32))
        swa_lens = tl.cast(tl.load(swa_lens_ptrs + group), tl.pointer_type(tl.int32))
        is_valid = tl.cast(tl.load(is_valid_ptrs + group), tl.pointer_type(tl.int1))
        slot_mapping = tl.cast(tl.load(slot_mapping_ptrs + group), tl.pointer_type(tl.int64))
        block_table = tl.cast(tl.load(block_table_ptrs + group), tl.pointer_type(tl.int32))
        # swa_meta_fused._swa_decode_fused_kernel from here on, line for line.
        pid = tl.program_id(0)
        if pid < num_decode:
            valid = tl.load(slot_mapping + pid) >= 0
            tl.store(is_valid + pid, valid)
            if not valid:
                tl.store(swa_lens + pid, 0)
                for i in range(0, index_width, TRITON_BLOCK_SIZE):
                    offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
                    tl.store(swa_indices + pid * swa_stride + offset, -1, mask=offset < index_width)
            else:
                # _compute_swa_indices_and_lens_kernel, causal, no image, token_offset 0
                req_idx = tl.load(token_to_req + pid)
                query_start = tl.load(query_start_loc + req_idx)
                query_end = tl.load(query_start_loc + req_idx + 1)
                query_len = query_end - query_start
                seq_len = tl.load(seq_lens + req_idx)
                prefix_len = seq_len - query_len
                pos = prefix_len + pid - query_start
                start_pos = tl.maximum(pos - (window_size - 1), 0)
                end_pos = pos + 1
                swa_len = end_pos - start_pos
                tl.store(swa_lens + pid, swa_len)
                for i in range(0, index_width, TRITON_BLOCK_SIZE):
                    offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
                    pos_offset = start_pos + offset
                    block_indices = pos_offset // block_size
                    block_numbers = tl.load(
                        block_table + req_idx * bt_stride + block_indices, mask=pos_offset < end_pos
                    )
                    block_offsets = pos_offset % block_size
                    slot_ids = block_numbers * block_size + block_offsets
                    slot_ids = tl.where(offset < swa_len, slot_ids, -1)
                    tl.store(swa_indices + pid * swa_stride + offset, slot_ids, mask=offset < index_width)
        else:
            e = num_decode + (pid - num_decode) * CHUNK + tl.arange(0, CHUNK)
            in_slots = e < num_slots
            slot = tl.load(slot_mapping + e, mask=in_slots, other=0)
            tl.store(is_valid + e, slot >= 0, mask=in_slots)
            tl.store(swa_lens + e, 0, mask=e < max_tokens)

    return _swa_decode_batched_kernel


class SwaBatched:
    def __init__(self, torch, triton, kernel, fused):
        self.torch = torch
        self.triton = triton
        self.kernel = kernel
        self.fused = fused  # swa_meta_fused.SwaFused: the reference launch for the verify path
        self._ptrs_key = None
        self._ptrs = None

    def _arrays(self, plan, device):
        key = (tuple(plan.static_ptrs), plan.key(), str(device))
        if key != self._ptrs_key:
            self._ptrs = self.torch.tensor(plan.pointers(), dtype=self.torch.int64, device=device)
            self._ptrs_key = key
        return self._ptrs, key[1][3]

    def launch(self, plan, qsl, seq_lens, t2r, nd) -> None:
        b0 = plan.members[0][1]
        ptrs, bt_stride = self._arrays(plan, b0.decode_swa_lens.device)
        max_tokens = b0.decode_swa_lens.shape[0]
        rest = max(plan.n_slots, max_tokens) - nd
        grid = (nd + max(0, self.triton.cdiv(rest, CHUNK)), len(plan.members))
        self.kernel[grid](
            ptrs[0], b0.decode_swa_indices.stride(0), ptrs[1], b0.window_size, b0.decode_swa_indices.shape[-1],
            qsl, seq_lens, t2r, ptrs[4], ptrs[2],
            ptrs[3], bt_stride, b0.block_size,
            nd, plan.n_slots, max_tokens,
            TRITON_BLOCK_SIZE=1024, CHUNK=CHUNK,
        )

    def _views(self, plan, nd):
        return [
            (b.decode_swa_indices[:nd], b.decode_swa_lens, b.is_valid_token[: plan.n_slots])
            for _, b in plan.members
        ]

    def verify(self, plan, qsl, seq_lens, t2r, nd) -> None:
        """Every group's fused launch, then the batched launch over poisoned buffers; compare all."""
        torch = self.torch
        for i, (_, b) in enumerate(plan.members):
            self.fused._launch(b, b.decode_swa_indices, qsl, seq_lens, t2r, plan.slot_row(i),
                               b.is_valid_token[: plan.n_slots], plan.block_tables[i], nd)
        views = self._views(plan, nd)
        ref = [tuple(t.clone() for t in v) for v in views]
        has_valid = bool((plan.slot_row(0)[:nd] >= 0).any())
        for ind, lens, valid in views:
            ind.fill_(-7)
            lens.fill_(-7)
            valid.fill_(True)
        err = None
        try:
            self.launch(plan, qsl, seq_lens, t2r, nd)
            for g, (got, want) in enumerate(zip(views, ref)):
                if not all(torch.equal(a, b) for a, b in zip(got, want)):
                    err = RuntimeError(f"batched != fused (group {plan.members[g][0]}, decode tokens {nd})")
                    break
        except Exception as exc:  # noqa: BLE001
            err = exc
        if err is not None:
            for got, want in zip(views, ref):
                for a, b in zip(got, want):
                    a.copy_(b)
            _disarm(err)
            raise err
        if has_valid:
            _STATE["verify_left"] -= 1
        if not _STATE["engaged"]:
            _STATE["engaged"] = True
            print(
                "dsv41: swa metadata batched self-check bit-exact (groups %d, decode tokens %d, valid tokens %s)"
                % (len(plan.members), nd, has_valid),
                flush=True,
            )

    def rows_written(self, builder, indices, qsl, seq_lens, t2r, slots, valid, block_table, nd) -> bool:
        """swa_meta_fused hook: True when this call's batched launch wrote this build's rows."""
        plan = _CALL["plan"]
        if plan is None or not _STATE["armed"]:
            return False
        i = plan.index.get(id(builder))
        if (
            i is None
            or plan.block_tables[i] is not block_table
            or indices is not builder.decode_swa_indices
            or slots.shape[0] != plan.n_slots
            or slots.data_ptr() != plan.slot_ptr(i)
        ):
            return False
        batch = _CALL["batch"]
        if batch is not None:
            return batch[0] is qsl and batch[1] is seq_lens and batch[2] == nd
        try:
            if _STATE["verify_left"] > 0:
                self.verify(plan, qsl, seq_lens, t2r, nd)
            else:
                self.launch(plan, qsl, seq_lens, t2r, nd)
        except Exception as exc:  # noqa: BLE001
            _disarm(exc)
            return False
        _CALL["batch"] = (qsl, seq_lens, nd)
        return True


def wrap(stock, make_plan):
    """build_attn_metadata that publishes the call's SWA plan to rows_written() while it runs."""

    def build_attn_metadata(*args, **kwargs):
        plan = None
        if not args and call_eligible(kwargs):
            try:
                plan = make_plan(kwargs)
            except Exception as exc:  # noqa: BLE001 - never let the planner break a step
                _disarm(exc)
        if plan is None:
            return stock(*args, **kwargs)
        _CALL["plan"], _CALL["batch"] = plan, None
        try:
            return stock(*args, **kwargs)
        finally:
            _CALL["plan"], _CALL["batch"] = None, None

    build_attn_metadata._dsv41_swa_batched = True
    build_attn_metadata.__wrapped__ = stock
    return build_attn_metadata


def install() -> str:
    import torch
    import swa_meta_fused as smf
    from vllm.triton_utils import tl, triton
    from vllm.v1.attention.backends.mla import sparse_swa
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec
    from vllm.v1.worker.gpu.model_states import default as model_state

    if getattr(model_state.build_attn_metadata, "_dsv41_swa_batched", False):
        return "already installed"
    build = sparse_swa.DeepseekSparseSWAMetadataBuilder.build
    fused = getattr(build, "__globals__", {}).get("_dsv41_swa")
    if not getattr(build, "_dsv41_swa_fused", False) or fused is None:
        raise RuntimeError("swa_meta_fused is not installed (DSV41_SWA_META_FUSED=1 installs it first)")
    _STATE["verify_left"] = verify_calls()
    helper = SwaBatched(torch, triton, _build_kernel(tl, triton), fused)
    cache = {}

    def make_plan(kwargs):
        if torch.cuda.is_current_stream_capturing() or smf._STATE["verify_left"] > 0 or not smf._STATE["armed"]:
            return None
        attn_groups, kv_cache_config = kwargs["attn_groups"], kwargs["kv_cache_config"]
        key = (id(attn_groups), id(kv_cache_config))
        if key not in cache:
            members = swa_groups(attn_groups, kv_cache_config, SlidingWindowMLASpec,
                                 sparse_swa.DeepseekSparseSWAMetadataBuilder)
            if len(members) < 2 or not shapes_agree([b for _, b in members]):
                members = []
            cache.clear()
            # Keep the keyed objects alive so their ids cannot be reused.
            cache[key] = (attn_groups, kv_cache_config, members, static_pointers([b for _, b in members]))
        _, _, members, static = cache[key]
        if not members:
            return None
        plan = Plan(members, static, kwargs["block_tables"], kwargs["slot_mappings"])
        if len({int(t.stride(0)) for t in plan.block_tables}) != 1:
            return None
        return plan

    smf._BATCH_HOOK = helper.rows_written
    model_state.build_attn_metadata = wrap(model_state.build_attn_metadata, make_plan)
    return f"model_states.default.build_attn_metadata batches causal SWA decode rows (verify={_STATE['verify_left']})"
