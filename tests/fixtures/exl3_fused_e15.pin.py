# Pinned from dsv41-flash-exl3-sm121:e15-kbits (sha256:7e2cec199bed),
# /usr/local/lib/python3.12/dist-packages/vllm_exl3/exl3.py (vllm-exl3 d3cfd39 + the p2b chain incl. widen_p2b_kbits):
# build_exl3_fused_state and apply_exl3_fused_moe verbatim, for tests/test_moe_prefill_tc.py (the lazy-scratch
# rewrite of moe_prefill_tc.py must match these anchors). Not imported by anything.

def build_exl3_fused_state(layer: torch.nn.Module, inners: list[dict[str, Any]]) -> None:
    """Pointer tables + fused temps, once after load. No per-token alloc."""
    try:
        exllamav3_ext = load_exllamav3_ext()
    except Exception:
        exllamav3_ext = None

    device = layer.w13_trellis.device
    n_exp = len(inners)
    # Gate/up input rotations are immutable after load. Cache this compatibility
    # fact once so fat-prefill dispatch never calls torch.equal on CUDA tensors
    # in the per-expert hot loop.
    for pack in inners:
        pack["_exl3_gate_up_shared_suh"] = bool(
            torch.equal(pack["gate"].suh, pack["up"].suh)
        )
    hidden = int(layer._exl3_hidden_size)
    intermediate = int(layer._exl3_intermediate_local)

    def _ptrs(which: str, attr: str) -> torch.Tensor:
        return torch.tensor(
            [int(getattr(pack[which], attr).data_ptr()) for pack in inners],
            dtype=torch.int64,
            device=device,
        )

    layer._exl3_ptrs = {
        "gate_trellis": _ptrs("gate", "trellis"),
        "gate_suh": _ptrs("gate", "suh"),
        "gate_svh": _ptrs("gate", "svh"),
        "up_trellis": _ptrs("up", "trellis"),
        "up_suh": _ptrs("up", "suh"),
        "up_svh": _ptrs("up", "svh"),
        "down_trellis": _ptrs("down", "trellis"),
        "down_suh": _ptrs("down", "suh"),
        "down_svh": _ptrs("down", "svh"),
    }
    # Short aliases match the native extension terminology and keep the table
    # ABI stable for callers that construct their own RoutedExperts wrapper.
    layer._exl3_ptrs.update(
        {
            "gate_t_ptrs": layer._exl3_ptrs["gate_trellis"],
            "gate_suh_ptrs": layer._exl3_ptrs["gate_suh"],
            "gate_svh_ptrs": layer._exl3_ptrs["gate_svh"],
            "up_t_ptrs": layer._exl3_ptrs["up_trellis"],
            "up_suh_ptrs": layer._exl3_ptrs["up_suh"],
            "up_svh_ptrs": layer._exl3_ptrs["up_svh"],
            "down_t_ptrs": layer._exl3_ptrs["down_trellis"],
            "down_suh_ptrs": layer._exl3_ptrs["down_suh"],
            "down_svh_ptrs": layer._exl3_ptrs["down_svh"],
        }
    )
    idx = int(device.index) if device.index is not None else 0
    if exllamav3_ext is not None and hasattr(
        exllamav3_ext, "exl3_moe_max_concurrency"
    ):
        concurrency = int(exllamav3_ext.exl3_moe_max_concurrency(idx))
        if concurrency < 1:
            concurrency = 1
    else:
        # Native p2b does not consume ExLlamaV3 scratch buffers.
        layer._exl3_fused_temps = None
        layer._exl3_fused_concurrency = 0
        layer._exl3_k = int(layer._exl3_bits)
        layer._exl3_k_d = int(getattr(layer, "_exl3_bits_d", layer._exl3_bits))  # widen_p2b_kbits
        return
    key = (str(device), hidden, intermediate, concurrency)
    temps = _FUSED_TEMP_CACHE.get(key)
    if temps is None:
        temps = (
            torch.empty((concurrency, TEMP_ROWS_FUSED, hidden), dtype=torch.float16, device=device),
            torch.empty((concurrency, TEMP_ROWS_FUSED, hidden), dtype=torch.float16, device=device),
            torch.empty((concurrency, TEMP_ROWS_FUSED, intermediate), dtype=torch.float16, device=device),
            torch.empty((concurrency, TEMP_ROWS_FUSED, intermediate), dtype=torch.float16, device=device),
        )
        _FUSED_TEMP_CACHE[key] = temps
    layer._exl3_fused_temps = temps
    layer._exl3_fused_concurrency = concurrency
    layer._exl3_k = int(layer._exl3_bits)
    layer._exl3_k_d = int(getattr(layer, "_exl3_bits_d", layer._exl3_bits))


def apply_exl3_fused_moe(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    layer: torch.nn.Module,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float | None = None,
) -> torch.Tensor:
    """One exl3_moe launch per layer. Experts with count > 128 fall back to LinearEXL3."""
    tokens, hidden = x2d.shape
    n_exp = len(inners)

    # Keep this entry point independently usable by callers that bypass
    # ``apply_exl3_experts`` (for example, custom vLLM runners).
    if get_moe_kernel_backend() == "native":
        try:
            native_out = _apply_native_fused_moe(
                x2d, ids, weights, layer, inners, expert_map, limit
            )
        except Exception as exc:
            native_out = None
            layer._exl3_native_error = repr(exc)
            logger.warning_once(
                "Native EXL3 MoE dispatch failed in fused entry point; "
                "falling back to ExLlamaV3/Python: %s",
                exc,
            )
        if native_out is not None:
            layer._exl3_last_apply = "native"
            return native_out

    try:
        import exllamav3_ext
    except Exception:
        # Direct callers may use this helper without installing ExLlamaV3.
        # Keep the same graceful fallback contract as ``apply_exl3_experts``.
        return apply_exl3_python_loop(
            x2d, ids, weights, inners, expert_map, limit
        )

    ptrs = getattr(layer, "_exl3_ptrs", None)
    temps = getattr(layer, "_exl3_fused_temps", None)
    if not ptrs or temps is None:
        raise RuntimeError("EXL3 fused pointer tables were not built after weight load")

    local = map_topk_to_local(ids, n_exp, expert_map)
    topk = int(ids.shape[-1])
    flat_token = torch.arange(tokens, device=x2d.device, dtype=torch.long).repeat_interleave(topk)
    flat_weight = weights.reshape(-1).to(dtype=torch.float16)
    # scatter_add stays on GPU. torch.bincount can host-stage and break CUDA graphs.
    expert_count = torch.zeros(n_exp + 1, dtype=torch.long, device=local.device)
    expert_count.scatter_add_(
        0, local.long(), torch.ones(local.shape, dtype=torch.long, device=local.device)
    )
    out = torch.zeros(tokens, hidden, dtype=torch.float32, device=x2d.device)
    xh = x2d.contiguous().half()

    counts = expert_count[:n_exp]

    if tokens > TEMP_ROWS_FUSED and bool((counts > TEMP_ROWS_FUSED).any().item()):
        logger.info_once("EXL3 fat-chunk slicing ACTIVE (tokens=%d)" % tokens)
        # Deep-context prefill chunks can route more than TEMP_ROWS_FUSED rows
        # to a single expert. The fused kernel covers at most TEMP_ROWS_FUSED
        # rows per expert, and the old fallback reconstructed whole experts
        # per chunk, stalling prefill by orders of magnitude past ~160k
        # context (the ">163k hang"). Within a slice of <= TEMP_ROWS_FUSED
        # tokens no expert can exceed TEMP_ROWS_FUSED rows (each token adds at
        # most one row per expert), so re-run the fused path per slice.
        # Prefill-only: decode batches are at most the largest capture size,
        # far below TEMP_ROWS_FUSED, and never reach this host sync.
        for s in range(0, tokens, TEMP_ROWS_FUSED):
            e = min(s + TEMP_ROWS_FUSED, tokens)
            out[s:e] = apply_exl3_fused_moe(
                x2d[s:e], ids[s:e], weights[s:e], layer, inners, expert_map, limit
            )
        return out

    fat = counts > FAT_EXPERT_THRESHOLD
    # counts[e] cannot exceed the row count, so with no more rows than the
    # threshold no expert can be fat. Testing that Python-side first keeps the
    # device sync below off the decode path, where graph capture forbids it.
    fat_possible = tokens > FAT_EXPERT_THRESHOLD
    fat_route = torch.zeros_like(local, dtype=torch.bool)
    if fat_possible and bool(fat.any().item()):
        safe_local = local.clamp(min=0, max=max(n_exp - 1, 0))
        fat_route = (local < n_exp) & fat.index_select(0, safe_local)

    # The standard kernel handles non-fat routes. Fat routes are represented by
    # the invalid sentinel with zero weight here and are dispatched exactly once
    # below through the fat GEMM path.
    standard_local = local.masked_fill(fat_route, n_exp)
    standard_weight = flat_weight.masked_fill(fat_route, 0)
    order = standard_local.argsort()
    token_sorted = flat_token[order]
    weight_sorted = standard_weight[order]
    standard_count = torch.zeros(
        n_exp + 1, dtype=torch.long, device=local.device
    )
    standard_count.scatter_add_(
        0,
        standard_local.long(),
        torch.ones(standard_local.shape, dtype=torch.long, device=local.device),
    )
    fn = exllamav3_ext.exl3_moe
    # -1 = unknown active count: max-concurrency grid, no .item() host sync.
    n_active_host = -1 if _exl3_moe_accepts_num_active(fn) else None

    k = int(getattr(layer, "_exl3_k", 4))
    k_d = int(getattr(layer, "_exl3_k_d", k))  # widen_p2b_kbits
    if k_d != k and bool(getattr(layer, "_exl3_codebook_flags", (True,))[0]):
        # exl3_moe's K = 0 instance switches K at runtime (~2x slower); vllm_exl3_c carries the same
        # kernel with compile-time (K_gu, K_d), exl3_moe's arguments.
        fn = getattr(_load_native_exl3_ext(), "exl3_moe_kmix", None) or fn
    args = (
        xh,
        out,
        standard_count,
        token_sorted,
        weight_sorted,
        temps[0],
        temps[1],
        temps[2],
        temps[3],
        MOE_ACT_SILU,
        k,
        k,
        k_d,
        ptrs["gate_trellis"],
        ptrs["gate_suh"],
        ptrs["gate_svh"],
        ptrs["up_trellis"],
        ptrs["up_suh"],
        ptrs["up_svh"],
        ptrs["down_trellis"],
        ptrs["down_suh"],
        ptrs["down_svh"],
        *getattr(layer, "_exl3_codebook_flags", (True, False, True, False, True, False)),
        float(limit) if (limit is not None and limit > 0) else 0.0,
    )
    if n_active_host is not None:
        fn(*args, n_active_host)
    else:
        fn(*args)

    if fat_possible and bool(fat.any().item()):
        fat_order = local.argsort()
        apply_exl3_batched_fat(
            xh,
            flat_token[fat_order],
            flat_weight[fat_order],
            counts.tolist(),
            inners,
            limit,
            FAT_EXPERT_THRESHOLD,
            out,
            use_kernel=_fat_kernel_available(),
        )
    return out
