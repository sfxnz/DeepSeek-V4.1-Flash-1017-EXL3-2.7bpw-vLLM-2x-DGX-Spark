# Pinned from image dsv41-flash-exl3-sm121:e15-kbits (vLLM 0.1.dev20904+g179dd0fa9, the 0909 base),
# vllm/v1/core/kv_cache_utils.py lines 1901-2072: _approximate_gcd and _get_packed_kv_cache_groups, verbatim
# (extracted copy: /home/sfxnz/projects/data/dsv41-1017/perf2/src/vllm-e15). docker/patch/kv_packed_groups.py
# (DSV41_KV_PACKED_GROUPS) rewrites this region; tests/test_kv_packed_groups.py checks its anchors here.
# --- excerpt ---
def _approximate_gcd(values: Sequence[int], *, lower_bound: int | None = None) -> int:
    """Pick a chunk size that minimizes total upward padding.

    Each x is rounded up to a multiple of d:

      x -> ceil(x / d) * d

    Total padding is:

      pad(d) = sum_i (ceil(x_i / d) * d - x_i)

    We brute-force d in [lower_bound, max(values)] (fine for small lists / small
    maxima) and return the d with minimum padding. Ties prefer larger d.
    """
    if not values:
        raise ValueError("values must be non-empty")
    if any(x <= 0 for x in values):
        raise ValueError(f"values must be positive, got: {list(values)!r}")

    min_d = max(1, lower_bound if lower_bound is not None else 1)
    max_d = max(values)
    if min_d > max_d:
        return min_d

    best_d = min_d
    best_pad: int | None = None
    for d in range(min_d, max_d + 1):
        pad = sum((d - (x % d)) % d for x in values)
        if best_pad is None or pad < best_pad or (pad == best_pad and d > best_d):
            best_pad = pad
            best_d = d

    return best_d


def _get_packed_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Group mixed-page-size layers for contiguous block-outermost packing.

    Greedily buckets layers into uniform-type specs. Buckets with equal layer
    counts per page size are treated as a repeating layer pattern (one layer
    per page size) and split into groups covering the same number of pattern
    repeats (picked by ``_approximate_gcd`` to minimize padding), so all
    groups pack into the same per-block layout. Mamba buckets are additionally
    split to fit the block the attention buckets already need.
    Returns None when the layout is not block-outermost or all layers already
    share one page size.
    """
    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    page_sizes = {spec.page_size_bytes for spec in kv_cache_spec.values()}
    if not layout.is_block_outermost or len(page_sizes) <= 1:
        return None

    buckets: list[dict[str, KVCacheSpec]] = []
    for name, spec in kv_cache_spec.items():
        for bucket in buckets:
            candidate = {**bucket, name: spec}
            if UniformTypeKVCacheSpecs.is_uniform_type(candidate):
                bucket[name] = spec
                break
        else:
            buckets.append({name: spec})

    bucketed = []
    for bucket in buckets:
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(bucket)
        assert uniform_spec is not None
        page_size_layers: dict[int, list[str]] = defaultdict(list)
        for layer_name, layer_spec in bucket.items():
            page_size_layers[layer_spec.page_size_bytes].append(layer_name)
        # Only 1:1 patterns (one layer of each page size per repeat) are
        # supported; counts sharing a gcd > 1 (e.g. 2:1) could in principle
        # repeat too, but such buckets are emitted whole instead.
        balanced = len(set(map(len, page_size_layers.values()))) == 1
        bucketed.append((uniform_spec, page_size_layers, balanced))

    # Balanced buckets that mix page sizes must stay whole, so the largest one
    # sets a floor on the repeats per group; larger single-size buckets are
    # split down toward it. No such bucket means nothing needs packing.
    min_repeats_per_group = max(
        (
            spec.get_max_layers_per_page_size()
            for spec, page_size_layers, balanced in bucketed
            if balanced and len(page_size_layers) > 1
        ),
        default=0,
    )
    repeats_per_group = (
        _approximate_gcd(
            [
                spec.get_max_layers_per_page_size()
                for spec, _, balanced in bucketed
                if balanced
            ],
            lower_bound=min_repeats_per_group,
        )
        if min_repeats_per_group
        else None
    )

    def num_groups_for(spec: UniformTypeKVCacheSpecs, balanced: bool) -> int:
        if balanced and repeats_per_group is not None:
            return cdiv(spec.get_max_layers_per_page_size(), repeats_per_group)
        return 1

    def widest_group_bytes(page_size_layers: dict[int, list[str]], n: int) -> int:
        """Page bytes of the largest of the n groups a bucket splits into."""
        return sum(
            cdiv(len(names), n) * page for page, names in page_size_layers.items()
        )

    # Bytes a block must hold however the state buckets end up split: mamba,
    # circular-buffer and unsplit sliding-window buckets can go down to one
    # state per group, every other bucket's split is fixed by the repeat pattern.
    def is_state_bucket(spec: UniformTypeKVCacheSpecs) -> bool:
        if isinstance(spec.first_spec, (MambaSpec, CircularBufferSpec)):
            return True
        return repeats_per_group is None and isinstance(
            spec.first_spec, SlidingWindowSpec
        )

    anchor_bytes = max(
        (
            widest_group_bytes(
                page_size_layers,
                len(spec.kv_cache_specs)
                if is_state_bucket(spec)
                else num_groups_for(spec, balanced),
            )
            for spec, page_size_layers, balanced in bucketed
        ),
        default=0,
    )

    groups = []
    for spec, page_size_layers, balanced in bucketed:
        num_groups = num_groups_for(spec, balanced)
        # Cap a state group at the states a block already fits rather than let
        # it widen the block.
        if anchor_bytes and is_state_bucket(spec):
            states_per_block = max(anchor_bytes // spec.first_spec.page_size_bytes, 1)
            num_groups = max(
                num_groups, cdiv(len(spec.kv_cache_specs), states_per_block)
            )
        if num_groups == 1:
            groups.append(KVCacheGroupSpec(list(spec.kv_cache_specs), spec))
            continue

        pattern_repeats = list(zip(*page_size_layers.values()))
        for i in range(num_groups):
            group_layer_names = [
                name for repeat in pattern_repeats[i::num_groups] for name in repeat
            ]
            group_layer_specs = {
                name: spec.kv_cache_specs[name] for name in group_layer_names
            }
            group_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert group_spec is not None
            groups.append(KVCacheGroupSpec(group_layer_names, group_spec))

    _annotate_eagle_groups(
        vllm_config,
        kv_cache_spec,
        groups,
        use_deepseek_v4_fallback=_is_deepseek_v4_eagle(vllm_config),
    )
    _warn_if_unannotated_eagle_mamba(vllm_config, groups)
    return groups


