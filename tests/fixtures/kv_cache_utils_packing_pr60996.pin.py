# The same region after vLLM PR #60996's v1/core/kv_cache_utils.py hunk (open upstream PR, "Size packed
# groups by pooled bytes per request, not layer padding") was applied to e15's file with patch(1) (offset 76,
# fuzz 1 on a context line): lines 1901-2070 of /home/sfxnz/projects/data/dsv41-1017/perf2/upstream/notes/pr60996/e15/kv_cache_utils.py.
# --- excerpt ---
def _get_packed_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Group mixed-page-size layers for contiguous block-outermost packing.

    Greedily buckets layers into uniform-type specs. Buckets with equal layer
    counts per page size are treated as a repeating layer pattern (one layer
    per page size) and split into groups covering the same number of pattern
    repeats, picked to minimize the bytes a max-length request draws from the
    shared pool, so all groups pack into the same per-block layout. Mamba
    buckets are additionally split to fit the block the attention buckets
    already need. Returns None when the layout is not block-outermost or all
    layers already share one page size.
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

    def num_groups_for(
        spec: UniformTypeKVCacheSpecs, balanced: bool, repeats: int | None
    ) -> int:
        if balanced and repeats is not None:
            return cdiv(spec.get_max_layers_per_page_size(), repeats)
        return 1

    def widest_group_bytes(page_size_layers: dict[int, list[str]], n: int) -> int:
        """Page bytes of the largest of the n groups a bucket splits into."""
        return sum(
            cdiv(len(names), n) * page for page, names in page_size_layers.items()
        )

    # A recycling bucket that the repeat pattern does not split may still be
    # split so that one of its states fills a block.
    def is_state_bucket(spec: UniformTypeKVCacheSpecs, repeats: int | None) -> bool:
        if isinstance(spec.first_spec, (MambaSpec, CircularBufferSpec)):
            return True
        return repeats is None and isinstance(spec.first_spec, SlidingWindowSpec)

    def anchor_bytes(repeats: int | None) -> int:
        """Widest block the state buckets could need under a split choice.

        Mamba, circular-buffer and unsplit sliding-window buckets can go down to
        one state per group; every other bucket's width is fixed by the repeat
        pattern.
        """
        return max(
            (
                widest_group_bytes(
                    page_size_layers,
                    len(spec.kv_cache_specs)
                    if is_state_bucket(spec, repeats)
                    else num_groups_for(spec, balanced, repeats),
                )
                for spec, page_size_layers, balanced in bucketed
            ),
            default=0,
        )

    def split_plan(repeats: int | None) -> list[int]:
        """The number of groups each bucket splits into."""
        anchor = anchor_bytes(repeats)
        plan = []
        for spec, _, balanced in bucketed:
            num_groups = num_groups_for(spec, balanced, repeats)
            # Cap a state group at the states a block already fits rather than
            # let it widen the block.
            if anchor and is_state_bucket(spec, repeats):
                states_per_block = max(anchor // spec.first_spec.page_size_bytes, 1)
                num_groups = max(
                    num_groups, cdiv(len(spec.kv_cache_specs), states_per_block)
                )
            plan.append(num_groups)
        return plan

    # Every group claims its blocks from one shared pool, so a block is as wide
    # as the widest group's page and the narrower groups only pad out the rest
    # of it. Splitting a bucket into more, narrower groups shrinks that block,
    # but every group claims its own blocks for a request - and a recycling
    # group (sliding window, circular buffer, mamba) reserves as many blocks for
    # one layer as for many - so the block count grows with the group count.
    # Price both against the bytes a max-length request draws from the pool.
    max_repeats_per_group = max(
        (
            spec.get_max_layers_per_page_size()
            for spec, _, balanced in bucketed
            if balanced
        ),
        default=0,
    )
    needs_packing = any(
        balanced and len(page_size_layers) > 1
        for _, page_size_layers, balanced in bucketed
    )

    def build_groups(repeats: int | None) -> list[KVCacheGroupSpec]:
        """The groups a split choice produces."""
        groups: list[KVCacheGroupSpec] = []
        for (spec, page_size_layers, _), num_groups in zip(
            bucketed, split_plan(repeats)
        ):
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
        return groups

    # Price a split by what the allocator actually charges for it: every group
    # claims its own blocks from the shared pool, and a block is rounded up to
    # the layers' stride alignment, which is what
    # ``_max_memory_usage_bytes_from_groups`` computes.
    repeats_per_group = (
        min(
            range(1, max_repeats_per_group + 1),
            key=lambda repeats: (
                _max_memory_usage_bytes_from_groups(vllm_config, build_groups(repeats)),
                -repeats,
            ),
        )
        if needs_packing
        else None
    )

    groups = build_groups(repeats_per_group)

    _annotate_eagle_groups(
        vllm_config,
        kv_cache_spec,
        groups,
        use_deepseek_v4_fallback=_is_deepseek_v4_eagle(vllm_config),
    )
    _warn_if_unannotated_eagle_mamba(vllm_config, groups)
    return groups


