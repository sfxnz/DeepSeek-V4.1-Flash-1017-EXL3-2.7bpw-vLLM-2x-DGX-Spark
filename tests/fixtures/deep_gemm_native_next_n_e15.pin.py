# Pinned read-only from dsv41-flash-exl3-sm121:e15-kbits vllm/utils/deep_gemm.py lines 560-569 (native_next_n_supported), verbatim
# (extracted copy: /home/sfxnz/projects/data/dsv41-1017/perf2/src/vllm-e15). indexer_expanded_lazy.py relies on it: True off SM90.
def native_next_n_supported(next_n: int) -> bool:
    """Whether the paged MQA logits kernel takes `next_n` Q rows per request.

    SM90 implements only {1, 2, 4}; SM100 and SM120 schedule any `next_n` via
    multi-atom tiles. Unsupported values must be flattened to one row per query.
    """
    if current_platform.is_device_capability_family(90):
        return next_n in (1, 2, 4)
    return True

