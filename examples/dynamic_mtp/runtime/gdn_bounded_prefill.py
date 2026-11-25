"""Execute large multi-sequence GDN prefills in independent groups."""

from functools import lru_cache
from types import SimpleNamespace

import torch

MAX_TOKENS = 4096


@lru_cache(maxsize=256)
def sequence_groups(cu, limit=MAX_TOKENS):
    """Split at sequence boundaries without splitting a recurrent sequence."""
    if len(cu) <= 2 or cu[-1] <= limit or any(a >= b for a, b in zip(cu, cu[1:])):
        return ()
    groups = []
    start = 0
    for end in range(1, len(cu)):
        if end > start + 1 and cu[end] - cu[start] > limit:
            groups.append((start, end - 1))
            start = end - 1
    groups.append((start, len(cu) - 1))
    return tuple(groups) if len(groups) > 1 else ()


@lru_cache(maxsize=512)
def metadata_for(cu, first, last):
    offsets = tuple(x - cu[first] for x in cu[first:last + 1])
    chunks = tuple(item for row, (a, b) in enumerate(zip(offsets, offsets[1:]))
                   for chunk in range((b - a + 63) // 64) for item in (row, chunk))
    return SimpleNamespace(cu_seqlens_host=offsets,
                           chunk_indices_chunk64_host=chunks,
                           cu_seqlens_kern=None, keep_meta=None)


def bounded_prefill(operation, *, q, k, v, g, beta, initial_state, scale, prebuilt_meta, fused_fwd):
    cu = prebuilt_meta.cu_seqlens_host
    groups = sequence_groups(cu)
    if not groups:
        return operation(q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state,
                         scale=scale, prebuilt_meta=prebuilt_meta, fused_fwd=fused_fwd)
    assert cu[0] == 0 and cu[-1] == q.shape[1] and len(cu) == initial_state.shape[0] + 1
    output = torch.empty_like(v, memory_format=torch.contiguous_format)
    states = torch.empty_like(initial_state, memory_format=torch.contiguous_format)
    for first, last in groups:
        begin, end = cu[first], cu[last]
        result, state = operation(
            q=q[:, begin:end], k=k[:, begin:end], v=v[:, begin:end],
            g=g[:, begin:end], beta=beta[:, begin:end],
            initial_state=initial_state[first:last], scale=scale,
            prebuilt_meta=metadata_for(cu, first, last), fused_fwd=fused_fwd,
        )
        output[:, begin:end].copy_(result)
        states[first:last].copy_(state)
        del result, state
    return output, states
