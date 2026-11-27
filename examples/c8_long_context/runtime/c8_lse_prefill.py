# SPDX-License-Identifier: Apache-2.0
"""Optional C8 continuation prefill with page-aligned BF16 FIA/LSE segments."""

import atexit
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import time

import torch
import torch_npu

from fused_gather import gather_c8, prepare_lengths
from segment_plan import split_history

SEGMENT_TOKENS = 16384
logger = logging.getLogger("vllm.c8_lse_prefill")
_stats = {
    "calls": 0, "attention_rounds": 0, "merge_calls": 0,
    "plan_builds": 0, "plan_reuses": 0, "plan_host_ms": 0.0,
    "table_preparations": 0, "packed_tables": 0, "table_host_ms": 0.0,
    "guard_fallbacks": 0, "short_bypasses": 0,
}
_bypass_reasons = set()


@atexit.register
def _report_stats():
    if not (_stats["calls"] or _stats["guard_fallbacks"] or _stats["short_bypasses"]):
        return
    result = {"kind": "c8_lse_prefill_stats", "segment_tokens": SEGMENT_TOKENS,
              "pid": os.getpid(), **_stats}
    print("C8 LSE segmented prefill stats " + json.dumps(result), flush=True)
    run = os.environ.get("C8_AUDIT_RUN")
    if run:
        with (Path(run) / "runtime-c8-lse-audit.jsonl").open("a") as out:
            out.write(json.dumps(result) + "\n")


@dataclass(frozen=True)
class RoundPlan:
    causal: bool
    rows: tuple
    qlens: tuple
    klens: tuple
    q_offsets: tuple
    kv_offsets: tuple
    active_spans: tuple
    first_pages: tuple
    page_counts: tuple
    prepared: tuple


@dataclass(frozen=True)
class PrefillPlan:
    rounds: tuple
    query_spans: tuple


def _spans(lengths):
    start = 0
    out = []
    for n in lengths:
        out.append(slice(start, start + n))
        start += n
    return tuple(out)


def _offsets(lengths):
    return tuple(s.stop for s in _spans(lengths))


def _prepare_plan(metadata, query, decodes, qlens, klens, block):
    fingerprint = (decodes, qlens, klens, block, query.device)
    cached = getattr(metadata, "_c8_lse_prefill_plan", None)
    if cached is not None and cached[0] == fingerprint:
        _stats["plan_reuses"] += 1
        return cached[1]

    start = time.perf_counter()
    rounds = []
    for group in split_history(qlens, klens, SEGMENT_TOKENS, block):
        rows = tuple(s.row for s in group.segments)
        local_q = tuple(qlens[row] for row in rows)
        local_kv = tuple(s.length for s in group.segments)
        rounds.append(RoundPlan(
            group.causal, rows, local_q, local_kv, _offsets(local_q), _offsets(local_kv),
            _spans(local_q),
            tuple(s.start // block for s in group.segments),
            tuple((s.length + block - 1) // block for s in group.segments),
            prepare_lengths(local_kv, query.device),
        ))
    plan = PrefillPlan(tuple(rounds), _spans(qlens))
    metadata._c8_lse_prefill_plan = (fingerprint, plan)
    _stats["plan_builds"] += 1
    _stats["plan_host_ms"] += (time.perf_counter() - start) * 1000
    return plan


def _table_for_round(table, group):
    # Do not cache copied physical page IDs across model forwards. A reused
    # metadata object can have the same lengths and different physical pages.
    # Only the logical segmentation and length tensors are cached.
    start = time.perf_counter()
    _stats["table_preparations"] += 1
    rows, first, counts = group.rows, group.first_pages, group.page_counts
    if (rows[-1] - rows[0] + 1 == len(rows)
            and len(set(first)) == 1 and len(set(counts)) == 1):
        result = table[rows[0]:rows[-1] + 1, first[0]:first[0]+counts[0]]
    else:
        _stats["packed_tables"] += 1
        result = torch.zeros((len(rows), max(counts)),
                             dtype=table.dtype, device=table.device)
        for out_row, (row, begin, count) in enumerate(zip(rows, first, counts)):
            result[out_row, :count].copy_(table[row, begin:begin + count])
    _stats["table_host_ms"] += (time.perf_counter() - start) * 1000
    return result


def _query_for_round(query, plan, rows, cache):
    if rows not in cache:
        if rows[-1] - rows[0] + 1 == len(rows):
            start = plan.query_spans[rows[0]].start
            end = plan.query_spans[rows[-1]].stop
            cache[rows] = query[start:end]
        else:
            cache[rows] = torch.cat([query[plan.query_spans[row]] for row in rows], dim=0)
    return cache[rows]


def _partial(backend, layer, query, table, metadata, plan, group, query_cache):
    part_query = _query_for_round(query, plan, group.rows, query_cache)
    part_table = _table_for_round(table, group)
    k, v = gather_c8(backend.key_cache, backend.value_cache, part_table, group.klens,
                     layer._c8_k_scale, layer._c8_v_scale, query.dtype,
                     tile=4096, prepared=group.prepared)
    kwargs = dict(input_layout="TND", actual_seq_lengths=list(group.q_offsets),
                  actual_seq_lengths_kv=list(group.kv_offsets),
                  num_heads=backend.num_heads, num_key_value_heads=backend.num_kv_heads,
                  scale=backend.scale, sparse_mode=3 if group.causal else 0,
                  inner_precise=0, softmax_lse_flag=True)
    if group.causal:
        kwargs["atten_mask"] = metadata.attn_mask
    out, lse = torch_npu.npu_fused_infer_attention_score(part_query, k, v, **kwargs)
    out = out.view(-1, backend.num_heads, backend.head_size)
    expected = {(out.shape[0], backend.num_heads, 1), (out.shape[0], backend.num_heads)}
    if tuple(lse.shape) not in expected or lse.dtype != torch.float32:
        raise RuntimeError(f"Unexpected C8 TND LSE shape/dtype: {lse.shape}, {lse.dtype}")
    _stats["attention_rounds"] += 1
    return out, lse.reshape(out.shape[0], backend.num_heads)


def _merge(out, lse, partial_out, partial_lse):
    # npu_attention_update performs a max-anchored logsumexp and weighted
    # output merge. Keep every intermediate in FP32 and return its LSE.
    width = out.shape[-1]
    merged, merged_lse = torch_npu.npu_attention_update(
        (lse.contiguous().view(-1), partial_lse.contiguous().view(-1)),
        (out.contiguous().view(-1, width), partial_out.float().contiguous().view(-1, width)),
        1,
    )
    if merged.dtype != torch.float32 or merged_lse.dtype != torch.float32:
        raise RuntimeError("C8 native LSE merge did not preserve FP32 state")
    _stats["merge_calls"] += 1
    return merged.view_as(out), merged_lse.view_as(lse)


def _guard(backend, layer, query, table, metadata, qlens, klens):
    if query.dtype != torch.bfloat16 or query.dim() != 3 or not query.is_contiguous():
        return "query must be BF16 TND"
    if query.shape[1:] != (backend.num_heads, backend.head_size):
        return "unexpected query head layout"
    if (not qlens or len(qlens) != len(klens) or sum(qlens) != query.shape[0]
            or any(not 0 < q <= k for q, k in zip(qlens, klens))):
        return "invalid query/KV length metadata"
    if metadata.attn_mask is None or not hasattr(torch_npu, "npu_attention_update"):
        return "causal mask or native LSE merge unavailable"
    k, v = backend.key_cache, backend.value_cache
    if k.dim() != 4 or k.shape != v.shape or k.dtype != torch.int8 or v.dtype != torch.int8:
        return "unexpected INT8 cache layout"
    _, block, heads, dim = k.shape
    width = heads * dim
    if heads != backend.num_kv_heads or dim != backend.head_size:
        return "unexpected KV head layout"
    if SEGMENT_TOKENS % block or block * width % 4096 or 4096 % width:
        return "page size or tile geometry is incompatible"
    page_size = block * width
    if (k.stride() != (2 * page_size, width, dim, 1) or v.stride() != k.stride()
            or v.data_ptr() - k.data_ptr() != page_size):
        return "validated interleaved ND page layout required"
    if (table.dim() != 2 or table.dtype != torch.int32 or table.shape[0] != len(klens)
            or table.stride(1) != 1 or max(klens) > table.shape[1] * block):
        return "incompatible block table geometry"
    for scale in (layer._c8_k_scale, layer._c8_v_scale):
        if scale.numel() != width or not scale.is_contiguous():
            return "incompatible per-channel scale layout"
    return None


def try_prefill(backend, layer, query, metadata, decodes, lengths, query_lens):
    """Return None before any candidate execution if this batch should bypass."""
    qlens, klens = tuple(query_lens), tuple(lengths)
    if not qlens or max((k - q for q, k in zip(qlens, klens)), default=0) <= SEGMENT_TOKENS:
        _stats["short_bypasses"] += 1
        return None
    table = metadata.block_tables[decodes:]
    reason = _guard(backend, layer, query, table, metadata, qlens, klens)
    if reason is not None:
        _stats["guard_fallbacks"] += 1
        if reason not in _bypass_reasons:
            _bypass_reasons.add(reason)
            logger.warning("C8 LSE segmented prefill bypass: %s", reason)
        return None

    plan = _prepare_plan(metadata, query, decodes, qlens, klens,
                         backend.key_cache.shape[1])
    if not getattr(backend, "_c8_lse_prefill_logged", False):
        logger.info("C8 LSE segmented prefill active for %s: rounds=%d history=%d segment=%d",
                    backend._layer_name, len(plan.rounds),
                    max(k-q for q, k in zip(qlens, klens)), SEGMENT_TOKENS)
        backend._c8_lse_prefill_logged = True
    _stats["calls"] += 1

    query_cache = {}
    # The final causal suffix contains every current Q position, so all
    # accumulators begin with a valid LSE. Historical rounds are noncausal.
    out, lse = _partial(backend, layer, query, table, metadata, plan,
                        plan.rounds[0], query_cache)
    out = out.float()
    all_rows = tuple(range(len(qlens)))
    for group in plan.rounds[1:]:
        part_out, part_lse = _partial(backend, layer, query, table, metadata, plan,
                                      group, query_cache)
        if group.rows == all_rows:
            out, lse = _merge(out, lse, part_out, part_lse)
        else:
            # Each participating sequence maps back to its original Query
            # rows. Rows whose history is already consumed stay unchanged.
            for row, source in zip(group.rows, group.active_spans):
                target = plan.query_spans[row]
                merged, merged_lse = _merge(out[target], lse[target],
                                            part_out[source], part_lse[source])
                out[target].copy_(merged)
                lse[target].copy_(merged_lse)
                del merged, merged_lse
        del part_out, part_lse
    # Any actual operator exception propagates to the benchmark/service.
    # Do not retry on a potentially corrupted NPU runtime.
    return out.to(query.dtype)
