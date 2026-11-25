"""Fused scale-only C8 paged gather for contiguous dense attention input."""

import torch
from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit
def _gather_c8_kernel(
    k_ptr, v_ptr, ks_ptr, vs_ptr, table_ptr, lens_ptr, starts_ptr,
    out_k, out_v, table_stride, k_page_stride, v_page_stride, num_tasks,
    MAX_PAGES: tl.constexpr, PAGE_TOKENS: tl.constexpr, WIDTH: tl.constexpr,
    TILE: tl.constexpr,
):
    tiles_per_page: tl.constexpr = PAGE_TOKENS * WIDTH // TILE
    lanes = tl.arange(0, TILE)
    ks = tl.load(ks_ptr + lanes % WIDTH).to(tl.float32)
    vs = tl.load(vs_ptr + lanes % WIDTH).to(tl.float32)
    for task in range(tl.program_id(0), num_tasks, tl.num_programs(0)):
        logical_page = task // tiles_per_page
        row = logical_page // MAX_PAGES
        page = logical_page % MAX_PAGES
        within = task % tiles_per_page * TILE
        length = tl.load(lens_ptr + row)
        base = page * PAGE_TOKENS * WIDTH + within
        if base < length * WIDTH:
            physical = tl.load(table_ptr + row * table_stride + page).to(tl.int64)
            start = tl.load(starts_ptr + row).to(tl.int64)
            mask = base + lanes < length * WIDTH
            dst = start * WIDTH + base + lanes
            kv = tl.load(k_ptr + physical * k_page_stride + within + lanes, mask=mask, other=0).to(tl.float32)
            tl.store(out_k + dst, kv * ks, mask=mask)
            vv = tl.load(v_ptr + physical * v_page_stride + within + lanes, mask=mask, other=0).to(tl.float32)
            tl.store(out_v + dst, vv * vs, mask=mask)


def prepare_lengths(lengths, device):
    starts = []
    offset = 0
    for length in lengths:
        starts.append(offset)
        offset += length
    return (torch.tensor(lengths, dtype=torch.int32, device=device),
            torch.tensor(starts, dtype=torch.int32, device=device))


def gather_c8(key, value, table, lengths, key_scale, value_scale, dtype, *, tile=4096, prepared=None):
    if key.dtype != torch.int8 or value.dtype != torch.int8 or key.shape != value.shape:
        raise ValueError("Expected paired INT8 KV page views")
    pages, block, heads, dim = key.shape
    width = heads * dim
    if key.stride()[1:] != (width, dim, 1) or value.stride()[1:] != (width, dim, 1):
        raise ValueError("Unexpected inner KV page layout")
    if table.stride(1) != 1 or key_scale.numel() != width or value_scale.numel() != width:
        raise ValueError("Unexpected C8 scale or block table layout")
    if block * width % tile or tile % width:
        raise ValueError("Tile must contain an integral number of token rows")
    total = sum(lengths)
    out_k = torch.empty((total, heads, dim), dtype=dtype, device=key.device)
    out_v = torch.empty_like(out_k)
    if not total:
        return out_k, out_v
    lens, starts = prepared if prepared is not None else prepare_lengths(lengths, key.device)
    pages_per_seq = table.shape[1]
    tasks = table.shape[0] * pages_per_seq * (block * width // tile)
    init_device_properties_triton()
    grid = (min(tasks, get_vectorcore_num()),)
    _gather_c8_kernel[grid](
        key, value, key_scale, value_scale, table, lens, starts, out_k, out_v,
        table.stride(0), key.stride(0), value.stride(0), tasks,
        MAX_PAGES=pages_per_seq, PAGE_TOKENS=block, WIDTH=width, TILE=tile, num_warps=8,
    )
    return out_k, out_v
