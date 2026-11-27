# SPDX-License-Identifier: Apache-2.0
"""Experimental four-group C8 hybrid KV planner, pinned to tested 0.30.0 sources.

Keep the BF16 draft, with the same logical block IDs as the INT8 target. Its
contiguous 2P-byte pages occupy an exclusive tail of the shared backing. GDN
shares only the 16 equal-P target slots, under the normal global BlockPool
ownership invariant. The special, proven mixed-page geometry does not require
block-outermost storage or padding the 16 INT8 target layers to BF16 size.
"""
from __future__ import annotations

from dataclasses import replace
import functools
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import sys

MODULE = "vllm.v1.core.kv_cache_utils"
SOURCE_SHA256 = {
    "vllm/v1/core/kv_cache_utils.py": "2666c9f113584e52e7521058efd2c0d9598544559dc942d3a3da87001a3fedf2",
    "vllm/v1/kv_cache_interface.py": "1cf202f1a44d5bc5c3832b70b41507687ecceea3784e67a4f003bc0210d6eecb",
    "vllm_ascend/worker/v2/attn_utils.py": "b1e7d441168a5c96f2364199f9092c8b0b95b297393c63dc31dbb19beae8e160",
    "vllm_ascend/patch/platform/patch_kv_cache_utils.py": "5fc49ebb9b64a2f14c4eb38b8c66d3daf1b7612961cf76392cf113d76f555aab",
}
TARGET = tuple(f"language_model.model.layers.{i}.self_attn.attn" for i in range(3, 64, 4))
DRAFT = "mtp.layers.0.self_attn.attn"
STATE = tuple(f"language_model.model.layers.{i}.linear_attn" for i in range(64) if i % 4 != 3)
ALL = frozenset((*TARGET, DRAFT, *STATE))
B = 1664
P = 3_407_872
S = P * 18

def require(condition, msg):
    if not condition:
        raise RuntimeError("[C8_GROUPING] rejected: " + msg)

def _keyed_groups(m, groups, *, normalized_ok=False):
    """Recognize exactly our four groups, never merely a mixed dtype model."""
    if len(groups) != 4:
        return None
    a, *state = groups
    if tuple(a.layer_names) != (*TARGET, DRAFT):
        return None
    if tuple(tuple(g.layer_names) for g in state) != tuple(STATE[i::3] for i in range(3)):
        return None
    if any(g.is_eagle_group or g.host_resident for g in groups):
        return None
    if any(type(g.kv_cache_spec) is not m.MambaSpec or g.kv_cache_spec.block_size != B
           or g.kv_cache_spec.page_size_bytes != P for g in state):
        return None
    a_spec = a.kv_cache_spec
    if isinstance(a_spec, m.UniformTypeKVCacheSpecs):
        ss = a_spec.kv_cache_specs
        if tuple(ss) != (*TARGET, DRAFT) or a_spec.block_size != B:
            return None
        if any(type(ss[n]) is not m.FullAttentionSpec or str(ss[n].dtype) != "torch.int8"
               or ss[n].page_size_bytes != P or ss[n].real_page_size_bytes != P
               or ss[n].block_size != B for n in TARGET):
            return None
        d = ss[DRAFT]
        if not (type(d) is m.FullAttentionSpec and str(d.dtype) == "torch.bfloat16"
                and d.block_size == B and d.page_size_bytes == d.real_page_size_bytes == 2*P):
            return None
        return "mixed"
    if normalized_ok and (type(a_spec) is m.FullAttentionSpec and str(a_spec.dtype)=="torch.int8"
                          and a_spec.block_size == B and a_spec.page_size_bytes == P):
        return "scheduler"
    return None

def _scope(config):
    return (os.environ.get("C8_GROUPING") == "1"
            and getattr(config.cache_config, "cache_dtype", None) == "int8"
            and getattr(config.speculative_config, "method", None) == "mtp")

def plan_groups(m, config, specs):
    """Recover pre-Ascend alignment C8 geometry without altering worker layers.

    Input has already been aligned by Ascend get_kv_cache_spec: target 3328/2P,
    BF16 draft 1664/2P and Mamba 1664/2P padded. Restore one logical token page
    size, keep per-layer physical page sizes and the exact GDN state contents.
    """
    parallel = config.parallel_config
    for name in ("tensor_parallel_size", "pipeline_parallel_size", "decode_context_parallel_size", "prefill_context_parallel_size", "data_parallel_size"):
        require(getattr(parallel, name, 1) == 1, f"TP1/PP1/CP1/DP1 only: {name}")
    require(config.kv_transfer_config is None, "KV transfer not validated")
    require(config.cache_config.mamba_cache_mode == "align", "align Mamba mode required")
    require(config.cache_config.block_size == B, f"configured manager block != {B}")
    require(config.cache_config.get_resolved_kv_cache_layout().name == "LBNHC", "expected contiguous LBNHC")
    require(config.speculative_config.num_speculative_tokens == 3, "dynamic K max 3 required")
    require(frozenset(specs) == ALL, f"expected exact model 65 layers; got {len(specs)}, missing={sorted(ALL-specs.keys())}, extra={sorted(specs.keys()-ALL)}")
    main = {}
    for name in TARGET:
        t = specs[name]
        require(type(t) is m.FullAttentionSpec and str(t.dtype) == "torch.int8", f"unexpected target spec {name}: {t}")
        require(t.block_size in (B, 2*B) and t.real_page_size_bytes == t.block_size * P//B, f"target geometry {name}: {t}")
        require(t.page_size_bytes in (P, 2*P) and t.num_kv_heads == 4 and t.head_size == t.head_size_v == 256, f"target dense layout {name}: {t}")
        require(t.sliding_window is None and t.attention_chunk_size is None and not t.non_causal, f"target full causal attention {name}")
        main[name] = replace(t, block_size=B, page_size_padded=None)
        require(main[name].page_size_bytes == P, f"target restored page {name}")
    d = specs[DRAFT]
    require(type(d) is m.FullAttentionSpec and str(d.dtype) == "torch.bfloat16", f"preserve draft BF16: {d}")
    require(d.block_size == B and d.real_page_size_bytes == d.page_size_bytes == 2*P and d.num_kv_heads == 4 and d.head_size == d.head_size_v == 256, f"draft dense geometry: {d}")
    require(d.sliding_window is None and d.attention_chunk_size is None and not d.non_causal, "draft full causal attention")
    main[DRAFT] = replace(d, page_size_padded=None)
    mix = m.UniformTypeKVCacheSpecs.from_specs(main)
    require(mix is not None and mix.block_size == B and mix.page_size_bytes == S, "full attention wrapper must share token slots and sum pages")
    # Keep target as first representative when the scheduler unwraps the spec.
    a = m.KVCacheGroupSpec([*TARGET, DRAFT], mix)
    states = {}
    for name in STATE:
        s = specs[name]
        require(type(s) is m.MambaSpec and s.block_size == B and s.real_page_size_bytes <= P
                and s.page_size_bytes in (P, 2*P), f"GDN real state does not fit original P: {name} {s}")
        require(s.num_speculative_blocks == 3 and s.mamba_cache_mode == "align", f"GDN mode or lease mismatch: {name} {s}")
        states[name] = replace(s, page_size_padded=P)
        require(states[name].page_size_bytes == P, f"GDN padded page {name}")
    state_groups=[]
    for i in range(3):
        names=STATE[i::3]
        representative=states[names[0]]
        require(all(states[n] == representative for n in names), f"GDN state specs differ in group {i}")
        state_groups.append(m.KVCacheGroupSpec(list(names), representative))
    groups=[a, *state_groups]
    require(_keyed_groups(m,groups)=="mixed", "constructed group signature mismatch")
    print("[C8_GROUPING] grouped target16+draft1 and GDN3x16; token page=1664, target/GDN page=3407872, draft page=6815744, shared pool stride=61341696", flush=True)
    return groups

def expected_intervals(nblocks):
    """Exact byte interval for each layer's whole contiguous region."""
    require(nblocks > 0, "empty pool")
    out={name:(i*P*nblocks,P*nblocks,P) for i,name in enumerate(TARGET)}
    out[DRAFT]=(16*P*nblocks,2*P*nblocks,2*P)
    for i in range(3):
        out.update({name:(j*P*nblocks,P*nblocks,P) for j,name in enumerate(STATE[i::3])})
    return out

def certify_alloc(m, config, available_memory):
    """Verify stock upstream allocator's exact returned physical descriptors.

    Equal P slots and unique global block IDs guarantee target/GDN disjointness
    while different block IDs are live. Draft tail is disjoint from *all* state
    and target slots. The check is before NPU allocation, and detects layout
    drift/corruption rather than skipping all mixed-page validation blindly.
    """
    require(_keyed_groups(m, config.kv_cache_groups)=="mixed", "allocation group signature changed")
    n=config.num_blocks; total=S*n
    require(total <= available_memory, "planned bytes exceed budget")
    expect=expected_intervals(n)
    actual={}
    for x in config.kv_cache_tensors:
        require(x.size==total, f"allocation descriptor total {x.size} != {total}")
        require(not x.host_resident, "host tensor unexpected")
        for i,name in enumerate(x.layers):
            require(name not in actual, f"duplicate descriptor layer {name}")
            off=x.offset + i*x.layer_stride
            actual[name]=(off,x.layer_stride,x.block_stride)
            require(0<=off<off+x.layer_stride<=total, f"layer out of range {name}")
    require(actual==expect, f"allocation offsets/strides mismatch: {[(k, actual.get(k), v) for k,v in expect.items() if actual.get(k)!=v][:5]}")
    print("[C8_GROUPING] certified allocation " + json.dumps({"groups":4,"logical_blocks":n,"budget":available_memory,"pool_bytes":total,"block_bytes":S,"target_region":[0,16*P*n],"draft_exclusive_region":[16*P*n,total],"GDN_overlays_only_target":True,"descriptors":len(config.kv_cache_tensors)}), flush=True)
    return config

def patch_module(m):
    if getattr(m,"_c8_grouping_installed",False):return
    old_group=m.get_kv_cache_groups
    old_validate=m.validate_kv_cache_layout
    old_bytes=m._get_kv_cache_bytes_per_block
    old_alloc=m.get_kv_cache_config_from_groups
    @functools.wraps(old_group)
    def grouped(vllm_config, kv_cache_spec):
        if _scope(vllm_config):
            return plan_groups(m,vllm_config,kv_cache_spec)
        return old_group(vllm_config,kv_cache_spec)
    @functools.wraps(old_validate)
    def validate(layout, groups):
        if _keyed_groups(m,groups)=="mixed":
            require(layout.name == "LBNHC", "special layout only validated for LBNHC")
            return
        return old_validate(layout,groups)
    @functools.wraps(old_bytes)
    def block_bytes(groups):
        mode=_keyed_groups(m,groups,normalized_ok=True)
        if mode:
            # Worker wrapper sums true mixed pages itself; scheduler unwraps
            # first target spec and must still account for the BF16 tail.
            if mode=="mixed":require(old_bytes(groups)==S,"stock mixed bytes accounting changed")
            return S
        return old_bytes(groups)
    @functools.wraps(old_alloc)
    def allocate(vllm_config, kv_cache_groups, available_memory):
        out=old_alloc(vllm_config,kv_cache_groups,available_memory)
        if _keyed_groups(m,kv_cache_groups)=="mixed":
            return certify_alloc(m,out,available_memory)
        return out
    m.get_kv_cache_groups=grouped
    m.validate_kv_cache_layout=validate
    m._get_kv_cache_bytes_per_block=block_bytes
    m.get_kv_cache_config_from_groups=allocate
    m._c8_grouping_installed=True

def _verify_hashes(origin):
    p=Path(origin).resolve()
    vllm_root=p.parents[3]
    # Locate the actual source trees on the tested container, not pip metadata.
    # The import hook verifies the core file before running it. Sibling sources
    # must match the read-only snapshots used to establish pointer invariants.
    ascend_root=Path(os.environ["C8_GROUPING_ASCEND_ROOT"])
    for rel, want in SOURCE_SHA256.items():
        src=(vllm_root if rel.startswith("vllm/") else ascend_root)/rel
        require(src.is_file() and hashlib.sha256(src.read_bytes()).hexdigest()==want, f"source hash changed: {src}")

class _Loader:
    def __init__(self, src_loader, origin):self.src_loader=src_loader;self.origin=origin
    def create_module(self,spec):return self.src_loader.create_module(spec)
    def exec_module(self,module):
        _verify_hashes(self.origin)
        self.src_loader.exec_module(module)
        patch_module(module)

class _Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != MODULE:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        require(spec is not None and spec.loader is not None and spec.origin is not None,"unable to locate KV planner source")
        spec.loader=_Loader(spec.loader, spec.origin)
        return spec

def install():
    if os.environ.get("C8_GROUPING")!="1":return
    require(MODULE not in sys.modules, "install grouping import hook before KV planner is imported; Ascend monkeypatch captures original functions")
    if any(isinstance(x,_Finder) for x in sys.meta_path):return
    sys.meta_path.insert(0,_Finder())
