# SPDX-License-Identifier: Apache-2.0
"""CPU-only planner regression using the exact installed source functions, no NPU.

Spec classes here are minimal host models of the verified dataclasses. The
extracted allocator/stride/capacity functions are unmodified source ASTs. An
interval simulation checks all cross-group view roles at representative block IDs; equal width/offset geometry proves disjointness for every distinct ID.
"""
from __future__ import annotations
import ast
from collections import defaultdict
from dataclasses import dataclass, replace
import copy
import contextlib
import io
import unittest
from unittest.mock import patch
import hashlib
import math
from math import prod
import os
from pathlib import Path
import sys
from types import SimpleNamespace
# Explicitly opt in to testing actual pinned upstream source. The ordinary
# host suite is independent of vLLM, torch, CANN and this environment variable.
RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
sys.path.insert(0, str(RUNTIME))
import c8_grouping as C
import hybrid_sparse_retention as retention

SOURCE_ROOT = os.environ.get("C8_TEST_VLLM_SOURCE")
def source_dir():
    if not SOURCE_ROOT:
        return None
    root = Path(SOURCE_ROOT)
    for candidate in (root / "vllm" / "v1", root / "v1", root):
        if (candidate / "core" / "kv_cache_utils.py").is_file():
            return candidate
    raise ValueError("C8_TEST_VLLM_SOURCE must point to the matching vLLM checkout or package")
SRC=source_dir()
B,P,S=C.B,C.P,C.S

@dataclass(frozen=True)
class FullAttentionSpec:
    block_size:int
    dtype:str
    num_kv_heads:int=4
    head_size:int=256
    head_size_v:int=256
    page_size_padded:int|None=None
    sliding_window:int|None=None
    attention_chunk_size:int|None=None
    non_causal:bool=False
    @property
    def num_heads(self):return self.num_kv_heads
    @property
    def state_content_size_bytes(self):return (self.head_size+self.head_size_v)*(1 if self.dtype=='torch.int8' else 2)
    @property
    def real_page_size_bytes(self):return self.num_heads*self.state_content_size_bytes*self.block_size
    @property
    def page_size_bytes(self):return self.page_size_padded or self.real_page_size_bytes
    def get_num_kernel_states(self,bs):return bs
    def max_memory_usage_bytes(self,v):return math.ceil(v.model_config.max_model_len/self.block_size)*self.page_size_bytes

@dataclass(frozen=True)
class MambaSpec:
    block_size:int
    real_page_size_bytes:int
    page_size_padded:int
    mamba_cache_mode:str='align'
    num_speculative_blocks:int=3
    num_prefill_checkpoint_blocks:int=0
    @property
    def page_size_bytes(self):return self.page_size_padded or self.real_page_size_bytes
    num_heads=1
    @property
    def state_content_size_bytes(self):return self.real_page_size_bytes
    def get_num_kernel_states(self,bs):return 1
    def max_memory_usage_bytes(self,v):return (2+self.num_speculative_blocks+self.num_prefill_checkpoint_blocks)*self.page_size_bytes

@dataclass(frozen=True)
class UniformTypeKVCacheSpecs:
    block_size:int
    kv_cache_specs:dict
    @staticmethod
    def from_specs(ss):
        b={x.block_size for x in ss.values()}
        if len(b)!=1 or not all(isinstance(x,FullAttentionSpec) for x in ss.values()):return None
        return UniformTypeKVCacheSpecs(b.pop(),ss)
    @property
    def page_size_bytes(self):return sum(x.page_size_bytes for x in self.kv_cache_specs.values())
    def max_memory_usage_bytes(self,v):return self.page_size_bytes*max(x.max_memory_usage_bytes(v)//x.page_size_bytes for x in self.kv_cache_specs.values())
    def max_memory_usage_pages(self,v):return self.max_memory_usage_bytes(v)//self.page_size_bytes

@dataclass
class KVCacheGroupSpec:
    layer_names:list
    kv_cache_spec:object
    is_eagle_group:bool=False
    host_resident:bool=False

@dataclass
class KVCacheTensor:
    size:int
    layers:list
    layer_stride:int
    block_stride:int
    offset:int=0
    host_resident:bool=False

@dataclass
class KVCacheConfig:
    num_blocks:int
    kv_cache_tensors:list
    kv_cache_groups:list
    prefix_cache_retention_interval:int|None=None
    hisparse_host_num_blocks:int|None=None

class HiSparseHotSpec:pass

def extract(path,names,ns):
    src=path.read_text();nodes=[n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    # Actual source annotations use vllm types not needed in this host fixture.
    # Future annotations avoid importing vLLM/torch; all function bodies remain intact.
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias('annotations')],level=0),*nodes],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)

def allocator_with_pinned_source():
    k=SRC/'core/kv_cache_utils.py';i=SRC/'kv_cache_interface.py'
    assert hashlib.sha256(k.read_bytes()).hexdigest()==C.SOURCE_SHA256['vllm/v1/core/kv_cache_utils.py']
    assert hashlib.sha256(i.read_bytes()).hexdigest()==C.SOURCE_SHA256['vllm/v1/kv_cache_interface.py']
    # Use a real module namespace so stock functions resolve the patched global
    # validate()/bytes() when invoked by the wrapped allocator.
    import types
    m=types.ModuleType('host_kv_utils');ns=m.__dict__
    ns.update(globals())
    exec((SRC/'kv_cache_layout.py').read_text(),ns)
    extract(i,{'compute_layout_strides','compute_layer_kv_cache_shape_bytes'},ns)
    ns.update({'_glm5_next_tensor_layout':lambda _:None,'logger':SimpleNamespace(info=lambda *a:None),'cdiv':lambda a,b:(a+b-1)//b,'round_up':lambda a,b:math.ceil(a/b)*b,'get_kv_cache_groups':lambda v,s: (_ for _ in ()).throw(AssertionError('unexpected stock fallback'))})
    extract(k,{'_get_per_layer_spec','_get_kv_cache_bytes_per_block','validate_kv_cache_layout','get_kv_cache_config_from_groups','may_override_num_blocks','generate_scheduler_kv_cache_config','_pool_bytes_per_block','_max_memory_usage_bytes_from_groups','get_max_concurrency_for_kv_cache_config'},ns)
    C.patch_module(m)
    cfg=SimpleNamespace(cache_config=SimpleNamespace(cache_dtype='int8',mamba_cache_mode='align',block_size=B,get_resolved_kv_cache_layout=lambda:ns['KVCacheLayout'].LBNHC,prefix_cache_retention_interval=None,num_gpu_blocks_override=None),attention_config=SimpleNamespace(hisparse_config=None),parallel_config=SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1,decode_context_parallel_size=1,data_parallel_size=1),speculative_config=SimpleNamespace(method='mtp',num_speculative_tokens=3),kv_transfer_config=None,model_config=SimpleNamespace(max_model_len=262144))
    specs={n:FullAttentionSpec(B*2,'torch.int8') for n in C.TARGET};specs[C.DRAFT]=FullAttentionSpec(B,'torch.bfloat16');specs.update({n:MambaSpec(B,3_200_000,2*P) for n in C.STATE})
    os.environ['C8_GROUPING']='1'
    groups=m.get_kv_cache_groups(cfg,specs)
    assert len(groups)==4 and len(groups[0].layer_names)==17
    assert m._get_kv_cache_bytes_per_block(groups)==S
    # Exact 25.43 GiB budget from 4006 stock 2P blocks, use same available bytes.
    budget=4006*2*P
    pool=m.get_kv_cache_config_from_groups(cfg,groups,budget)
    assert pool.num_blocks==budget//S==445
    assert m._max_memory_usage_bytes_from_groups(cfg,groups)==S*(math.ceil(262144/B)+15)
    scheduler=m.generate_scheduler_kv_cache_config([pool]);assert type(scheduler.kv_cache_groups[0].kv_cache_spec) is FullAttentionSpec
    assert m._get_kv_cache_bytes_per_block(scheduler.kv_cache_groups)==S
    assert m.get_max_concurrency_for_kv_cache_config(cfg,pool)==m.get_max_concurrency_for_kv_cache_config(cfg,scheduler)
    # Exact allocator offsets tested by certify_alloc. Boundary-interval proof:
    # select several live blocks (one global block id belongs to one group).
    # All 65 layer storage regions share slots intentionally, but no actual live
    # page interval overlaps with another live group's distinct block ID.
    intervals=C.expected_intervals(pool.num_blocks)
    sample=[0,1,2,pool.num_blocks//2,pool.num_blocks-1]
    for gid_a,ga in enumerate(groups):
        for gid_b,gb in enumerate(groups):
            if gid_a==gid_b:continue
            for a in ga.layer_names:
                for b in gb.layer_names:
                    for ia in sample:
                        for ib in sample:
                            if ia==ib:continue
                            sa,_,pa=intervals[a];sb,_,pb=intervals[b]
                            left=sa+ia*pa;right=sb+ib*pb
                            assert left+pa<=right or right+pb<=left,(a,b,ia,ib)
    # All per-layer raw slices are contiguous. For 128-token physical blocks,
    # exact K/V pointer intervals within manager block satisfy C8 ND alias bounds.
    for name,(base,nbytes,page) in intervals.items():
        if name in C.STATE:continue
        physical=page//13;half=physical//2;pages=pool.num_blocks*13
        last_k=base+(pages-1)*physical
        last_v=last_k+half
        assert last_v+half==base+nbytes
        assert (last_k-base)==(2*pages-2)*half
        assert half==128*4*256*(2 if name==C.DRAFT else 1)
    # Fail closed on unsupported state-size / geometry and corrupt allocations.
    bad=dict(specs);bad[C.STATE[0]]=replace(bad[C.STATE[0]],real_page_size_bytes=P+1)
    try:m.get_kv_cache_groups(cfg,bad);raise AssertionError('bad GDN state passed')
    except RuntimeError as e:assert 'GDN real state' in str(e)
    corrupt=copy.deepcopy(pool);corrupt.kv_cache_tensors[0].offset+=P
    try:C.certify_alloc(m,corrupt,budget);raise AssertionError('overlap offset passed')
    except RuntimeError as e:assert 'offsets/strides mismatch' in str(e)
    print('PASS 4-group layout, actual upstream allocator/stride/capacity functions, 445 blocks, 61,341,696 bytes/block, exact descriptor and alias interval proof, all failure guards')
@unittest.skipUnless(SOURCE_ROOT, "set C8_TEST_VLLM_SOURCE for pinned-source AST regression")
class PinnedUpstreamTests(unittest.TestCase):
    def test_actual_allocator_and_intervals(self):
        with patch.dict(os.environ, {"C8_GROUPING": "1"}), contextlib.redirect_stdout(io.StringIO()):
            allocator_with_pinned_source()

    def test_native_reachable_block_mask(self):
        # Extract the exact native mask without importing device runtime modules.
        path = SRC / "core" / "single_type_kv_cache_manager.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "0233531d85aeced9e57a38c76d83db970519c92902b7e55484390cb2b64985bf")
        tree = ast.parse(path.read_text())
        manager = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "MambaManager")
        mask = next(x for x in manager.body if isinstance(x, ast.FunctionDef) and x.name == "reachable_block_mask")
        mask.decorator_list = []
        script = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), mask],
                            type_ignores=[])
        ns = {"MambaSpec": MambaSpec}
        exec(compile(ast.fix_missing_locations(script), str(path), "exec"), ns)
        native = ns["reachable_block_mask"]
        for block, page, _ in retention.GEOMETRIES.values():
            h = 38 * block
            spec = MambaSpec(block, page, page)
            self.assertIsNone(native(None, 0, 38, 128, spec, False, None, (h,)))
            for prompt, expected in ((h, [36, 37]), (h + 1, [37])):
                with self.subTest(prompt=prompt):
                    selected = native(None, 0, 38, 128, spec, False, 0,
                                      retention.boundaries(prompt, 128))
                    self.assertEqual([i for i, keep in enumerate(selected) if keep], expected)

if __name__ == "__main__":
    unittest.main()
