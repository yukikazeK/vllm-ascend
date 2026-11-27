# SPDX-License-Identifier: Apache-2.0
"""Optional startup-only runtime tensor-view validation for the four-group planner."""
from __future__ import annotations
from functools import wraps
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import sys
from c8_grouping import (ALL, B, DRAFT, P, S, SOURCE_SHA256, STATE, TARGET,
                         _keyed_groups, expected_intervals, require)
MODULE="vllm_ascend.worker.v2.attn_utils"

def patch_worker(m):
    old=m._allocate_kv_cache
    reshape=m._reshape_kv_cache_v2
    @wraps(old)
    def allocate(kv_cache_config, shared_layers, device):
        raw=old(kv_cache_config,shared_layers,device)
        if _keyed_groups(m,kv_cache_config.kv_cache_groups)!="mixed":return raw
        n=kv_cache_config.num_blocks
        require(set(raw)==ALL and not shared_layers,"actual layer storage coverage mismatch")
        bases={t.untyped_storage().data_ptr() for t in raw.values()}
        require(len(bases)==1,"actual raw backing not shared")
        base=bases.pop()
        require(base!=0,"actual backing pointer unavailable")
        # No pointer addresses in logs; offsets are relative to one allocation.
        for name,(offset,layer_size,page) in expected_intervals(n).items():
            t=raw[name]
            require(str(t.dtype)=="torch.int8" and t.is_contiguous(),f"{name} raw dtype/contiguity")
            require(t.numel()==layer_size and t.data_ptr()-base==offset,f"{name} raw bytes/pointer mismatch")
        print("[C8_GROUPING] actual raw tensor pointer intervals certified " + json.dumps({"views":len(raw),"blocks":n,"pool_bytes":S*n}),flush=True)
        return raw
    @wraps(reshape)
    def checked(*args,**kwargs):
        out=reshape(*args,**kwargs)
        cfg=kwargs.get("kv_cache_config")
        # Current pinned source calls by keyword. If this changes, fail closed
        # when a caller passes the grouped config positionally.
        if cfg is None and len(args)>=6:cfg=args[5]
        if cfg is None or _keyed_groups(m,cfg.kv_cache_groups)!="mixed":return out
        raw=kwargs.get("kv_cache_raw_tensors",args[1] if len(args)>1 else None)
        kbs=kwargs.get("kernel_block_sizes",args[3] if len(args)>3 else None)
        require(kbs is not None and list(kbs)==[128,B,B,B],f"kernel block geometries differ: {kbs}")
        n=cfg.num_blocks
        for name in (*TARGET,DRAFT):
            k,v=out[name];base=raw[name].data_ptr()
            dtype="torch.bfloat16" if name==DRAFT else "torch.int8"
            half=128*4*256*(2 if name==DRAFT else 1)
            require(tuple(k.shape)==(13*n,128,4,256) and tuple(k.shape)==tuple(v.shape),f"{name} shape")
            require(str(k.dtype)==str(v.dtype)==dtype,f"{name} dtype")
            require(k.stride()==v.stride()==(2*128*4*256,4*256,256,1),f"{name} interleaving")
            require(k.data_ptr()==base and v.data_ptr()-base==half,f"{name} K/V pointer mismatch")
            require(v.data_ptr()+((13*n-1)*2*half+half)==base+raw[name].numel(),f"{name} final V bound")
        for name in STATE:
            base=raw[name].data_ptr()
            for t in out[name]:
                require(t.shape[0]==n and t.stride(0)*t.element_size()==P,f"{name} GDN stride")
                start=t.data_ptr()-base
                payload=t[0].numel()*t.element_size()  # view only, no device copy/compute
                require(0<=start and start+payload<=P,f"{name} GDN state out of page")
        print("[C8_GROUPING] actual K/V aliases and GDN views certified " + json.dumps({"target_int8":16,"draft_bf16":1,"gdn":48,"kernel_block_sizes":list(kbs),"blocks":n}),flush=True)
        return out
    m._allocate_kv_cache=allocate
    m._reshape_kv_cache_v2=checked

class _Loader:
    def __init__(self,loader,origin):self.loader=loader;self.origin=origin
    def create_module(self,spec):return self.loader.create_module(spec)
    def exec_module(self,module):
        rel="vllm_ascend/worker/v2/attn_utils.py"
        require(hashlib.sha256(Path(self.origin).read_bytes()).hexdigest()==SOURCE_SHA256[rel],"worker pointer auditor source hash mismatch")
        self.loader.exec_module(module)
        patch_worker(module)

class _Finder:
    def find_spec(self,fullname,path=None,target=None):
        if fullname!=MODULE:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        require(spec is not None and spec.loader is not None and spec.origin is not None,"worker source absent")
        spec.loader=_Loader(spec.loader,spec.origin)
        return spec

def install():
    if os.environ.get("C8_GROUPING")!="1":return
    require(MODULE not in sys.modules,"worker module already imported")
    if not any(isinstance(x,_Finder) for x in sys.meta_path):sys.meta_path.insert(0,_Finder())
