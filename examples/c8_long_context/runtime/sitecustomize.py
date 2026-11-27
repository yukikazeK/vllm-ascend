# SPDX-License-Identifier: Apache-2.0
"""Verify the candidate's scales and cache dtype in the isolated load-check service."""
import importlib.machinery
import json
import os
from pathlib import Path
import re
import runpy
import sys

# Terminal hook in dynamic-runtime -> capacity-audit -> C8 chain.
# Do not rediscover/chainsitecustomize; that would recurse back into the dynamic runtime.
from audit_io import append,rank

TARGET = "vllm_ascend.quantization.methods.kv_cache.kv_c8"
PATH = "vllm_ascend.attention.attention_v1"


def log(run, row):
    append(run/"runtime-nd-audit.jsonl",row)


def audit_load(module, run):
    from safetensors.torch import load_file
    import torch

    source = Path(os.environ["C8_CHECKPOINT"]) / "quant_model_weights-kv-cache-c8.safetensors"
    expected = load_file(source)
    original = module.AscendC8KVCacheAttentionMethod.process_weights_after_loading

    def check(self, layer):
        original(self, layer)
        idx = re.search(r"\.layers\.(\d+)\.", layer.layer_name).group(1)
        for prefix, tag in (("k", "k_proj"), ("v", "v_proj")):
            for field in ("scale", "offset"):
                lookup = f".layers.{idx}.self_attn.{tag}.kv_cache_{field}"
                [target] = [val for key, val in expected.items() if key.endswith(lookup)]
                actual = getattr(layer, f"{prefix}_cache_{field}").detach().flatten().cpu()
                want=target.flatten()
                if actual.numel()!=want.numel():
                    shard_size=layer.impl.num_kv_heads*layer.impl.head_size
                    if actual.numel()!=shard_size:
                        raise RuntimeError(f"C8 calibration unexpected TP shard size: {lookup} {actual.shape}")
                    want=want[rank()*shard_size:(rank()+1)*shard_size]
                if not torch.equal(actual,want.to(actual.dtype)):
                    raise RuntimeError(f"C8 calibration parameter did not load: {lookup} {actual.shape}")
        log(run, {"kind": "scales_loaded", "layer": int(idx), "channels": layer.k_cache_scale.numel()})

    module.AscendC8KVCacheAttentionMethod.process_weights_after_loading = check


def audit_cache(module, run):
    import torch

    original = module.AscendC8AttentionBackendImpl.forward
    seen = set()

    def check(self, layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs):
        if attn_metadata is not None and layer.layer_name not in seen:
            if len(kv_cache) < 2 or kv_cache[0].dtype != torch.int8 or kv_cache[1].dtype != torch.int8:
                raise RuntimeError(f"Cache is not INT8 for {layer.layer_name}: {[k.dtype for k in kv_cache]}")
            seen.add(layer.layer_name)
            idx = int(re.search(r"\.layers\.(\d+)\.", layer.layer_name).group(1))
            log(run, {"kind": "int8_cache", "layer": idx, "shape": list(kv_cache[0].shape)})
        return original(self, layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs)

    module.AscendC8AttentionBackendImpl.forward = check

    # The MTP draft full-attention layer is uncalibrated and must retain
    # BF16 KV. Detect incorrect INT8 inheritance explicitly.
    base_original=module.AscendAttentionBackendImpl.forward
    non_c8_seen=set()
    def base_check(self,layer,query,key,value,kv_cache,attn_metadata,*args,**kwargs):
        if attn_metadata is not None and layer.layer_name not in non_c8_seen and len(kv_cache)>=2:
            if any(t.dtype==torch.int8 for t in kv_cache[:2]):
                raise RuntimeError(f"Uncalibrated non-C8 full-attention cache is INT8: {layer.layer_name}; "
                                   "skip MTP layer0 cache dtype or verify actual draft prefix")
            non_c8_seen.add(layer.layer_name)
            log(run,{"kind":"non_c8_cache","layer_name":layer.layer_name,
                     "dtypes":[str(t.dtype) for t in kv_cache[:2]],"shape":list(kv_cache[0].shape)})
        return base_original(self,layer,query,key,value,kv_cache,attn_metadata,*args,**kwargs)
    module.AscendAttentionBackendImpl.forward=base_check


class Loader(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        super().exec_module(module)
        run = Path(os.environ["C8_AUDIT_RUN"])
        if module.__name__ == TARGET:
            audit_load(module, run)
        elif module.__name__ == PATH:
            audit_cache(module, run)
            if os.environ.get("C8_ND_RUN"):
                from c8_nd_backend import adapt_attention
                adapt_attention(module)


class Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in (TARGET, PATH):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec and spec.origin:
                spec.loader = Loader(fullname, spec.origin)
                return spec
        return None


if os.environ.get("C8_AUDIT_RUN") and not getattr(sys,"_c8_capacity_overlay_hook",False):
    sys._c8_capacity_overlay_hook=True
    sys.meta_path.insert(0, Finder())
