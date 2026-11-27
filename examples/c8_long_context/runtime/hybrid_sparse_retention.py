# SPDX-License-Identifier: Apache-2.0
"""Opt-in grouped C8 and W8A8 standalone retention correction, 0.30.0 only.

Ascend's hybrid coordinator reads a retired vLLM env instead of KVCacheConfig's
prefix_cache_retention_interval, silently selecting dense GDN checkpoints.
Restore the config value (0), then select the local-reader replay boundaries
instead of the inherited coarse EAGLE-drop boundary calculation. The existing
Mamba cache_blocks/physical-page mask and native writes remain unchanged.
"""
from __future__ import annotations
import functools
import json
import os
from types import MethodType

MATCH=128
GEOMETRIES={
    "torch.int8": (1664,3_407_872,"C8"),
    "torch.bfloat16": (896,3_670_016,"W8A8"),
}
TARGET=tuple(f'language_model.model.layers.{i}.self_attn.attn' for i in range(3,64,4))
DRAFT='mtp.layers.0.self_attn.attn'
GDN=tuple(f'language_model.model.layers.{i}.linear_attn' for i in range(64) if i%4!=3)


def require(cond,message):
    if not cond:raise RuntimeError('[C8_SPARSE_RETENTION] unsupported: '+message)


def boundaries(prompt_tokens:int, alignment:int)->tuple[int,...]:
    """Largest local hash boundaries reachable by resend and continuation.

    Selecting a boundary is merely permission for the native cache writer to
    retain an already-computed state. It does not create a partial state or
    insert a key for an unwritten/null block. Mamba's own reachable mask maps
    fine boundaries to full physical state pages; its normal partial-tail
    writer may separately cache a genuinely exported fine state.
    """
    require(prompt_tokens>=1 and alignment>0,'invalid replay geometry')
    return tuple(sorted({((prompt_tokens-1)//alignment)*alignment,
                         (prompt_tokens//alignment)*alignment}))


def local_replay_boundaries(self,request):
    return boundaries(int(request.num_prompt_tokens),int(self._cache_hit_alignment_tokens))


def _is_candidate(cfg):
    groups=cfg.kv_cache_groups
    return len(groups)==4 and tuple(groups[0].layer_names)==(*TARGET,DRAFT) and str(getattr(groups[0].kv_cache_spec,'dtype',None)) in GEOMETRIES


def configure(kvm):
    cfg=kvm.kv_cache_config
    if not _is_candidate(cfg):return False
    groups=cfg.kv_cache_groups;c=kvm.coordinator
    B,P,variant=GEOMETRIES[str(groups[0].kv_cache_spec.dtype)]
    # Only the two exact observed Qwen 4-group TP1/DP1 geometries. The independent
    # grouping overlay enforces parallel settings and real physical pointers.
    require(type(c).__name__=='AscendHybridKVCacheCoordinator','actual hybrid coordinator class')
    require(tuple(tuple(g.layer_names) for g in groups[1:])==tuple(GDN[i::3] for i in range(3)),'GDN group layers')
    require(type(groups[0].kv_cache_spec).__name__=='FullAttentionSpec','normalized FA group required')
    require(groups[0].kv_cache_spec.block_size==B and groups[0].kv_cache_spec.page_size_bytes==P,'FA block/page geometry')
    require(all(type(g.kv_cache_spec).__name__=='MambaSpec' and g.kv_cache_spec.block_size==B and g.kv_cache_spec.page_size_bytes==P and g.kv_cache_spec.mamba_cache_mode=='align' and g.kv_cache_spec.num_prefill_checkpoint_blocks==0 and g.kv_cache_spec.num_speculative_blocks==3 for g in groups[1:]),'GDN geometry/mode/checkpoint scope')
    require(getattr(cfg,'kv_transfer_config',None) is None and getattr(c,'skips_eagle_block_drop',False),'standalone verified local reader only')
    require(c._cache_hit_alignment_tokens==MATCH and c.scheduler_block_size==B,'exact fine/hash and scheduler alignment')
    require(c.enable_partial_hash_hits and c.enable_caching,'prefix caching required')
    require(c.eagle_group_ids=={0},'single grouped FA EAGLE fallback')
    require(c.dcp_world_size==1 and len(c.single_type_managers)==4,'TP1/DCP1 grouped managers')
    require(c.num_reprefillable_tokens==0,'cannot optimize unverified multi-module MTP write boundaries')
    require(all(not getattr(g,'host_resident',False) for g in groups),'host cache not tested')
    require(len({id(m.block_pool) for m in c.single_type_managers})==1,'single pool')
    require(cfg.prefix_cache_retention_interval==0,'explicit --prefix-cache-retention-interval 0 needed')
    require(all(type(m).__name__ in ('MambaManager','AscendMambaManager') and not m.has_prefill_checkpoint_blocks for m in c.single_type_managers[1:]),'no in-flight checkpoint override')
    from vllm.v1.core.kv_cache_coordinator import _validate_prefix_cache_retention_interval
    _validate_prefix_cache_retention_interval(cfg.prefix_cache_retention_interval,c.scheduler_block_size,cfg)
    before=c.retention_interval
    c.retention_interval=cfg.prefix_cache_retention_interval
    c.get_replay_boundaries=MethodType(local_replay_boundaries,c)
    print('[C8_SPARSE_RETENTION] '+json.dumps({'previous_retention':before,'new_retention':c.retention_interval,'variant':variant,'policy':'local-resend-and-extension','gdn_groups':3,'scheduler_page_tokens':B,'hash_alignment':MATCH,'example_aligned_history':B*38,'example_replay_boundaries':boundaries(B*38,MATCH)}),flush=True)
    return True


def install():
    if os.environ.get('QWEN_HYBRID_SPARSE_RETENTION')!='1':return
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    if getattr(KVCacheManager,'_qwen_hybrid_sparse_retention_installed',False):return
    orig=KVCacheManager.__init__
    @functools.wraps(orig)
    def init(self,*a,**kw):
        orig(self,*a,**kw)
        configure(self)
    KVCacheManager.__init__=init
    KVCacheManager._qwen_hybrid_sparse_retention_installed=True
