# SPDX-License-Identifier: Apache-2.0
"""Verified primitive folded GQA LSE-prefix + causal-tail service graph adapter.

Opt-in only; capture TP1 uniform Q4 C1 by default. Graph replay uses existing
V2 persistent slot_mapping, not newly allocated capture-only metadata buffers.
"""
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any


def enabled():return os.environ.get('C8_FOLD')=='1'
def widths():return tuple(map(int,os.environ.get('C8_FOLD_WIDTHS','4').split(',')))
def max_reqs():return int(os.environ.get('C8_FOLD_MAX_REQS','1'))


def eligible(q_ends, actual_tokens, physical_tokens, allowed_widths=(4,), limit=1):
    """Select immutable graph capture topology, not a runtime route switch."""
    q_ends=list(map(int,q_ends));physical_tokens=int(physical_tokens)
    if physical_tokens not in range(2,limit*max(allowed_widths)+1):return None,'physical_tokens_outside_fold_scope'
    q_sizes=[b-a for a,b in zip([0,*q_ends[:-1]],q_ends)]
    if not q_sizes or physical_tokens!=sum(q_sizes) or actual_tokens!=physical_tokens:
        return None,'non_exact_capture_token_geometry'
    if len(set(q_sizes))!=1 or q_sizes[0] not in allowed_widths:
        return None,'not_supported_uniform_capture_width'
    b=len(q_sizes);q=q_sizes[0]
    return ((b,q),'uniform_capture') if 1<=b<=limit else (None,'request_count_outside_fold_scope')


def real_prefixes(q_ends,kv_lens,actual_tokens,physical_tokens,b,width):
    """Runtime validation; capture topology is immutable and cannot be rerouted here.
    Real width must match captured uniform graph. Pads at tail may be absent or
    represented by extra cumulative boundaries. Padded output is never consumed.
    """
    a=int(actual_tokens);physical_tokens=int(physical_tokens)
    if physical_tokens!=b*width or not 0<a<=physical_tokens or a%width:
        raise RuntimeError(f'Invalid fold runtime token counts: actual={a}, physical={physical_tokens}, B={b}, Q={width}')
    n=a//width
    q_ends=list(map(int,q_ends));ls=list(map(int,kv_lens))
    if q_ends[:n]!=[width*(i+1) for i in range(n)] or len(ls)<n:
        raise RuntimeError(f'Fold runtime uniform width mismatch: {q_ends}, {ls}, {a}/{physical_tokens}')
    # A real decode verification must have >=1 prior token. Full-decode-only
    # uniform Q>1 never includes first-token prefill. Refuse unsafe routing.
    prefix=[l-width for l in ls[:n]]
    if any(l<1 for l in prefix):raise RuntimeError(f'Fold runtime has no common history: {prefix}')
    return prefix+[1]*(b-n),n

@dataclass(eq=False)
class FoldParams:
    layer_name:str
    batch:int
    width:int
    slot_view:Any
    # C8 full graph output buffers are captured for optional diagnostics only.
    snap:Any=None
    runtime_meta:Any=None
    proof_count:int=0
    @property
    def name(self):return self.layer_name
    def resolve(self, metadata):
        from c8_nd_backend import flat_args, trace
        m=metadata[self.layer_name]
        try:
            pref,n=real_prefixes(m.actual_seq_lengths_q,m.seq_lens_list,m.num_actual_tokens,self.batch*self.width,self.batch,self.width)
            if m.slot_mapping is None or m.slot_mapping.data_ptr()!=self.slot_view.data_ptr() or m.slot_mapping.dtype!=self.slot_view.dtype:
                raise RuntimeError(f'V2 slot_mapping captured buffer address/dtype changed: saved={slot_info(self.slot_view)}, live={slot_info(m.slot_mapping)}')
            if len(m.slot_mapping)<int(m.num_actual_tokens):raise RuntimeError('Missing fold live slot mapping entries')
            # Existing flat metadata has correct even-address aliases and
            # padded null rows. Shared per-metadata cache avoids per-layer copies.
            cached=getattr(m,'_c8_fold_entry',None)
            if cached is None or cached[:2]!=(self.batch,self.width):
                f=flat_args(m,self.batch*self.width,require_prepared=False)
                table=f['block_table'][::self.width].contiguous()
                m._c8_fold_entry=(self.batch,self.width,pref,table,n)
            else:_,_,pref,table,n=cached
        except Exception as exc:
            trace('fold_graph_replay_rejected',layer_name=self.layer_name,error=str(exc),
                physical_queries=self.batch*self.width,actual_tokens=int(m.num_actual_tokens))
            raise
        key=(self.batch,self.width,n)
        if key not in _REPLAY:
            _REPLAY.add(key)
            trace('fold_graph_replay',layer_name=self.layer_name,physical_queries=self.batch*self.width,
                  q_cumulative=list(m.actual_seq_lengths_q),q_lengths=[self.width],actual_tokens=int(m.num_actual_tokens),
                  kv_lengths=list(m.seq_lens_list)[:32],prefix_lens=pref[:32],real_requests=n,padded_requests=self.batch-n,
                  slot_mapping_pointer_stable=True,slot_source=slot_info(m.slot_mapping))
        if self.snap is not None and _quality_active() and self.proof_count<_proof_limit():self.runtime_meta=m
        else:self.runtime_meta=None
        return {'block_table':table,'actual_seq_lengths_kv':pref}

_REPLAY=set();_CAPTURE=set();_QUALITY_INSTALLED=False

def slot_info(slot):
    # Only view metadata; never inspect device values or copy to host.
    if slot is None:return None
    return {'dtype':str(slot.dtype),'shape':list(slot.shape),'stride':list(slot.stride()),'contiguous':bool(slot.is_contiguous()),'device':str(slot.device),'address':int(slot.data_ptr())}

def _quality_active():
    run=os.environ.get('C8_AUDIT_RUN')
    return bool(run and (Path(run)/'.enable_fold_quality').exists())
def _proof_limit():return int(os.environ.get('C8_FOLD_QUALITY_STEPS','3'))
def _quality_requested(name):
    return os.environ.get('C8_FOLD_VALIDATE')=='1' and any(name.endswith(f'layers.{n}.self_attn.attn') for n in os.environ.get('C8_FOLD_QUALITY_LAYERS','3,63').split(','))


def attach_scales(layer,torch):
    if not enabled() or hasattr(layer,'_c8_fold_masks'):return
    device=layer._c8_nd_combined_scale.device
    layer._c8_fold_k_scale=layer._c8_nd_combined_scale[0].reshape(1,1,-1,256).float()
    layer._c8_fold_v_scale=layer._c8_nd_combined_scale[1].reshape(1,1,-1,256).float()
    layer._c8_fold_masks={q:torch.triu(torch.ones((q,q),dtype=torch.bool),diagonal=1).to(device=device) for q in widths()}


def _tail_merge(torch,query,k,v,slot_view,layer,prefix,lse,b,width,hq,hk,d,block):
    # In V2 slot_mapping is physical_page*block+offset, refreshed in place
    # by BlockTables.compute_slot_mappings before graph execution. The even
    # alias used by both FIA and this gather maps physical page p -> 2*p.
    # Padding slots are PAD_SLOT_ID (-1); read harmless slot0, never write.
    # AscendBlockTables owns int32 (not the upstream int64); cast only Q slots
    # on device for the index_select compatibility tested by the primitive.
    real=slot_view[:b*width].to(torch.int64).clamp_min(0)
    ids=real+torch.div(real,block,rounding_mode='floor')*block
    K=k.reshape(-1,hk,d).index_select(0,ids).view(b,width,hk,d).float()*layer._c8_fold_k_scale
    V=v.reshape(-1,hk,d).index_select(0,ids).view(b,width,hk,d).float()*layer._c8_fold_v_scale
    group=hq//hk
    K=K.permute(0,2,1,3).repeat_interleave(group,dim=1)
    V=V.permute(0,2,1,3).repeat_interleave(group,dim=1)
    Q=query[:b*width].view(b,width,hq,d).permute(0,2,1,3).float()
    logits=(Q.unsqueeze(3)*K.unsqueeze(2)).sum(-1)*(d**-.5)
    logits=logits.masked_fill(layer._c8_fold_masks[width],float('-inf'))
    pref=prefix.view(b,hq,width,d).float();lse=lse.view(b,hq,width).float()
    peak=torch.maximum(logits.amax(-1),lse)
    pm=torch.exp(lse-peak);tw=torch.exp(logits-peak.unsqueeze(-1));mass=pm+tw.sum(-1)
    y=(pm.unsqueeze(-1)*pref+(tw.unsqueeze(-1)*V.unsqueeze(2)).sum(3))/mass.unsqueeze(-1)
    return y.transpose(1,2).reshape(b*width,hq,d).to(query.dtype)


def fold_capture(module,self,query,metadata,output,layer,args_for_decode):
    from c8_nd_backend import trace
    if not enabled():return False
    torch,torch_npu=module.torch,module.torch_npu
    physical=int(query.shape[0]);shape,reason=eligible(metadata.actual_seq_lengths_q,metadata.num_actual_tokens,physical,widths(),max_reqs())
    # Validated scope only. Disable when split context, non-causal,
    # per-layer head geometry differs, noncontiguous slot source, or KV transfer.
    cfg=self.vllm_config;par=cfg.parallel_config
    if shape is not None and (os.environ.get('C8_GROUPING')!='1' or any(getattr(par,k,1)!=1 for k in ('tensor_parallel_size','pipeline_parallel_size','decode_context_parallel_size','prefill_context_parallel_size','data_parallel_size')) or cfg.kv_transfer_config is not None):
        shape=None;reason='unsupported_parallel_or_transfer'
    elif shape is not None and ((self.num_heads,self.num_kv_heads,self.head_size)!=(24,4,256) or metadata.causal is not True or query.dtype!=torch.bfloat16):
        shape=None;reason='unsupported_head_geometry_or_dtype'
    elif shape is not None and (metadata.slot_mapping is None or metadata.slot_mapping.dtype not in (torch.int32,torch.int64) or metadata.slot_mapping.ndim!=1 or metadata.slot_mapping.numel()<physical or not metadata.slot_mapping.is_contiguous()):
        shape=None;reason='capture_slot_mapping_not_stable_shape'
    if shape is None:
        key=('flat',physical,tuple(metadata.actual_seq_lengths_q),reason)
        # q_cumulative tuples contain dummy capture geometry only, never prompts.
        if key not in _CAPTURE:_CAPTURE.add(key);trace('fold_capture_fallback',layer_name=self._layer_name,physical_queries=physical,reason=reason,q_cumulative=list(metadata.actual_seq_lengths_q),actual_tokens=int(metadata.num_actual_tokens),slot_source=slot_info(metadata.slot_mapping))
        return False
    b,width=shape
    # The V2 manager has completed its import by capture time; hooking here
    # avoids importing a partially initialized graph/attention module.
    install_quality_hook()
    if not hasattr(layer,'_c8_fold_masks') or width not in layer._c8_fold_masks:
        raise RuntimeError('Fold scale/mask cache not prepared before capture')
    base=args_for_decode(self,query,metadata,layer,physical,capture=True)
    # Dummy capture may have KV=Q or PAD_SLOT_ID. Prefix1 reads block0 and
    # dummy output is discarded. Real replay validates positive history.
    prefixes=[max(1,base['actual_seq_lengths_kv'][i*width]-1) for i in range(b)]
    table=base['block_table'][::width].contiguous()
    packed=query[:physical].view(b,width,self.num_heads,self.head_size).transpose(1,2).contiguous().view(b,self.num_heads*width,1,self.head_size)
    args={**base,'query':packed,'num_heads':self.num_heads*width,'block_table':table,
          'actual_seq_lengths_kv':prefixes,'softmax_lse_flag':True}
    k,v,block=base['key'],base['value'],base['block_size']
    prefix=torch.empty_like(packed);lse=torch.empty((b,self.num_heads*width,1,1),dtype=torch.float32,device=query.device)
    ws=module.get_capture_resource(('c8.fold.fia.workspace',b,width),
          lambda:torch_npu._npu_fused_infer_attention_score_get_max_workspace(**args),self._use_max_workspace_for_fia_graph)
    provider=FoldParams(self._layer_name,b,width,metadata.slot_mapping[:physical])
    module.register_task(torch_npu.npu_fused_infer_attention_score.out,{**args,'workspace':ws,'out':[prefix,lse]},provider)
    result=_tail_merge(torch,query,k,v,provider.slot_view,layer,prefix,lse,b,width,self.num_heads,self.num_kv_heads,self.head_size,block)
    output[:physical].copy_(result)
    if _quality_requested(self._layer_name):
        # Small captured snapshot prevents later layers reusing Q/out storage.
        # Reference op is NOT captured; it runs after replay only under marker.
        qs=torch.empty_like(query[:physical]);ys=torch.empty_like(output[:physical]);qs.copy_(query[:physical]);ys.copy_(output[:physical])
        provider.snap=(qs,ys,k,v,block,base['antiquant_scale'])
    key=('fold',b,width)
    if key not in _CAPTURE:
        _CAPTURE.add(key)
        trace('fold_graph_capture',layer_name=self._layer_name,physical_queries=physical,requests=b,width=width,virtual_heads=self.num_heads*width,quality_snapshot=provider.snap is not None,slot_source=slot_info(provider.slot_view),source_contract='AscendBlockTables.__init__ persistent int32; inherited get_dummy_slot_mappings returns same allocation')
    return True


def _validate_provider(provider):
    from c8_nd_backend import trace,flat_args
    import torch,torch_npu
    m=provider.runtime_meta
    if m is None:return
    provider.runtime_meta=None
    q,out,k,v,block,scale=provider.snap
    n=int(m.num_actual_tokens);width=provider.width
    # Match exact logical positions in actual per-group block table vs the
    # persistent physical slot mapping used by graph-captured tail gather.
    pos=torch.tensor([[l-width+j for j in range(width)] for l in m.seq_lens_list[:n//width]],dtype=torch.int64,device=q.device)
    expect=m.block_tables[:n//width].gather(1,torch.div(pos,block,rounding_mode='floor').long()).long()*block+torch.remainder(pos,block)
    slots_ok=bool(torch.equal(expect.flatten(),provider.slot_view[:n].to(torch.int64)))
    padding_slots_ok=bool((provider.slot_view[n:provider.batch*width]<0).all()) if n<provider.batch*width else True
    f=flat_args(m,provider.batch*width,require_prepared=False)
    ref,_=torch_npu.npu_fused_infer_attention_score(query=q.unsqueeze(2),key=k,value=v,
        **f,block_size=block,input_layout='BNSD',num_heads=24,num_key_value_heads=4,scale=256**-.5,
        sparse_mode=0,antiquant_scale=scale,antiquant_mode=0)
    a=out[:n].float();r=ref[:n].squeeze(2).float()
    finite=bool(torch.isfinite(a).all() and torch.isfinite(r).all());rel=float((a-r).norm()/r.norm().clamp_min(1e-6)) if finite else None
    mx=float((a-r).abs().max()) if finite else None
    limit=float(os.environ.get('C8_FOLD_REL_L2_MAX','.01'))
    ok=slots_ok and padding_slots_ok and finite and rel<=limit
    provider.proof_count+=1
    trace('fold_live_quality',layer_name=provider.layer_name,physical_queries=provider.batch*width,actual_tokens=n,
          kv_lens=list(m.seq_lens_list)[:n//width],slot_mapping_equals_live_block_table=slots_ok,padding_slots_ok=padding_slots_ok,real_requests=n//width,padded_requests=provider.batch-n//width,finite=finite,relative_l2=rel,max_abs=mx,limit=limit,ok=ok,diagnostic_stage='after_manager_submitted_replay_and_update')
    if not ok:raise RuntimeError(f'C8 fold live same-cache numerical check failed: slot={slots_ok}, pad={padding_slots_ok}, finite={finite}, rel_l2={rel}')


def install_quality_hook():
    global _QUALITY_INSTALLED
    if _QUALITY_INSTALLED or not enabled() or os.environ.get('C8_FOLD_VALIDATE')!='1':return
    from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
    from vllm_ascend.compilation.updatable_graph import UpdatableGraph
    orig=ModelAclGraphManager._updatable_graph_replay
    def replay(self,desc,attn_metadata,*a,**kw):
        # Target V2 queues replay BEFORE graph.update (the latter records
        # external events). Never synchronize/reference from NPUGraph.replay
        # itself: the external event on this stream has not been submitted!
        result=orig(self,desc,attn_metadata,*a,**kw)
        if _quality_active():
            graph=self.graphs[desc]
            if isinstance(graph,UpdatableGraph):
                # Both replay and update have now been submitted. Reference
                # op on the caller stream follows completed graph work; the
                # diagnostics may safely synchronize. Draft manager is untouched.
                for task in graph.tasks:
                    p=task.provider
                    if isinstance(p,FoldParams) and p.snap is not None and p.runtime_meta is not None:_validate_provider(p)
        return result
    ModelAclGraphManager._updatable_graph_replay=replay
    _QUALITY_INSTALLED=True
