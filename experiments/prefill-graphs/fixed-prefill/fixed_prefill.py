import copy,dataclasses,json,os,time
from pathlib import Path
import torch
from vllm.forward_context import get_forward_context
root=Path(os.environ['FIXED_GRAPH_DIR'])
graphs={}
pool=None
inside_capture=False
def log(event,**data):
    with (root/'replay.jsonl').open('a') as f:print(json.dumps({'event':event,**data}),file=f)
    print('FIXED_PREFILL',event,data,flush=True)

class Changed(Exception):
    pass

# A per-graph shadow of metadata. Shared layer metadata is memoized on each traversal.
def shadow(src,dst,memo,path,init):
    if src is None or isinstance(src,(bool,int,float,str)):
        if not init and src!=dst:raise Changed(path)
        return src
    key=id(src)
    if key in memo:return memo[key]
    if isinstance(src,torch.Tensor):
        if init:
            result=src.clone()
        else:
            if src.shape!=dst.shape or src.dtype!=dst.dtype or src.device!=dst.device:raise Changed(path)
            if src.device.type=='cpu':
                if not torch.equal(src,dst):raise Changed(path)
            else:dst.copy_(src,non_blocking=True)
            result=dst
    elif isinstance(src,dict):
        if not init and src.keys()!=dst.keys():raise Changed(path)
        result={k:shadow(v,None if init else dst[k],memo,path+'/'+str(k),init) for k,v in src.items()}
    elif isinstance(src,(list,tuple)):
        if not init and len(src)!=len(dst):raise Changed(path)
        values=[shadow(x,None if init else dst[i],memo,path,init) for i,x in enumerate(src)]
        result=type(src)(values)
    elif dataclasses.is_dataclass(src):
        result=copy.copy(src) if init else dst
        names={f.name for f in dataclasses.fields(src)}
        names.update(getattr(src,'__dict__',{}))
        if not init and set(getattr(src,'__dict__',{}))!=set(getattr(dst,'__dict__',{})):raise Changed(path+'/dynamic_fields')
        for name in sorted(names):
            val=shadow(getattr(src,name),None if init else getattr(dst,name),memo,path+'/'+name,init)
            object.__setattr__(result,name,val)
    else:
        if not init and src is not dst and src!=dst:raise Changed(path)
        result=src
    memo[key]=result
    return result

def run(runner,meta,q,computed,ids,pos,inter,emb,kwargs):
    global pool,inside_capture
    ordinary=lambda:runner._model_forward(q,ids,pos,inter,emb,**kwargs)
    tagfile=root/'tag';tag=tagfile.read_text().strip() if tagfile.exists() else ''
    if not tag.startswith('graph'):return ordinary()
    key=(q,computed)
    context=get_forward_context()
    if context.cudagraph_runtime_mode.name!='NONE' or context.attn_metadata is not meta:
        log('fallback',tag=tag,reason='unexpected dispatch or metadata');return ordinary()
    g=graphs.get(key)
    # Capture only when explicitly armed, then reuse that exact sequence position.
    if g is None and 'capture' not in tag:
        log('fallback',tag=tag,reason='uncaptured position',offset=computed);return ordinary()
    inputs=(ids,pos,inter,emb,kwargs)
    if g is None:
        if pool is None:pool=torch.npu.graph_pool_handle()
        mem0=(torch.npu.memory_allocated(),torch.npu.memory_reserved())
        pinned=shadow(meta,None,{},'meta',True)
        inp=shadow(inputs,None,{},'inputs',True)
        # One synchronised graph pool shared by these serial single-request forwards.
        torch.npu.synchronize()
        graph=torch.npu.NPUGraph()
        log('capture_begin',tag=tag,q=q,offset=computed)
        original=context.attn_metadata
        backups=[]
        layers=runner.vllm_config.compilation_config.static_forward_context
        for layer,md in meta.items():
            if hasattr(md,'prefill_state_indices') and layer in layers:
                indices=(md.non_spec_prefill_metadata.causal_conv1d.cache_indices.reshape(1,-1)[:,0],md.prefill_state_indices.reshape(-1))
                for cache,idx in zip(layers[layer].kv_cache[:2],indices):
                    idx=idx.to(cache.device,dtype=torch.long)
                    backups.append((cache,idx,cache.index_select(0,idx)))
        torch.npu.synchronize()
        try:
            context.attn_metadata=pinned
            inside_capture=True
            with torch.npu.graph(graph,pool=pool):
                out=runner._model_forward(q,*inp[:4],**inp[4])
        finally:
            inside_capture=False
            context.attn_metadata=original
        torch.npu.synchronize()
        for cache,idx,backup in backups:
            cache.index_copy_(0,idx,backup)
        graph.replay()
        torch.npu.synchronize()
        del backups,backup
        graphs[key]=(graph,pinned,inp,out)
        log('captured',tag=tag,q=q,offset=computed,allocated_change=int(torch.npu.memory_allocated()-mem0[0]),reserved_change=int(torch.npu.memory_reserved()-mem0[1]))
        return out
    graph,pinned,inp,out=g
    try:
        shadow(meta,pinned,{},'meta',False)
        shadow(inputs,inp,{},'inputs',False)
    except Changed as exc:
        log('fallback',tag=tag,q=q,offset=computed,reason=str(exc));return ordinary()
    graph.replay()
    log('replay',tag=tag,q=q,offset=computed)
    return out

def observe(runner,meta,logits,computed,scheduled,prompt):
    path=root/'tag'
    if not path.exists():return
    tag=path.read_text().strip()
    if not tag or 'perf' in tag:return
    if len(computed)!=1:return
    start,n,total=int(computed[0]),int(scheduled[0]),int(prompt[0])
    if start>=total+1:return
    layers=runner.vllm_config.compilation_config.static_forward_context
    gnames=sorted(k for k,v in meta.items() if hasattr(v,'prefill_state_indices') and k in layers)
    fnames=sorted(k for k,v in meta.items() if hasattr(v,'block_tables') and k in layers and hasattr(v,'slot_mapping'))
    snap={'logits':logits.detach().cpu()[:1],'gdn_rows':{},'kv_pages':[]}
    if fnames:
        att=meta[fnames[0]]
        snap['kv_pages']=att.block_tables[0,:max(1,(start+n+895)//896)].detach().cpu().tolist()
    for k in dict.fromkeys([gnames[0],gnames[len(gnames)//2],gnames[-1]]):
        m=meta[k];ix=m.prefill_state_indices if start<total else m.non_spec_state_indices_tensor
        idx=int(ix.reshape(-1)[0].cpu())
        snap['gdn_rows'][k]=idx
        for j,cache in enumerate(layers[k].kv_cache[:2]):
            snap[k+str(j)]=cache[idx].detach().cpu()
    torch.save(snap,root/f'{tag}_{start}_{n}.pt')
