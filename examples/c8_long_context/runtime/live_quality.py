# SPDX-License-Identifier: Apache-2.0
"""Optional eager same-real-pages numerical diagnostic; disable for timing."""
import os
from pathlib import Path
import torch
from audit_io import append
_seen=set()
def eager_check(name,args,output,metadata):
    if not name.endswith((".layers.3.self_attn.attn",".layers.63.self_attn.attn")):
        return
    lens=args["actual_seq_lengths_kv"]
    if not lens or max(lens)>8192 or len(lens)>16:
        return
    ends=list(metadata.actual_seq_lengths_q)
    mult=max((b-a for a,b in zip([0]+ends[:-1],ends)),default=0)
    if mult<=1 or (name,mult) in _seen:
        return
    _seen.add((name,mult))
    q=args["query"].float()
    scale=args["antiquant_scale"].float()
    nh=args["num_key_value_heads"]
    width=args["block_size"]
    refs=[]
    for i,n in enumerate(lens):
        ids=args["block_table"][i,:(n+width-1)//width].long()
        k=args["key"][ids].reshape(-1,nh,q.shape[-1])[:n].float()*scale[0,:,0]
        v=args["value"][ids].reshape(-1,nh,q.shape[-1])[:n].float()*scale[1,:,0]
        qq=q[i,:,0,:].reshape(nh,q.shape[1]//nh,q.shape[-1])
        scores=torch.einsum("hgd,lhd->hgl",qq,k)*args["scale"]
        refs.append(torch.einsum("hgl,lhd->hgd",scores.softmax(-1),v).reshape(q.shape[1],1,q.shape[-1]))
    gold=torch.stack(refs).float();actual=output.float()
    finite=bool(torch.isfinite(gold).all() and torch.isfinite(actual).all())
    rel=float((actual-gold).norm()/gold.norm().clamp_min(1e-9)) if finite else None
    append(Path(os.environ["C8_ND_DIAG"])/"runtime-c8-live-quality.jsonl",
           {"layer_name":name,"query_len_max":mult,"kv_lengths":lens,"relative_l2":rel,"finite":finite,
            "ok":finite and rel<=.04})
    if not finite or rel>.04:
        raise RuntimeError("C8 eager numerical same-real-cache proof failed")
