import os,json
from pathlib import Path
root=Path(os.environ['GRAPH_TREND_DIR'])
calls={}
def snap(runner,metadata,logits,mode,computed,prompt_lens,scheduled):
    import torch
    keyfile=root/'diag-tag'
    if not keyfile.exists(): return
    tag=keyfile.read_text().strip()
    if not tag:return
    i=calls.get(tag,0);calls[tag]=i+1
    n=len(scheduled)
    before=[int(x) for x in computed[:n]]
    prompt=[int(x) for x in prompt_lens[:n]]
    types=['P' if a<b else 'D' for a,b in zip(before,prompt)]
    with (root/'calls.jsonl').open('a') as f:
        print(json.dumps([tag,i,mode.name,before,list(map(int,scheduled)),types]),file=f)
    if tag.startswith('mixed') or i>=4:return
    snapshot={'logits':logits.detach().cpu()[:n].clone(),'computed':before,'scheduled':list(map(int,scheduled))}
    layers=runner.vllm_config.compilation_config.static_forward_context
    names=sorted(k for k,v in metadata.items() if hasattr(v,'prefill_state_indices') and k in layers)
    if names:
        for k in dict.fromkeys([names[0],names[len(names)//2],names[-1]]):
            meta=metadata[k];ix=meta.prefill_state_indices if 'P' in types else meta.non_spec_state_indices_tensor
            if ix is None:continue
            ids=ix.reshape(-1)[:n].cpu().tolist()
            if any(int(a)<0 for a in ids):continue
            state=layers[k].kv_cache[1]
            snapshot[k]=torch.stack([state[int(a)].detach().cpu() for a in ids])
    torch.save(snapshot,root/f'{tag}-{i}.pt')
