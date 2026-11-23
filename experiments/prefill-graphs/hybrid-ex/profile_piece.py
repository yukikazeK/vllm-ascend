import os,json
from pathlib import Path
import torch
import torch_npu
def wrapped(candidate,reference,key):
    root=Path(os.environ['GRAPH_TREND_DIR']);seen=set()
    def call(*args):
        if len(seen)<2 and not torch.npu.is_current_stream_capturing():
            shapes=tuple(tuple(x.shape) for x in args if isinstance(x,torch.Tensor))
            n=next((int(arg) for arg in args if isinstance(arg,int)),None)
            # Restrict profiler to two small capture warm-up / execution shapes.
            if n is None:n=next((shape[0] for shape in shapes if len(shape)==1),None)
            if n and n<=128 and n not in seen:
                seen.add(n)
                results=[]
                for tag,fn in [('basic',reference),('ex',candidate)]:
                    torch.npu.synchronize()
                    prof=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                           on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(root/f'profile-{n}-{tag}')))
                    with prof:
                        val=fn(*args);torch.npu.synchronize()
                    results.append(val)
                from torch.utils._pytree import tree_flatten
                a,b=[tree_flatten(x)[0] for x in results]
                report=[]
                for x,y in zip(a,b,strict=True):
                    if isinstance(x,torch.Tensor):
                        a0=x.detach().float();b0=y.detach().float()
                        if a0.numel():
                            delta=(a0-b0).norm()/a0.norm().clamp_min(1e-9)
                            report.append(float(delta))
                        assert x.shape==y.shape
                    else:assert x==y
                (root/f'piece-{n}.json').write_text(json.dumps({'compiler_key':key,'shape':shapes,'relative_l2':report}))
                print('EX_PIECE_EXECUTED',n,key,report,flush=True)
                return results[1]
        return candidate(*args)
    return call
