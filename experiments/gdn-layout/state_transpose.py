import torch
from vllm.triton_utils import triton,tl

@triton.jit
def _state_io(SRC,DST,IDS,VALID, SRC_BATCH:tl.constexpr,DST_BATCH:tl.constexpr,
              ID_STRIDE:tl.constexpr,MASK_STRIDE:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
              READ:tl.constexpr,TILE:tl.constexpr):
    tile=tl.program_id(0)
    batch=tl.program_id(1)
    head=tile//(D//TILE)
    ix=tl.load(IDS+batch*ID_STRIDE).to(tl.int64)
    a=(tile%(D//TILE))*TILE+tl.arange(0,TILE)
    b=tl.arange(0,D)
    src_slice=head*D*D+a[:,None]*D+b[None,:]
    dst_slice=head*D*D+b[:,None]*D+a[None,:]
    if READ:
        valid=tl.load(VALID+batch*MASK_STRIDE).to(tl.int1)
        x=tl.load(SRC+ix*SRC_BATCH+src_slice,mask=valid,other=0.)
        tl.store(DST+batch*DST_BATCH+dst_slice,tl.trans(x))
    else:
        x=tl.load(SRC+batch*SRC_BATCH+src_slice)
        tl.store(DST+ix*DST_BATCH+dst_slice,tl.trans(x))

def gather_transposed(state,indices,valid):
    n=indices.numel()
    h,d=state.shape[1:3]
    assert state.shape[-1]==d==128 and state.stride()[1:]==(d*d,d,1)
    out=torch.empty((n,h,d,d),dtype=state.dtype,device=state.device)
    if n:
        _state_io[(h*4,n)](state,out,indices,valid,state.stride(0),out.stride(0),indices.stride(0),valid.stride(0),h,d,True,32)
    return out

def scatter_transposed(state,indices,source):
    n=indices.numel()
    h,d=state.shape[1:3]
    assert source.shape==(n,h,d,d) and source.is_contiguous()
    if n:
        _state_io[(h*4,n)](source,state,indices,indices,source.stride(0),state.stride(0),indices.stride(0),1,h,d,False,32)
