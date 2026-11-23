from contextlib import contextmanager
import vllm_ascend.compilation.acl_graph as acl
from full_policy import event

def capture_tiers(spec, ranges, capture):
    maximum=spec.num_speculative_steps
    main=spec.decode_cudagraph_manager
    original=acl._draft_graph_params
    spec._depth_graphs={maximum:(main,original)}
    try:
        for k in sorted({v for _,_,v in ranges} - {0,1,maximum}):
            if not 1<k<maximum:raise ValueError(k)
            acl._draft_graph_params=None
            mgr=type(main)(main.vllm_config,spec.device,main.cudagraph_mode,decode_query_len=1)
            mgr.speculator=spec;mgr.update_stream=spec.update_stream
            spec.num_speculative_steps=k;spec.decode_cudagraph_manager=mgr
            capture(mgr)
            spec._depth_graphs[k]=(mgr,acl._draft_graph_params)
            event('depth_graph_captured',k=k)
    finally:
        spec.num_speculative_steps=maximum;spec.decode_cudagraph_manager=main;acl._draft_graph_params=original

@contextmanager
def tier(spec,k):
    previous=spec.num_speculative_steps
    main=spec.decode_cudagraph_manager
    original=acl._draft_graph_params
    try:
        if k!=previous:
            spec.num_speculative_steps=k
            if k>1:
                spec.decode_cudagraph_manager,acl._draft_graph_params=spec._depth_graphs[k]
        yield
    finally:
        spec.num_speculative_steps=previous;spec.decode_cudagraph_manager=main;acl._draft_graph_params=original
