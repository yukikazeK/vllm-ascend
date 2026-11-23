import torch
def fla_gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
    prebuilt_meta,
    fused_fwd,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the A2/A3 Phase6 FLA NPU GDN prefill kernel."""
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

    q, k, v = (x.transpose(1,2).contiguous() for x in (q,k,v))
    q = l2norm_fwd(q)
    k = l2norm_fwd(k)
    g = g.to(torch.float32).contiguous()
    beta = beta.to(v.dtype).contiguous()
    initial_state = initial_state.contiguous()
    cu_seqlens = prebuilt_meta.cu_seqlens_host
    chunk_indices = prebuilt_meta.chunk_indices_chunk64_host
    keep_meta = prebuilt_meta.keep_meta
    # stateVFirst=false takes [N, Hv, K, V], i.e. the transpose of ssm_state.
    initial_state_kern = initial_state.transpose(-1, -2).contiguous()
    if keep_meta is not None:
        cu_seqlens = prebuilt_meta.cu_seqlens_kern
        initial_state_kern = initial_state[keep_meta].transpose(-1, -2).contiguous()

    output, final_state, *_ = fused_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state_kern,
        output_final_state=True,
        chunk_size=64,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        layout="BNSD",
        use_exp2=False,
        use_qk_l2norm_in_kernel=False,
        allow_neg_eigval=False,
        disable_recompute=True,
        state_v_first=False,
    )
    if keep_meta is not None:
        # Scatter in the kernel layout so empty segments keep their initial state,
        # then flip back to the caller's [.., Dv, Dk].
        full_final_state = initial_state.transpose(-1, -2).contiguous()
        full_final_state[keep_meta] = final_state
        final_state = full_final_state
    return output, final_state.transpose(-1, -2).contiguous()


def fla_state_transposed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
    prebuilt_meta,
    fused_fwd,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the A2/A3 Phase6 FLA NPU GDN prefill kernel."""
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

    q, k, v = (x.transpose(1,2).contiguous() for x in (q,k,v))
    q = l2norm_fwd(q)
    k = l2norm_fwd(k)
    g = g.to(torch.float32).contiguous()
    beta = beta.to(v.dtype).contiguous()
    initial_state = initial_state.contiguous()
    cu_seqlens = prebuilt_meta.cu_seqlens_host
    chunk_indices = prebuilt_meta.chunk_indices_chunk64_host
    keep_meta = prebuilt_meta.keep_meta
    # stateVFirst=false takes [N, Hv, K, V], i.e. the transpose of ssm_state.
    assert keep_meta is None
    initial_state_kern=initial_state
    output, final_state, *_ = fused_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state_kern,
        output_final_state=True,
        chunk_size=64,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        layout="BNSD",
        use_exp2=False,
        use_qk_l2norm_in_kernel=False,
        allow_neg_eigval=False,
        disable_recompute=True,
        state_v_first=False,
    )
    return output, final_state
