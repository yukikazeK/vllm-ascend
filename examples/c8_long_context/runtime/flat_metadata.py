# SPDX-License-Identifier: Apache-2.0
"""Pure host-side causal plan for flattened single-query C8 speculative attention."""
def flat_plan(q_ends, kv_lens, actual_tokens, token_count, request_count=None):
    q_ends=list(map(int,q_ends)); kv_lens=list(map(int,kv_lens))
    actual_tokens=int(actual_tokens); token_count=int(token_count)
    request_count=len(q_ends) if request_count is None else int(request_count)
    if not 0<=request_count<=len(q_ends) or token_count<0 or actual_tokens<0:
        raise RuntimeError("invalid flat C8 request/token counts")
    if request_count and len(kv_lens)<request_count:
        raise RuntimeError("missing C8 KV lengths")
    indices=[]; lengths=[]; padding=[]; start=0
    for row,end in enumerate(q_ends[:request_count]):
        if end<start:
            raise RuntimeError("nonmonotone cumulative Q in C8 metadata")
        if start>=token_count:
            break
        real_end=max(start,min(end,actual_tokens))
        real_q=real_end-start
        if min(end,token_count)<real_end:
            raise RuntimeError("C8 flattened count cuts a real speculative request")
        L=kv_lens[row]
        if real_q and L<real_q:
            raise RuntimeError(f"invalid C8 real Q len {real_q}, final KV len {L}")
        for j in range(start,min(end,token_count)):
            is_pad=j>=real_end
            indices.append(0 if is_pad else row)
            lengths.append(1 if is_pad else L-real_q+(j-start)+1)
            padding.append(is_pad)
        start=end
    # If capture padded extra slots are not represented by cumQ, they must
    # not include any real query. Mixed decode uses an explicit prefix count.
    if len(lengths)<token_count:
        if min(actual_tokens,token_count)>len(lengths):
            raise RuntimeError("missing real C8 decode query metadata")
        extra=token_count-len(lengths)
        indices.extend([0]*extra);lengths.extend([1]*extra);padding.extend([True]*extra)
    if len(lengths)!=token_count or any(n<1 for n in lengths):
        raise RuntimeError("C8 flattened metadata shape/length mismatch")
    return indices,lengths,padding
