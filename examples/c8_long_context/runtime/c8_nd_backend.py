# SPDX-License-Identifier: Apache-2.0
"""Experimental 910B4 dense C8 backend for the 256-dimension GQA model."""

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import time

logger = logging.getLogger("vllm.c8_nd_attention")
_REPLAY_BATCHES = set()


from flat_metadata import flat_plan
from audit_io import append, rank


def trace(kind, **fields):
    run=os.environ.get("C8_AUDIT_RUN")
    if run:
        append(Path(run)/"runtime-nd-audit.jsonl",dict(kind=kind,**fields))


def build_flat(metadata, token_count, request_count=None):
    import torch
    rows,lens,pad=flat_plan(metadata.actual_seq_lengths_q,metadata.seq_lens_list,
                            metadata.num_actual_tokens,token_count,request_count)
    original=metadata.block_tables
    if original is None or original.ndim!=2 or not rows:
        raise RuntimeError("C8 flattened decode needs nonempty block table")
    if original.shape[0]==0 or max(rows)>=original.shape[0]:
        raise RuntimeError("C8 flattened request exceeds block table")
    ids=torch.tensor(rows,dtype=torch.long,device=original.device)
    table=original.index_select(0,ids)*2
    if any(pad):
        padding=torch.tensor(pad,dtype=torch.bool,device=original.device)
        table.masked_fill_(padding[:,None],0)
    entry=(int(token_count),request_count,table,lens,pad)
    metadata._c8_flat_decode_entry=entry
    return entry


def flat_args(metadata, token_count, request_count=None, require_prepared=False):
    saved=getattr(metadata,"_c8_flat_decode_entry",None)
    if saved is None or saved[0]<int(token_count) or (request_count is not None and saved[1] not in (None,request_count)):
        if require_prepared:
            raise RuntimeError("C8 graph capture lacks precomputed flattened page table")
        saved=build_flat(metadata,token_count,request_count)
    return {"block_table":saved[2][:token_count],"actual_seq_lengths_kv":saved[3][:token_count]}


@dataclass(frozen=True)
class DecodeParams:
    layer_name: str
    batch: int
    @property
    def name(self):
        return self.layer_name
    def resolve(self, metadata):
        current=metadata[self.layer_name]
        args=flat_args(current,self.batch,require_prepared=False)
        ends=list(current.actual_seq_lengths_q)
        q_lengths=tuple(sorted(set(b-a for a,b in zip([0]+ends[:-1],ends)
                                   if a<int(current.num_actual_tokens) and b>a)))
        key=(self.batch,q_lengths,rank())
        if key not in _REPLAY_BATCHES:
            _REPLAY_BATCHES.add(key)
            trace("flatten_graph_replay",layer_name=self.layer_name,physical_queries=self.batch,
                  q_cumulative=ends,q_lengths=list(q_lengths),actual_tokens=int(current.num_actual_tokens),
                  kv_lengths=args["actual_seq_lengths_kv"][:32])
        return args



def adapt_attention(module):
    torch, torch_npu = module.torch, module.torch_npu
    cls = module.AscendC8AttentionBackendImpl
    from c8_fold import attach_scales, fold_capture
    original_prep = cls._prepare_c8_scales
    original_c8_fia = cls._forward_c8_fused_infer_attention
    builder=module.AscendAttentionMetadataBuilder
    orig_build=builder.build
    orig_graph_build=builder.build_for_graph_capture
    def prepare(self,metadata,all_queries=False):
        if getattr(getattr(self,"kv_cache_spec",None),"dtype",None)!=torch.int8 or metadata.block_tables is None:
            return metadata
        if all_queries or metadata.num_prefills==0:
            n=int(metadata.actual_seq_lengths_q[-1]) if metadata.actual_seq_lengths_q else 0
            if n:
                build_flat(metadata,n,None)
        elif metadata.num_decodes>0 and metadata.num_decode_tokens>0:
            build_flat(metadata,int(metadata.num_decode_tokens),int(metadata.num_decodes))
        return metadata
    def build(self,*args,**kwargs):
        return prepare(self,orig_build(self,*args,**kwargs))
    def build_graph(self,*args,**kwargs):
        return prepare(self,orig_graph_build(self,*args,**kwargs),all_queries=True)
    builder.build=build
    builder.build_for_graph_capture=build_graph

    def begin_prefill_memory(self, lengths, query_lens, decode_tokens):
        # The first full-attention layer is layer 3 in the validated model.
        # This optional audit is identical for both benchmark arms.
        if os.environ.get("C8_PREFILL_MEMORY_AUDIT") != "1":
            return None
        suffix = os.environ.get("C8_PREFILL_MEMORY_LAYER", "layers.3.self_attn.attn")
        run = os.environ.get("C8_AUDIT_RUN")
        if not run or not self._layer_name.endswith(suffix):
            return None
        torch.npu.reset_peak_memory_stats()
        self._c8_memory_step = getattr(self, "_c8_memory_step", 0) + 1
        return {
            "kind": "c8_prefill_memory", "pid": os.getpid(),
            "layer": self._layer_name, "step": self._c8_memory_step,
            "query_lengths": [int(q) for q in query_lens],
            "kv_lengths": [int(k) for k in lengths],
            "history_lengths": [int(k - q) for q, k in zip(query_lens, lengths)],
            "decode_tokens": int(decode_tokens),
            "lse_requested": os.environ.get("C8_PREFILL_LSE",
                                             os.environ.get("VLLM_ASCEND_C8_LSE_PREFILL")) == "1",
            "timestamp_monotonic_s": time.perf_counter(),
            "before_bytes": torch.npu.memory_allocated(),
        }

    def end_prefill_memory(record, route):
        if record is None:
            return
        peak = torch.npu.max_memory_allocated()
        record.update(route=route, peak_bytes=peak,
                      additional_peak_bytes=max(0, peak - record["before_bytes"]),
                      after_bytes=torch.npu.memory_allocated())
        append(Path(os.environ["C8_AUDIT_RUN"]) / "runtime-c8-prefill-memory.jsonl",record)

    def prepare_scales(self, layer, device):
        original_prep(self, layer, device)
        if not hasattr(layer, "_c8_nd_combined_scale"):
            expected=self.num_kv_heads*self.head_size
            for name in ("k","v"):
                raw=getattr(layer,f"{name}_cache_scale").detach().flatten().cpu()
                want=raw
                if raw.numel()!=expected:
                    first=rank()*expected
                    want=raw[first:first+expected]
                actual=getattr(layer,f"_c8_{name}_scale").detach().flatten().cpu()
                off=getattr(layer,f"_c8_{name}_offset")
                if actual.numel()!=expected or not torch.equal(actual,want.to(actual.dtype)) or not bool((off==0).all()):
                    raise RuntimeError("C8 rank shard/zero-offset verification failed")
            layer._c8_nd_combined_scale = torch.stack(
                (layer._c8_k_aq_scale_nz_bnsd, layer._c8_v_aq_scale_nz_bnsd)
            )
            trace("scales_rank_sharded",layer_name=layer.layer_name,channels=expected,
                  num_heads=self.num_heads,num_kv_heads=self.num_kv_heads)
        attach_scales(layer,torch)

    def args_for_decode(self, query, metadata, layer, batch, request_count=None, capture=False):
        if not hasattr(self, "_c8_nd_read_alias"):
            source_k, source_v = self.key_cache, self.value_cache
            pages, block, heads, dim = source_k.shape
            row = heads * dim
            size = block * row
            if (source_k.stride() != (2 * size, row, dim, 1)
                    or source_v.stride() != source_k.stride()
                    or source_v.data_ptr() - source_k.data_ptr() != size):
                raise ValueError("Unexpected interleaved C8 KV layout")
            # The even page indices select K at 2*i and V at 2*i+1 using
            # two contiguous views into the original shared hybrid storage.
            shape = (2 * pages - 1, block, row)
            stride = (size, row, 1)
            self._c8_nd_read_alias = (
                torch.as_strided(source_k, shape, stride),
                torch.as_strided(source_v, shape, stride),
                block,
            )
        k, v, block = self._c8_nd_read_alias
        return {
            "query": query[:batch].unsqueeze(2), "key": k, "value": v,
            **flat_args(metadata,batch,request_count,require_prepared=capture), "block_size": block,
            "input_layout": "BNSD",
            "num_heads": self.num_heads, "num_key_value_heads": self.num_kv_heads, "scale": self.scale,
            "sparse_mode": 0, "antiquant_scale": layer._c8_nd_combined_scale, "antiquant_mode": 0,
        }

    def decode(self, query, metadata, output, layer, batch=None):
        count=(int(metadata.num_actual_tokens) if batch is None else int(metadata.num_decode_tokens))
        if count==0:
            return output
        args=args_for_decode(self,query,metadata,layer,count,batch)
        y,_=torch_npu.npu_fused_infer_attention_score(**args)
        output[:count]=y.squeeze(2)
        diag=os.environ.get("C8_ND_DIAG")
        if diag and (Path(diag)/".enable_live_quality").exists():
            from live_quality import eager_check
            eager_check(self._layer_name,args,y,metadata)
        return output

    def full_graph(self, query, key, value, metadata, output, layer=None):
        batch=query.shape[0]
        if fold_capture(module,self,query,metadata,output,layer,args_for_decode):
            return output,batch
        args=args_for_decode(self,query,metadata,layer,batch,capture=True)
        scratch=module.get_capture_resource(
            "c8.nd.flat-q1.fia.workspace",
            lambda: torch_npu._npu_fused_infer_attention_score_get_max_workspace(**args),
            self._use_max_workspace_for_fia_graph,
        )
        lse=torch.empty(1,dtype=query.dtype,device=query.device)
        module.register_task(torch_npu.npu_fused_infer_attention_score.out,
                             {**args,"workspace":scratch,"out":[output[:batch].unsqueeze(2),lse]},
                             DecodeParams(self._layer_name,batch))
        return output,batch

    def c8_fia(self,query,key,value,metadata,output,layer):
        if metadata.attn_state==module.AscendAttentionState.SpecDecoding and metadata.num_prefills==0:
            return decode(self,query,metadata,output,layer)
        return original_c8_fia(self,query,key,value,metadata,output,layer)

    def write(self, query, key, value, cache, metadata, output):
        return module.AscendAttentionBackendImpl.reshape_and_cache(
            self, query, key, value, cache, metadata, output)

    def gather(self, key, value, table, lengths, target_dtype, layer):
        # Only the prefill continuation path needs BF16 K/V. Decode reads INT8 directly.
        k = key.view_as(self.key_cache)
        v = value.view_as(self.value_cache)
        batch, cols = table.shape
        slots = table.reshape(-1).long()
        block = k.shape[1]
        padded = cols * block
        positions = torch.arange(padded, device=k.device)
        mask = (positions < torch.tensor(lengths, device=k.device)[:, None]).flatten()
        shape = (batch * padded, self.num_kv_heads, self.head_size)
        dk = k[slots].reshape(shape)[mask].to(target_dtype) * layer._c8_k_scale
        dv = v[slots].reshape(shape)[mask].to(target_dtype) * layer._c8_v_scale
        return dk, dv

    def chunked(self, query, key, value, metadata, output, layer):
        decode_tokens = metadata.num_decode_tokens
        decodes = metadata.num_decodes
        if decode_tokens:
            decode(self, query, metadata, output, layer, decodes)
        if not metadata.num_prefills:
            return output
        lengths = metadata.seq_lens_list[decodes:]
        offsets = [end - decode_tokens for end in metadata.actual_seq_lengths_q[decodes:]]
        query_lens = [end - start for start, end in zip([0, *offsets[:-1]], offsets)]
        end = decode_tokens + offsets[-1]
        memory_record = begin_prefill_memory(self, lengths, query_lens, decode_tokens)
        if lengths == query_lens and key is not None and value is not None:
            k, v = key[decode_tokens:end], value[decode_tokens:end]
            route = "all-new"
        else:
            if os.environ.get("C8_PREFILL_LSE", os.environ.get("VLLM_ASCEND_C8_LSE_PREFILL")) == "1":
                from c8_lse_prefill import try_prefill
                segmented = try_prefill(self, layer, query[decode_tokens:end], metadata,
                                        decodes, lengths, query_lens)
                if segmented is not None:
                    output[decode_tokens:end] = segmented
                    end_prefill_memory(memory_record, "lse-segmented")
                    return output
            if os.environ.get("C8_PREFILL_FUSED_GATHER"):
                from fused_gather import gather_c8, prepare_lengths
                table = metadata.block_tables[decodes:]
                fingerprint = (decodes, tuple(lengths))
                cached = getattr(metadata, "_c8_fused_prefill_meta", None)
                if cached is None or cached[0] != fingerprint:
                    prepared = prepare_lengths(lengths, query.device)
                    metadata._c8_fused_prefill_meta = (fingerprint, prepared)
                else:
                    prepared = cached[1]
                if not getattr(self, "_c8_fused_prefill_logged", False):
                    logger.info("C8 fused paged gather active for %s", self._layer_name)
                    self._c8_fused_prefill_logged = True
                k, v = gather_c8(self.key_cache, self.value_cache, table, lengths,
                                 layer._c8_k_scale, layer._c8_v_scale, query.dtype,
                                 tile=4096, prepared=prepared)
                route = "fused-gather"
            else:
                k, v = gather(self, self.key_cache, self.value_cache, metadata.block_tables[decodes:],
                              lengths, query.dtype, layer)
                route = "legacy-gather"
        kv_offsets = [sum(lengths[:i]) for i in range(1, len(lengths) + 1)]
        y, _ = torch_npu.npu_fused_infer_attention_score(
            query=query[decode_tokens:end], key=k.contiguous(), value=v.contiguous(),
            atten_mask=metadata.attn_mask, input_layout="TND",
            actual_seq_lengths=offsets, actual_seq_lengths_kv=kv_offsets,
            num_heads=self.num_heads, num_key_value_heads=self.num_kv_heads, scale=self.scale,
            sparse_mode=3,
        )
        output[decode_tokens:end] = y.view(-1, self.num_heads, self.head_size)
        end_prefill_memory(memory_record, route)
        return output

    cls._prepare_c8_scales = prepare_scales
    cls._forward_c8_decode = decode
    cls.full_graph_fia = full_graph
    cls._reshape_and_cache = write
    cls._forward_c8_chunked_prefill = chunked
    cls._forward_c8_fused_infer_attention = c8_fia
    cls._dequant_paged_kv_to_dense = gather
