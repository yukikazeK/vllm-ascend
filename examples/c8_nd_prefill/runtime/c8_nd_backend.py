"""Experimental 910B4 dense C8 backend for the 256-dimension GQA model."""

from dataclasses import dataclass
import logging
import os

logger = logging.getLogger("vllm.c8_nd_attention")
_REPLAY_BATCHES = set()


@dataclass(frozen=True)
class DecodeParams:
    name: str
    batch: int

    def resolve(self, metadata):
        current = metadata[self.name]
        if self.batch not in _REPLAY_BATCHES:
            _REPLAY_BATCHES.add(self.batch)
            logger.info("C8 ND graph replay batch=%d", self.batch)
        return {
            "block_table": even_pages(current)[: self.batch],
            "actual_seq_lengths_kv": current.seq_lens_list[: self.batch],
        }


def even_pages(metadata):
    if not hasattr(metadata, "_c8_nd_even_pages"):
        metadata._c8_nd_even_pages = metadata.block_tables * 2
    return metadata._c8_nd_even_pages


def adapt_attention(module):
    torch, torch_npu = module.torch, module.torch_npu
    cls = module.AscendC8AttentionBackendImpl
    original_prep = cls._prepare_c8_scales

    def prepare_scales(self, layer, device):
        original_prep(self, layer, device)
        if not hasattr(layer, "_c8_nd_combined_scale"):
            layer._c8_nd_combined_scale = torch.stack(
                (layer._c8_k_aq_scale_nz_bnsd, layer._c8_v_aq_scale_nz_bnsd)
            )

    def args_for_decode(self, query, metadata, layer, batch):
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
            "block_table": even_pages(metadata)[:batch], "block_size": block,
            "actual_seq_lengths_kv": metadata.seq_lens_list[:batch], "input_layout": "BNSD",
            "num_heads": self.num_heads, "num_key_value_heads": self.num_kv_heads, "scale": self.scale,
            "sparse_mode": 0, "antiquant_scale": layer._c8_nd_combined_scale, "antiquant_mode": 0,
        }

    def decode(self, query, metadata, output, layer, batch=None):
        token_count = len(metadata.seq_lens_list) if batch is None else batch
        y, _ = torch_npu.npu_fused_infer_attention_score(
            **args_for_decode(self, query, metadata, layer, token_count))
        output[:token_count] = y.squeeze(2)
        return output

    def full_graph(self, query, key, value, metadata, output, layer=None):
        batch = query.shape[0]
        args = args_for_decode(self, query, metadata, layer, batch)
        scratch = module.get_capture_resource(
            "c8.nd.gqa.fia.workspace",
            lambda: torch_npu._npu_fused_infer_attention_score_get_max_workspace(**args),
            self._use_max_workspace_for_fia_graph,
        )
        lse = torch.empty(1, dtype=query.dtype, device=query.device)
        module.register_task(torch_npu.npu_fused_infer_attention_score.out,
                             {**args, "workspace": scratch, "out": [output[:batch].unsqueeze(2), lse]},
                             DecodeParams(self._layer_name, batch))
        return output, batch

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
            if decode_tokens != decodes:
                raise RuntimeError("C8 ND prototype needs one decode token per request")
            decode(self, query, metadata, output, layer, decodes)
        if not metadata.num_prefills:
            return output
        lengths = metadata.seq_lens_list[decodes:]
        offsets = [end - decode_tokens for end in metadata.actual_seq_lengths_q[decodes:]]
        query_lens = [end - start for start, end in zip([0, *offsets[:-1]], offsets)]
        end = decode_tokens + offsets[-1]
        if lengths == query_lens and key is not None and value is not None:
            k, v = key[decode_tokens:end], value[decode_tokens:end]
        else:
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
            else:
                k, v = gather(self, self.key_cache, self.value_cache, metadata.block_tables[decodes:],
                              lengths, query.dtype, layer)
        kv_offsets = [sum(lengths[:i]) for i in range(1, len(lengths) + 1)]
        y, _ = torch_npu.npu_fused_infer_attention_score(
            query=query[decode_tokens:end], key=k.contiguous(), value=v.contiguous(),
            atten_mask=metadata.attn_mask, input_layout="TND",
            actual_seq_lengths=offsets, actual_seq_lengths_kv=kv_offsets,
            num_heads=self.num_heads, num_key_value_heads=self.num_kv_heads, scale=self.scale,
            sparse_mode=3,
        )
        output[decode_tokens:end] = y.view(-1, self.num_heads, self.head_size)
        return output

    cls._prepare_c8_scales = prepare_scales
    cls._forward_c8_decode = decode
    cls.full_graph_fia = full_graph
    cls._reshape_and_cache = write
    cls._forward_c8_chunked_prefill = chunked
    cls._dequant_paged_kv_to_dense = gather
