"""Opt-in, narrow 910B4 GDN PCP forward-path experiment."""
import importlib.machinery as loader
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
MODULES = {
    'vllm_ascend.attention.context_parallel.common_cp',
    'vllm_ascend.worker.v2.pcp_manager',
    'vllm_ascend.ops.gdn_attn_builder',
    'vllm_ascend.worker.v2.aclgraph_utils',
}
prior_lengths = None

def validate(config):
    parallel = config.parallel_config
    assert parallel.tensor_parallel_size == 1
    assert parallel.prefill_context_parallel_size == 2
    assert parallel.decode_context_parallel_size == 1
    assert parallel.pipeline_parallel_size == parallel.data_parallel_size == 1
    assert config.model_config.hf_text_config.model_type == 'qwen3_5_text'
    assert config.scheduler_config.max_num_seqs == 1
    assert config.compilation_config.cudagraph_mode.name == 'FULL_DECODE_ONLY'
    assert config.cache_config.mamba_cache_mode == 'none'
    assert not config.cache_config.enable_prefix_caching
    assert config.speculative_config is None

def install_manager(module):
    cls = module.AscendPCPManager
    original_validate = cls.validate_config
    original_partition = cls.partition_batch

    def validate_pilot(config, supports_mm_inputs):
        original_validate(config, supports_mm_inputs)
        validate(config)

    def contiguous(self, rank, counts, prefilling):
        for req, count in enumerate(counts):
            size = int(count)
            if size <= 0:
                continue
            if not prefilling[req]:
                # Both replicas advance the same decode convolution and SSM state.
                yield req, 0, size
                continue
            if size < 6:
                raise ValueError('This PCP trial requires at least three new prefill tokens per rank')
            middle = (size + 1) // 2
            start = 0 if rank == 0 else middle
            end = middle if rank == 0 else size
            yield req, start, end - start

    def partition(self, input_batch, *args, **kwargs):
        global prior_lengths
        prior_lengths = None
        result = original_partition(self, input_batch, *args, **kwargs)
        if bool(input_batch.is_prefilling_np.any()):
            prior_lengths = input_batch.num_computed_tokens_np[: input_batch.num_reqs].copy()
            name = ROOT / f'partition-{os.getpid()}.jsonl'
            row = {'rank': self.pcp_rank, 'before': prior_lengths.tolist(),
                   'global': input_batch.num_scheduled_tokens.tolist(),
                   'local': result.num_scheduled_tokens.tolist()}
            with name.open('a') as output:
                print(json.dumps(row), file=output)
        return result

    cls.validate_config = staticmethod(validate_pilot)
    cls._iter_rank_chunks = contiguous
    cls.partition_batch = partition

def install_gdn(module):
    cls = module.AscendGDNAttentionMetadataBuilder
    original = cls._build_prefill_has_initial_state

    def has_state(self, *, context_lens_tensor, non_spec_sequence_indices):
        # Rank 1's positional offset within this segment is used by GQA,
        # but is not committed recurrent history at entry to the GDN step.
        if prior_lengths is None or len(prior_lengths) != context_lens_tensor.numel():
            raise RuntimeError('GDN metadata could not identify the global PCP prefill rows')
        history = context_lens_tensor.new_tensor(prior_lengths)
        return original(self, context_lens_tensor=history, non_spec_sequence_indices=non_spec_sequence_indices)

    module.AscendGDNAttentionBackend.supports_pcp = classmethod(lambda cls: True)
    cls._build_prefill_has_initial_state = has_state

def install_graph_probe(module):
    cls = module.ModelAclGraphManager
    original = cls.run_fullgraph
    def first_replay(self, desc):
        result = original(self, desc)
        from vllm.distributed import get_pcp_group
        (ROOT / f'replay-{os.getpid()}.json').write_text(json.dumps({
            'rank':get_pcp_group().rank_in_group,'graph':desc.cg_mode.name,'tokens':desc.num_tokens}))
        cls.run_fullgraph = original
        return result
    cls.run_fullgraph = first_replay

class PilotLoader(loader.SourceFileLoader):
    def exec_module(self, module):
        super().exec_module(module)
        installer = {
            'vllm_ascend.attention.context_parallel.common_cp': lambda obj: setattr(obj,'is_pcp_decode_sharding_enabled',lambda config: False),
            'vllm_ascend.worker.v2.pcp_manager': install_manager,
            'vllm_ascend.ops.gdn_attn_builder': install_gdn,
            'vllm_ascend.worker.v2.aclgraph_utils': install_graph_probe,
        }[module.__name__]
        installer(module)

class PilotFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in MODULES:
            spec = loader.PathFinder.find_spec(fullname, path)
            if spec and spec.origin:
                spec.loader = PilotLoader(fullname, spec.origin)
            return spec

if os.getenv('GDN_PCP_PILOT') == '1':
    sys.meta_path.insert(0, PilotFinder())
