"""Export and import the installed AscendStore backend's whole-key objects."""
import fcntl
from hashlib import sha256
import importlib.machinery as machinery
import logging
import sys
import time

try:
    from .cache_package import local_objects, read_json, verified_payload, write_atomic, write_json
except ImportError:
    from cache_package import local_objects, read_json, verified_payload, write_atomic, write_json

_BACKEND = 'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend'
_WORKER = 'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker'
logger = logging.getLogger('vllm.local_prefix_cache')


def import_objects(backend, store, manifest, rank=None):
    start = time.monotonic()
    objects = local_objects(manifest)
    if rank is not None:
        records = read_json(manifest)
        objects = [o for o in objects if records[o[0]].get('rank', rank) == rank]
    placement = backend._build_replicate_config()
    placement.preferred_segment = backend.local_seg
    # The whole-object put API rejects node-affinity in Mooncake 0.3.13.
    placement.prefer_alloc_in_same_node = False
    for key, path, length, expected in objects:
        payload = verified_payload(path, length, expected)
        if store.is_exist(key) != 1:
            rc = store.put(key, payload, placement)
            if rc != 0 and store.is_exist(key) != 1:
                raise RuntimeError(f'Mooncake rejected cache object {key}: {rc}')
        stored = store.get(key)
        if len(stored) != length or sha256(stored).hexdigest() != expected:
            raise RuntimeError(f'Mooncake object differs from its package: {key}')
    logger.info('Loaded local prefix package %s: %s objects, %s bytes in %.3f s',
                manifest, len(objects), sum(o[2] for o in objects), time.monotonic() - start)


def capture_put(backend, directory):
    original_put = backend.put
    index = directory / 'objects.json'

    def put(self, keys, addresses, sizes):
        result = original_put(self, keys, addresses, sizes)
        try:
            if len(keys) != len(sizes):
                raise RuntimeError('AscendStore returned inconsistent object metadata')
            with (directory / 'export.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                entries = read_json(index) if index.exists() else {}
                changed = False
                for key, parts in zip(keys, sizes):
                    if key in entries:
                        continue
                    payload = self.store.get(key)
                    if len(payload) != sum(parts):
                        raise RuntimeError(f'incomplete export for Store key {key}')
                    checksum = sha256(payload).hexdigest()
                    name = sha256(key.encode()).hexdigest() + '.bin'
                    write_atomic(directory / name, payload)
                    entries[key] = {'file': name, 'bytes': len(payload), 'sha256': checksum,
                                    'rank': self._agent_cache_rank}
                    changed = True
                if changed:
                    write_json(index, entries)
        except Exception as exc:
            write_json(directory / 'export-error.json', {'error': str(exc)})
            raise
        return result

    backend.put = put


def worker_layout(worker, caches):
    # AscendStore reports the exported per-object byte ranges. Record the
    # actual registered layout, plus precision and shard ownership per layer.
    config = worker.vllm_config.parallel_config
    tensors = {}
    for layer, value in caches.items():
        parts = value if isinstance(value, (tuple, list)) else (value,)
        tensors[layer] = [
            {'shape_per_block': list(x.shape[1:]), 'stride': list(x.stride()), 'dtype': str(x.dtype)}
            for x in parts
        ]
    return {
        'tp': config.tensor_parallel_size,
        'pp': config.pipeline_parallel_size,
        'pcp': config.prefill_context_parallel_size,
        'dcp': config.decode_context_parallel_size,
        'groups': list(worker.grouped_block_size),
        'hash_block': worker.hash_block_size,
        'cacheable_groups': list(worker.cacheable_group_ids),
        'block_lengths': {str(k): list(v) for k, v in worker.group_block_len.items()},
        'block_strides': {str(k): list(v) for k, v in worker.group_block_stride.items()},
        'tensors': tensors,
    }


def register_layout(worker_class, manifest, export):
    original = worker_class.register_kv_caches

    def register(self, caches):
        result = original(self, caches)
        rank = f'{self.pp_rank}-{self.pcp_rank}-{self.tp_rank}'
        layout = worker_layout(self, caches)
        if export:
            self.m_store._agent_cache_rank = rank
            write_json(export / f'layout-{rank}.json', layout)
        else:
            expected = read_json(manifest.with_name('layouts.json'))
            if expected.get(rank) != layout:
                raise RuntimeError(f'local prefix cache has an incompatible layout for rank {rank}')
            import_objects(self.m_store, self.m_store.store, manifest, rank)
        return result

    worker_class.register_kv_caches = register


def install(manifest=None, export=None):
    managed = manifest is not None and manifest.with_name('layouts.json').is_file()

    class Loader(machinery.SourceFileLoader):
        def exec_module(self, module):
            super().exec_module(module)
            if module.__name__ == _WORKER:
                register_layout(module.KVPoolWorker, manifest, export)
            elif export:
                capture_put(module.MooncakeBackend, export)
            else:
                backend = module.MooncakeBackend
                original = backend._setup_store

                def load_store(self):
                    store = original(self)
                    if self._contribute_memory:
                        import_objects(self, store, manifest)
                    return store

                backend._setup_store = load_store

    modules = {_WORKER} if managed else {_BACKEND, _WORKER} if export else {_BACKEND}

    class Finder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname in modules:
                spec = machinery.PathFinder.find_spec(fullname, path)
                if spec and spec.origin:
                    spec.loader = Loader(fullname, spec.origin)
                return spec
            return None

    sys.meta_path.insert(0, Finder())
