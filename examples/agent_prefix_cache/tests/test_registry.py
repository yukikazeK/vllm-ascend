"""Regression checks for disk validation and atomic activation."""
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from agent_cache.engine import profile_id
from agent_cache.registry import Registry, check_package, copy_package, new_metadata
from bootstrap.cache_package import read_json, write_json
from bootstrap.local_prefix_loader import import_objects


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.registry = Registry(self.base / 'registry')
        self.profile = {'model': 'model', 'model_revision': 'revision-1',
                        'serve_args': ['/models/model', '--tensor-parallel-size', '2']}
        self.fingerprint = profile_id(self.profile)

    def package(self, agent, tokens, payload=b'block bytes'):
        path = self.registry.stage(agent)
        objects = {
            'test-object': {'file': 'cache.bin', 'bytes': len(payload),
                            'sha256': sha256(payload).hexdigest(), 'rank': '0-0-0'},
        }
        layouts = {'0-0-0': {'dtype': 'test'}}
        record = new_metadata(agent, self.fingerprint, 'revision-1', {'token_ids': tokens},
                              tokens, layouts, objects,
                              {'external_prefix_hit_tokens': 1, 'output_tokens_equal': True})
        (path/'cache.bin').write_bytes(payload)
        write_json(path/'probe.json', {'token_ids': tokens, 'output_ids': list(range(16))})
        write_json(path/'objects.json', objects)
        write_json(path/'layouts.json', layouts)
        write_json(path/'package.json', record)
        return record['version'], self.registry.publish(path)

    def test_activation_and_rollback(self):
        old, path = self.package('agent', [1, 2])
        new, _ = self.package('agent', [1, 3], b'new bytes')
        active = self.registry.activate(f'agent@{new}')
        self.assertEqual(self.registry.current(self.fingerprint), {'agent': new})
        self.registry.activate(f'agent@{old}')
        self.assertEqual(self.registry.current(self.fingerprint), {'agent': old})
        filename = read_json(active/'objects.json')['test-object']['file']
        self.assertEqual((active/filename).read_bytes(), (path/'cache.bin').read_bytes())
        with self.assertRaisesRegex(ValueError, 'active version'):
            self.registry.remove(f'agent@{old}')

    def test_collision_keeps_existing_active_package(self):
        old, _ = self.package('agent-a', [1, 2])
        self.package('agent-b', [1, 2], b'incompatible')
        self.registry.activate('agent-a')
        with self.assertRaisesRegex(ValueError, 'conflicting bytes'):
            self.registry.activate('agent-b')
        self.assertEqual(self.registry.current(self.fingerprint), {'agent-a': old})

    def test_export_and_checksum(self):
        _, src = self.package('agent', [1, 2])
        (src/'unrelated.log').write_text('not part of the package')
        export = self.base/'exported'
        copy_package(src, export)
        self.assertFalse((export/'unrelated.log').exists())
        with (export/'cache.bin').open('r+b') as f:
            f.write(b'X')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            check_package(export)

    def test_profile_ignores_service_address(self):
        new = dict(self.profile, serve_args=[*self.profile['serve_args'], '--port', '18080', '--host=127.0.0.1'])
        self.assertEqual(profile_id(new), self.fingerprint)
        new['serve_args'] = ['/models/model', '--tensor-parallel-size', '4']
        self.assertNotEqual(profile_id(new), self.fingerprint)


    def test_import_uses_rank_local_segment(self):
        _, package = self.package('agent', [1, 2])
        index = read_json(package/'objects.json')
        index['other-rank'] = dict(index['test-object'], rank='0-0-1')
        write_json(package/'objects.json', index)

        class Store:
            def __init__(self):
                self.data = {}

            def is_exist(self, key):
                return int(key in self.data)

            def put(self, key, payload, placement):
                if placement.preferred_segment != 'rank0' or placement.prefer_alloc_in_same_node:
                    return -600
                self.data[key] = payload
                return 0

            def get(self, key):
                return self.data[key]

        store = Store()
        backend = SimpleNamespace(
            local_seg='rank0', _build_replicate_config=lambda: SimpleNamespace(prefer_alloc_in_same_node=True),
        )
        import_objects(backend, store, package/'objects.json', rank='0-0-0')
        self.assertEqual(store.data, {'test-object': b'block bytes'})


if __name__ == '__main__':
    unittest.main()
