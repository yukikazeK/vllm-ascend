# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the C8 runtime overlay's pure planning and fail-closed guards.

These tests deliberately use lightweight host fixtures, not torch/NPU tensors.
They do not replace the documented numerical, graph replay or allocation tests
on the pinned NPU environment.
"""
from __future__ import annotations

import contextlib
import io
import os
import random
import sys
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
sys.path.insert(0, str(RUNTIME))

from flat_metadata import flat_plan  # noqa: E402
from segment_plan import Segment, split_history  # noqa: E402
import c8_fold as fold  # noqa: E402
import c8_grouping as grouping  # noqa: E402
import hybrid_sparse_retention as retention  # noqa: E402


class FlatMetadataTests(unittest.TestCase):
    def test_causal_speculation_and_dummy_rows(self):
        self.assertEqual(
            flat_plan([3, 4, 7], [100, 20, 1], 4, 7),
            ([0, 0, 0, 1, 0, 0, 0], [98, 99, 100, 20, 1, 1, 1],
             [False] * 4 + [True] * 3),
        )
        self.assertEqual(flat_plan([4], [75], 2, 4),
                         ([0] * 4, [74, 75, 1, 1], [False, False, True, True]))
        self.assertEqual(flat_plan([2], [80], 2, 4),
                         ([0] * 4, [79, 80, 1, 1], [False, False, True, True]))

    def test_mixed_prefill_excludes_non_decode_rows(self):
        self.assertEqual(flat_plan([4, 6, 100], [100, 200, 200], 100, 6, 2),
                         ([0] * 4 + [1] * 2, [97, 98, 99, 100, 199, 200], [False] * 6))

    def test_fail_closed(self):
        for args in (([4], [2], 4, 4), ([4], [100], 4, 3),
                     ([2, 1], [100, 100], 3, 3), ([4], [], 4, 4),
                     ([4], [100], 6, 6), ([4], [100], -1, 4),
                     ([4], [100], 4, 4, 2)):
            with self.subTest(args=args), self.assertRaises(RuntimeError):
                flat_plan(*args)


class SegmentPlanTests(unittest.TestCase):
    def test_short_history_and_aligned_segments(self):
        plan = split_history([4, 4], [16388, 32780], 16384, 128)
        self.assertEqual(plan[0].segments, (Segment(0, 0, 16388), Segment(1, 32768, 12)))
        self.assertTrue(plan[0].causal)
        self.assertEqual([round_.segments for round_ in plan[1:]],
                         [(Segment(1, 0, 16384),), (Segment(1, 16384, 16384),)])
        self.assertTrue(all(not round_.causal for round_ in plan[1:]))

    def test_random_ragged_partition_covers_exact_history(self):
        rng = random.Random(7421)
        for _ in range(100):
            qlens = [rng.randrange(1, 13377) for _ in range(rng.randrange(1, 9))]
            klens = [q + rng.randrange(0, 262144 - q) for q in qlens]
            plan = split_history(qlens, klens, 16384, 128)
            for row, (q, kv) in enumerate(zip(qlens, klens)):
                parts = sorted((s for r in plan for s in r.segments if s.row == row),
                               key=lambda s: s.start)
                self.assertEqual(parts[0].start, 0)
                self.assertEqual(sum(p.length for p in parts), kv)
                self.assertTrue(all(p.start % 128 == 0 for p in parts))
                self.assertTrue(all(a.start + a.length == b.start for a, b in zip(parts, parts[1:])))
                self.assertOnOrBeforeQuery(plan[0].segments[row], kv, q)

    def assertOnOrBeforeQuery(self, segment, kv, q):
        self.assertLessEqual(segment.start, kv - q)
        self.assertGreaterEqual(segment.length, q)
        self.assertEqual(segment.start + segment.length, kv)

    def test_invalid_geometry(self):
        for args in (([], [], 16384, 128), ([5], [4], 16384, 128),
                     ([1, 2], [1], 16384, 128), ([1], [1], 16385, 128),
                     ([1], [1], 0, 128)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                split_history(*args)


class FoldTopologyTests(unittest.TestCase):
    def test_verified_q4_shapes(self):
        for batch in (1, 2, 4, 8):
            ends = [4 * i for i in range(1, batch + 1)]
            histories = [116480 + 128 * i for i in range(batch)]
            with self.subTest(batch=batch):
                self.assertEqual(fold.eligible(ends, batch * 4, batch * 4, (4,), 8),
                                 ((batch, 4), "uniform_capture"))
                self.assertEqual(fold.real_prefixes(ends, [h + 4 for h in histories],
                                                    batch * 4, batch * 4, batch, 4),
                                 (histories, batch))

    def test_padded_replay_is_distinct_from_capture(self):
        ends = list(range(4, 33, 4))
        self.assertIsNone(fold.eligible(ends, 20, 32, (4,), 8)[0])
        for meta_ends, lengths in ((ends, [58244] * 5 + [1] * 3),
                                   (ends[:5], [58244] * 5)):
            self.assertEqual(fold.real_prefixes(meta_ends, lengths, 20, 32, 8, 4),
                             ([58240] * 5 + [1] * 3, 5))

    def test_incompatible_width_or_length_is_rejected(self):
        for ends in ([1, 2, 3, 4], [2, 4], [1, 4], []):
            self.assertIsNone(fold.eligible(ends, 4, 4, (4,), 8)[0])
        self.assertIsNone(fold.eligible([4, 8], 8, 8, (4,), 1)[0])
        for args in (([4], [4], 4, 4, 1, 4), ([2, 4], [9, 10], 4, 4, 1, 4),
                     ([4], [8], 3, 4, 1, 4), ([4], [8], 4, 8, 1, 4),
                     ([4], [8], 0, 4, 1, 4), ([4, 8], [8], 8, 8, 2, 4)):
            with self.subTest(args=args), self.assertRaises(RuntimeError):
                fold.real_prefixes(*args)

    def test_quality_hook_follows_replay_and_update_submission(self):
        # Verify hook ordering without importing torch, scheduling or a graph.
        # The old placement in graph.replay() would validate before update.
        steps = []
        class Graph:
            pass

        class Manager:
            def _updatable_graph_replay(self, desc, attn_metadata):
                steps.extend(("replay", "update"))
                return "original-result"

        provider = fold.FoldParams("layer", 1, 4, None, snap=object(), runtime_meta=object())
        graph = Graph()
        graph.tasks = [SimpleNamespace(provider=provider)]
        manager = Manager()
        manager.graphs = {"decode": graph}
        acl = ModuleType("vllm_ascend.worker.v2.aclgraph_utils")
        acl.ModelAclGraphManager = Manager
        updatable = ModuleType("vllm_ascend.compilation.updatable_graph")
        updatable.UpdatableGraph = Graph
        with patch.dict(sys.modules, {acl.__name__: acl, updatable.__name__: updatable}), \
             patch.dict(os.environ, {"C8_FOLD": "1", "C8_FOLD_VALIDATE": "1"}), \
             patch.object(fold, "_QUALITY_INSTALLED", False), \
             patch.object(fold, "_quality_active", return_value=True) as active, \
             patch.object(fold, "_validate_provider", side_effect=lambda p: steps.append("validate")) as check:
            fold.install_quality_hook()
            wrapper = Manager._updatable_graph_replay
            fold.install_quality_hook()
            self.assertIs(wrapper, Manager._updatable_graph_replay)
            self.assertEqual(manager._updatable_graph_replay("decode", None), "original-result")
            self.assertEqual(steps, ["replay", "update", "validate"])
            check.assert_called_once_with(provider)
            steps.clear()
            active.return_value = False
            manager._updatable_graph_replay("decode", None)
            self.assertEqual(steps, ["replay", "update"])


# Minimal dataclasses with the page arithmetic of the pinned vLLM descriptors.
# The runtime planner/guards themselves are exercised directly; no fake device
# allocation is claimed by this test.
@dataclass(frozen=True)
class FullAttentionSpec:
    block_size: int
    dtype: str
    num_kv_heads: int = 4
    head_size: int = 256
    head_size_v: int = 256
    page_size_padded: int | None = None
    sliding_window: int | None = None
    attention_chunk_size: int | None = None
    non_causal: bool = False

    @property
    def real_page_size_bytes(self):
        return self.block_size * self.num_kv_heads * (self.head_size + self.head_size_v) * (1 if self.dtype == "torch.int8" else 2)

    @property
    def page_size_bytes(self):
        return self.page_size_padded or self.real_page_size_bytes


@dataclass(frozen=True)
class MambaSpec:
    block_size: int
    real_page_size_bytes: int
    page_size_padded: int
    mamba_cache_mode: str = "align"
    num_speculative_blocks: int = 3
    num_prefill_checkpoint_blocks: int = 0

    @property
    def page_size_bytes(self):
        return self.page_size_padded or self.real_page_size_bytes


@dataclass(frozen=True)
class UniformTypeKVCacheSpecs:
    block_size: int
    kv_cache_specs: dict

    @staticmethod
    def from_specs(specs):
        block_sizes = {s.block_size for s in specs.values()}
        if len(block_sizes) != 1 or not all(isinstance(s, FullAttentionSpec) for s in specs.values()):
            return None
        return UniformTypeKVCacheSpecs(block_sizes.pop(), specs)

    @property
    def page_size_bytes(self):
        return sum(s.page_size_bytes for s in self.kv_cache_specs.values())


@dataclass
class KVCacheGroupSpec:
    layer_names: list
    kv_cache_spec: object
    is_eagle_group: bool = False
    host_resident: bool = False


@dataclass
class KVCacheTensor:
    size: int
    layers: list
    layer_stride: int
    block_stride: int
    offset: int = 0
    host_resident: bool = False


def mock_geometry():
    b, page = grouping.B, grouping.P
    module = SimpleNamespace(FullAttentionSpec=FullAttentionSpec, MambaSpec=MambaSpec,
                             UniformTypeKVCacheSpecs=UniformTypeKVCacheSpecs, KVCacheGroupSpec=KVCacheGroupSpec)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1,
                                        decode_context_parallel_size=1, prefill_context_parallel_size=1,
                                        data_parallel_size=1),
        kv_transfer_config=None,
        cache_config=SimpleNamespace(cache_dtype="int8", mamba_cache_mode="align", block_size=b,
                                     get_resolved_kv_cache_layout=lambda: SimpleNamespace(name="LBNHC")),
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=3),
    )
    specs = {name: FullAttentionSpec(b * 2, "torch.int8") for name in grouping.TARGET}
    specs[grouping.DRAFT] = FullAttentionSpec(b, "torch.bfloat16")
    specs.update({name: MambaSpec(b, 3_200_000, 2 * page) for name in grouping.STATE})
    return module, config, specs


def mock_allocation(groups, nblocks=445):
    p, n = grouping.P, nblocks
    total = grouping.S * n
    tensors = [KVCacheTensor(total, list(grouping.TARGET), p * n, p),
               KVCacheTensor(total, [grouping.DRAFT], 2 * p * n, 2 * p, 16 * p * n)]
    tensors.extend(KVCacheTensor(total, list(grouping.STATE[i::3]), p * n, p) for i in range(3))
    return SimpleNamespace(kv_cache_groups=groups, num_blocks=n, kv_cache_tensors=tensors)


class GroupingTests(unittest.TestCase):
    def setUp(self):
        self.module, self.config, self.specs = mock_geometry()
        # These messages reflect host fixtures only, not actual allocation logs.
        with contextlib.redirect_stdout(io.StringIO()):
            self.groups = grouping.plan_groups(self.module, self.config, self.specs)

    def test_exact_four_group_plan_and_descriptor_bounds(self):
        groups = self.groups
        self.assertEqual([len(g.layer_names) for g in groups], [17, 16, 16, 16])
        self.assertEqual(grouping._keyed_groups(self.module, groups), "mixed")
        specs = groups[0].kv_cache_spec.kv_cache_specs
        self.assertTrue(all(specs[n].dtype == "torch.int8" and specs[n].page_size_bytes == grouping.P
                            for n in grouping.TARGET))
        self.assertEqual(specs[grouping.DRAFT].dtype, "torch.bfloat16")
        self.assertEqual(specs[grouping.DRAFT].page_size_bytes, 2 * grouping.P)
        self.assertEqual(groups[0].kv_cache_spec.page_size_bytes, grouping.S)
        pool = mock_allocation(groups)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIs(grouping.certify_alloc(self.module, pool, grouping.S * pool.num_blocks), pool)
        n, p = pool.num_blocks, grouping.P
        intervals = grouping.expected_intervals(n)
        self.assertEqual(set(intervals), grouping.ALL)
        self.assertEqual(intervals[grouping.DRAFT], (16 * p * n, 2 * p * n, 2 * p))
        # Check representative global block IDs: state/target roles can overlay
        # the same slot, but distinct live block IDs never overlap physically.
        for a, (offset_a, size_a, page_a) in intervals.items():
            self.assertEqual(size_a, n * page_a)
            self.assertLessEqual(offset_a + size_a, grouping.S * n)
            for b, (offset_b, _, page_b) in intervals.items():
                for ia, ib in ((0, 1), (n // 2, n - 1), (n - 1, 0)):
                    left = offset_a + ia * page_a
                    right = offset_b + ib * page_b
                    self.assertTrue(left + page_a <= right or right + page_b <= left,
                                    f"overlap {a}/{b} at IDs {ia}/{ib}")

    def test_invalid_layouts_and_geometry(self):
        for name, spec in ((grouping.STATE[0], replace(self.specs[grouping.STATE[0]], real_page_size_bytes=grouping.P + 1)),
                           (grouping.DRAFT, replace(self.specs[grouping.DRAFT], dtype="torch.int8")),
                           (grouping.TARGET[0], replace(self.specs[grouping.TARGET[0]], num_kv_heads=8))):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                grouping.plan_groups(self.module, self.config, {**self.specs, name: spec})
        self.config.parallel_config.tensor_parallel_size = 2
        with self.assertRaisesRegex(RuntimeError, "TP1"):
            grouping.plan_groups(self.module, self.config, self.specs)
        with self.assertRaises(RuntimeError):
            grouping.expected_intervals(0)

    def test_corrupted_allocator_descriptors_and_budget(self):
        budget = grouping.S * 445
        for mutation in (lambda p: setattr(p.kv_cache_tensors[0], "offset", grouping.P),
                         lambda p: setattr(p.kv_cache_tensors[0], "block_stride", grouping.P + 1),
                         lambda p: setattr(p.kv_cache_tensors[0], "host_resident", True),
                         lambda p: p.kv_cache_tensors.append(p.kv_cache_tensors[0])):
            pool = mock_allocation(self.groups)
            mutation(pool)
            with self.assertRaises(RuntimeError):
                grouping.certify_alloc(self.module, pool, budget)
        with self.assertRaisesRegex(RuntimeError, "exceed budget"):
            grouping.certify_alloc(self.module, mock_allocation(self.groups), budget - 1)

    def test_scheduler_normalized_spec_keeps_full_mixed_page_charge(self):
        module = self.module
        module.get_kv_cache_groups = Mock(return_value="stock-groups")
        module.validate_kv_cache_layout = Mock(return_value="stock-validation")
        module._get_kv_cache_bytes_per_block = Mock(return_value=grouping.S)
        module.get_kv_cache_config_from_groups = Mock(return_value="stock-allocation")
        grouping.patch_module(module)
        identity = module.get_kv_cache_groups
        grouping.patch_module(module)
        self.assertIs(identity, module.get_kv_cache_groups)
        normalized = [replace(self.groups[0], kv_cache_spec=self.groups[0].kv_cache_spec.kv_cache_specs[grouping.TARGET[0]]),
                      *self.groups[1:]]
        self.assertEqual(grouping._keyed_groups(module, normalized, normalized_ok=True), "scheduler")
        self.assertEqual(module._get_kv_cache_bytes_per_block(normalized), grouping.S)
        self.assertEqual(module._get_kv_cache_bytes_per_block(self.groups), grouping.S)
        with patch.dict(os.environ, {"C8_GROUPING": "0"}):
            self.assertEqual(module.get_kv_cache_groups(self.config, self.specs), "stock-groups")

    def test_grouping_import_hook_fails_if_module_loaded_early(self):
        with patch.dict(os.environ, {"C8_GROUPING": "1"}), \
             patch.dict(sys.modules, {grouping.MODULE: ModuleType(grouping.MODULE)}):
            with self.assertRaisesRegex(RuntimeError, "before KV planner"):
                grouping.install()


class AscendHybridKVCacheCoordinator:
    pass


class AscendMambaManager:
    pass


def mock_retention(dtype):
    b, page, _ = retention.GEOMETRIES[dtype]
    full = FullAttentionSpec(b, dtype, page_size_padded=page)
    state = MambaSpec(b, 3_200_000, page)
    cfg = SimpleNamespace(
        kv_cache_groups=[KVCacheGroupSpec([*retention.TARGET, retention.DRAFT], full)]
        + [KVCacheGroupSpec(list(retention.GDN[i::3]), state) for i in range(3)],
        prefix_cache_retention_interval=0, kv_transfer_config=None,
    )
    coordinator = AscendHybridKVCacheCoordinator()
    coordinator.skips_eagle_block_drop = True
    coordinator._cache_hit_alignment_tokens = retention.MATCH
    coordinator.scheduler_block_size = b
    coordinator.enable_partial_hash_hits = coordinator.enable_caching = True
    coordinator.eagle_group_ids = {0}
    coordinator.dcp_world_size = 1
    coordinator.retention_interval = None
    coordinator.num_reprefillable_tokens = 0
    pool = object()
    coordinator.single_type_managers = [SimpleNamespace(block_pool=pool)] + [AscendMambaManager() for _ in range(3)]
    for manager in coordinator.single_type_managers[1:]:
        manager.block_pool = pool
        manager.has_prefill_checkpoint_blocks = False
    return SimpleNamespace(coordinator=coordinator, kv_cache_config=cfg)


class SparseRetentionTests(unittest.TestCase):
    def test_boundaries(self):
        for block, _, _ in retention.GEOMETRIES.values():
            h = 38 * block
            self.assertEqual(retention.boundaries(h, 128), (h - 128, h))
            self.assertEqual(retention.boundaries(h + 1, 128), (h,))
        for args in ((0, 128), (-1, 128), (1, 0)):
            with self.subTest(args=args), self.assertRaises(RuntimeError):
                retention.boundaries(*args)

    def test_scope_and_native_validator_order(self):
        coordinator_module = ModuleType("vllm.v1.core.kv_cache_coordinator")
        validate = Mock()
        coordinator_module._validate_prefix_cache_retention_interval = validate
        with patch.dict(sys.modules, {coordinator_module.__name__: coordinator_module}):
            for dtype in retention.GEOMETRIES:
                kvm = mock_retention(dtype)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertTrue(retention.configure(kvm))
                self.assertEqual(kvm.coordinator.retention_interval, 0)
                h = kvm.coordinator.scheduler_block_size * 38
                self.assertEqual(kvm.coordinator.get_replay_boundaries(SimpleNamespace(num_prompt_tokens=h)), (h - 128, h))
                validate.assert_called_with(0, kvm.coordinator.scheduler_block_size, kvm.kv_cache_config)
        kvm = mock_retention("torch.int8")
        kvm.kv_cache_config.kv_cache_groups.pop()
        self.assertFalse(retention.configure(kvm))
        self.assertIsNone(kvm.coordinator.retention_interval)

    def test_reject_unsupported_modes_before_changes(self):
        for mutation in (lambda k: setattr(k.coordinator, "dcp_world_size", 2),
                         lambda k: setattr(k.coordinator, "_cache_hit_alignment_tokens", 256),
                         lambda k: setattr(k.kv_cache_config, "prefix_cache_retention_interval", 1),
                         lambda k: setattr(k.coordinator, "skips_eagle_block_drop", False),
                         lambda k: setattr(k.coordinator.single_type_managers[1], "has_prefill_checkpoint_blocks", True)):
            kvm = mock_retention("torch.int8")
            mutation(kvm)
            with self.assertRaises(RuntimeError):
                retention.configure(kvm)
            self.assertIsNone(kvm.coordinator.retention_interval)

    def test_install_is_opt_in_idempotent_and_after_manager_init(self):
        order = []
        class Manager:
            def __init__(self, arg):
                self.arg = arg
                order.append("native-init")

        module = ModuleType("vllm.v1.core.kv_cache_manager")
        module.KVCacheManager = Manager
        def configure(manager):
            self.assertEqual(manager.arg, 32)
            order.append("configure")
        with patch.dict(sys.modules, {module.__name__: module}), \
             patch.dict(os.environ, {"QWEN_HYBRID_SPARSE_RETENTION": "0"}), \
             patch.object(retention, "configure", side_effect=configure):
            original = Manager.__init__
            retention.install()
            self.assertIs(original, Manager.__init__)
            os.environ["QWEN_HYBRID_SPARSE_RETENTION"] = "1"
            retention.install()
            wrapped = Manager.__init__
            self.assertIsNot(original, wrapped)
            retention.install()
            self.assertIs(wrapped, Manager.__init__)
            Manager(32)
            self.assertEqual(order, ["native-init", "configure"])


if __name__ == "__main__":
    unittest.main()
