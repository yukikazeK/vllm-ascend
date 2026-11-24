# SPDX-License-Identifier: Apache-2.0
"""Check that the experimental replay guards and per-forward lifetimes hold."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from gdn_k2_exact import is_eligible
from gdn_shared_indices import flatten_once


class GDNIndexingTests(unittest.TestCase):
    def test_exact_graph_rejects_padding_prefill_and_wrong_width(self):
        self.assertTrue(is_eligible(32, 96, 32, 96, 3, 3))
        for batch in ((31, 93, 3, 3), (32, 96, None, 3), (32, 95, 3, 3), (32, 96, 3, 4)):
            self.assertFalse(is_eligible(32, 96, *batch))

    def test_cache_respects_index_group_lengths_and_forward_lifetime(self):
        calls = []

        def build(indices, n, qsl):
            result = object()
            calls.append(result)
            return result

        first = SimpleNamespace()
        indices, qsl = object(), object()
        result = flatten_once(first, indices, 96, qsl, build)
        for _ in range(47):
            self.assertIs(result, flatten_once(first, indices, 96, qsl, build))
        self.assertIsNot(result, flatten_once(first, indices, 93, qsl, build))
        self.assertIsNot(result, flatten_once(first, object(), 96, qsl, build))
        self.assertIsNot(result, flatten_once(first, indices, 96, object(), build))
        self.assertIsNot(result, flatten_once(SimpleNamespace(), indices, 96, qsl, build))
        self.assertEqual(len(calls), 5)


if __name__ == "__main__":
    unittest.main()
