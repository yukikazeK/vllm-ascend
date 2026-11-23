# SPDX-License-Identifier: Apache-2.0

import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

EXAMPLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE))
from launch import BATCH_DEPTHS, serving_args, validate_policy


class LauncherTests(unittest.TestCase):

    def setUp(self):
        self.args = [
            "serve", "/models/model", "--tensor-parallel-size", "1",
            "--no-async-scheduling",
            "--compilation-config", json.dumps({"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}),
            "--speculative-config", json.dumps({
                "method": "mtp",
                "num_speculative_tokens": 3,
                "num_speculative_tokens_per_batch_size": BATCH_DEPTHS,
            }),
        ]

    def test_accepts_synchronous_full_decode_tp1(self):
        self.assertEqual(serving_args(self.args), self.args)

    def test_rejects_incompatible_parallel_graph_and_scheduler(self):
        variants = ((3, "2"), (4, "--async-scheduling"),
                    (6, '{"mode":0,"cudagraph_mode":"PIECEWISE"}'))
        for index, replacement in variants:
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                args = self.args.copy()
                args[index] = replacement
                serving_args(args)

    def test_rejects_out_of_range_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = Path(tmp) / "test.json"
            policy.write_text('{"64": {"1": 4}}')
            with self.assertRaisesRegex(ValueError, "invalid MTP depth"):
                validate_policy(policy)

    def test_zero_draft_retains_one_scratch_slot_and_discards_proposals(self):
        with patch.dict("os.environ", {"DYN_FULL_POLICY": str(EXAMPLE / "policy.example.json")}):
            sys.path.insert(0, str(EXAMPLE / "runtime"))
            try:
                policy = importlib.import_module("autok")
                running = [SimpleNamespace(num_prompt_tokens=64, spec_token_ids=[3]) for _ in range(64)]
                requests = {str(i): request for i, request in enumerate(running)}
                scheduler = SimpleNamespace(requests=requests, running=running, waiting=[])
                policy.prepare(scheduler)
                self.assertTrue(all(not request.spec_token_ids for request in running))
                self.assertEqual(policy.physical_depth("0", 3), 1)
                output = SimpleNamespace(num_scheduled_tokens={"0": 1}, scheduled_spec_decode_tokens={})
                policy.choose(scheduler, output)
                self.assertEqual(output.num_spec_tokens_to_schedule, 0)
            finally:
                sys.path.remove(str(EXAMPLE / "runtime"))
                sys.modules.pop("autok", None)


if __name__ == "__main__":
    unittest.main()
