# SPDX-License-Identifier: Apache-2.0
"""Host-only launch compatibility and command construction checks."""
import importlib.util
import json
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import os

SPEC = importlib.util.spec_from_file_location("c8_launch", Path(__file__).resolve().parents[1] / "launch.py")
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "model with spaces"
        self.model.mkdir()
        self.config = {"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": {
            "model_type": "qwen3_5_text", "num_hidden_layers": 64,
            "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
            "hidden_size": 5120, "full_attention_interval": 4,
            "mtp_num_hidden_layers": 1, "max_position_embeddings": 262144,
            "layer_types": ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)],
        }}
        (self.model / "quant_model_weights-kv-cache-c8.safetensors").touch()
        self.write_model()
        self.run = self.root / "run with spaces"
        self.sources = {"vllm": self.root / "installed core" / "vllm",
                        "vllm_ascend": self.root / "installed ascend" / "vllm_ascend"}

    def write_model(self):
        (self.model / "config.json").write_text(json.dumps(self.config))

    def options(self, *args):
        return launch.parse_args([str(self.model), "--state-dir", str(self.root), *args])[1]

    def test_model_geometry(self):
        self.assertEqual(launch.validate_model(self.model), self.model)
        self.config["text_config"]["num_key_value_heads"] = 8
        self.write_model()
        with self.assertRaisesRegex(ValueError, "num_key_value_heads"):
            launch.validate_model(self.model)
        self.config["text_config"]["num_key_value_heads"] = 4
        self.write_model()
        (self.model / "quant_model_weights-kv-cache-c8.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "calibrated C8"):
            launch.validate_model(self.model)

    def test_source_check_rejects_mismatch_and_missing(self):
        name = "vllm/v1/a.py"
        source = self.sources["vllm"] / "v1/a.py"
        source.parent.mkdir(parents=True)
        good = b"# exact source\n"
        source.write_bytes(good)
        expected = {name: sha256(good).hexdigest()}
        self.assertEqual(launch.validate_release(self.sources, expected), self.sources)
        source.write_bytes(b"# drift\n")
        with self.assertRaisesRegex(ValueError, name):
            launch.validate_release(self.sources, expected)
        source.unlink()
        with self.assertRaisesRegex(ValueError, name):
            launch.validate_release(self.sources, expected)

    def test_source_manifest_covers_dynamic_patches_and_c8_hooks(self):
        sources = launch.expected_sources()
        self.assertGreaterEqual(len(sources), 17)
        for filename in launch.SOURCE_SHA256:
            self.assertEqual(sources[filename], launch.SOURCE_SHA256[filename])
        for filename in ("manifest.json", "gdn_k2/manifest.json", "gdn_workspace/manifest.json"):
            manifest = json.loads((launch.DYNAMIC / "patches" / filename).read_text())
            for module, record in manifest.items():
                self.assertEqual(sources[module.replace(".", "/") + ".py"], record["source_sha256"])

    def test_cann_version_and_custom_opp_refused(self):
        opp = self.root / "opp"
        opp.mkdir()
        info = opp / "version.info"
        info.write_text("Version=9.2.1\nrequired_package_runtime_version=>=8.5\n")
        launch.validate_cann({"ASCEND_OPP_PATH": str(opp)})
        launch.validate_cann({"ASCEND_HOME_PATH": str(self.root)})
        with self.assertRaisesRegex(ValueError, "custom OPP"):
            launch.validate_cann({"ASCEND_OPP_PATH": str(opp), "ASCEND_CUSTOM_OPP_PATH": "modified"})
        info.write_text("Version=9.2.0\n")
        with self.assertRaisesRegex(ValueError, "9.2.1"):
            launch.validate_cann({"ASCEND_OPP_PATH": str(opp)})

    def test_only_installed_fla_vendor_is_allowed(self):
        opp = self.root / "opp"
        opp.mkdir()
        (opp / "version.info").write_text("Version=9.2.1\n")
        fla = self.root / "fla_npu"
        vendor = fla / "opp/vendors/fla_npu_transformer"
        lib = vendor / "op_api/lib"
        lib.mkdir(parents=True)
        env = {"ASCEND_OPP_PATH": str(opp), "ASCEND_CUSTOM_OPP_PATH": os.pathsep.join(map(str, (vendor, lib)))}
        with patch.object(launch.util, "find_spec", return_value=SimpleNamespace(submodule_search_locations=[str(fla)])), \
             patch.object(launch.metadata, "version", return_value=launch.FLA_RELEASE):
            launch.validate_cann(env)
            env["ASCEND_CUSTOM_OPP_PATH"] += os.pathsep + str(self.root / "experimental")
            with self.assertRaisesRegex(ValueError, "custom OPP"):
                launch.validate_cann(env)

    def test_fixed_command_matches_dynamic_validator(self):
        opts = self.options()
        cmd = launch.build_command(opts, self.model, self.run)
        args = cmd[cmd.index("--") + 1:]
        self.assertEqual(launch.load_dynamic_launcher().serving_args(args), args)
        def value(flag):
            return args[args.index(flag) + 1]
        self.assertEqual(value("--max-num-batched-tokens"), "8384")
        self.assertEqual(value("--kv-cache-dtype"), "int8")
        self.assertEqual(value("--kv-cache-dtype-skip-layers"), "0")
        self.assertEqual(value("--max-model-len"), "262144")
        self.assertEqual(value("--max-num-seqs"), "64")
        self.assertEqual(value("--gpu-memory-utilization"), "0.9")
        self.assertEqual(value("--prefix-cache-retention-interval"), "0")
        self.assertEqual(value("--tensor-parallel-size"), "1")
        self.assertEqual(value("--data-parallel-size"), "1")
        self.assertIn("--enable-prefix-caching", args)
        self.assertIn("--enable-chunked-prefill", args)
        self.assertNotIn("--enforce-eager", args)
        self.assertNotIn("--kv-cache-memory-bytes", args)
        self.assertEqual(json.loads(value("--compilation-config"))["cudagraph_mode"], "FULL_DECODE_ONLY")
        self.assertEqual(json.loads(value("--speculative-config"))["num_speculative_tokens"], 3)
        self.assertEqual(args[1], str(self.model))
        self.assertEqual(opts.batch_budget, 8384)
        self.assertEqual(self.options("--batch-budget", "8192").batch_budget, 8192)

    def test_run_environment_isolated_and_portable(self):
        opts = self.options()
        env = launch.build_env(opts, self.model, self.run, self.sources, {
            "ASCEND_RT_VISIBLE_DEVICES": "1", "C8_TEST_TIER_FILE": "/old/file",
            "PYTHONPATH": "/installed/lib", "C8_FOLD_VALIDATE": "1", "CAPACITY_AUDIT_DIR": "/old/audit",
        })
        self.assertNotIn("C8_TEST_TIER_FILE", env)
        self.assertNotIn("CAPACITY_AUDIT_DIR", env)
        self.assertEqual(env["C8_FOLD_VALIDATE"], "0")
        self.assertEqual(env["C8_GROUPING_ASCEND_ROOT"], str(self.sources["vllm_ascend"].parent))
        self.assertEqual(env["C8_AUDIT_RUN"], str(self.run / "audit"))
        self.assertEqual(env["PYTHONPATH"].split(launch.os.pathsep)[0], str(launch.HERE / "runtime"))
        self.assertEqual(env["ASCEND_RT_VISIBLE_DEVICES"], "1")
        for flag in ("C8_GROUPING", "C8_FOLD", "QWEN_HYBRID_SPARSE_RETENTION", "C8_PREFILL_LSE"):
            self.assertEqual(env[flag], "1")
        self.assertEqual(env["C8_FOLD_MAX_REQS"], "8")
        self.assertEqual(env["C8_FOLD_WIDTHS"], "4")
        self.assertEqual(launch.build_env(self.options("--validate-fold"), self.model, self.run,
                                         self.sources, {})["C8_FOLD_VALIDATE"], "1")
        with self.assertRaisesRegex(ValueError, "exactly one NPU"):
            launch.build_env(opts, self.model, self.run, self.sources, {"ASCEND_RT_VISIBLE_DEVICES": "0,1"})

    def test_default_policy_is_valid(self):
        self.assertEqual(launch.load_dynamic_launcher().validate_policy(launch.HERE / "policy.example.json")["8192"]["8"], 3)


if __name__ == "__main__":
    unittest.main()
