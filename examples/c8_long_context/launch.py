#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch the validated 910B4-1/Qwen 27B C8 long-context experiment.

No installed source is edited. The existing dynamic-MTP launcher and this
experiment's import hooks operate in the isolated child process.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from importlib import metadata, util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
DYNAMIC = HERE.parent / "dynamic_mtp"
RELEASE = "0.30.0"
CANN_RELEASE = "9.2.1"
FLA_DISTRIBUTION = "flash-linear-attention-npu-a2"
FLA_RELEASE = "26.9.1"
MAX_MODEL_LEN = 262144
MAX_SEQS = 64
BATCH_BUDGETS = (8192, 8384)
GRAPH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
BATCH_DEPTHS = [[1, 1, 3], [2, 3, 2], [4, 7, 1], [8, 128, 0]]

# Exact installed 0.30.0 container sources read during the verified service runs.
# Dynamic MTP's separate manifest additionally covers all of its patched files.
SOURCE_SHA256 = {
    "vllm/v1/core/kv_cache_utils.py": "2666c9f113584e52e7521058efd2c0d9598544559dc942d3a3da87001a3fedf2",
    "vllm/v1/core/kv_cache_coordinator.py": "bf342f79f2bc8d6acb69d3becefb334509e84cae888d42bbd421a8dc8ca9eca3",
    "vllm/v1/core/kv_cache_manager.py": "58800535eda854580d84e1025bf32fe01b2a43ce0bba501b4f3d125e4f2c8afa",
    "vllm/v1/core/single_type_kv_cache_manager.py": "0233531d85aeced9e57a38c76d83db970519c92902b7e55484390cb2b64985bf",
    "vllm/v1/core/block_pool.py": "34c2fd67148cecb7763ff49ad694a57073759b0fa67572a4270dc995b4acf8fc",
    "vllm/v1/kv_cache_interface.py": "1cf202f1a44d5bc5c3832b70b41507687ecceea3784e67a4f003bc0210d6eecb",
    "vllm/v1/core/sched/scheduler.py": "c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884",
    "vllm/v1/worker/gpu/block_table.py": "61c004315d5af7e7eae4e2a9e6be92ea82c520327690a7f55a73bb9ce95f520a",
    "vllm_ascend/attention/attention_v1.py": "bea4e410f797557e29cedbb495024147606e74b0e056d725194760ca69296eba",
    "vllm_ascend/quantization/methods/kv_cache/kv_c8.py": "1e85e37adbf497f934642ec80823cef566010806f3a930b53a1ef914e561b78f",
    "vllm_ascend/worker/v2/attn_utils.py": "b1e7d441168a5c96f2364199f9092c8b0b95b297393c63dc31dbb19beae8e160",
    "vllm_ascend/worker/v2/block_table.py": "54770439398a23d9e18dc6bb7923b2fb9b251090512fe9c59db70a869e3868e4",
    "vllm_ascend/worker/v2/aclgraph_utils.py": "b2271eb4aab5876e85370897236a99ece7e9a9541b05f2fc0dd923b32bb8193d",
    "vllm_ascend/worker/v2/model_runner.py": "533986687a50caa40afe7d3610d5fd9fb7d64d0ef167d95fe2c656bf0a3ccec5",
    "vllm_ascend/compilation/updatable_graph.py": "d41d912ce8f4183228fdc182dbdb6a504f07642fddb8872c657e7f4429ac0729",
    "vllm_ascend/patch/platform/patch_kv_cache_coordinator.py": "828169b4c131a6e42f01be91ce87a36fd1b6025d9f6607eb75e68df9dbf17634",
    "vllm_ascend/patch/platform/patch_kv_cache_utils.py": "5fc49ebb9b64a2f14c4eb38b8c66d3daf1b7612961cf76392cf113d76f555aab",
}


def load_dynamic_launcher():
    spec = util.spec_from_file_location("c8_dynamic_launcher", DYNAMIC / "launch.py")
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the bundled dynamic MTP launcher")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_sources():
    """Merge the actual input source hashes of the selected dynamic patch set."""
    result = dict(SOURCE_SHA256)
    merged = {}
    for filename in ("manifest.json", "gdn_k2/manifest.json", "gdn_workspace/manifest.json"):
        merged.update(json.loads((DYNAMIC / "patches" / filename).read_text()))
    for module, record in merged.items():
        filename = module.replace(".", "/") + ".py"
        digest = record["source_sha256"]
        if filename in result and result[filename] != digest:
            raise ValueError(f"conflicting source hash for {filename}")
        result[filename] = digest
    return result


def validate_release(sources=None, expected=None):
    """Resolve package directories, not a container-specific checkout path."""
    if sources is None:
        sources = {}
        for distribution, package in (("vllm", "vllm"), ("vllm-ascend", "vllm_ascend")):
            version = metadata.version(distribution)
            if version.partition("+")[0] != RELEASE:
                raise ValueError(f"{distribution} {RELEASE} is required, found {version}")
            spec = util.find_spec(package)
            if spec is None or not spec.submodule_search_locations:
                raise ValueError(f"cannot find the installed {package} package")
            sources[package] = Path(next(iter(spec.submodule_search_locations))).resolve()
    for filename, digest in (expected_sources() if expected is None else expected).items():
        package, suffix = filename.split("/", 1)
        source = sources[package] / suffix
        if not source.is_file() or sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f"installed source differs from the validated {RELEASE}: {filename}")
    return sources


def validate_cann(env):
    if env.get("ASCEND_CUSTOM_OPP_PATH"):
        # The tested Python environment automatically registers the vendor
        # directory of the *installed* FLA NPU dependency at interpreter
        # startup, even when ASCEND_CUSTOM_OPP_PATH was unset in the shell.
        # Allow only that already-installed package, never arbitrary or
        # experimental CANN OPP overrides.
        spec = util.find_spec("fla_npu")
        roots = [] if spec is None else list(spec.submodule_search_locations or [])
        if len(roots) != 1:
            raise ValueError("custom OPP is not validated (installed FLA package not found)")
        try:
            fla_version = metadata.version(FLA_DISTRIBUTION)
        except metadata.PackageNotFoundError as exc:
            raise ValueError("custom OPP is not validated (FLA distribution not found)") from exc
        if fla_version != FLA_RELEASE:
            raise ValueError(f"custom OPP requires the tested FLA NPU {FLA_RELEASE}, found {fla_version}")
        vendor = Path(roots[0]).resolve() / "opp/vendors/fla_npu_transformer"
        allowed = {vendor.resolve(), (vendor / "op_api/lib").resolve()}
        entries = {Path(path).resolve() for path in env["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep) if path}
        if not entries or not entries <= allowed or not all(path.is_dir() for path in entries):
            raise ValueError("custom OPP is not validated; only installed FLA NPU vendor paths are allowed")
    opp = env.get("ASCEND_OPP_PATH")
    if not opp:
        home = env.get("ASCEND_TOOLKIT_HOME") or env.get("ASCEND_HOME_PATH")
        if not home:
            raise ValueError("source the installed CANN environment first (ASCEND_OPP_PATH)")
        opp = str(Path(home) / "opp")
    version_file = Path(opp) / "version.info"
    if not version_file.is_file() or f"Version={CANN_RELEASE}" not in version_file.read_text().splitlines():
        raise ValueError(f"validated CANN OPP version {CANN_RELEASE} required: {version_file}")


def validate_model(model):
    """Guard the model geometry; runtime also validates scale values/actual views."""
    model = Path(model).expanduser().resolve()
    if not model.is_dir() or not (model / "config.json").is_file():
        raise ValueError("MODEL must be a local checkpoint directory containing config.json")
    if not (model / "quant_model_weights-kv-cache-c8.safetensors").is_file():
        raise ValueError("MODEL must contain the calibrated C8 scale safetensors")
    config = json.loads((model / "config.json").read_text())
    if not isinstance(config, dict) or not isinstance(config.get("text_config"), dict):
        raise ValueError("MODEL config.json must contain a text_config mapping")
    text = config["text_config"]
    required = {
        "model_type": "qwen3_5_text", "num_hidden_layers": 64,
        "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
        "hidden_size": 5120, "full_attention_interval": 4,
        "mtp_num_hidden_layers": 1, "max_position_embeddings": MAX_MODEL_LEN,
        "layer_types": ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)],
    }
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("MODEL architecture differs from the validated Qwen 27B checkpoint")
    for key, value in required.items():
        if text.get(key) != value:
            raise ValueError(f"MODEL geometry differs from the validated checkpoint: {key}")
    return model


def build_command(opts, model, run):
    command = [sys.executable, str(DYNAMIC / "launch.py"), "--state-dir", str(run / "dynamic"),
               "--gdn-k2-indexing", "--gdn-prefill-grouping", "--policy", str(opts.policy.resolve()), "--",
               "serve", str(model), "--served-model-name", opts.served_model_name or model.name,
               "--host", opts.host, "--port", str(opts.port),
               "--tensor-parallel-size", "1", "--data-parallel-size", "1", "--pipeline-parallel-size", "1",
               "--prefill-context-parallel-size", "1", "--decode-context-parallel-size", "1",
               "--dtype", "bfloat16", "--quantization", "ascend", "--kv-cache-dtype", "int8",
               "--kv-cache-dtype-skip-layers", "0", "--max-model-len", str(MAX_MODEL_LEN),
               "--max-num-seqs", str(MAX_SEQS), "--max-num-batched-tokens", str(opts.batch_budget),
               "--gpu-memory-utilization", "0.9", "--safetensors-load-strategy", "lazy",
               "--language-model-only", "--trust-remote-code", "--enable-prefix-caching",
               "--enable-prompt-tokens-details", "--prefix-match-unit", "128", "--mamba-cache-mode", "align",
               "--enable-chunked-prefill", "--mamba-ssm-cache-dtype", "float32",
               "--prefix-cache-retention-interval", "0", "--no-async-scheduling",
               "--compilation-config", json.dumps({"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                                  "cudagraph_capture_sizes": GRAPH_SIZES}),
               "--additional-config", '{"enable_cpu_binding":false}',
               "--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 3,
                                                  "num_speculative_tokens_per_batch_size": BATCH_DEPTHS})]
    return command


def build_env(opts, model, run, sources, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    # Do not accidentally inherit a test-only tier selector or stale overlay.
    for key in ("VLLM_ENABLE_STARTUP_PLAN", "VLLM_SERVER_DEV_MODE", "ASCEND_LAUNCH_BLOCKING", "C8_ND_DIAG",
                "C8_TEST_TIER_FILE", "C8_PREFILL_MEMORY_AUDIT", "C8_FOLD_QUALITY_STEPS",
                "C8_FOLD_QUALITY_LAYERS", "C8_FOLD_REL_L2_MAX", "DYN_FULL_ROOT", "DYN_FULL_POLICY",
                "MTP_K2_TARGET_EXACT", "MTP_GDN_EXACT_K2", "CAPACITY_AUDIT_DIR", "CAPACITY_DYNAMIC_RUNTIME"):
        env.pop(key, None)
    visible = env.get("ASCEND_RT_VISIBLE_DEVICES", "0")
    if not visible.isdecimal():
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES must designate exactly one NPU index")
    env["ASCEND_RT_VISIBLE_DEVICES"] = visible
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(HERE / "runtime") + (os.pathsep + previous if previous else "")
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")
    env.update({
        "PYTHONUNBUFFERED": "1", "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_ENGINE_READY_TIMEOUT_S": "1200",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True", "VLLM_CACHE_ROOT": str(run / "compile-cache"),
        "C8_CHECKPOINT": str(model), "C8_AUDIT_RUN": str(run / "audit"), "C8_ND_RUN": "1",
        "C8_PREFILL_FUSED_GATHER": "1", "C8_PREFILL_LSE": "1", "C8_GROUPING": "1",
        "QWEN_HYBRID_SPARSE_RETENTION": "1", "C8_FOLD": "1", "C8_FOLD_MAX_REQS": "8", "C8_FOLD_WIDTHS": "4",
        "C8_FOLD_VALIDATE": "1" if opts.validate_fold else "0",
        "C8_GROUPING_ASCEND_ROOT": str(sources["vllm_ascend"].parent),
    })
    return env


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--state-dir", type=Path, required=True, help="Directory for unique runtime/audit records")
    parser.add_argument("--policy", type=Path, default=HERE / "policy.example.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--served-model-name")
    parser.add_argument("--batch-budget", type=int, choices=BATCH_BUDGETS, default=8384)
    parser.add_argument("--validate-fold", action="store_true", help="Capture opt-in numerical diagnostic snapshots")
    parser.add_argument("--verify-only", action="store_true", help="Check compatibility without starting a service")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return parser, args


def main(argv=None):
    parser, opts = parse_args(argv)
    try:
        model = validate_model(opts.model)
        sources = validate_release()
        validate_cann(os.environ)
        dynamic = load_dynamic_launcher()
        policy = dynamic.validate_policy(opts.policy)
        if not shutil.which("vllm") or not shutil.which("patch"):
            raise ValueError("vllm and GNU patch must be on PATH within the verified Python environment")
        for filename in ("sitecustomize.py", "c8_grouping.py", "c8_grouping_worker_audit.py",
                         "hybrid_sparse_retention.py", "c8_nd_backend.py", "c8_fold.py"):
            if not (HERE / "runtime" / filename).is_file():
                raise ValueError(f"missing bundled runtime: {filename}")
        # Validate the exact CLI and environment without creating a run in verify-only mode.
        prospective = opts.state_dir.resolve() / "compatibility-check"
        probe_command = build_command(opts, model, prospective)
        dynamic.serving_args(probe_command[probe_command.index("--") + 1:])
        build_env(opts, model, prospective, sources)
        if opts.verify_only:
            print(f"Compatibility verified: {len(expected_sources())} installed source files; CANN {CANN_RELEASE}; model geometry and policy")
            return
        opts.state_dir.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix="c8-", dir=opts.state_dir.resolve()))
        for name in ("audit", "compile-cache", "dynamic"):
            (run / name).mkdir()
        # Freeze the policy per service to prevent in-flight external edits.
        frozen = run / "policy.json"
        frozen.write_text(json.dumps(policy, indent=2) + "\n")
        opts.policy = frozen
        env = build_env(opts, model, run, sources)
        command = build_command(opts, model, run)
        (run / "launch-config.json").write_text(json.dumps({"command": command, "policy": policy,
            "batch_budget": opts.batch_budget, "source_packages": {k: str(v) for k, v in sources.items()},
            "cann": CANN_RELEASE, "diagnostic_snapshots": opts.validate_fold}, indent=2) + "\n")
    except (ValueError, OSError, argparse.ArgumentError, metadata.PackageNotFoundError) as exc:
        parser.error(str(exc))
    print(f"C8 long-context run: {run}", flush=True)
    print(f"Audit: {run / 'audit'}", flush=True)
    if opts.validate_fold:
        print(f"For live same-cache checks, enable marker {run / 'audit' / '.enable_fold_quality'}; remove before timing", flush=True)
    os.execve(sys.executable, command, env)


if __name__ == "__main__":
    main()
