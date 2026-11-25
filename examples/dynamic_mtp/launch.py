#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Enable the optional, version-matched dynamic MTP overlay for vLLM serve."""

import argparse
from hashlib import sha256
from importlib import metadata, util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
RELEASE = "0.30.0"
BATCH_DEPTHS = [[1, 1, 3], [2, 3, 2], [4, 7, 1], [8, 128, 0]]
PARALLEL_FLAGS = (
    "--tensor-parallel-size",
    "--data-parallel-size",
    "--pipeline-parallel-size",
    "--prefill-context-parallel-size",
    "--decode-context-parallel-size",
)


def validate_policy(path):
    policy = json.loads(path.read_text())
    if not isinstance(policy, dict) or not policy:
        raise ValueError("the MTP policy needs at least one input-length row")
    for length, row in policy.items():
        if not length.isdecimal() or int(length) <= 0 or str(int(length)) != length:
            raise ValueError(f"invalid input length {length!r}")
        if not isinstance(row, dict) or not row:
            raise ValueError(f"missing pressure points for input length {length}")
        for pressure, depth in row.items():
            if not pressure.isdecimal() or int(pressure) <= 0 or str(int(pressure)) != pressure:
                raise ValueError(f"invalid pressure ceiling {pressure!r}")
            if type(depth) is not int or depth not in range(4):
                raise ValueError(f"invalid MTP depth {depth!r}; choose an integer from 0 through 3")
    return policy


def serving_args(command):
    if len(command) < 2 or command[0] != "serve" or command[1].startswith("-"):
        raise ValueError("pass vllm arguments as: -- serve MODEL [options]")
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    for flag in PARALLEL_FLAGS:
        parser.add_argument(flag, action="append", type=int)
    parser.add_argument("--speculative-config", action="append", type=json.loads)
    parser.add_argument("--compilation-config", "-cc", action="append", type=json.loads)
    parser.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction)
    settings, rest = parser.parse_known_args(command[2:])
    for flag in PARALLEL_FLAGS:
        values = getattr(settings, flag[2:].replace("-", "_"))
        if values and any(value != 1 for value in values):
            raise ValueError(f"{flag} must be 1; run separate TP1 processes for two cards")
    if settings.async_scheduling is not False:
        raise ValueError("dynamic MTP requires --no-async-scheduling")
    if any(arg == "--enforce-eager" or arg.startswith(("-cc.", "--compilation-config."))
           for arg in rest):
        raise ValueError("use the single JSON --compilation-config option and enable graph execution")
    speculative = settings.speculative_config or []
    if len(speculative) != 1 or not isinstance(speculative[0], dict):
        raise ValueError("specify a single JSON --speculative-config")
    if speculative[0].get("method") != "mtp" or speculative[0].get("num_speculative_tokens") != 3:
        raise ValueError("use method=mtp and num_speculative_tokens=3")
    if speculative[0].get("num_speculative_tokens_per_batch_size") != BATCH_DEPTHS:
        raise ValueError("use the capture depth ranges shown in the README")
    compilation = settings.compilation_config or []
    if len(compilation) != 1 or not isinstance(compilation[0], dict):
        raise ValueError("specify a single JSON --compilation-config")
    if compilation[0].get("mode") != 0 or compilation[0].get("cudagraph_mode") != "FULL_DECODE_ONLY":
        raise ValueError("use mode=0 and cudagraph_mode=FULL_DECODE_ONLY")
    return command


def installed_sources():
    sources = {}
    for distribution, package in (("vllm", "vllm"), ("vllm-ascend", "vllm_ascend")):
        version = metadata.version(distribution)
        if version.partition("+")[0] != RELEASE:
            raise ValueError(f"{distribution} {RELEASE} is required, found {version}")
        spec = util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            raise ValueError(f"cannot locate the installed {package} source")
        sources[package] = Path(next(iter(spec.submodule_search_locations)))
    return sources


def prepare_overlay(state_dir, policy, sources, *, gdn_k2=False, gdn_prefill_grouping=False):
    patch_dir = HERE / "patches"
    modules = json.loads((patch_dir / "manifest.json").read_text())
    if gdn_k2:
        modules.update(json.loads((patch_dir / "gdn_k2" / "manifest.json").read_text()))
    if gdn_prefill_grouping:
        if not gdn_k2:
            raise ValueError("GDN prefill grouping requires --gdn-k2-indexing")
        modules.update(json.loads((patch_dir / "gdn_workspace" / "manifest.json").read_text()))
    state_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=state_dir) as tmp:
        working = Path(tmp)
        for module, record in modules.items():
            package, suffix = module.split(".", 1)
            source = sources[package] / (suffix.replace(".", "/") + ".py")
            if sha256(source.read_bytes()).hexdigest() != record["source_sha256"]:
                raise ValueError(f"installed module differs from the validated release source: {module}")
            output = working / f"{module}.py"
            subprocess.run(
                ["patch", "--batch", "--forward", "--fuzz=0", "--silent",
                 "--output", str(output), str(source), str(patch_dir / record["patch"])],
                check=True,
            )
            result = output.read_bytes()
            if sha256(result).hexdigest() != record["result_sha256"]:
                raise ValueError(f"patched module does not match the validated implementation: {module}")
            compile(result, str(output), "exec")
        (working / "modules.json").write_text(json.dumps(sorted(modules)))
        (working / "policy.json").write_text(json.dumps(policy))
        # Emit policy transitions only, avoiding per-step file I/O in the hot path.
        (working / "timing").touch()
        # Use an independent directory so two replicas can never replace one another's modules.
        overlay = Path(tempfile.mkdtemp(prefix="run-", dir=state_dir))
        for path in working.iterdir():
            path.rename(overlay / path.name)
    return overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=HERE / "policy.example.json")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument(
        "--gdn-k2-indexing", action="store_true",
        help="Opt in to the experimental shared GDN indices and exact K2 graph.",
    )
    parser.add_argument(
        "--gdn-prefill-grouping", action="store_true",
        help="Group large multi-sequence GDN prefills to reduce the workspace peak.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    try:
        command = options.command
        if command and command[0] == "--":
            command = command[1:]
        serving_args(command)
        exe = shutil.which("vllm")
        if not exe:
            raise ValueError("vllm must be on PATH in the same Python environment as this launcher")
        policy = validate_policy(options.policy)
        overlay = prepare_overlay(
            options.state_dir.resolve(), policy, installed_sources(), gdn_k2=options.gdn_k2_indexing,
            gdn_prefill_grouping=options.gdn_prefill_grouping,
        )
    except (ValueError, OSError, argparse.ArgumentError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    env = os.environ.copy()
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(HERE / "runtime") + (os.pathsep + previous if previous else "")
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    env["DYN_FULL_ROOT"] = str(overlay)
    env["DYN_FULL_POLICY"] = str(overlay / "policy.json")
    if options.gdn_k2_indexing:
        env["MTP_K2_TARGET_EXACT"] = "1"
        env["MTP_GDN_EXACT_K2"] = "1"
    print(f"Dynamic MTP overlay: {overlay}", flush=True)
    os.execve(exe, [exe, *command], env)


if __name__ == "__main__":
    main()
