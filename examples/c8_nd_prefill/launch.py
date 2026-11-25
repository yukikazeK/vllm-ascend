#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the validated 256-dimension ND C8 TP1 backend for vLLM-Ascend 0.30.0."""

import argparse
from hashlib import sha256
from importlib import metadata, util
import json
import os
from pathlib import Path
import shutil
import tempfile

HERE = Path(__file__).resolve().parent
SOURCE_SHA256 = {
    "attention/attention_v1.py": "bea4e410f797557e29cedbb495024147606e74b0e056d725194760ca69296eba",
    "quantization/methods/kv_cache/kv_c8.py": "1e85e37adbf497f934642ec80823cef566010806f3a930b53a1ef914e561b78f",
}


def validate_command(command):
    if len(command) < 2 or command[0] != "serve" or command[1].startswith("-"):
        raise ValueError("pass vLLM arguments as: -- serve MODEL [options]")
    model = Path(command[1]).resolve()
    if not (model / "quant_model_weights-kv-cache-c8.safetensors").is_file():
        raise ValueError("MODEL must contain the calibrated C8 scale safetensors")
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    for flag in ("tensor-parallel-size", "data-parallel-size", "pipeline-parallel-size",
                 "prefill-context-parallel-size", "decode-context-parallel-size"):
        parser.add_argument("--" + flag, type=int, default=1)
    parser.add_argument("--dtype")
    parser.add_argument("--quantization")
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--speculative-config")
    parser.add_argument("--compilation-config", "-cc", type=json.loads)
    parser.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction)
    settings, rest = parser.parse_known_args(command[2:])
    if any(v != 1 for key, v in vars(settings).items() if key.endswith("parallel_size")):
        raise ValueError("the validated C8 backend requires a single TP1 replica")
    if (settings.dtype, settings.quantization, settings.kv_cache_dtype) != ("bfloat16", "ascend", "int8"):
        raise ValueError("use --dtype bfloat16 --quantization ascend --kv-cache-dtype int8")
    if settings.speculative_config is not None:
        raise ValueError("the validated C8 ND backend handles MTP0")
    if settings.async_scheduling is not False or settings.enable_prefix_caching is not False:
        raise ValueError("use --no-async-scheduling --no-enable-prefix-caching for this experiment")
    cfg = settings.compilation_config
    if not isinstance(cfg, dict) or cfg.get("mode") != 0 or cfg.get("cudagraph_mode") != "FULL_DECODE_ONLY":
        raise ValueError("use a JSON --compilation-config with mode=0 and FULL_DECODE_ONLY")
    if "--enforce-eager" in rest or any(s.startswith(("-cc.", "--compilation-config.")) for s in rest):
        raise ValueError("use the single JSON graph configuration option")
    return model


def validate_release():
    for package in ("vllm", "vllm-ascend"):
        if metadata.version(package).partition("+")[0] != "0.30.0":
            raise ValueError(f"{package} 0.30.0 is required")
    spec = util.find_spec("vllm_ascend")
    if spec is None or not spec.submodule_search_locations:
        raise ValueError("cannot find the installed vllm_ascend package")
    source = Path(next(iter(spec.submodule_search_locations)))
    for filename, digest in SOURCE_SHA256.items():
        if sha256((source / filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"installed source differs from validated 0.30.0: {filename}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--fused-gather", action="store_true",
                        help="Enable the fused paged INT8 gather and BF16 dequantization kernel.")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    opts = parser.parse_args()
    command = opts.command[1:] if opts.command[:1] == ["--"] else opts.command
    try:
        model = validate_command(command)
        validate_release()
        exe = shutil.which("vllm")
        if not exe:
            raise ValueError("vllm must be on PATH within this Python environment")
        opts.state_dir.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix="c8-", dir=opts.state_dir.resolve()))
    except (ValueError, OSError, argparse.ArgumentError, metadata.PackageNotFoundError) as error:
        parser.error(str(error))
    env = os.environ.copy()
    runtime = str(HERE / "runtime")
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = runtime + (os.pathsep + previous if previous else "")
    env["C8_CHECKPOINT"] = str(model)
    env["C8_AUDIT_RUN"] = str(run)
    env["C8_ND_RUN"] = "1"
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    env.pop("C8_PREFILL_FUSED_GATHER", None)
    if opts.fused_gather:
        env["C8_PREFILL_FUSED_GATHER"] = "1"
    print(f"C8 validation records: {run}", flush=True)
    os.execve(exe, [exe, *command], env)


if __name__ == "__main__":
    main()
