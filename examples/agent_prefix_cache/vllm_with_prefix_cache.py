#!/usr/bin/env python3
"""Start vLLM with an optional local AscendStore prefix package."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

BOOTSTRAP = Path(__file__).resolve().parent / 'bootstrap'
MANIFEST_ENV = 'VLLM_LOCAL_PREFIX_CACHE_MANIFEST'


def configured_connector(args, parser):
    positions = [i for i,arg in enumerate(args)
                 if arg == '--kv-transfer-config' or arg.startswith('--kv-transfer-config=')]
    if len(positions) > 1:
        parser.error('specify --kv-transfer-config once')
    if positions:
        i = positions[0]
        flag, separator, raw = args[i].partition('=')
        if not separator:
            if i+1==len(args):
                parser.error(f'{flag} requires a JSON value')
            raw = args[i+1]
        try:
            config = json.loads(raw)
            extra = config.get('kv_connector_extra_config', {})
            supported = (
                config.get('kv_connector') == 'AscendStoreConnector'
                and config.get('kv_role') == 'kv_both'
                and extra.get('backend', 'mooncake') == 'mooncake'
                and not extra.get('use_layerwise', False)
                and not extra.get('load_async', False)
            )
        except (AttributeError, ValueError):
            parser.error('--kv-transfer-config must be a JSON object')
        if not supported:
            parser.error('local prefix loading requires the synchronous, whole-key AscendStoreConnector Mooncake backend with kv_role=kv_both')
        return
    config = {
        'kv_connector':'AscendStoreConnector',
        'kv_role':'kv_both',
        'kv_connector_extra_config':{'backend':'mooncake','use_layerwise':False,'load_async':False},
    }
    args.extend(['--kv-transfer-config',json.dumps(config)])


def main():
    parser = argparse.ArgumentParser(
        prog='vllm_with_prefix_cache',add_help=False,allow_abbrev=False,
        description='Usage: vllm_with_prefix_cache.py serve MODEL [vLLM flags] [local cache flags]',
    )
    parser.add_argument('--load-local-prefix-cache',action=argparse.BooleanOptionalAction,default=False,
                        help='Import a local prefix package before vLLM accepts requests. Default: off.')
    parser.add_argument('--local-prefix-cache-path',type=Path,metavar='PATH',
                        help='An objects.json manifest or the directory containing it.')
    parser.add_argument('--prefix-cache-help',action='store_true',help='Show these optional flags.')
    args, command = parser.parse_known_args()
    if args.prefix_cache_help:
        parser.print_help()
        return
    exe = shutil.which('vllm')
    if exe is None:
        parser.error('vllm is not on PATH')
    env = os.environ.copy()
    env.pop(MANIFEST_ENV,None)
    if args.load_local_prefix_cache:
        if not command or command[0] != 'serve':
            parser.error('--load-local-prefix-cache is supported with vllm serve')
        if args.local_prefix_cache_path is None:
            parser.error('--load-local-prefix-cache requires --local-prefix-cache-path')
        if '--no-enable-prefix-caching' in command:
            parser.error('--load-local-prefix-cache requires prefix caching')
        manifest = args.local_prefix_cache_path.expanduser()
        if manifest.is_dir():
            manifest /= 'objects.json'
        try:
            sys.path.insert(0,str(BOOTSTRAP))
            from local_prefix_loader import local_objects
            manifest = manifest.resolve(strict=True)
            local_objects(manifest)
            config = Path(env['MOONCAKE_CONFIG_PATH']).resolve(strict=True)
            if not config.is_file():
                raise ValueError(f'Mooncake config is not a regular file: {config}')
        except KeyError:
            parser.error('set MOONCAKE_CONFIG_PATH to the configuration for a running Mooncake Store')
        except (OSError,ValueError) as exc:
            parser.error(str(exc))
        configured_connector(command,parser)
        env[MANIFEST_ENV] = str(manifest)
        env['MOONCAKE_CONFIG_PATH'] = str(config)
        existing = env.get('PYTHONPATH')
        env['PYTHONPATH'] = str(BOOTSTRAP) + (os.pathsep+existing if existing else '')
    os.execve(exe,[exe,*command],env)


if __name__ == '__main__':
    main()
