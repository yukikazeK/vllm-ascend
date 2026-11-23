"""Build and verify packages using isolated vLLM and Mooncake processes."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from bootstrap.cache_package import fingerprint, local_objects, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'vllm_with_prefix_cache.py'
LOOPBACK = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def profile(path):
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError('profile must be a JSON object')
    for field in ('name', 'model', 'model_revision'):
        if not isinstance(data.get(field), str) or not data[field]:
            raise ValueError(f'profile needs a nonempty {field}')
    args = data.get('serve_args')
    if not isinstance(args, list) or not args or not all(isinstance(x, str) for x in args) or args[0].startswith('-'):
        raise ValueError('serve_args must contain a model path followed by vLLM options')
    seed = data.get('prefix_hash_seed')
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError('prefix_hash_seed must be an integer between 0 and 2^32-1')
    return data


def profile_id(data):
    args = list(data['serve_args'])
    for name in ('--port', '--host'):
        args = replace_option(args, name, '')[:-2]
    identity = {'model': data['model'], 'model_revision': data['model_revision'], 'serve_args': args}
    if data.get('prefix_hash_seed') is not None:
        identity['prefix_hash_seed'] = data['prefix_hash_seed']
    return fingerprint(identity)


def profile_env(data):
    env = os.environ.copy()
    if data.get('prefix_hash_seed') is not None:
        env['PYTHONHASHSEED'] = str(data['prefix_hash_seed'])
    return env


def bootstrap_env(env):
    previous = env.get('PYTHONPATH')
    env['PYTHONPATH'] = str(ROOT / 'bootstrap') + (os.pathsep + previous if previous else '')
    return env


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def replace_option(args, key, val):
    result = []
    pos = 0
    while pos < len(args):
        arg = args[pos]
        pos += 1
        if arg == key:
            if pos >= len(args):
                raise ValueError(f'{key} requires a value')
            pos += 1
        elif not arg.startswith(key+'='):
            result.append(arg)
    return [*result, key, str(val)]


@contextmanager
def process(command, logfile, env=None):
    with logfile.open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            yield child
        finally:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()


def wait_until(ready, process_, logfile, timeout):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if process_.poll() is not None:
            raise RuntimeError(f'process exited with status {process_.returncode}; see {logfile}')
        if ready():
            return
        time.sleep(.4)
    raise TimeoutError(f'process did not become ready within {timeout}s; see {logfile}')


def query(port, route, data=None, timeout=15):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{route}',
        data=json.dumps(data).encode() if data is not None else None,
        headers={'Content-Type': 'application/json'})
    with LOOPBACK.open(req, timeout=timeout) as result:
        return result.read()


def is_ready(port):
    try:
        query(port, '/health', timeout=2)
    except OSError:
        return False
    return True


def accepts_tcp(port):
    with socket.socket() as sock:
        return sock.connect_ex(('127.0.0.1', port)) == 0


@contextmanager
def private_engine(cfg, run, mode, timeout, package):
    moon_config = os.environ.get('MOONCAKE_CONFIG_PATH')
    if not moon_config:
        raise ValueError('set MOONCAKE_CONFIG_PATH to a Mooncake configuration template')
    master = shutil.which('mooncake_master')
    if master is None:
        raise ValueError('mooncake_master is not on PATH')
    master_port, port = free_port(), free_port()
    for _ in range(5):
        if port != master_port:
            break
        port = free_port()
    else:
        raise RuntimeError('could not allocate distinct test ports')
    config = read_json(Path(moon_config))
    config.update(master_server_address=f'127.0.0.1:{master_port}', enable_ssd_offload=False)
    write_json(run / f'{mode}-mooncake.json', config)
    env = profile_env(cfg)
    for name in ('VLLM_LOCAL_PREFIX_CACHE_EXPORT', 'VLLM_LOCAL_PREFIX_CACHE_MANIFEST'):
        env.pop(name, None)
    env['MOONCAKE_CONFIG_PATH'] = str(run / f'{mode}-mooncake.json')
    args = replace_option(cfg['serve_args'], '--port', port)
    args = replace_option(args, '--host', '127.0.0.1')
    # Configure exactly the backend that supports the export/import wire format.
    from vllm_with_prefix_cache import configured_connector
    import argparse
    configured_connector(args, argparse.ArgumentParser(prog='agent-cache'))
    command = [sys.executable, str(LAUNCHER), 'serve', *args]
    if mode == 'build':
        env['VLLM_LOCAL_PREFIX_CACHE_EXPORT'] = str(package)
        bootstrap_env(env)
    else:
        command += ['--load-local-prefix-cache', '--local-prefix-cache-path', str(package)]
    master_log = run / f'{mode}-master.log'
    server_log = run / f'{mode}-vllm.log'
    with process([master, f'--rpc_port={master_port}', '--rpc_address=127.0.0.1',
                  '--metrics_host=127.0.0.1', f'--metrics_port={free_port()}'], master_log) as store:
        wait_until(lambda: accepts_tcp(master_port), store, master_log, 30)
        with process(command, server_log, env) as server:
            wait_until(lambda: is_ready(port), server, server_log, timeout)
            yield port


def prompt_tokens(config, cfg, port):
    if 'token_ids' in config:
        result = config['token_ids']
    else:
        if not isinstance(config.get('messages'), list) or not config['messages']:
            raise ValueError('agent config requires token_ids or messages')
        allowed = {'messages', 'tools', 'chat_template_kwargs', 'add_generation_prompt'}
        request = {key: val for key, val in config.items() if key in allowed}
        request.setdefault('add_generation_prompt', False)
        request['model'] = cfg['model']
        result = json.loads(query(port, '/tokenize', request, timeout=60))['tokens']
    if not isinstance(result, list) or not result or any(type(t) is not int or t < 0 for t in result):
        raise ValueError('token_ids must be a nonempty array of nonnegative integers')
    return result


def run_request(cfg, port, tokens, salt=None):
    data = {'model': cfg['model'], 'prompt': tokens, 'max_tokens': 16,
            'ignore_eos': True, 'temperature': 0, 'return_token_ids': True}
    if salt is not None:
        data['cache_salt'] = salt
    res = json.loads(query(port, '/v1/completions', data, timeout=180))
    actual = res['choices'][0].get('token_ids')
    usage = res['usage']
    if len(actual or []) != 16 or usage['prompt_tokens'] != len(tokens) or usage['completion_tokens'] != 16:
        raise RuntimeError(f'unexpected tokenization or generation result: {usage}')
    return actual


def count_hits(port):
    metrics = query(port, '/metrics').decode()
    numbers = re.findall(r'^vllm:external_prefix_cache_hits_total(?:\{[^}]*\})?\s+([\d.eE+-]+)$', metrics, re.M)
    return int(sum(map(float, numbers)))


def await_export(package):
    index, error = package / 'objects.json', package / 'export-error.json'
    # Writes from multiple cache groups may finish after the response. A stable
    # manifest is followed by an actual fresh-engine cache-hit check.
    until = time.monotonic() + 45
    while time.monotonic() < until:
        if error.exists():
            raise RuntimeError(read_json(error)['error'])
        if index.exists() and time.time() - index.stat().st_mtime >= 2:
            local_objects(index)
            return
        time.sleep(.25)
    raise TimeoutError('no stable exported prefix manifest; check the backend and cacheable prefix length')


def finish_layout(package):
    paths = sorted(package.glob('layout-*.json'))
    resolved = {p.stem.removeprefix('layout-'): read_json(p) for p in paths}
    if not resolved:
        raise RuntimeError('the engine did not publish its cache layout; see the build log')
    first = next(iter(resolved.values()))
    rank_keys = {f'{pp}-{pcp}-{tp}' for pp in range(first['pp'])
                 for pcp in range(first['pcp']) for tp in range(first['tp'])}
    if resolved.keys() != rank_keys:
        raise RuntimeError(f'cache layouts missing ranks: {sorted(rank_keys - resolved.keys())}')
    write_json(package / 'layouts.json', resolved)
    for p in paths:
        p.unlink()


def cold_restore(cfg, run, package, tokens, expected, minimum, timeout, salt=None):
    with private_engine(cfg, run, 'restore', timeout, package) as port:
        before = count_hits(port)
        actual = run_request(cfg, port, tokens, salt)
        until = time.monotonic() + 5
        hits = count_hits(port) - before
        while hits < minimum and time.monotonic() < until:
            time.sleep(.25)
            hits = count_hits(port) - before
    proof = {'external_prefix_hit_tokens': hits, 'output_tokens_equal': actual == expected}
    write_json(run / 'restore-proof.json', proof)
    if hits < minimum or actual != expected:
        raise RuntimeError(f'cold restore failed: {proof}; expected at least {minimum} external hits and equal tokens')
    return proof
