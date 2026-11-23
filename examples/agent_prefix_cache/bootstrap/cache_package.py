"""On-disk object format shared by the startup loader and the management CLI."""
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_atomic(path, payload):
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
        sync_dir(path.parent)
    finally:
        Path(tmp).unlink(missing_ok=True)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    write_atomic(path, (json.dumps(value, sort_keys=True, indent=2) + '\n').encode())


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def local_objects(manifest):
    records = read_json(manifest)
    if not isinstance(records, dict) or not records:
        raise ValueError(f'empty or invalid cache manifest: {manifest}')
    root = manifest.parent.resolve()
    objects = []
    for key, rec in records.items():
        if not key or not isinstance(rec, dict):
            raise ValueError('invalid cache key or record')
        name, length, checksum = (rec.get(f) for f in ('file', 'bytes', 'sha256'))
        if (
            not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..')
            or type(length) is not int or length <= 0
            or not isinstance(checksum, str) or not re.fullmatch(r'[0-9a-f]{64}', checksum)
        ):
            raise ValueError(f'invalid file, size or checksum for cache key {key}')
        path = (root / name).resolve(strict=True)
        if path.parent != root or not path.is_file() or path.stat().st_size != length:
            raise ValueError(f'cache file outside package or unexpected size: {path}')
        objects.append((key, path, length, checksum))
    return objects


def verified_payload(path, length, checksum):
    payload = path.read_bytes()
    if len(payload) != length or sha256(payload).hexdigest() != checksum:
        raise ValueError(f'cache file failed checksum verification: {path}')
    return payload
