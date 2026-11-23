"""Immutable agent packages and atomic per-profile deployment snapshots."""
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import os
import re
import shutil
from tempfile import mkdtemp
from uuid import uuid4

from bootstrap.cache_package import fingerprint, local_objects, read_json, verified_payload, sync_dir, write_atomic, write_json

NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z')
DIGEST = re.compile(r'[0-9a-f]{64}\Z')


def identifier(value):
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ValueError(f'invalid identifier: {value!r}')
    return value


def metadata(package):
    record = read_json(package / 'package.json')
    identifier(record['agent'])
    expected = version_id(record['profile_fingerprint'], record['prefix_sha256'], record.get('salt_sha256'))
    if (
        record['schema'] != 1 or record['version'] != expected
        or not DIGEST.fullmatch(record['profile_fingerprint'])
        or not DIGEST.fullmatch(record['prefix_sha256'])
        or record['external_prefix_hit_tokens'] <= 0
        or not record['output_tokens_equal']
    ):
        raise ValueError(f'unverified or invalid package metadata: {package}')
    return record


def check_package(package):
    record = metadata(package)
    probe = read_json(package / 'probe.json')
    if fingerprint(probe['token_ids']) != record['prefix_sha256'] or len(probe['token_ids']) != record['prefix_tokens']:
        raise ValueError(f'prefix tokens differ from package metadata: {package}')
    if salt_hash(probe.get('cache_salt')) != record.get('salt_sha256'):
        raise ValueError(f'cache salt differs from package metadata: {package}')
    if not isinstance(probe['output_ids'], list) or len(probe['output_ids']) != 16:
        raise ValueError(f'invalid saved verification output: {package}')
    layouts = read_json(package / 'layouts.json')
    if not layouts or fingerprint(layouts) != record['layout_fingerprint']:
        raise ValueError(f'resolved layouts differ from package metadata: {package}')
    manifest = package / 'objects.json'
    objects = local_objects(manifest)
    index = read_json(manifest)
    if fingerprint(index) != record['objects_fingerprint']:
        raise ValueError(f'object index differs from package metadata: {package}')
    if any('rank' in entry and entry['rank'] not in layouts for entry in index.values()):
        raise ValueError(f'object index refers to an unknown rank: {package}')
    for _, path, length, checksum in objects:
        verified_payload(path, length, checksum)
    return record


def copy_package(src, dest):
    check_package(src)
    dest = dest.resolve()
    if dest.exists():
        raise FileExistsError(dest)
    files = {item[1].name for item in local_objects(src/'objects.json')}
    files.update(('objects.json', 'layouts.json', 'package.json', 'probe.json'))
    tmp = Path(mkdtemp(prefix=f'.{dest.name}-', dir=dest.parent))
    try:
        for name in sorted(files):
            write_atomic(tmp/name, (src/name).read_bytes())
        check_package(tmp)
        tmp.rename(dest)
        sync_dir(dest.parent)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


class Registry:
    def __init__(self, root):
        self.root = root.resolve()
        self.versions = self.root / 'agents'
        self.builds = self.root / 'builds'
        self.active = self.root / 'active'
        self.deployments = self.root / 'deployments'
        for p in (self.versions, self.builds, self.active, self.deployments):
            p.mkdir(parents=True, exist_ok=True)

    def lock(self):
        lock = (self.root / '.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX)
        return lock

    def all(self, agent=None):
        versions = self.versions / identifier(agent) if agent else self.versions
        pattern = '*/package.json' if agent else '*/*/package.json'
        return sorted(((metadata(p.parent), p.parent) for p in versions.glob(pattern)),
                      key=lambda entry: entry[0]['created_utc'], reverse=True)

    def resolve(self, ref, fp=None):
        agent, _, version = ref.partition('@')
        identifier(agent)
        if version:
            identifier(version)
        for info, path in self.all(agent):
            if (not version or info['version'] == version) and (not fp or info['profile_fingerprint'] == fp):
                return path
        raise ValueError(f'no matching package: {ref}')

    def stage(self, agent):
        return Path(mkdtemp(prefix=identifier(agent) + '-', dir=self.builds))

    def publish(self, package):
        info = check_package(package)
        dest = self.versions / info['agent'] / info['version']
        dest.parent.mkdir(exist_ok=True)
        with self.lock():
            if dest.exists():
                check_package(dest)
                return dest
            package.rename(dest)
            sync_dir(dest.parent)
            sync_dir(self.versions)
        return dest

    def current(self, fp):
        link = self.active / fp
        return read_json(link / 'active.json') if link.is_dir() else {}

    def activate(self, ref, fp=None):
        with self.lock():
            chosen = self.resolve(ref, fp)
            info = check_package(chosen)
            fp = info['profile_fingerprint']
            agents = self.current(fp)
            agents[info['agent']] = info['version']
            dest = Path(mkdtemp(prefix=fp[:12]+'-', dir=self.deployments))
            try:
                layouts, objects = None, {}
                for agent, version in agents.items():
                    package = self.resolve(f'{agent}@{version}', fp)
                    check_package(package)
                    candidate = read_json(package / 'layouts.json')
                    if layouts is not None and candidate != layouts:
                        raise ValueError(f'agents have different resolved cache layouts: {agent}')
                    layouts = candidate
                    manifest = package/'objects.json'
                    entries = read_json(manifest)
                    for key, path, length, checksum in local_objects(manifest):
                        if key in objects:
                            if objects[key]['sha256'] != checksum:
                                raise ValueError(f'conflicting bytes for a shared Store key: {key}')
                            continue
                        name = checksum + '.bin'
                        output = dest / name
                        if not output.exists():
                            os.link(path, output)
                        objects[key] = {**entries[key], 'file': name}
                write_json(dest / 'objects.json', objects)
                write_json(dest / 'layouts.json', layouts)
                write_json(dest / 'active.json', agents)
                link = self.active / ('.' + uuid4().hex)
                link.symlink_to(os.path.relpath(dest, self.active), target_is_directory=True)
                try:
                    os.replace(link, self.active / fp)
                finally:
                    link.unlink(missing_ok=True)
            except BaseException:
                shutil.rmtree(dest)
                raise
            sync_dir(self.active)
        return self.active / fp

    def remove(self, ref):
        if '@' not in ref:
            raise ValueError('remove requires AGENT@VERSION')
        with self.lock():
            path = self.resolve(ref)
            info = metadata(path)
            current = self.current(info['profile_fingerprint'])
            if current.get(info['agent']) == info['version']:
                raise ValueError('switch away from the active version before removing it')
            shutil.rmtree(path)


def salt_hash(salt):
    return fingerprint(salt) if salt else None


def version_id(fp, prefix, salt_digest):
    fields = {'profile': fp, 'prefix': prefix}
    if salt_digest:
        fields['salt'] = salt_digest
    return fingerprint(fields)[:16]


def new_metadata(agent, fp, revision, config, tokens, layouts, objects, proof):
    prefix = fingerprint(tokens)
    salt = salt_hash(config.get('cache_salt'))
    return {
        'schema': 1, 'agent': identifier(agent),
        'version': version_id(fp, prefix, salt),
        'salt_sha256': salt,
        'profile_fingerprint': fp, 'model_revision': revision,
        'layout_fingerprint': fingerprint(layouts), 'objects_fingerprint': fingerprint(objects),
        'prefix_tokens': len(tokens), 'prefix_sha256': prefix,
        'agent_config_sha256': fingerprint(config), **proof,
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
