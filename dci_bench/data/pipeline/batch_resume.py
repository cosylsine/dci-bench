"""Read-only reuse of completed v1 phases in a separate, frozen batch run.

The old run is never mutated.  Incomplete export/pool directories are ignored;
only completed embedding shards may be copied to the new cache.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import uuid

from . import batch


def _origin(entry, plan, source_root):
    reuse = plan.get('reuse')
    if not reuse:
        return None
    bundle = Path(reuse['bundle'])
    old = batch.load_plan(bundle)
    if batch.sha(bundle / 'plan.json') != reuse['plan_sha256']:
        raise ValueError('Reuse plan identity changed')
    root = Path(reuse['run_root'])
    if root != Path(old['run_root']):
        raise ValueError('Reuse run does not match the frozen parent plan')
    if batch.read(root / 'run.json') != {'bundle': str(bundle), 'plan_sha256': reuse['plan_sha256']}:
        raise ValueError('Reuse run identity mismatch')
    old_entry = next((e for e in old['sources'] if e['source_id'] == entry['source_id']), None)
    if old_entry != entry or not batch.compatible_parameters(old['parameters'], plan['parameters']):
        raise ValueError('Cannot reuse different source content or model/pool parameters')
    source = root / 'sources' / entry['source_id']
    new_root = Path(source_root).parent.parent.resolve()
    old_root = root.resolve()
    if new_root == old_root or new_root in old_root.parents or old_root in new_root.parents:
        raise ValueError('Reuse requires an independent new run directory')
    return source


def _binding(entry, plan):
    return {'entry_hash': batch.digest(entry), 'parameters_hash': batch.digest(plan['parameters'])}


def _regular(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f'Reuse artifact must be an independent regular file: {path}')
    return info


def _parent_binding(entry, plan):
    old_plan = batch.read(Path(plan['reuse']['bundle']) / 'plan.json')
    return _binding(entry, old_plan)


def _parent_pools(entry, plan, source):
    """Resolve inherited pool locations, including v2 references into v1."""
    targets = {str(capacity): source / 'pools' / str(capacity)
               for capacity in plan['parameters']['capacities']}
    for receipt in (source / 'reuse.json', source / 'phases/pools.json', source / '_COMPLETE.json'):
        if not receipt.is_file():
            continue
        value = batch.read(receipt)
        if any(value.get(k) != v for k, v in _parent_binding(entry, plan).items()):
            raise ValueError('Parent pool receipt binding differs')
        rows = value.get('pools', {})
        rows = rows.items() if isinstance(rows, dict) else ((str(row['capacity']), row) for row in rows)
        for capacity, row in rows:
            if capacity not in targets:
                raise ValueError('Parent pool receipt capacity differs')
            targets[capacity] = Path(row['path'])
    return targets


def _snapshot(entry, plan, source):
    result = {'schema_version': 'dci-kalm-pool-reuse-v1', 'source_id': entry['source_id'],
              **_binding(entry, plan), 'reuse_plan_hash': batch.digest(plan['reuse']),
              'origin': str(source), 'phases': {}, 'pools': {}, 'cache_files': {}}
    if not source.exists():
        return result
    bound = source / 'binding.json'
    if not bound.exists():
        # An unstarted source may only have its directory/lock so far.
        if any((source / 'phases').glob('*.json')) or (source / '_COMPLETE.json').exists():
            raise ValueError('Reuse source has receipts without a source binding')
        return result
    if batch.read(bound) != _parent_binding(entry, plan):
        raise ValueError('Reuse source binding differs')
    for name in ('export', 'rank'):
        path = source / 'phases' / (name + '.json')
        if not path.exists():
            continue
        _regular(path)
        value = batch.read(path)
        if name == 'export':
            marker = Path(value['suite']) / 'metadata' / entry['task'] / '_COMPLETE.json'
            _regular(marker)
            if batch.sha(marker) != value['completion_sha256']:
                raise ValueError('Reuse export completion changed')
        else:
            if 'export' not in result['phases']:
                raise ValueError('Reuse ranking has no successful export')
            _regular(Path(value['path']))
            if not batch._ranking_receipt_valid(value, plan):
                raise ValueError('Reuse ranking changed')
        result['phases'][name] = value
    if 'rank' in result['phases']:
        for capacity, pool in _parent_pools(entry, plan, source).items():
            marker = pool / 'metadata' / entry['task'] / '_COMPLETE.json'
            if marker.is_file():
                _regular(marker)
                result['pools'][capacity] = {
                    'path': str(pool), 'completion_sha256': batch.sha(marker)}
    else:
        cache = source / 'embedding_cache'
        parent_parameters = batch.read(Path(plan['reuse']['bundle']) / 'plan.json')['parameters']
        cache_compatible = parent_parameters.get('batch_size') == plan['parameters'].get('batch_size')
        if not cache_compatible:
            result['cache_reuse_skipped'] = 'batch_size_changed; original cache identity retained'
        if cache.exists() and cache_compatible:
            for path in sorted(cache.rglob('*.npz')):
                info = _regular(path)
                if path.resolve() != path.absolute():
                    raise ValueError(f'Reuse cache must not traverse symlinks: {path}')
                result['cache_files'][str(path.relative_to(cache))] = {
                    'bytes': info.st_size,
                    'sha256': batch.sha(path) if batch.content_hashes_enabled(plan) else None}
    result['content_hashes_verified'] = batch.content_hashes_enabled(plan)
    result['parent_parameters_hash'] = _parent_binding(entry, plan)['parameters_hash']
    return result


def _check_snapshot(snapshot, entry, plan):
    expected = {'schema_version': 'dci-kalm-pool-reuse-v1', 'source_id': entry['source_id'],
                **_binding(entry, plan), 'reuse_plan_hash': batch.digest(plan['reuse'])}
    if any(snapshot.get(k) != value for k, value in expected.items()):
        raise ValueError('Saved reuse snapshot differs from the new plan')
    origin = Path(plan['reuse']['run_root']) / 'sources' / entry['source_id']
    if snapshot.get('origin') != str(origin):
        raise ValueError('Saved reuse origin differs')
    if set(snapshot['phases']) - {'export', 'rank'}:
        raise ValueError('Invalid reused phase')
    for name, value in snapshot['phases'].items():
        original = origin / 'phases' / (name + '.json')
        _regular(original)
        if batch.read(original) != value:
            raise ValueError('Saved reuse phase differs from its immutable origin')
    if set(snapshot['pools']) - {str(c) for c in plan['parameters']['capacities']}:
        raise ValueError('Invalid reused pool capacity')
    for capacity, row in snapshot['pools'].items():
        pool = _parent_pools(entry, plan, origin)[capacity]
        if row['path'] != str(pool):
            raise ValueError('Saved reuse pool path differs from its origin')
        marker = pool / 'metadata' / entry['task'] / '_COMPLETE.json'
        _regular(marker)
        if batch.sha(marker) != row['completion_sha256']:
            raise ValueError('Reused pool completion changed')


def import_source_receipts(entry, plan, source_root):
    """Pin then import successful phases; the caller must hold the new source lock.

    Paths in receipts keep pointing to the original immutable export/ranking.
    Full corpus and pool verification remains the phase worker's responsibility.
    """
    if not plan.get('reuse'):
        return
    source_root = Path(source_root)
    source = _origin(entry, plan, source_root)
    path = source_root / 'reuse.json'
    if path.exists():
        snapshot = batch.read(path)
        _check_snapshot(snapshot, entry, plan)
    else:
        snapshot = _snapshot(entry, plan, source)
        # Pin the set before importing anything, so an interrupted import is replayable.
        batch.write(path, snapshot)
    for name, value in snapshot['phases'].items():
        target = source_root / 'phases' / (name + '.json')
        if target.exists():
            if batch.read(target) != value:
                raise ValueError(f'Imported {name} receipt differs')
        else:
            batch.write(target, value)
    for relative, info in snapshot['cache_files'].items():
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Invalid cache shard path in reuse snapshot')
        source_file = source / 'embedding_cache' / rel
        target = source_root / 'embedding_cache' / rel
        if target.exists():
            _regular(target)
            if (target.stat().st_size != info['bytes']
                    or (batch.content_hashes_enabled(plan) and batch.sha(target) != info['sha256'])):
                raise ValueError('Imported cache shard differs')
            continue
        source_info = _regular(source_file)
        if source_info.st_size != info['bytes']:
            raise ValueError('Source cache shard changed')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name('.' + target.name + '.' + uuid.uuid4().hex + '.tmp')
        # The source is read-only; the new cache receives independent regular files.
        with source_file.open('rb') as reader, temporary.open('xb') as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        if (temporary.stat().st_size != info['bytes']
                or (batch.content_hashes_enabled(plan) and batch.sha(temporary) != info['sha256'])):
            raise ValueError('Copied cache shard hash mismatch; partial copy preserved')
        os.replace(temporary, target)


def pool_target(entry, plan, source_root, capacity):
    """Prefer a local published pool, otherwise a pinned completed parent pool."""
    source_root = Path(source_root)
    target = source_root / 'pools' / str(capacity)
    if target.exists() or not plan.get('reuse'):
        return target
    path = source_root / 'reuse.json'
    if not path.exists():
        return target
    snapshot = batch.read(path)
    _check_snapshot(snapshot, entry, plan)
    row = snapshot['pools'].get(str(capacity))
    if row is None:
        return target
    # A different export/ranking can never inherit a previous pool.
    for name in ('export', 'rank'):
        if batch.read(source_root / 'phases' / (name + '.json')) != snapshot['phases'].get(name):
            raise ValueError('Reused pool no longer matches source phase receipts')
    pool = Path(row['path'])
    expected = _parent_pools(entry, plan, Path(snapshot['origin']))[str(capacity)]
    if pool != expected:
        raise ValueError('Reused pool path differs from its origin')
    marker = pool / 'metadata' / entry['task'] / '_COMPLETE.json'
    _regular(marker)
    if batch.sha(marker) != row['completion_sha256']:
        raise ValueError('Reused pool completion changed')
    return pool
