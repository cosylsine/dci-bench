#!/usr/bin/env python3
"""Create a fresh, byte-identical local copy of complete pilot corpora.

Only the workspace location changes. Catalogs, metadata, canonical data and
document names/bytes stay frozen; labels remain outside tool-visible workspaces.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def copy_file(source, destination, expected=None):
    info = source.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f'Staging requires an independent regular file: {source}')
    if expected and info.st_size != expected['bytes']:
        raise ValueError(f'Source size changed: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with source.open('rb') as src, destination.open('xb') as dst:
        for chunk in iter(lambda: src.read(4 * 1024**2), b''):
            h.update(chunk)
            dst.write(chunk)
    expected_hash = expected['sha256'] if expected else h.hexdigest()
    if h.hexdigest() != expected_hash or digest(destination) != expected_hash:
        raise ValueError(f'Staging hash mismatch: {source}')
    destination.chmod(0o444)


def local_filesystem(path):
    """Reject remote mounts even when they are bind-mounted under /tmp."""
    kind = subprocess.check_output(['stat', '-f', '-c', '%T', str(path)], text=True).strip()
    if kind not in {'ext2/ext3', 'xfs', 'btrfs', 'overlayfs', 'tmpfs', 'zfs'}:
        raise ValueError(f'Local corpus root must be node-local storage, got {kind}: {path}')
    return kind


def stage(inventory, destination, *, workers=16):
    """Copy selected full corpora; never reuse, alter, or delete existing data."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Refusing to reuse local corpus directory: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    filesystem = local_filesystem(destination.parent)
    suites = sorted({row['suite'] for row in inventory.values()})
    targets = {suite: destination / f'suite-{i}' for i, suite in enumerate(suites)}
    needed = 0
    file_count = 0
    for row in inventory.values():
        with (Path(row['metadata']) / 'corpus_files.jsonl').open() as handle:
            for line in handle:
                size = json.loads(line)['bytes']
                needed += ((size + 4095) // 4096) * 4096
                file_count += 1
        needed += sum(p.stat().st_size for p in Path(row['canonical']).rglob('*') if p.is_file())
    # Reserve room for metadata, inode storage and filesystem overhead.
    required = int(needed * 1.25) + 2 * 1024**3
    if shutil.disk_usage(destination.parent).free < required:
        raise ValueError(f'Local staging needs at least {required / 1024**3:.1f} GiB free')
    fs = os.statvfs(destination.parent)
    if fs.f_files and fs.f_favail < file_count + 10000:
        raise ValueError(f'Insufficient local inodes for {file_count} corpus files')
    destination.mkdir()
    started = time.monotonic()
    for source, target in targets.items():
        source = Path(source)
        completion = json.loads((source / '_COMPLETE.json').read_text())
        for name in ['catalog.json', '_COMPLETE.json', *completion.get('audit_hashes', {})]:
            if Path(name).name != name:
                raise ValueError(f'Invalid suite attachment: {name}')
            copy_file(source / name, target / name)
            expected = completion.get('audit_hashes', {}).get(name)
            if expected and digest(target / name) != expected:
                raise ValueError(f'Suite audit changed: {name}')
        if digest(target / 'catalog.json') != completion['catalog_sha256']:
            raise ValueError('Suite catalog hash mismatch')
        # The frozen registrar checks completion records for ALL catalog tasks.
        # Copy host-only metadata for those tasks, but no additional corpora.
        for path in sorted((source / 'metadata').rglob('*')):
            if path.is_file() or path.is_symlink():
                copy_file(path, target / path.relative_to(source))
    relocated = {}
    records = {}
    for task, row in inventory.items():
        start = time.monotonic()
        target = targets[row['suite']]
        src_corpus = Path(row['workspace']) / 'corpus'
        dst_corpus = target / 'workspaces' / task / 'corpus'
        if src_corpus.is_symlink() or not src_corpus.is_dir():
            raise ValueError(f'Corpus must be a real directory: {src_corpus}')
        dst_corpus.mkdir(parents=True)
        copy_file(Path(row['workspace']) / 'README.md', dst_corpus.parent / 'README.md')
        for path in sorted(Path(row['canonical']).rglob('*')):
            if path.is_file() or path.is_symlink():
                copy_file(path, target / 'canonical' / task / path.relative_to(row['canonical']))
        seen = set()
        pending = deque()
        total_bytes = 0
        # Read the copied integrity index, then verify both copied bytes and closure.
        with ThreadPoolExecutor(max_workers=workers) as pool, \
             (target / 'metadata' / task / 'corpus_files.jsonl').open() as handle:
            for line in handle:
                entry = json.loads(line)
                name = entry['filename']
                if (name in seen or Path(name).name != name or not name.endswith('.md')
                        or '/' in name or '\\' in name):
                    raise ValueError(f'Invalid/duplicate corpus filename: {name}')
                seen.add(name)
                total_bytes += entry['bytes']
                pending.append(pool.submit(copy_file, src_corpus / name, dst_corpus / name, entry))
                if len(pending) >= workers * 4:
                    pending.popleft().result()
                if len(seen) % 50000 == 0:
                    print(f'[pilot] staging {task}: {len(seen)}/{row["num_docs"]} files submitted', flush=True)
            for future in pending:
                future.result()
        if (len(seen) != row['num_docs'] or seen != set(os.listdir(src_corpus))
                or seen != set(os.listdir(dst_corpus))):
            raise ValueError(f'Full corpus closure changed: {task}')
        relocated[task] = dict(row, suite=str(target), metadata=str(target / 'metadata' / task),
                              canonical=str(target / 'canonical' / task), workspace=str(dst_corpus.parent))
        records[task] = dict(source_workspace=row['workspace'], local_workspace=str(dst_corpus.parent),
            documents=len(seen), bytes=total_bytes, elapsed_seconds=time.monotonic() - start,
            corpus_index_sha256=digest(target / 'metadata' / task / 'corpus_files.jsonl'),
            catalog_sha256=digest(target / 'catalog.json'),
            corpus_hash=json.loads((target / 'metadata' / task / 'manifest.json').read_text())['corpus_hash'],
            metadata_completion_sha256=digest(target / 'metadata' / task / '_COMPLETE.json'))
        print(f'[pilot] staged {task}: {len(seen)} documents, {records[task]["elapsed_seconds"]:.2f}s', flush=True)
    record = dict(schema_version='dci-local-corpus-stage-v1', filesystem=filesystem,
                  destination=str(destination), tasks=records, elapsed_seconds=time.monotonic() - started,
                  policy='Full independent byte-verified copies; unchanged frozen metadata; no labels in workspace')
    with (destination / '_STAGED.json').open('x') as handle:
        json.dump(record, handle, indent=2)
        handle.write('\n')
    return relocated, record
