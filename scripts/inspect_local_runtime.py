#!/usr/bin/env python3
"""Provision a private, complete Inspect tools bundle before parallel runners.

Inspect 0.3.260 injects into a fixed host path even for separate local sandboxes.
Only its installation path constants are relocated in memory. No installed
package, frozen solver, server-state directory or protocol is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
import runpy
import sys
import tarfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(destination):
    import inspect_ai
    from inspect_ai.tool._sandbox_tools_utils.sandbox import _get_executable_name

    if inspect_ai.__version__ != '0.3.260':
        raise ValueError('Inspect installation-path adapter requires reviewed version 0.3.260')
    archive = Path(inspect_ai.__file__).parent / 'binaries' / _get_executable_name('amd64', False, False)
    return extract_bundle(archive, Path(destination))


def extract_bundle(archive, destination):
    """Publish only a complete verified bundle; never reuse any old installation."""
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not (member.isdir() or member.isfile()):
                raise ValueError(f'Unsafe tools archive member: {member.name}')
            if str(name) in files:
                raise ValueError(f'Duplicate tools archive member: {name}')
            target = destination / str(name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f'Archive file unavailable: {name}')
            data = stream.read()
            with target.open('xb') as handle:
                handle.write(data)
            target.chmod(0o555 if member.mode & 0o111 else 0o444)
            digest = hashlib.sha256(data).hexdigest()
            if sha(target) != digest:
                raise ValueError(f'Extracted tools bytes differ: {name}')
            files[str(name)] = digest
    if 'inspect-sandbox-tools' not in files:
        raise ValueError('Missing sandbox tools launcher')
    record = dict(schema_version='dci-inspect-local-tools-v1', inspect_version='0.3.260',
                  source_archive=str(archive), archive_sha256=sha(archive),
                  directory=str(destination.resolve()), files=files)
    with (destination / '_READY.json').open('x') as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write('\n')
    for path in destination.rglob('*'):
        if path.is_dir():
            path.chmod(0o555)
    destination.chmod(0o555)
    return record


def activate(directory):
    """Redirect existing aliases and future imports to the verified private tree."""
    directory = Path(directory).resolve()
    record = json.loads((directory / '_READY.json').read_text())
    if record['directory'] != str(directory) or record['inspect_version'] != '0.3.260':
        raise ValueError('Inspect tools manifest identity mismatch')
    for name, digest in record['files'].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory) or path.is_symlink() or sha(path) != digest:
            raise ValueError(f'Inspect tools changed: {name}')
    import inspect_ai
    if inspect_ai.__version__ != record['inspect_version']:
        raise ValueError('Inspect package differs from prepared tools')
    cli = importlib.import_module('inspect_ai.util._sandbox._cli')
    old_cli, old_dir = cli.SANDBOX_CLI, cli.SANDBOX_TOOLS_DIR
    new_cli = str(directory / 'inspect-sandbox-tools')
    redirected = []
    # Inspect's eager imports can already have copied these constants. Updating
    # the defining module alone would leave bridge/transport/cleanup inconsistent.
    for name, module in list(sys.modules.items()):
        if not name.startswith('inspect_ai.') or module is None:
            continue
        changed = False
        for key, before, after in [('SANDBOX_CLI', old_cli, new_cli),
                                   ('SANDBOX_TOOLS_DIR', old_dir, str(directory))]:
            if getattr(module, key, None) == before:
                setattr(module, key, after)
                changed = True
        if changed:
            redirected.append(name)
    if cli.SANDBOX_CLI != new_cli:
        raise RuntimeError('Inspect path relocation was not applied')
    return redirected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tools-directory', required=True, type=Path)
    parser.add_argument('--runner', required=True, type=Path)
    parser.add_argument('runner_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    modules = activate(args.tools_directory)
    print(json.dumps({'inspect_tools_directory': str(args.tools_directory),
                      'redirected_modules': modules}), flush=True)
    runner = args.runner.resolve()
    remaining = args.runner_args[1:] if args.runner_args[:1] == ['--'] else args.runner_args
    sys.path.insert(0, str(runner.parent))
    sys.argv = [str(runner), *remaining]
    runpy.run_path(str(runner), run_name='__main__')


if __name__ == '__main__':
    main()
