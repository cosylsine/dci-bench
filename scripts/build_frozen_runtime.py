#!/usr/bin/env python3
"""Build a pinned Debian runtime and publish a local OCI archive plus offline bundle.

This is an explicit, network-enabled preparation command, never an evaluation
dependency. It requires a previously verified crane binary and system bwrap.
Existing output directories and prepared input trees are never overwritten.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE_IMAGE = 'docker.io/library/debian@sha256:7378260c0a098d27ac206225db5e7d92b8c7c84ea7afccc86eeb513756bbd392'
SNAPSHOT = '20261104T212339Z'
RIPGREP_VERSION = '14.1.1-1+b4'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_oci_layout(bundle: Path, output: Path) -> str:
    """Package exactly the merged bundle as one linux/amd64 OCI image layer."""
    layout = output / 'oci-layout'
    blobs = layout / 'blobs/sha256'
    blobs.mkdir(parents=True)
    compressed = output / 'layer.tar.gz'
    with bundle.open('rb') as source, compressed.open('xb') as target:
        with gzip.GzipFile(filename='', mode='wb', fileobj=target, mtime=0) as stream:
            shutil.copyfileobj(source, stream)
    layer_hash = sha256(compressed)
    compressed.rename(blobs / layer_hash)

    def json_blob(value: dict, media_type: str) -> dict:
        data = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
        digest = hashlib.sha256(data).hexdigest()
        (blobs / digest).write_bytes(data)
        return {'mediaType': media_type, 'digest': 'sha256:' + digest, 'size': len(data)}

    config = json_blob({
        'architecture': 'amd64', 'os': 'linux', 'config': {},
        'rootfs': {'type': 'layers', 'diff_ids': ['sha256:' + sha256(bundle)]},
    }, 'application/vnd.oci.image.config.v1+json')
    manifest = json_blob({
        'schemaVersion': 2, 'mediaType': 'application/vnd.oci.image.manifest.v1+json',
        'config': config,
        'layers': [{'mediaType': 'application/vnd.oci.image.layer.v1.tar+gzip',
                    'digest': 'sha256:' + layer_hash, 'size': (blobs / layer_hash).stat().st_size}],
    }, 'application/vnd.oci.image.manifest.v1+json')
    (layout / 'oci-layout').write_text('{"imageLayoutVersion":"1.0.0"}\n')
    (layout / 'index.json').write_text(json.dumps({
        'schemaVersion': 2, 'manifests': [{**manifest,
            'platform': {'architecture': 'amd64', 'os': 'linux'},
            'annotations': {'org.opencontainers.image.ref.name': 'frozen-v1'}}],
    }, sort_keys=True, separators=(',', ':')) + '\n')
    with tarfile.open(output / 'runtime-image.tar', 'w') as archive:
        for path in sorted(layout.rglob('*')):
            info = archive.gettarinfo(str(path), str(path.relative_to(layout)))
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ''
            info.mode = 0o755 if info.isdir() else 0o644
            if info.isfile():
                with path.open('rb') as stream:
                    archive.addfile(info, stream)
            else:
                archive.addfile(info)
    return manifest['digest']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crane', type=Path, required=True)
    parser.add_argument('--crane-sha256', required=True, help='Expected SHA-256 of the crane executable')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-image', default=BASE_IMAGE)
    parser.add_argument('--snapshot', default=SNAPSHOT)
    parser.add_argument('--ripgrep-version', default=RIPGREP_VERSION)
    parser.add_argument('--prepared-rootfs', type=Path,
                        help='Explicit trusted staging tree, copied before normalization; skips package acquisition')
    args = parser.parse_args()
    if not hasattr(tarfile, 'data_filter'):
        parser.error('Use a maintained Python with tarfile.data_filter support (the benchmark Python supports it)')
    if not re.fullmatch(r'.+@sha256:[a-f0-9]{64}', args.base_image):
        parser.error('--base-image must be a complete digest reference')
    if not re.fullmatch(r'\d{8}T\d{6}Z', args.snapshot):
        parser.error('--snapshot must be an immutable Debian snapshot timestamp')
    crane = args.crane.resolve(strict=True)
    if sha256(crane) != args.crane_sha256:
        raise ValueError('crane executable SHA-256 mismatch')
    from dci_bench.runtime.manager import freeze_tree, tree_digest

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rootfs = output / 'build-rootfs'
    env = {'PATH': '/usr/bin:/bin', 'HOME': '/root', 'LANG': 'C.UTF-8',
           'LC_ALL': 'C.UTF-8', 'TZ': 'UTC', 'DEBIAN_FRONTEND': 'noninteractive'}
    if args.prepared_rootfs is not None:
        shutil.copytree(args.prepared_rootfs.resolve(strict=True), rootfs, symlinks=True)
    else:
        base_tar = output / 'base-rootfs.tar'
        subprocess.run([str(crane), 'export', '--platform', 'linux/amd64', args.base_image, str(base_tar)], check=True)
        rootfs.mkdir()
        # Debian alternatives use absolute symlinks. Only the trusted build
        # boundary normalizes them; runtime imports accept relative links only.
        with tarfile.open(base_tar) as archive:
            for member in archive:
                if member.issym() and member.linkname.startswith('/'):
                    member.linkname = posixpath.relpath(member.linkname.lstrip('/'),
                                                       posixpath.dirname(member.name) or '.')
                archive.extract(member, rootfs, filter='data')
        sources = rootfs / 'etc/apt/sources.list.d/debian.sources'
        sources.write_text(
            f'Types: deb\nURIs: http://snapshot.debian.org/archive/debian/{args.snapshot}\n'
            'Suites: trixie\nComponents: main\n'
            'Signed-By: /usr/share/keyrings/debian-archive-keyring.pgp\nCheck-Valid-Until: no\n')
        build_args = ['/usr/bin/bwrap', '--unshare-user', '--uid', '0', '--gid', '0',
                      '--unshare-pid', '--unshare-ipc', '--unshare-uts',
                      '--bind', str(rootfs), '/', '--dev', '/dev',
                      '--ro-bind', '/etc/resolv.conf', '/etc/resolv.conf',
                      '--die-with-parent', '--new-session']
        # Build-only network access and APT user choice; neither is reused by Pi.
        apt = ['apt-get', '-o', 'APT::Sandbox::User=root', '-o', 'Acquire::Retries=1',
               '-o', 'Acquire::http::Timeout=30']
        subprocess.run([*build_args, *apt, 'update'], env=env, check=True, timeout=1200)
        subprocess.run([*build_args, *apt, 'install', '-y', '--no-install-recommends',
                        f'ripgrep={args.ripgrep_version}'], env=env, check=True, timeout=1200)

    # Normalize the private build copy only. Immutable input bundles are retained.
    for path in sorted(rootfs.rglob('*')):
        if path.is_symlink() and os.readlink(path).startswith('/'):
            target = os.readlink(path).lstrip('/')
            relative = posixpath.relpath(target, str(path.parent.relative_to(rootfs)))
            path.unlink()
            path.symlink_to(relative)
    for relative in ('var/lib/apt/lists', 'var/cache/apt/archives', 'var/log'):
        target = rootfs / relative
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
    for relative in ('workspace/corpus', 'scratch', 'tmp', 'sandbox-bin'):
        (rootfs / relative).mkdir(parents=True, exist_ok=True)
    # bwrap cannot create a file mountpoint after the root was bound read-only.
    # Stable-rg profiles bind their protocol config over this inert placeholder.
    (rootfs / 'sandbox-bin/rg.conf').touch()
    probe = ['/usr/bin/bwrap', '--unshare-user', '--uid', '65534', '--gid', '65534',
             '--unshare-pid', '--unshare-net', '--unshare-ipc', '--unshare-uts',
             '--cap-drop', 'ALL', '--ro-bind', str(rootfs), '/', '--dev', '/dev',
             '--die-with-parent', '--new-session']
    package_list = subprocess.check_output([*probe, '/usr/bin/dpkg-query', '-W'], env=env, text=True)
    (output / 'packages.txt').write_text(package_list)
    if f'ripgrep\t{args.ripgrep_version}\n' not in package_list:
        raise ValueError('Prepared rootfs does not contain the pinned ripgrep package')
    tools = {}
    for name, relative in (('bash', 'usr/bin/bash'), ('rg', 'usr/bin/rg')):
        version = subprocess.check_output([*probe, '/' + relative, '--version'], env=env, text=True)
        tools[name] = {'path': relative, 'sha256': sha256(rootfs / relative), 'version': version.strip()}
    freeze_tree(rootfs)
    tree_hash = tree_digest(rootfs)
    inventory = {
        'executables': [str(path.relative_to(rootfs)) for path in sorted(rootfs.rglob('*'))
                        if path.is_file() and path.stat().st_mode & 0o111],
        'bash_builtins': subprocess.check_output(
            [*probe, '/bin/bash', '--noprofile', '--norc', '-c', 'compgen -b'], env=env, text=True).splitlines(),
        'scope': 'Runtime executable paths and Bash builtins; not a command-name allowlist',
    }
    (output/'tool-inventory.json').write_text(json.dumps(inventory, indent=2, sort_keys=True)+'\n')
    bundle = output / 'rootfs.tar'

    def canonical_header(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = info.mtime = 0
        info.uname = info.gname = ''
        return info

    with tarfile.open(bundle, 'w', format=tarfile.PAX_FORMAT) as archive:
        archive.add(rootfs, arcname='.', filter=canonical_header)
    image_archive = output / 'runtime-image.tar'
    image_digest = write_oci_layout(bundle, output)
    config = {
        'schema_version': 'dci-runtime-v1', 'mode': 'frozen-rootfs',
        'image': 'localhost/dci-runtime@' + image_digest, 'platform': 'linux/amd64',
        'bundle_sha256': sha256(bundle), 'rootfs_tree_sha256': tree_hash,
        'cache_dir': './cache', 'network_during_eval': 'disabled',
        'provisioning': 'prepare-before-eval', 'allow_download_during_eval': False,
        'tool_policy': 'open', 'filesystem_policy': 'dci-bwrap-frozen-ro-v1',
    }
    provenance = {
        'schema_version': 'dci-runtime-build-v1', 'source_base_image': args.base_image,
        'source_kind': 'explicit-prepared-tree' if args.prepared_rootfs else 'pinned-oci-and-apt-snapshot',
        'snapshot': args.snapshot, 'ripgrep_version': args.ripgrep_version,
        'crane_version': subprocess.check_output([str(crane), 'version'], text=True).strip(),
        'crane_sha256': args.crane_sha256, 'builder_sha256': sha256(Path(__file__)),
        'image_digest': image_digest, 'image_archive_sha256': sha256(image_archive),
        'platform': 'linux/amd64', 'bundle_sha256': config['bundle_sha256'],
        'rootfs_tree_sha256': tree_hash, 'tools': tools, 'packages_sha256': sha256(output/'packages.txt'),
        'tool_inventory_sha256': sha256(output/'tool-inventory.json'),
        'registry_published': False,
    }
    for filename, payload in (('runtime-config.json', config), ('release.json', provenance)):
        (output/filename).write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'config': str(output/'runtime-config.json'), 'image_digest': image_digest,
                      'rootfs_tree_sha256': tree_hash}))


if __name__ == '__main__':
    main()
