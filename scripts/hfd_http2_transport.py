#!/usr/bin/env python3
"""Optional HTTP/2 transport for hfd's generated aria2 input and HF manifest.

hfd still selects the pinned files, maintains its cache and shows progress.
Only incomplete files are transferred; existing partials are preserved first.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import uuid


def main():
    args = sys.argv[1:]
    input_file = Path(args[args.index('-i') + 1])
    jobs = int(args[args.index('-j') + 1])
    root = Path.cwd()
    metadata = json.loads((root / '.hfd/repo_metadata.json').read_text())
    expected = {row['rfilename']: row for row in metadata['siblings']}
    entries = []
    for block in input_file.read_text().strip().split('\n\n'):
        if not block.strip():
            continue
        lines = block.splitlines()
        options = dict(line.strip().split('=', 1) for line in lines[1:])
        relative = PurePosixPath(options.get('dir', '')) / options['out']
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix() not in expected:
            raise ValueError('Unsafe or unknown hfd destination')
        entries.append((lines[0], relative.as_posix(), options.get('header')))
    backup = root / '.hfd' / f'partials-before-http2-{uuid.uuid4().hex}'

    def download(entry):
        url, relative, header = entry
        row = expected[relative]
        if os.environ.get('DCI_HFD_USE_RAW') == '1' and not row.get('lfs'):
            # HF raw serves Git blobs directly; resolve may redirect through a
            # separately failing cache. LFS must retain its resolve endpoint.
            url = url.replace('/resolve/', '/raw/', 1)
        output = root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        for path in (output, Path(str(output) + '.aria2')):
            if path.exists():
                saved = backup / path.relative_to(root)
                saved.parent.mkdir(parents=True, exist_ok=True)
                os.rename(path, saved)
        command = ['curl', '-4', '--http2', '--fail', '--location', '--silent', '--show-error',
                   '--connect-timeout', '15', '--max-time', '180', '--retry', '3', '--retry-delay', '2',
                   '--output', str(output)]
        if header:
            command.extend(['--header', header])
        subprocess.run(command + [url], check=True)
        if output.stat().st_size != row['size']:
            raise ValueError(f'Download size mismatch: {relative}')
        lfs = row.get('lfs', {})
        expected_digest = lfs.get('sha256') or lfs.get('oid')
        if expected_digest:
            digest = hashlib.sha256()
            expected_digest = expected_digest.removeprefix('sha256:')
        else:
            digest = hashlib.sha1(f'blob {row["size"]}\0'.encode())
            expected_digest = row.get('blobId')
        with output.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected_digest:
            raise ValueError(f'Upstream digest mismatch: {relative}')

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(download, entries))


if __name__ == '__main__':
    main()
