#!/usr/bin/env python3
"""Download pinned Chinese gapfill inputs with checked, resumable byte ranges."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import requests

SOURCES = {
    "t2ranking": {
        "repo": "THUIR/T2Ranking", "revision": "2a369a430a70979223f1b9a41b1919774d46b432",
        "path": "data/collection.tsv", "bytes": 3659243528,
        "sha256": "07b84e543e9ba696124a727c00629d5bce586631648c436c98dd6e9b146da212",
        "small": ["README.md", "data/queries.dev.tsv", "data/qrels.dev.tsv", "data/queries.train.tsv", "data/qrels.train.tsv"],
    },
    "finrag": {
        "repo": "zhaosuifeng/FinRAGBench-V", "revision": "d0d65255c94e687caa81ac9da7758ed25ff046a5",
        "path": "pdfs_for_QA/pdf_ch.tar.gz", "bytes": 4668203104,
        "sha256": "73b6cc2a53fb12a3810a54fba541494cf7fcb882b146f16148e9cd88832b1963",
        "small": ["README.md", "queries/queries_ch.json", "qrels/qrels_ch.tsv"],
    },
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def acquire(root: Path, name: str, workers: int) -> None:
    spec = SOURCES[name]
    target = root / name
    prefix = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/"
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for rel in spec["small"]:
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            response = requests.get(prefix + rel, timeout=120)
            response.raise_for_status()
            with path.open("xb") as stream:
                stream.write(response.content)
        records.append({"path": rel, "url": prefix + rel, "bytes": path.stat().st_size,
                        "sha256": digest(path)})
    output = target / spec["path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = 64 * 1024 * 1024
    chunks = [(start, min(start + chunk_size, spec["bytes"]) - 1)
              for start in range(0, spec["bytes"], chunk_size)]

    def chunk(bounds: tuple[int, int]) -> Path:
        start, end = bounds
        directory = target / "ranges" / f"{start:012d}"
        directory.mkdir(parents=True, exist_ok=True)
        complete = directory / "chunk"
        if complete.exists():
            if complete.stat().st_size != end - start + 1:
                raise ValueError(f"Wrong completed chunk size: {complete}")
            return complete
        part = directory / "chunk.part"
        offset = part.stat().st_size if part.exists() else 0
        if offset > end - start + 1:
            raise ValueError(f"Oversized partial chunk: {part}")
        failures = 0
        session = requests.Session()
        while offset < end - start + 1:
            begin = start + offset
            # The observed connection closes during long responses. Bound each
            # request and resume verified offsets instead of discarding bytes.
            request_end = min(end, begin + 4 * 1024 * 1024 - 1)
            try:
                response = session.get(prefix + spec["path"],
                                        headers={"Range": f"bytes={begin}-{request_end}"},
                                        timeout=(30, 120), stream=True)
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {begin}-{request_end}/{spec['bytes']}":
                    raise ValueError("Server did not honor the exact requested byte range")
                with response, part.open("ab") as stream:
                    for block in response.iter_content(65536):
                        stream.write(block)
                failures = 0
            except requests.RequestException as error:
                failures += 1
                print(json.dumps({"source": name, "retry": failures, "chunk_start": start,
                                  "error": type(error).__name__}), flush=True)
                if failures >= 8:
                    raise
                time.sleep(min(2 ** failures, 20))
            offset = part.stat().st_size if part.exists() else 0
        session.close()
        if part.stat().st_size != end - start + 1:
            raise ValueError(f"Incomplete byte range: {part}")
        part.rename(complete)
        print(json.dumps({"source": name, "chunk_start": start, "bytes": end - start + 1}), flush=True)
        return complete

    if not output.exists():
        with ThreadPoolExecutor(max_workers=workers) as executor:
            paths = list(executor.map(chunk, chunks))
        assembled = output.with_name(output.name + ".assembled")
        if assembled.exists():
            raise FileExistsError(f"Preserved prior assembly requires review: {assembled}")
        with assembled.open("xb") as stream:
            for path in paths:
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        stream.write(block)
        if assembled.stat().st_size != spec["bytes"] or digest(assembled) != spec["sha256"]:
            raise ValueError("Assembled bytes disagree with pinned upstream LFS SHA256")
        assembled.rename(output)
    if output.stat().st_size != spec["bytes"] or digest(output) != spec["sha256"]:
        raise ValueError("Existing file disagrees with pinned upstream LFS SHA256")
    records.append({"path": spec["path"], "url": prefix + spec["path"],
                    "bytes": spec["bytes"], "sha256": spec["sha256"], "upstream_lfs_verified": True})
    receipt = target / "acquisition-receipt.json"
    value = {"schema_version": "dci-zh-source-acquisition-v1", "repo": spec["repo"],
             "revision": spec["revision"], "files": records,
             "license_basis": "Apache-2.0 in pinned upstream dataset card; underlying documents retain their attribution",
             "download_scope": "Declared files only; source corpus scope is audited separately"}
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if receipt.exists() and receipt.read_text() != payload:
        raise ValueError("Acquisition receipt changed")
    if not receipt.exists():
        receipt.write_text(payload)
    print(json.dumps({"source": name, "complete": True, "receipt": str(receipt)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    acquire(args.raw_root, args.source, args.workers)
