#!/usr/bin/env python3
"""Acquire and publish the bounded 2026-10-23 BEIR domain releases.

This wrapper keeps the network and parsing path in ``scripts/data_pipeline.py``
and records the exact commands, raw snapshot manifest, release verification,
and any failure trace under the allowed research directory.  It never removes
or overwrites an existing snapshot or release.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dci_bench.data.pipeline.download import download_url
DEFAULT_CONFIG = ROOT / "configs/data_pipeline/domains_20261023.json"
DEFAULT_PYTHON = Path("/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python")
DEFAULT_RESEARCH_ROOT = Path(
    "/mnt/afs2/202609/embedding_models/dci-bench/eval_data/"
    "source_research_20261023/domains"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_dump(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _chunked_archive(spec: dict[str, Any], *, raw_root: Path) -> dict[str, Any]:
    """Materialise a slow official archive with independent HTTP ranges.

    The UKP archive host can leave a streaming response open after roughly one
    MiB.  Each request below is still made by the pipeline's direct
    ``download_url`` helper (which ignores the legacy proxy), but a bounded
    range lets the server close the response and keeps a failed chunk as a
    diagnosable artifact.  The assembled archive and chunks live under the
    source-specific ``raw_data/domain_*`` directory.
    """

    options = spec.get("options") if isinstance(spec.get("options"), dict) else {}
    url = str(options.get("archive", ""))
    expected_bytes = int(options.get("archive_bytes", 0))
    if not url or expected_bytes <= 0:
        raise ValueError(f"chunked acquisition requires archive_bytes: {spec['id']}")
    source_root = raw_root / str(spec["id"]) / "download_stage"
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage = source_root / attempt
    stage.mkdir(parents=True, exist_ok=False)
    chunk_size = 1024 * 1024
    ranges = [
        (start, min(start + chunk_size, expected_bytes) - 1)
        for start in range(0, expected_bytes, chunk_size)
    ]
    archive_name = Path(urlsplit(url).path).name or f"{spec['id']}.zip"
    chunk_dir = stage / "chunks"
    chunk_dir.mkdir()

    def fetch_range(item: tuple[int, int]) -> dict[str, Any]:
        start, end = item
        size = end - start + 1
        path = chunk_dir / f"{start:012d}-{end:012d}.part"
        started = time.monotonic()
        digest, actual = download_url(
            url,
            path,
            timeout=float(options.get("timeout", 60)),
            retries=int(options.get("retries", 3)),
            max_bytes=size,
            expected_bytes=size,
            headers={"Range": f"bytes={start}-{end}"},
            overwrite=False,
        )
        return {
            "start": start,
            "end": end,
            "bytes": actual,
            "sha256": digest,
            "path": str(path),
            "duration_seconds": round(time.monotonic() - started, 3),
        }

    chunks: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    workers = min(8, len(ranges))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_range, item): item for item in ranges}
        for future in as_completed(futures):
            item = futures[future]
            try:
                chunks.append(future.result())
            except Exception as exc:
                errors.append({
                    "start": item[0],
                    "end": item[1],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
    if errors:
        receipt = {
            "source": spec["id"],
            "url": url,
            "expected_bytes": expected_bytes,
            "stage": str(stage),
            "chunk_size": chunk_size,
            "workers": workers,
            "chunks": sorted(chunks, key=lambda item: item["start"]),
            "errors": errors,
            "status": "failed",
        }
        _json_dump(stage / "chunk_receipt.json", receipt)
        raise RuntimeError(f"range download failed for {spec['id']}: {errors[0]['error']}")

    archive = stage / archive_name
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with archive.open("wb") as output:
        for item in sorted(chunks, key=lambda row: row["start"]):
            with Path(item["path"]).open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
                    sha256.update(block)
                    md5.update(block)
    actual_bytes = archive.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"assembled archive size mismatch for {spec['id']}: "
            f"expected {expected_bytes}, got {actual_bytes}"
        )
    receipt = {
        "source": spec["id"],
        "url": url,
        "expected_bytes": expected_bytes,
        "actual_bytes": actual_bytes,
        "stage": str(stage),
        "archive": str(archive),
        "archive_sha256": sha256.hexdigest(),
        "archive_md5": md5.hexdigest(),
        "chunk_size": chunk_size,
        "workers": workers,
        "chunks": sorted(chunks, key=lambda item: item["start"]),
        "status": "complete",
    }
    _json_dump(stage / "chunk_receipt.json", receipt)
    return receipt


def _run(command: list[str], *, cwd: Path) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
        env=dict(os.environ),
    )
    return {
        "command": command,
        "cwd": str(cwd),
        "started_at": _now(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _last_json(stdout: str) -> dict[str, Any] | None:
    # The pipeline emits one pretty-printed JSON object.  Keep this tolerant
    # of future progress lines while refusing to infer a result from malformed
    # output.  First try the complete stream, then each suffix beginning at a
    # likely object boundary.
    candidates = [stdout.strip()]
    candidates.extend(
        stdout[index:].strip()
        for index, character in enumerate(stdout)
        if character == "{" and index > 0
    )
    for candidate in reversed(candidates):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _source_entry(spec: dict[str, Any], release_path: Path) -> dict[str, Any]:
    options = spec.get("options") if isinstance(spec.get("options"), dict) else {}
    metadata = spec.get("metadata") if isinstance(spec.get("metadata"), dict) else {}
    archive_url = metadata.get("archive_url") or options.get("archive")
    return {
        "id": spec["id"],
        "domain": spec.get("domain"),
        "language": spec.get("language"),
        "dataset": options.get("dataset"),
        "revision": spec.get("revision"),
        "license": spec.get("license"),
        "urls": {
            "archive": archive_url,
            "official_homepage": metadata.get("official_homepage"),
            "dataset_card": metadata.get("dataset_card"),
            "license_source": metadata.get("license_source"),
            "beir_catalog": metadata.get("beir_catalog"),
        },
        "expected_counts": {
            "documents": metadata.get("expected_documents"),
            "queries": metadata.get("expected_test_queries"),
            "relevance_per_query": metadata.get("expected_test_relevance_per_query"),
        },
        "source_archive_bytes": metadata.get("source_archive_bytes"),
        "source_archive_sha256": metadata.get("source_archive_sha256"),
        "derived_input_sha256": metadata.get("derived_input_sha256"),
        "acquisition_status": "pending",
        "release_path": str(release_path),
        "limitations": [
            metadata.get("label_scope"),
            metadata.get("coverage_limit"),
            "Only the configured test qrels split is published; the full source corpus is retained.",
        ],
    }


def _ensure_qasper_fullcorpus(raw_root: Path) -> Path:
    """Derive the train+dev document view without creating train labels.

    The existing official QASPER snapshot is deliberately kept untouched.  A
    domain-scoped derived input removes only the train ``qas`` fields, keeps
    every train/dev paper record and every dev annotation byte-for-byte in
    content, and is then consumed by the normal QASPER connector.
    """

    source_snapshot = raw_root / "qasper_dev" / (
        "2afb05053bce7c7aa61add963277d9222c5507806da98a8bdd275efaf6c889d4"
    ) / "qasper" / "extracted"
    train_path = source_snapshot / "qasper-train-v0.3.json"
    dev_path = source_snapshot / "qasper-dev-v0.3.json"
    output_dir = raw_root / "domain_qasper_dev_fullcorpus" / "input"
    output = output_dir / "qasper-full-v0.3.json"
    derivation = output_dir / "derivation_manifest.json"
    if output.is_file():
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        expected = "d30e470275b7a2929eb74c90d962394f5a60dbd1040009a3733e533bfc13ea3e"
        if digest != expected:
            raise RuntimeError(
                f"existing QASPER derived input hash mismatch: {output}: {digest}"
            )
        return output
    if not train_path.is_file() or not dev_path.is_file():
        raise FileNotFoundError(
            "QASPER train/dev source snapshot is required before deriving the full corpus"
        )
    train = _load_json(train_path)
    dev = _load_json(dev_path)
    if not isinstance(train, dict) or not isinstance(dev, dict):
        raise ValueError("QASPER train/dev source files must be paper-ID mappings")
    combined: dict[str, Any] = {}
    train_qas = 0
    for paper_id, paper in train.items():
        if not isinstance(paper, dict):
            raise ValueError(f"QASPER train paper {paper_id!r} is not an object")
        row = copy.deepcopy(paper)
        train_qas += len(row.get("qas", [])) if isinstance(row.get("qas", []), list) else 0
        row.pop("qas", None)
        combined[str(paper_id)] = row
    for paper_id, paper in dev.items():
        if str(paper_id) in combined:
            raise ValueError(f"QASPER train/dev paper ID overlap: {paper_id!r}")
        combined[str(paper_id)] = copy.deepcopy(paper)
    output_dir.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, ensure_ascii=False, sort_keys=False, separators=(",", ":"))
        handle.write("\n")
    _json_dump(
        derivation,
        {
            "schema_version": "dci-domain-derived-input-v1",
            "source_id": "domain_qasper_dev_fullcorpus",
            "source_archive": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
            "source_archive_sha256": "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a",
            "source_snapshot": str(source_snapshot.parent.parent),
            "input_files": [
                {"path": str(train_path), "documents": len(train), "qas": train_qas},
                {"path": str(dev_path), "documents": len(dev), "qas": sum(len(row.get("qas", [])) for row in dev.values() if isinstance(row, dict))},
            ],
            "derivation": "copy all train and dev paper records; remove only train qas fields; preserve all dev qas exactly; no labels or evidence synthesized",
            "counts": {
                "train_documents": len(train),
                "dev_documents": len(dev),
                "combined_documents": len(combined),
                "train_qas_stripped": train_qas,
                "dev_qas_retained": sum(len(row.get("qas", [])) for row in dev.values() if isinstance(row, dict)),
            },
            "derived_file": {"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "bytes": output.stat().st_size},
        },
    )
    return output


def acquire(
    *,
    config_path: Path,
    python_path: Path,
    research_root: Path,
    source_ids: list[str] | None,
    release_name: str,
    chunked: bool,
) -> dict[str, Any]:
    config = _load_json(config_path)
    if not isinstance(config, dict) or not isinstance(config.get("sources"), list):
        raise ValueError(f"invalid source config: {config_path}")
    sources = {str(spec["id"]): spec for spec in config["sources"]}
    selected = source_ids or sorted(sources)
    unknown = sorted(set(selected) - set(sources))
    if unknown:
        raise ValueError(f"unknown source IDs: {unknown}")

    # Fail before acquisition when a retry would overwrite its earlier receipt.
    # A retry needs a new release name and a separate research directory.
    for source_id in selected:
        release = Path(config["eval_root"]) / "pipeline_v1" / source_id / release_name
        receipt = research_root / f"{source_id}.receipt.json"
        variant = research_root / f"{source_id}.chunked-config.json"
        for existing in (release, receipt, variant):
            if existing.exists():
                raise FileExistsError(f"Refusing to overwrite prior acquisition: {existing}")

    research_root.mkdir(parents=True, exist_ok=True)
    pipeline = ROOT / "scripts/data_pipeline.py"
    python = str(python_path)
    summary: dict[str, Any] = {
        "schema_version": "dci-domain-acquisition-v1",
        "created_at": _now(),
        "config_path": str(config_path),
        "python": python,
        "pipeline": str(pipeline),
        "release_name": release_name,
        "sources": [],
        "commands": [],
        "receipts": [],
    }
    previous_path = research_root / "research.json"
    if previous_path.is_file():
        try:
            previous = _load_json(previous_path)
        except (OSError, json.JSONDecodeError):
            previous = {}
        if isinstance(previous, dict):
            for key in ("sources", "commands", "receipts"):
                value = previous.get(key)
                if isinstance(value, list):
                    summary[key].extend(value)

    def replace_source(item: dict[str, Any]) -> None:
        summary["sources"] = [
            old for old in summary["sources"] if old.get("id") != item.get("id")
        ]
        summary["sources"].append(item)

    def record_receipt(path: Path) -> None:
        value = str(path)
        if value not in summary["receipts"]:
            summary["receipts"].append(value)

    if "domain_qasper_dev_fullcorpus" in selected:
        _ensure_qasper_fullcorpus(Path(config["raw_root"]))

    for source_id in selected:
        spec = sources[source_id]
        release_path = Path(config["eval_root"]) / "pipeline_v1" / source_id / release_name
        entry = _source_entry(spec, release_path)
        run_config = config_path
        receipt: dict[str, Any] = {
            "source": source_id,
            "started_at": _now(),
            "steps": [],
            "raw_snapshot": None,
            "release": None,
        }
        if chunked:
            try:
                chunk_receipt = _chunked_archive(
                    spec,
                    raw_root=Path(config["raw_root"]),
                )
            except Exception as exc:
                entry["acquisition_status"] = "range_download_failed"
                receipt["chunked_download"] = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                receipt["finished_at"] = _now()
                receipt_path = research_root / f"{source_id}.receipt.json"
                _json_dump(receipt_path, receipt)
                replace_source(entry)
                record_receipt(receipt_path)
                continue
            receipt["chunked_download"] = chunk_receipt
            entry["archive_bytes"] = chunk_receipt.get("actual_bytes")
            entry["archive_sha256"] = chunk_receipt.get("archive_sha256")
            entry["archive_md5"] = chunk_receipt.get("archive_md5")
            variant = json.loads(json.dumps(config))
            variant_source = next(item for item in variant["sources"] if item["id"] == source_id)
            variant_source["options"] = dict(variant_source.get("options") or {})
            variant_source["metadata"] = dict(variant_source.get("metadata") or {})
            variant_source["metadata"]["archive_url"] = spec.get("options", {}).get("archive")
            variant_source["options"]["archive"] = chunk_receipt["archive"]
            variant_path = research_root / f"{source_id}.chunked-config.json"
            _json_dump(variant_path, variant)
            receipt["run_config"] = str(variant_path)
            run_config = variant_path
        fetch = _run(
            [python, str(pipeline), "--config", str(run_config), "fetch", "--source", source_id],
            cwd=ROOT,
        )
        receipt["steps"].append(fetch)
        summary["commands"].append(fetch["command"])
        fetch_result = _last_json(fetch["stdout"])
        if fetch["returncode"] != 0 or not fetch_result or not fetch_result.get("snapshot"):
            entry["acquisition_status"] = "fetch_failed"
            receipt["finished_at"] = _now()
            receipt_path = research_root / f"{source_id}.receipt.json"
            _json_dump(receipt_path, receipt)
            receipt["receipt_path"] = str(receipt_path)
            replace_source(entry)
            record_receipt(receipt_path)
            continue
        snapshot = Path(fetch_result["snapshot"])
        receipt["raw_snapshot"] = str(snapshot)
        manifest_path = snapshot / "source_manifest.json"
        if manifest_path.is_file():
            receipt["raw_manifest"] = _load_json(manifest_path)
            entry["raw_snapshot_path"] = str(snapshot)
            entry["archive_bytes"] = next(
                (
                    item.get("bytes")
                    for item in receipt["raw_manifest"].get("files", [])
                    if str(item.get("path", "")).startswith("beir/archive")
                ),
                None,
            )

        build = _run(
            [
                python,
                str(pipeline),
                "--config",
                str(run_config),
                "build",
                "--source",
                source_id,
                "--release-name",
                release_name,
            ],
            cwd=ROOT,
        )
        receipt["steps"].append(build)
        summary["commands"].append(build["command"])
        build_result = _last_json(build["stdout"])
        if build["returncode"] != 0:
            entry["acquisition_status"] = "build_failed"
            receipt["finished_at"] = _now()
            receipt_path = research_root / f"{source_id}.receipt.json"
            _json_dump(receipt_path, receipt)
            receipt["receipt_path"] = str(receipt_path)
            replace_source(entry)
            record_receipt(receipt_path)
            continue

        verify = _run(
            [python, str(pipeline), "verify", str(release_path)],
            cwd=ROOT,
        )
        receipt["steps"].append(verify)
        summary["commands"].append(verify["command"])
        verify_result = _last_json(verify["stdout"])
        receipt["release"] = verify_result or build_result
        if verify["returncode"] != 0 or not isinstance(verify_result, dict):
            entry["acquisition_status"] = "verify_failed"
        else:
            entry["acquisition_status"] = "published_verified"
            entry["counts"] = {
                key: verify_result.get(key)
                for key in ("num_documents", "num_queries", "num_positive_pairs", "state")
            }
            entry["release_state"] = verify_result.get("state")
            entry["release_manifest_sha256"] = verify_result.get("bundle_hash")
        receipt["finished_at"] = _now()
        receipt_path = research_root / f"{source_id}.receipt.json"
        _json_dump(receipt_path, receipt)
        receipt["receipt_path"] = str(receipt_path)
        replace_source(entry)
        record_receipt(receipt_path)

    summary["completed_at"] = _now()
    summary["acquisition_status"] = (
        "published_verified"
        if summary["sources"] and all(
            item["acquisition_status"] == "published_verified" for item in summary["sources"]
        )
        else "partial_or_failed"
    )
    _json_dump(research_root / "research.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python", dest="python_path", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--research-root", type=Path, default=DEFAULT_RESEARCH_ROOT)
    parser.add_argument("--source", dest="source_ids", nargs="+")
    parser.add_argument("--release-name", default="expansion-v1")
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="download official archives using bounded HTTP ranges through the pipeline helper",
    )
    args = parser.parse_args(argv)
    result = acquire(
        config_path=args.config,
        python_path=args.python_path,
        research_root=args.research_root,
        source_ids=args.source_ids,
        release_name=args.release_name,
        chunked=args.chunked,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0 if result["acquisition_status"] == "published_verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
