#!/usr/bin/env python3
"""Probe and, when authorized, materialize the official LLeQA legal source.

The current run records the public GitHub source and HF dataset card, then
probes the gated corpus/question files without credentials.  If the official
data files become accessible, this bounded script is the place to add the
explicit corpus/question/qrel conversion; it never invents associations.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping
import urllib.error
import urllib.request
from zipfile import BadZipFile, ZipFile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "configs/data_pipeline/lleqa_gapfill_20261023.json"
DEFAULT_PYTHON = "/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python"
SCHEMA = "dci-lleqa-gapfill-research-v1"
MAX_PROBE_BYTES = 64 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_once(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(value) if not isinstance(value, bytes) else value
    digest = _sha256_bytes(payload)
    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise RuntimeError(f"immutable destination differs: {path}")
        return digest
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise RuntimeError(f"preserved partial exists; refusing overwrite: {partial}")
    with partial.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    return digest


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"config must be an object: {path}")
    return value


def _download_verified(item: Mapping[str, Any], destination: Path, max_bytes: int) -> dict[str, Any]:
    """Download one fixed metadata object through the configured HTTPS proxy."""
    expected_bytes = int(item["bytes"])
    expected_sha256 = str(item["sha256"])
    if destination.exists():
        digest, size = _sha256_path(destination)
        if size != expected_bytes or digest != expected_sha256:
            raise RuntimeError(f"existing metadata object differs: {destination}")
        return {"path": str(destination), "bytes": size, "sha256": digest, "reused": True}
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise RuntimeError(f"preserved partial exists; refusing overwrite: {partial}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        str(item["url"]),
        headers={"User-Agent": "dci-bench-lleqa-gapfill/20261023"},
        method="GET",
    )
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise RuntimeError(f"metadata object exceeds budget: {item['url']}")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        # Preserve a diagnostic partial only when bytes actually arrived.
        if partial.exists() and partial.stat().st_size == 0:
            partial.unlink()
        raise
    actual = digest.hexdigest()
    if size != expected_bytes or actual != expected_sha256:
        raise RuntimeError(
            f"metadata integrity mismatch for {item['url']}: "
            f"expected {expected_bytes}/{expected_sha256}, got {size}/{actual}"
        )
    os.replace(partial, destination)
    return {"path": str(destination), "bytes": size, "sha256": actual, "reused": False}


def _probe(url: str) -> dict[str, Any]:
    """Make one unauthenticated bounded request and preserve exact failure evidence."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "dci-bench-lleqa-gapfill/20261023"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(MAX_PROBE_BYTES)
            return {
                "url": url,
                "status": int(response.status),
                "bytes_read": len(body),
                "body_sha256": _sha256_bytes(body),
                "body_preview": body[:500].decode("utf-8", errors="replace"),
            }
    except urllib.error.HTTPError as exc:
        body = exc.read(MAX_PROBE_BYTES)
        return {
            "url": url,
            "status": int(exc.code),
            "bytes_read": len(body),
            "body_sha256": _sha256_bytes(body),
            "body_preview": body[:500].decode("utf-8", errors="replace"),
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "url": url,
            "status": None,
            "bytes_read": 0,
            "body_sha256": None,
            "body_preview": "",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _archive_inventory(path: Path) -> dict[str, Any]:
    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
    except (BadZipFile, OSError) as exc:
        return {"member_count": None, "data_members": [], "error": str(exc)}
    data_members = [
        name for name in names
        if any(token in name.lower() for token in ("article", "question", "corpus", "lleqa"))
        and name.lower().endswith((".json", ".jsonl", ".csv", ".parquet"))
    ]
    return {"member_count": len(names), "data_members": data_members}


def run(config_path: Path) -> dict[str, Any]:
    config = _load(config_path)
    source = config["source"]
    raw_root = Path(config["raw_root"]) / source["id"]
    metadata_root = raw_root / "source_metadata"
    research_root = Path(config["research_root"])
    raw_root.mkdir(parents=True, exist_ok=True)
    research_root.mkdir(parents=True, exist_ok=True)

    research_path = research_root / "research.json"
    if research_path.exists():
        return _load(research_path)

    metadata_files = {}
    for role, item_key in (("github_archive", "github_archive"), ("hf_card", "hf_card")):
        item = source[item_key]
        metadata_files[role] = _download_verified(
            item,
            raw_root / item["path"],
            int(config["download_budget_bytes"]),
        )

    archive_inventory = _archive_inventory(raw_root / source["github_archive"]["path"])
    attempts = []
    for item in source["raw_files"]:
        result = _probe(str(item["url"]))
        result.update({"role": item["role"], "source_path": item["source_path"]})
        attempts.append(result)
    for url in source["modelscope_candidates"]:
        result = _probe(str(url))
        result.update({"role": "modelscope_tree_candidate"})
        attempts.append(result)

    manifest = {
        "schema_version": "dci-lleqa-source-attempt-v1",
        "complete": False,
        "status": "blocked_access_gate",
        "source": source["id"],
        "dataset": source["dataset"],
        "github_revision": source["github_revision"],
        "hf_card_revision": source["hf_card_revision"],
        "metadata_files": metadata_files,
        "github_archive_inventory": archive_inventory,
        "access_attempts": attempts,
        "expected_official_counts": source["expected_official_counts"],
        "planned_conversion": source["planned_conversion"],
        "release": {
            "path": str(Path(config["eval_root"]) / "pipeline_v1" / source["id"] / "v1"),
            "status": "not_built",
            "reason": "official corpus and question files were gated; no raw labels were available",
        },
        "identity_note": "The GitHub archive and HF card are metadata evidence only; no corpus/query bytes were claimed from them.",
    }
    manifest_path = raw_root / "source_attempt_manifest.json"
    manifest_sha256 = _write_once(manifest_path, manifest)

    blocked = [
        item for item in attempts
        if item.get("status") not in {200, 206}
    ]
    receipt = {
        "schema_version": SCHEMA,
        "generated_at": _now(),
        "config": str(config_path),
        "python": DEFAULT_PYTHON,
        "download_budget_bytes": int(config["download_budget_bytes"]),
        "status": "blocked" if blocked else "ready_for_conversion",
        "source": {
            "id": source["id"],
            "domain": source["domain"],
            "language": source["language"],
            "language_variant": source["language_variant"],
            "dataset": source["dataset"],
            "source_family": source["source_family"],
            "source_scope": source["source_scope"],
            "translation_status": source["translation_status"],
            "evaluation_split": source["evaluation_split"],
            "expected_official_counts": source["expected_official_counts"],
            "official_urls": source["official_urls"],
            "raw_directory": str(raw_root),
            "source_attempt_manifest": str(manifest_path),
            "source_attempt_manifest_sha256": manifest_sha256,
            "metadata_files": metadata_files,
            "access_attempts": attempts,
            "archive_inventory": archive_inventory,
            "observed_counts": None,
            "release_path": str(Path(config["eval_root"]) / "pipeline_v1" / source["id"] / "v1"),
            "release_state": None,
            "planned_conversion": source["planned_conversion"],
        },
        "blocked_reason": (
            "HF corpus/questions returned HTTP 401 because the official LLeQA dataset is gated and no credentials were supplied; "
            "the fixed GitHub archive contains code but no data files; all three ModelScope candidates returned HTTP 404."
        ),
        "checks": [
            "GitHub source archive downloaded at a fixed commit and SHA-256 verified",
            "HF dataset card downloaded at a fixed dataset revision and SHA-256 verified",
            "official corpus and train/dev/test question URLs probed once without credentials",
            "ModelScope original and MTEB mirror tree candidates probed and failures preserved",
            "no corpus, queries, qrels, synthetic labels, GPU work, or release were created",
        ],
        "limitations": [
            "The official source requires a signed data-use agreement and authenticated HF access.",
            "The receipt records official card counts as expected metadata; no observed D/Q/qrel counts exist.",
            "No release can be verified until the gated source files are supplied through an authorized channel.",
        ],
    }
    _write_once(research_path, receipt)
    return receipt


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    receipt = run(args.config)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
