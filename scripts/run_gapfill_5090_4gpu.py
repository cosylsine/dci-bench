#!/usr/bin/env python3
"""Resume the frozen gapfill document pools on four RTX 5090 GPUs.

The launcher has two deliberately separate boundaries.  ``--check-only``
loads and validates the prepared source metadata and existing embedding
shards, then prints the deterministic four-worker plan without creating any
files or loading a model.  Full mode first performs a four-GPU CUDA check and
a tiny real encode, runs the missing document shards in isolated worker cache
directories, merges verified shards into the immutable source cache one at a
time, and finally invokes the existing ``rank_source``/pool publication path.

Only document encoding is parallel.  Query encoding, dense search, and pool
publication remain in the controller so no two processes write the same
cache, ranking, or pool directory.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Sequence
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = Path("/mnt/afs2/202609/embedding_models/dci-bench/eval_data")
RUN = BASE / "benchmark_gapfill_20261031_v1"
MODEL_PATH = "/mnt/afs/share/embedding_models/KaLM-embedding-multilingual-mini-instruct-v2_5"
REQUIRED_PYTHON = "/mnt/afs/250010100/miniconda/envs/lychee_reranker_5090_kalm451/bin/python"
EMBED_PY = REQUIRED_PYTHON
QUERY_PROMPT = "Instruct: Given a query, retrieve documents that answer the query \n Query: "
DOCUMENT_PROMPT = ""
MAX_LENGTH = 8192
SHARD_SIZE = 1024
GPU_COUNT = 4
POOL_CAPACITIES = (200, 1000)
SEED = 42
SCHEMA = "dci-gapfill-5090-4gpu-v1"
# The old cached vectors were produced on a different GPU/runtime.  A
# float32 rerun should be nearly identical, but this gate leaves room for
# harmless kernel rounding while rejecting a changed model/prompt path.
SMOKE_MAX_ABS_TOL = 5e-4
SMOKE_MIN_COSINE = 0.99999


@dataclass(frozen=True)
class SourceSpec:
    name: str
    batch_size: int
    receipt_name: str
    ranking_name: str
    raw_ranking_name: str
    pool_names: tuple[str, str]
    exclude_self: bool = False


SOURCE_SPECS: dict[str, SourceSpec] = {
    "gapfill_browsecomp_plus_en": SourceSpec(
        name="gapfill_browsecomp_plus_en",
        batch_size=8,
        receipt_name="catalog-row.json",
        ranking_name="ranking.json",
        raw_ranking_name="ranking.json",
        pool_names=("pools-200", "pools-1000"),
    ),
    "gapfill_cedar_zh_v2": SourceSpec(
        name="gapfill_cedar_zh_v2",
        batch_size=2,
        receipt_name="catalog-row-v2.json",
        ranking_name="ranking-no-self-v2.json",
        raw_ranking_name="ranking-raw.json",
        pool_names=("pools-200-v2", "pools-1000-v2"),
        exclude_self=True,
    ),
    "gapfill_t2ranking_software_zh": SourceSpec(
        name="gapfill_t2ranking_software_zh",
        batch_size=8,
        receipt_name="catalog-row.json",
        ranking_name="ranking.json",
        raw_ranking_name="ranking.json",
        pool_names=("pools-200", "pools-1000"),
    ),
    "gapfill_cfqa_enterprise_zh": SourceSpec(
        name="gapfill_cfqa_enterprise_zh",
        batch_size=8,
        receipt_name="catalog-row.json",
        ranking_name="ranking.json",
        raw_ranking_name="ranking.json",
        pool_names=("pools-200", "pools-1000"),
    ),
}
DEFAULT_SOURCES = ("gapfill_browsecomp_plus_en", "gapfill_cedar_zh_v2")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_new(path: Path, value: Any) -> None:
    """Write one new receipt without replacing an existing artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def source_suite(run_root: Path) -> Path:
    return Path(run_root) / "source"


def source_root(run_root: Path, source: str) -> Path:
    return Path(run_root) / source


def _metadata_dir(run_root: Path, source: str) -> Path:
    return source_suite(run_root) / "metadata" / source


def _visible_gpu_slots(value: str | None = None) -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES") if value is None else value
    if raw is None or not raw.strip():
        slots = [str(index) for index in range(GPU_COUNT)]
    else:
        slots = [item.strip() for item in raw.split(",")]
    if len(slots) != GPU_COUNT or any(not item or item == "-1" for item in slots):
        raise RuntimeError(
            f"requires exactly {GPU_COUNT} visible GPUs; CUDA_VISIBLE_DEVICES={raw!r}"
        )
    if len(set(slots)) != GPU_COUNT:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must contain four distinct physical GPU slots")
    return slots


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("torch", "transformers", "sentence-transformers", "numpy", "pyarrow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def runtime_provenance(*, visible_devices: Sequence[str] | None = None,
                       gpu_info: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Record runtime facts outside the cache identity.

    The source cache intentionally remains compatible with the existing
    config/batch identity.  Runtime versions and physical GPU assignment are
    therefore recorded here and in the final source receipt instead of being
    injected into ``EmbeddingConfig``.
    """

    slots = list(visible_devices) if visible_devices is not None else (
        [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
        or [str(index) for index in range(GPU_COUNT)]
    )
    return {
        "schema_version": "dci-gapfill-runtime-provenance-v1",
        "at": now(),
        "python": sys.executable,
        "required_python": REQUIRED_PYTHON,
        "python_matches_required": Path(sys.executable).resolve() == Path(REQUIRED_PYTHON).resolve(),
        "packages": package_versions(),
        "cuda_visible_devices": slots,
        "gpu_info": [dict(row) for row in (gpu_info or [])],
        "logical_encoder_device": "cuda:0",
        "output_dtype": "float32",
        "cache_identity_runtime_bound": False,
        "cache_identity_excludes": ["package_versions", "CUDA_VISIBLE_DEVICES", "GPU_name", "GPU_compute_capability"],
    }


def _validate_python_environment() -> None:
    actual = Path(sys.executable).resolve()
    expected = Path(REQUIRED_PYTHON).resolve()
    if actual != expected:
        raise RuntimeError(
            f"wrong Python environment: expected {REQUIRED_PYTHON}, got {sys.executable}; "
            "launch through scripts/launch_gapfill_5090_4gpu.sh"
        )


def _embedding_config(spec: SourceSpec, *, model_path: str = MODEL_PATH):
    from dci_bench.data.pipeline.retrieval import EmbeddingConfig

    return EmbeddingConfig(
        model_path=model_path,
        trust_remote_code=True,
        query_prompt=QUERY_PROMPT,
        document_prompt=DOCUMENT_PROMPT,
        max_length=MAX_LENGTH,
        pooling="mean",
        normalize=True,
        truncation_unit="tokens",
        character_max_length=None,
        device="cuda:0",
        batch_size=spec.batch_size,
        chunk_size=4096,
        weight_identity=None,
    )


def _load_source_bundle(run_root: Path, source: str, *, verify_content_hashes: bool = False) -> dict[str, Any]:
    from dci_bench.data.pipeline.retrieval import load_prepared_source

    return load_prepared_source(source_suite(run_root), source, verify_content_hashes=verify_content_hashes)


def _read_existing_weight_identity(cache_dir: Path, config: Any) -> str | None:
    """Read a prior namespace's weight identity without changing it."""

    import numpy as np

    shards_root = Path(cache_dir) / "shards"
    if not shards_root.is_dir():
        return None
    found: set[str] = set()
    for path in sorted(shards_root.glob("*/document-*.npz")):
        try:
            with np.load(path, allow_pickle=False) as payload:
                raw = payload["metadata"]
                metadata = json.loads(str(raw.item() if hasattr(raw, "item") else raw))
            identity = metadata.get("identity", {})
            value = identity.get("weight_identity")
            if isinstance(value, str) and value:
                found.add(value)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"existing embedding cache metadata is unreadable: {path}") from exc
    if len(found) > 1:
        raise ValueError(f"existing embedding cache has conflicting weight identities: {cache_dir}")
    return next(iter(found), None)


def _cache_for(entry: Mapping[str, Any]):
    from dci_bench.data.pipeline.retrieval import EmbeddingShardCache

    spec = SOURCE_SPECS[str(entry["source"])]
    config = _embedding_config(spec, model_path=str(entry["model_path"]))
    return EmbeddingShardCache(
        Path(entry["cache_dir"]),
        config,
        SHARD_SIZE,
        weight_identity=entry.get("weight_identity"),
    )


def _model_weight_identity(model_path: str, cache_dir: Path, config: Any) -> str:
    from dci_bench.data.pipeline.retrieval import model_weight_identity

    existing = _read_existing_weight_identity(cache_dir, config)
    return existing or model_weight_identity(model_path)


def _validate_prepared_metadata(run_root: Path, source: str) -> dict[str, Any]:
    """Validate the cheap immutable metadata boundary for one source."""

    metadata = _metadata_dir(run_root, source)
    required = (
        "_COMPLETE.json", "manifest.json", "source_manifest.json", "queries.jsonl",
        "qrels.json", "source_doc_ids.json", "corpus_files.jsonl",
    )
    missing = [name for name in required if not (metadata / name).is_file()]
    if missing:
        raise FileNotFoundError(f"prepared source {source} is missing metadata: {missing}")
    manifest = read_json(metadata / "manifest.json")
    if manifest.get("task") != source or manifest.get("state") not in (None, "labels_ready"):
        raise ValueError(f"prepared source manifest is not labels_ready for {source}")
    if manifest.get("num_docs", 0) < 1000:
        raise ValueError(f"source {source} has fewer than 1000 documents")
    complete = read_json(metadata / "_COMPLETE.json")
    if complete.get("schema_version") != "dci-dataset-expansion-20261014-v1":
        raise ValueError(f"prepared source completion schema changed for {source}")
    return manifest


@dataclass(frozen=True)
class PreparedShardReader:
    """Read only the document texts needed for one contiguous cache shard.

    The controller uses this bounded reader during preflight and merge.  The
    regular prepared-source loader materializes every document, which is
    useful at ranking time but can exceed host memory for BrowseComp+.
    """

    aliases: tuple[str, ...]
    files: Mapping[str, tuple[Path, int]]
    query_count: int

    @property
    def document_count(self) -> int:
        return len(self.aliases)

    def texts_for_shard(self, shard_index: int) -> list[str]:
        if isinstance(shard_index, bool) or not isinstance(shard_index, int) or shard_index < 0:
            raise ValueError(f"invalid document shard index: {shard_index!r}")
        start = shard_index * SHARD_SIZE
        if start >= self.document_count:
            raise ValueError(f"document shard index is out of range: {shard_index}")
        aliases = self.aliases[start:min(start + SHARD_SIZE, self.document_count)]
        texts: list[str] = []
        for alias in aliases:
            path, expected_bytes = self.files[alias]
            raw = path.read_bytes()
            if len(raw) != expected_bytes:
                raise ValueError(f"prepared source corpus size changed for alias {alias!r}: {path}")
            try:
                texts.append(raw.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise ValueError(f"prepared source corpus is not UTF-8 for alias {alias!r}: {path}") from exc
        return texts


def _prepared_shard_reader(run_root: Path, source: str) -> PreparedShardReader:
    """Build a bounded text reader from the frozen source inventory."""

    metadata = _metadata_dir(run_root, source)
    mapping = read_json(metadata / "source_doc_ids.json")
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError(f"prepared source document mapping is invalid for {source}")
    aliases = tuple(sorted(str(alias) for alias in mapping))
    if len(set(aliases)) != len(aliases) or any(not alias for alias in aliases):
        raise ValueError(f"prepared source document aliases are invalid for {source}")
    corpus_dir = source_suite(run_root) / "workspaces" / source / "corpus"
    files: dict[str, tuple[Path, int]] = {}
    inventory = metadata / "corpus_files.jsonl"
    with inventory.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"prepared source inventory row is invalid for {source}")
            alias = row.get("doc_id")
            filename = row.get("filename")
            expected_bytes = row.get("bytes")
            if (
                not isinstance(alias, str)
                or alias not in mapping
                or not isinstance(filename, str)
                or Path(filename).name != filename
                or isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, int)
                or expected_bytes < 0
            ):
                raise ValueError(f"prepared source inventory row is invalid for {source}: {row}")
            if alias in files:
                raise ValueError(f"prepared source inventory contains duplicate alias: {alias!r}")
            path = corpus_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"prepared source corpus file is missing: {path}")
            files[alias] = (path, expected_bytes)
    if set(files) != set(aliases):
        missing = sorted(set(aliases) - set(files))
        extra = sorted(set(files) - set(aliases))
        raise ValueError(f"prepared source inventory IDs differ for {source}: missing={missing[:5]} extra={extra[:5]}")
    query_count = 0
    with (metadata / "queries.jsonl").open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.strip():
                query_count += 1
    return PreparedShardReader(aliases=aliases, files=files, query_count=query_count)


def _pool_completion_path(pool: Path, source: str) -> Path:
    return Path(pool) / "metadata" / source / "_COMPLETE.json"


def completed_source_status(
    spec: SourceSpec,
    run_root: Path,
    *,
    pool_validator: Callable[..., Any] | None = None,
) -> dict[str, Any] | None:
    """Return a validated completion receipt, or ``None`` when absent.

    A present but invalid receipt fails closed.  This keeps a historical
    failure visible and prevents the launcher from replacing an artifact whose
    identity has not been established.
    """

    root = source_root(run_root, spec.name)
    receipt_path = root / spec.receipt_name
    if not receipt_path.exists():
        return None
    value = read_json(receipt_path)
    if value.get("source") != spec.name or value.get("task") != spec.name:
        raise ValueError(f"completed source receipt identity mismatch: {receipt_path}")
    ranking = Path(value.get("ranking", ""))
    if ranking != root / spec.ranking_name or not ranking.is_file():
        raise ValueError(f"completed source ranking path mismatch: {receipt_path}")
    expected_ranking_sha = value.get("ranking_sha256")
    if not isinstance(expected_ranking_sha, str) or sha256_file(ranking) != expected_ranking_sha:
        raise ValueError(f"completed source ranking changed: {ranking}")
    pools = value.get("pools")
    if not isinstance(pools, list) or len(pools) != len(POOL_CAPACITIES):
        raise ValueError(f"completed source pool receipt is incomplete: {receipt_path}")
    validator = pool_validator
    if validator is None:
        from dci_bench.data.document_pools import verify_document_pool_suite

        validator = verify_document_pool_suite
    checked: list[dict[str, Any]] = []
    by_capacity = {row.get("capacity"): row for row in pools if isinstance(row, Mapping)}
    for capacity in POOL_CAPACITIES:
        row = by_capacity.get(capacity)
        if row is None:
            raise ValueError(f"completed source is missing pool capacity {capacity}: {receipt_path}")
        target = Path(row.get("path", ""))
        if target != root / (spec.pool_names[0] if capacity == 200 else spec.pool_names[1]):
            raise ValueError(f"completed source pool path mismatch: {target}")
        completion = _pool_completion_path(target, spec.name)
        expected = row.get("completion_sha256")
        if not completion.is_file() or not isinstance(expected, str) or sha256_file(completion) != expected:
            raise ValueError(f"completed source pool changed: {target}")
        validator(target, task=spec.name, verify_content_hashes=False)
        checked.append({"capacity": capacity, "path": str(target), "completion_sha256": expected})
    return {"status": "complete", "receipt": str(receipt_path), "ranking": str(ranking), "pools": checked}


def _validate_canonical_shards(
    entry: Mapping[str, Any],
    texts: Sequence[str] | None = None,
    *,
    shard_loader: Callable[[int], Sequence[str]] | None = None,
    total_documents: int | None = None,
) -> list[int]:
    if shard_loader is None:
        if texts is None:
            raise ValueError("canonical shard validation requires texts or a shard loader")
        total_documents = len(texts)
    elif total_documents is None:
        raise ValueError("canonical shard validation requires total_documents with a shard loader")
    assert total_documents is not None
    cache = _cache_for(entry)
    total = (total_documents + SHARD_SIZE - 1) // SHARD_SIZE
    complete: list[int] = []
    for index in range(total):
        # Missing paths do not need their source text read.  Existing paths
        # are always opened and identity/hash checked before reuse.
        if not cache._path("document", index).exists():
            continue
        if shard_loader is None:
            assert texts is not None
            start, stop = index * SHARD_SIZE, min((index + 1) * SHARD_SIZE, len(texts))
            shard_texts = texts[start:stop]
        else:
            shard_texts = shard_loader(index)
        if cache.get_shard(shard_texts, kind="document", shard_index=index) is not None:
            complete.append(index)
    return complete


def partition_missing_shards(
    sources: Sequence[Mapping[str, Any]], *, worker_count: int = GPU_COUNT,
) -> dict[int, list[dict[str, Any]]]:
    """Assign every missing shard to exactly one stable worker.

    Ownership is based on source order and global shard index, so completion
    of an earlier shard cannot move a later shard to another worker on resume.
    """

    if worker_count != GPU_COUNT:
        raise ValueError("gapfill launcher is contracted for exactly four workers")
    workers = {index: [] for index in range(worker_count)}
    for source_index, entry in enumerate(sources):
        if entry.get("status") == "complete":
            continue
        pending = sorted(int(value) for value in entry.get("missing_shards", []))
        for shard_index in pending:
            owner = (source_index + shard_index) % worker_count
            workers[owner].append({"source": entry["source"], "shard": shard_index})
    return workers


def build_plan(
    *,
    run_root: Path = RUN,
    source_names: Sequence[str] = DEFAULT_SOURCES,
    visible_devices: Sequence[str] | None = None,
    state_root: Path | None = None,
    model_path: str = MODEL_PATH,
) -> dict[str, Any]:
    """Build a read-only plan and validate existing canonical shards."""

    run_root = Path(run_root).absolute()
    names = list(source_names)
    if not names:
        raise ValueError("at least one source is required")
    if len(set(names)) != len(names) or any(name not in SOURCE_SPECS for name in names):
        raise ValueError(f"unknown or duplicate source: {names}")
    slots = list(visible_devices) if visible_devices is not None else _visible_gpu_slots()
    if len(slots) != GPU_COUNT or len(set(slots)) != GPU_COUNT:
        raise ValueError("the plan requires four distinct physical GPU slots")
    if not Path(model_path).is_dir():
        raise FileNotFoundError(f"local embedding model directory does not exist: {model_path}")
    state_root = Path(state_root or (run_root / ".gapfill_5090_4gpu_state")).absolute()
    entries: list[dict[str, Any]] = []
    for source in names:
        spec = SOURCE_SPECS[source]
        completed = completed_source_status(spec, run_root)
        base = {
            "source": source,
            "task": source,
            "batch_size": spec.batch_size,
            "model_path": model_path,
            # Keep the prepared suite root in the plan.  Worker/merge readers
            # derive its parent run root before adding the source name.
            "suite": str(source_suite(run_root)),
            "canonical_cache_dir": str(source_root(run_root, source) / "embedding-cache"),
            "cache_dir": str(source_root(run_root, source) / "embedding-cache"),
            "state_root": str(state_root),
            "shard_size": SHARD_SIZE,
            "max_length": MAX_LENGTH,
            "normalize": True,
            "query_prompt": QUERY_PROMPT,
            "document_prompt": DOCUMENT_PROMPT,
            "logical_device": "cuda:0",
        }
        if completed is not None:
            entries.append({**base, "status": "complete", "completed": completed,
                            "documents": None, "queries": None, "total_shards": 0,
                            "completed_shards": [], "missing_shards": [], "weight_identity": None})
            continue
        manifest = _validate_prepared_metadata(run_root, source)
        shard_reader = _prepared_shard_reader(run_root, source)
        document_count = shard_reader.document_count
        if document_count != manifest.get("num_docs") or shard_reader.query_count != manifest.get("num_queries"):
            raise ValueError(
                f"prepared source counts changed for {source}: "
                f"manifest=({manifest.get('num_docs')}, {manifest.get('num_queries')}), "
                f"inventory=({document_count}, {shard_reader.query_count})"
            )
        config = _embedding_config(spec, model_path=model_path)
        cache_dir = Path(base["cache_dir"])
        weight_identity = _model_weight_identity(model_path, cache_dir, config)
        entry = {**base,
                 "status": "pending",
                 "documents": document_count,
                 "queries": shard_reader.query_count,
                 "total_shards": (document_count + SHARD_SIZE - 1) // SHARD_SIZE,
                 "weight_identity": weight_identity}
        completed_shards = _validate_canonical_shards(
            entry, shard_loader=shard_reader.texts_for_shard, total_documents=document_count
        )
        completed_set = set(completed_shards)
        entry["completed_shards"] = completed_shards
        entry["missing_shards"] = [index for index in range(entry["total_shards"]) if index not in completed_set]
        entries.append(entry)
    plan = {
        "schema_version": SCHEMA,
        "mode": "check-only-plan",
        "run_root": str(run_root),
        "state_root": str(state_root),
        "source_names": names,
        "parameters": {
            "required_python": REQUIRED_PYTHON,
            "model_path": model_path,
            "max_length": MAX_LENGTH,
            "normalize": True,
            "query_prompt": QUERY_PROMPT,
            "document_prompt": DOCUMENT_PROMPT,
            "shard_size": SHARD_SIZE,
            "seed": SEED,
            "gpu_count": GPU_COUNT,
            "logical_device": "cuda:0",
            "output_dtype": "float32",
        },
        "visible_devices": slots,
        "sources": entries,
    }
    worker_tasks = partition_missing_shards(entries)
    plan["workers"] = [
        {
            "worker_index": index,
            "physical_gpu": slots[index],
            "logical_device": "cuda:0",
            "environment": {"CUDA_VISIBLE_DEVICES": slots[index]},
            "tasks": worker_tasks[index],
            "cache_root": str(state_root / "workers" / f"worker-{index:02d}"),
        }
        for index in range(GPU_COUNT)
    ]
    arguments = ["--run-root", str(run_root), "--state-root", str(state_root),
                 "--model-path", model_path]
    for source in names:
        arguments.extend(["--source", source])
    plan["controller_command"] = [REQUIRED_PYTHON, str(Path(__file__).resolve()), *arguments]
    plan["check_only_command"] = [*plan["controller_command"], "--check-only"]
    plan["shell_command"] = ["bash", str(ROOT / "scripts" / "launch_gapfill_5090_4gpu.sh"), *arguments]
    plan["boundary"] = (
        "CPU/data/env preflight only; no model load, CUDA tensor, encoding, ranking, pool publication, or formal benchmark"
    )
    plan["runtime_provenance"] = runtime_provenance(visible_devices=slots)
    return plan


def _encoder_for(spec: SourceSpec, entry: Mapping[str, Any], encoder_factory: Callable[..., Any] | None):
    if encoder_factory is None:
        from dci_bench.data.pipeline.retrieval import SentenceTransformerEncoder

        encoder_factory = SentenceTransformerEncoder
    encoder = encoder_factory(_embedding_config(spec, model_path=str(entry["model_path"])))
    expected = entry.get("weight_identity")
    if expected and hasattr(encoder, "_weight_identity"):
        encoder._weight_identity = expected
    return encoder


def encode_document_shards(
    entry: Mapping[str, Any], shard_indices: Sequence[int], *,
    encoder_factory: Callable[..., Any] | None = None,
    bundle_loader: Callable[..., dict[str, Any]] = _load_source_bundle,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Encode assigned document shards into one isolated worker cache.

    This function is intentionally dependency-injectable for CPU contract
    tests.  A real worker uses ``SentenceTransformerEncoder`` and the same
    identity-bound ``EmbeddingShardCache`` as ``rank_source``.
    """

    source = str(entry["source"])
    spec = SOURCE_SPECS[source]
    if bundle_loader is _load_source_bundle:
        shard_reader = _prepared_shard_reader(Path(entry["suite"]).parent, source)
        document_count = shard_reader.document_count
        shard_texts_for = shard_reader.texts_for_shard
    else:
        bundle = bundle_loader(Path(entry["suite"]).parent, source, verify_content_hashes=False)
        texts = [row["text"] for row in bundle["documents"]]
        document_count = len(texts)

        def shard_texts_for(index: int) -> Sequence[str]:
            start, stop = index * SHARD_SIZE, min((index + 1) * SHARD_SIZE, document_count)
            return texts[start:stop]

    cache_dir = Path(entry.get("worker_cache_dir") or entry["cache_dir"])
    worker_entry = {**entry, "cache_dir": str(cache_dir)}
    cache = _cache_for(worker_entry)
    wanted = sorted(set(int(index) for index in shard_indices))
    total = (document_count + SHARD_SIZE - 1) // SHARD_SIZE
    if any(index < 0 or index >= total for index in wanted):
        raise ValueError(f"worker assignment contains an invalid shard for {source}: {wanted}")
    reused: list[int] = []
    missing: list[int] = []
    for index in wanted:
        shard_texts = shard_texts_for(index)
        if cache.get_shard(shard_texts, kind="document", shard_index=index) is None:
            missing.append(index)
        else:
            reused.append(index)
    encoded: list[int] = []
    if missing:
        encoder = _encoder_for(spec, entry, encoder_factory)
        for index in missing:
            shard_texts = shard_texts_for(index)
            values = encoder.encode(shard_texts, kind="document")
            import numpy as np

            values = np.asarray(values, dtype="float32")
            if values.ndim != 2 or values.shape[0] != len(shard_texts) or not np.isfinite(values).all():
                raise ValueError(f"encoder returned an invalid float32 matrix for {source} shard {index}")
            audit = {
                "truncation": getattr(encoder, "last_metadata", {}).get("truncation", {}),
                "requested_batch_size": spec.batch_size,
                "effective_batch_size": int(getattr(encoder, "last_metadata", {}).get("effective_batch_size", spec.batch_size)),
                "oom_retries": int(getattr(encoder, "last_metadata", {}).get("oom_retries", 0)),
                "runtime": dict(runtime or {}),
                "logical_device": "cuda:0",
                "output_dtype": "float32",
            }
            cache.set_shard(shard_texts, values, kind="document", shard_index=index, audit=audit)
            encoded.append(index)
    return {"source": source, "cache_dir": str(cache_dir), "reused_shards": reused,
            "encoded_shards": encoded, "requested_shards": wanted}


def compare_embedding_vectors(
    expected: Any, actual: Any, *, max_abs_tol: float = SMOKE_MAX_ABS_TOL,
    min_cosine: float = SMOKE_MIN_COSINE,
) -> dict[str, Any]:
    """Compare normalized float32 vectors across runtime/GPU versions."""

    import numpy as np

    expected = np.asarray(expected, dtype="float32")
    actual = np.asarray(actual, dtype="float32")
    if expected.shape != actual.shape or expected.ndim != 2 or not expected.size:
        raise ValueError(f"smoke vector shape mismatch: expected={expected.shape}, actual={actual.shape}")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("smoke vectors contain non-finite values")
    deltas = np.abs(expected - actual)
    max_abs = float(deltas.max())
    expected_norm = np.linalg.norm(expected, axis=1)
    actual_norm = np.linalg.norm(actual, axis=1)
    if np.any(expected_norm == 0) or np.any(actual_norm == 0):
        raise ValueError("smoke vectors contain a zero norm")
    cosine = np.sum(expected * actual, axis=1) / (expected_norm * actual_norm)
    min_cosine_observed = float(cosine.min())
    result = {
        "rows": int(expected.shape[0]),
        "dimensions": int(expected.shape[1]),
        "max_abs_diff": max_abs,
        "min_cosine": min_cosine_observed,
        "max_abs_tolerance": max_abs_tol,
        "min_cosine_threshold": min_cosine,
        "passed": max_abs <= max_abs_tol and min_cosine_observed >= min_cosine,
    }
    if not result["passed"]:
        raise RuntimeError(
            "cached-vector compatibility smoke failed: "
            f"max_abs_diff={max_abs:.8g} (limit {max_abs_tol}), "
            f"min_cosine={min_cosine_observed:.8g} (limit {min_cosine}); "
            "preserving cached shards and refusing bulk encoding"
        )
    return result


def merge_document_shards(
    entry: Mapping[str, Any], worker_shard_indices: Mapping[int, Sequence[int]], *,
    runtime: Mapping[str, Any] | None = None,
    bundle_loader: Callable[..., dict[str, Any]] = _load_source_bundle,
) -> dict[str, Any]:
    """Verify worker artifacts and publish them into the canonical cache serially."""

    source = str(entry["source"])
    if bundle_loader is _load_source_bundle:
        shard_reader = _prepared_shard_reader(Path(entry["suite"]).parent, source)
        document_count = shard_reader.document_count
        shard_texts_for = shard_reader.texts_for_shard
    else:
        bundle = bundle_loader(Path(entry["suite"]).parent, source, verify_content_hashes=False)
        texts = [row["text"] for row in bundle["documents"]]
        document_count = len(texts)

        def shard_texts_for(index: int) -> Sequence[str]:
            start, stop = index * SHARD_SIZE, min((index + 1) * SHARD_SIZE, document_count)
            return texts[start:stop]

    canonical_entry = {**entry, "cache_dir": entry["canonical_cache_dir"]}
    canonical = _cache_for(canonical_entry)
    merged: list[int] = []
    reused: list[int] = []
    for worker_index in sorted(worker_shard_indices):
        worker_cache_dir = Path(entry["state_root"]) / "workers" / f"worker-{worker_index:02d}" / source / "embedding-cache"
        worker_entry = {**entry, "cache_dir": str(worker_cache_dir)}
        worker_cache = _cache_for(worker_entry)
        for index in sorted(set(int(value) for value in worker_shard_indices[worker_index])):
            print(f"merge {source}: worker={worker_index} shard={index} checking", flush=True)
            shard_texts = shard_texts_for(index)
            existing = canonical.get_shard(shard_texts, kind="document", shard_index=index, with_metadata=True)
            if existing[0] is not None:
                reused.append(index)
                continue
            vectors, metadata = worker_cache.get_shard(shard_texts, kind="document", shard_index=index, with_metadata=True)
            if vectors is None or metadata is None:
                raise FileNotFoundError(f"worker {worker_index} did not produce {source} shard {index}")
            audit = dict(metadata.get("audit", {})) if isinstance(metadata.get("audit"), Mapping) else {}
            audit["merged_at"] = now()
            audit["merge_runtime"] = dict(runtime or {})
            canonical.set_shard(shard_texts, vectors, kind="document", shard_index=index, audit=audit)
            merged.append(index)
            print(f"merge {source}: shard={index} published", flush=True)
    # Final verification catches a missing assignment before rank_source can
    # fall through to model inference for an unprepared shard.
    all_indices = range((document_count + SHARD_SIZE - 1) // SHARD_SIZE)
    for index in all_indices:
        if canonical.get_shard(shard_texts_for(index), kind="document", shard_index=index) is None:
            raise ValueError(f"canonical cache is incomplete after merge: {source} shard {index}")
    return {"source": source, "merged_shards": sorted(merged), "reused_shards": sorted(reused),
            "total_shards": len(list(all_indices)), "cache_dir": str(entry["canonical_cache_dir"])}


def recover_completed_workers(plan_path: Path) -> dict[str, Any]:
    """Resume an interrupted merge on CPU, preserving the original attempt."""

    plan_path = Path(plan_path).absolute()
    plan = read_json(plan_path)
    if plan.get("schema_version") != SCHEMA or plan.get("mode") != "full":
        raise ValueError("merge-only requires a saved full-mode plan")
    expected = stable_digest({key: value for key, value in plan.items() if key != "plan_sha256"})
    if plan.get("plan_sha256") != expected:
        raise ValueError(f"saved worker plan hash mismatch: {plan_path}")
    worker_receipts = []
    for worker in plan["workers"]:
        index = int(worker["worker_index"])
        path = plan_path.parent / f"worker-{index:02d}.json"
        receipt = read_json(path)
        if receipt.get("worker_index") != index or receipt.get("tasks") != worker["tasks"]:
            raise ValueError(f"worker receipt does not match saved assignments: {path}")
        worker_receipts.append({"path": str(path), "sha256": sha256_file(path)})
    pending = [row for row in plan["sources"] if row.get("status") != "complete"]
    for entry in pending:
        manifest = _validate_prepared_metadata(Path(plan["run_root"]), entry["source"])
        if (manifest["num_docs"], manifest["num_queries"]) != (entry["documents"], entry["queries"]):
            raise ValueError(f"source counts changed since worker plan: {entry['source']}")
    attempt = Path(plan["state_root"]) / "attempts" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-merge-" + uuid.uuid4().hex[:8]
    )
    attempt.mkdir(parents=True, exist_ok=False)
    provenance = {
        "at": now(), "mode": "merge-only", "original_plan": str(plan_path),
        "original_plan_sha256": sha256_file(plan_path), "worker_receipts": worker_receipts,
        "inference_executed": False, "runtime": runtime_provenance(visible_devices=[]),
    }
    write_json_new(attempt / "recovery.json", provenance)
    print(f"CPU merge recovery: {attempt}", flush=True)
    rows = []
    for entry in pending:
        assignments = {
            int(worker["worker_index"]): [int(task["shard"]) for task in worker["tasks"]
                                          if task["source"] == entry["source"]]
            for worker in plan["workers"]
        }
        rows.append(merge_document_shards(entry, assignments, runtime=provenance))
    result = {
        "status": "document_cache_complete", "sources": rows, "attempt_root": str(attempt),
        "inference_executed": False,
        "boundary": "document cache merge only; rankings, query encoding and candidate pools not executed",
    }
    write_json_new(attempt / "merge.json", result)
    write_json_new(attempt / "_MERGE_COMPLETE.json", result)
    return result


def _gpu_preflight(visible_devices: Sequence[str]) -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != GPU_COUNT:
        count = torch.cuda.device_count()
        raise RuntimeError(f"full mode requires exactly four visible CUDA GPUs; found {count}")
    info: list[dict[str, Any]] = []
    for index in range(GPU_COUNT):
        props = torch.cuda.get_device_properties(index)
        capability = (int(props.major), int(props.minor))
        if "5090" not in str(props.name) or capability != (12, 0):
            raise RuntimeError(
                f"GPU {index} must be RTX 5090 sm120; got name={props.name!r}, capability={capability}"
            )
        value = (torch.ones(2, device=f"cuda:{index}") + 1).sum().item()
        if value != 4:
            raise RuntimeError(f"CUDA tensor preflight failed on logical GPU {index}")
        info.append({"logical_index": index, "name": str(props.name),
                     "capability": [capability[0], capability[1]],
                     "total_memory": int(props.total_memory),
                     "physical_slot": visible_devices[index]})
    return info


def _smoke_source(plan: Mapping[str, Any], source: str, *, encoder_factory: Callable[..., Any] | None = None) -> dict[str, Any]:
    entry = next(row for row in plan["sources"] if row["source"] == source)
    shard_reader = _prepared_shard_reader(Path(plan["run_root"]), source)
    texts = shard_reader.texts_for_shard(0)[:2]
    if not texts:
        raise ValueError(f"cannot smoke empty source: {source}")
    encoder = _encoder_for(SOURCE_SPECS[source], entry, encoder_factory)
    import numpy as np

    cache = _cache_for({**entry, "cache_dir": entry["canonical_cache_dir"]})
    compared = None
    compared_shard = None
    cached_runtime = None
    for shard_index in sorted(int(value) for value in entry.get("completed_shards", [])):
        shard_texts = shard_reader.texts_for_shard(shard_index)
        saved, metadata = cache.get_shard(
            shard_texts, kind="document",
            shard_index=shard_index, with_metadata=True,
        )
        if saved is None or metadata is None:
            continue
        audit = metadata.get("audit") if isinstance(metadata, Mapping) else None
        if isinstance(audit, Mapping) and isinstance(audit.get("runtime"), Mapping):
            cached_runtime = dict(audit["runtime"])
        sample_count = min(2, len(shard_texts))
        fresh = encoder.encode(shard_texts[:sample_count], kind="document")
        fresh = np.asarray(fresh, dtype="float32")
        compared = compare_embedding_vectors(saved[:sample_count], fresh)
        compared_shard = shard_index
        break
    if compared is None:
        fresh = encoder.encode(texts, kind="document")
        fresh = np.asarray(fresh, dtype="float32")
        if fresh.ndim != 2 or fresh.shape[0] != len(texts) or not np.isfinite(fresh).all():
            raise ValueError(f"smoke encoder returned invalid output for {source}")
    else:
        fresh = np.asarray(fresh, dtype="float32")
    return {
        "source": source,
        "documents": int(fresh.shape[0]),
        "shape": list(fresh.shape),
        "dtype": str(fresh.dtype),
        "inference_executed": True,
        "device": "cuda:0",
        "cached_vector_comparison": compared or {
            "status": "not_applicable",
            "reason": "no completed canonical document shard was available for comparison",
        },
        "cached_vector_runtime_provenance": cached_runtime or {
            "status": "unrecorded",
            "reason": "legacy cached shard did not record runtime provenance",
        },
        "cached_shard": compared_shard,
        "runtime_provenance": runtime_provenance(visible_devices=[os.environ.get("CUDA_VISIBLE_DEVICES", "unknown")]),
    }


def _worker_command(plan_path: Path, worker_index: int) -> list[str]:
    return [REQUIRED_PYTHON, str(Path(__file__).resolve()), "--worker",
            "--plan", str(plan_path), "--worker-index", str(worker_index)]


def _run_worker(plan_path: Path, worker_index: int) -> dict[str, Any]:
    plan = read_json(plan_path)
    workers = plan["workers"]
    worker = next(row for row in workers if int(row["worker_index"]) == worker_index)
    physical = str(worker["physical_gpu"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != physical:
        raise RuntimeError(
            f"worker {worker_index} must see exactly one physical GPU; "
            f"got CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}, expected {physical!r}"
        )
    runtime = runtime_provenance(visible_devices=[physical])
    results: list[dict[str, Any]] = []
    by_source = {row["source"]: row for row in plan["sources"]}
    for source in sorted({task["source"] for task in worker["tasks"]}):
        entry = by_source[source]
        indices = [int(task["shard"]) for task in worker["tasks"] if task["source"] == source]
        entry = {**entry, "worker_cache_dir": str(Path(worker["cache_root"]) / source / "embedding-cache")}
        results.append(encode_document_shards(entry, indices, runtime=runtime))
    receipt = {"schema_version": SCHEMA, "worker_index": worker_index,
               "physical_gpu": physical, "logical_device": "cuda:0",
               "tasks": worker["tasks"], "results": results,
               "runtime_provenance": runtime, "at": now()}
    receipt_path = plan_path.parent / f"worker-{worker_index:02d}.json"
    write_json_new(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
    return receipt


def _launch_workers(plan: Mapping[str, Any], plan_path: Path) -> dict[int, dict[str, Any]]:
    attempt_dir = plan_path.parent
    logs = attempt_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes: dict[int, tuple[subprocess.Popen[Any], Any, str]] = {}
    for worker in plan["workers"]:
        index = int(worker["worker_index"])
        log_path = logs / f"worker-{index:02d}-gpu-{worker['physical_gpu']}.log"
        log_handle = log_path.open("x", encoding="utf-8")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(worker["physical_gpu"])
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        process = subprocess.Popen(_worker_command(plan_path, index), env=env,
                                   stdout=log_handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        processes[index] = (process, log_handle, str(log_path))
    failures: dict[int, str] = {}
    receipts: dict[int, dict[str, Any]] = {}
    try:
        for index, (process, log_handle, log_path) in processes.items():
            code = process.wait()
            log_handle.close()
            if code != 0:
                failures[index] = f"exit_code={code}, log={log_path}"
            else:
                receipt_path = plan_path.parent / f"worker-{index:02d}.json"
                if not receipt_path.is_file():
                    failures[index] = f"missing worker receipt, log={log_path}"
                else:
                    receipts[index] = read_json(receipt_path)
    except BaseException:
        for process, log_handle, _ in processes.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process, log_handle, _ in processes.values():
            if process.poll() is None:
                process.wait()
            if not log_handle.closed:
                log_handle.close()
        raise
    if failures:
        raise RuntimeError(f"worker failures preserved in {attempt_dir}: {failures}")
    return receipts


def _finalize_source(plan: Mapping[str, Any], entry: Mapping[str, Any], *, runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Run existing ranking and pool publication after canonical caches close."""

    started = time.monotonic()
    from dci_bench.data.pipeline.stages import rank_source
    from dci_bench.data.document_pools import prepare_document_pools, verify_document_pool_suite

    source = str(entry["source"])
    spec = SOURCE_SPECS[source]
    run_root = Path(plan["run_root"])
    root = source_root(run_root, source)
    root.mkdir(parents=True, exist_ok=True)
    ranking = root / spec.ranking_name
    raw = root / spec.raw_ranking_name
    if spec.exclude_self:
        if not ranking.exists():
            if not raw.exists():
                rank_source(raw, suite=source_suite(run_root), task=source,
                            cache_dir=Path(entry["canonical_cache_dir"]), device="cuda:0",
                            model_path=str(entry["model_path"]),
                            batch_size=spec.batch_size, max_length=MAX_LENGTH,
                            top_k=1001, query_prompt=QUERY_PROMPT, document_prompt=DOCUMENT_PROMPT,
                            cache_shard_size=SHARD_SIZE, search_device="cuda:0",
                            search_query_batch_size=32)
            from scripts.prepare_gapfill_20261031 import filter_query_self_matches

            metadata = read_json(_metadata_dir(run_root, source) / "host_labels.json")["query_metadata"]
            self_documents = {qid: row["self_match_document_ids"][0] for qid, row in metadata.items()}
            filter_query_self_matches(
                raw, ranking, _metadata_dir(run_root, source) / "source_doc_ids.json",
                self_documents=self_documents,
                qrels=read_json(_metadata_dir(run_root, source) / "qrels.json"),
            )
    elif not ranking.exists():
        rank_source(ranking, suite=source_suite(run_root), task=source,
                    cache_dir=Path(entry["canonical_cache_dir"]), device="cuda:0",
                    model_path=str(entry["model_path"]),
                    batch_size=spec.batch_size, max_length=MAX_LENGTH, top_k=1000,
                    query_prompt=QUERY_PROMPT, document_prompt=DOCUMENT_PROMPT,
                    cache_shard_size=SHARD_SIZE, search_device="cuda:0",
                    search_query_batch_size=32)
    if not ranking.is_file():
        raise ValueError(f"rank_source did not publish ranking: {ranking}")
    pool_rows: list[dict[str, Any]] = []
    for capacity, pool_name in zip(POOL_CAPACITIES, spec.pool_names):
        target = root / pool_name
        if target.exists():
            verify_document_pool_suite(target, task=source)
        else:
            prepare_document_pools(source_suite(run_root), target, source,
                                   capacity=capacity, seed=SEED, rankings_path=ranking)
        completion = _pool_completion_path(target, source)
        if not completion.is_file():
            raise ValueError(f"pool publication missing completion marker: {target}")
        pool_rows.append({"capacity": capacity, "path": str(target),
                          "completion_sha256": sha256_file(completion)})
    manifest = read_json(_metadata_dir(run_root, source) / "manifest.json")
    receipt = {
        "content_hashes_verified": True,
        "documents": manifest["num_docs"],
        "elapsed_seconds": time.monotonic() - started,
        "export": str(source_suite(run_root)),
        "pools": pool_rows,
        "queries": manifest["num_queries"],
        "ranking": str(ranking),
        "ranking_sha256": sha256_file(ranking),
        "source": source,
        "task": source,
        "runtime_provenance": dict(runtime),
        "cache_identity_runtime_bound": False,
        "worker_plan_sha256": plan.get("plan_sha256"),
    }
    receipt_path = root / spec.receipt_name
    if receipt_path.exists():
        if read_json(receipt_path) != receipt:
            raise ValueError(f"refusing to replace existing source receipt: {receipt_path}")
    else:
        write_json_new(receipt_path, receipt)
    return receipt


def run_full(plan: dict[str, Any], *, visible_devices: Sequence[str]) -> dict[str, Any]:
    _validate_python_environment()
    gpu_info = _gpu_preflight(visible_devices)
    pending = [row for row in plan["sources"] if row.get("status") != "complete"]
    needs_encoding = [row for row in pending if row.get("missing_shards")]
    needs_ranking_smoke = []
    for row in pending:
        spec = SOURCE_SPECS[str(row["source"])]
        root = source_root(Path(plan["run_root"]), str(row["source"]))
        if spec.exclude_self:
            requires_rank_source = not (root / spec.ranking_name).is_file() and not (root / spec.raw_ranking_name).is_file()
        else:
            requires_rank_source = not (root / spec.ranking_name).is_file()
        if requires_rank_source:
            needs_ranking_smoke.append(row)
    if not pending:
        return {"status": "complete", "skipped": [row["source"] for row in plan["sources"]],
                "boundary": "existing source receipts validated; no model work required",
                "runtime_provenance": runtime_provenance(visible_devices=visible_devices, gpu_info=gpu_info)}
    state_root = Path(plan["state_root"])
    attempt = state_root / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True, exist_ok=False)
    full_plan = {**plan, "mode": "full", "attempt_root": str(attempt),
                 "runtime_provenance": runtime_provenance(visible_devices=visible_devices, gpu_info=gpu_info)}
    full_plan["plan_sha256"] = stable_digest({key: value for key, value in full_plan.items() if key != "plan_sha256"})
    plan_path = attempt / "plan.json"
    write_json_new(plan_path, full_plan)
    write_json_new(attempt / "preflight.json", {"gpu_info": gpu_info,
                                                 "runtime_provenance": full_plan["runtime_provenance"],
                                                 "at": now()})
    receipts: dict[int, dict[str, Any]] = {}
    final_runtime = dict(full_plan["runtime_provenance"])
    final_runtime["cached_vector_compatibility"] = {
        "status": "not_run",
        "reason": "no missing document shards required a compatibility smoke",
    }
    smoke_sources = needs_encoding or needs_ranking_smoke
    if smoke_sources:
        # Smoke exactly one real model encode before any expensive encoding or ranking.
        smoke_source = smoke_sources[0]["source"]
        smoke_log = (attempt / "smoke.log").open("x", encoding="utf-8")
        smoke_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(visible_devices[0]),
                         HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        smoke_command = [REQUIRED_PYTHON, str(Path(__file__).resolve()), "--smoke",
                         "--plan", str(plan_path), "--smoke-source", smoke_source]
        try:
            smoke = subprocess.run(smoke_command, env=smoke_env, stdout=smoke_log,
                                   stderr=subprocess.STDOUT, check=False)
        finally:
            smoke_log.close()
        if smoke.returncode != 0:
            raise RuntimeError(f"real encoder smoke failed; inspect {attempt / 'smoke.log'}")
        smoke_lines = [line for line in (attempt / "smoke.log").read_text(encoding="utf-8").splitlines() if line.strip()]
        try:
            smoke_result = json.loads(smoke_lines[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"real encoder smoke produced no JSON receipt; inspect {attempt / 'smoke.log'}") from exc
        smoke_value = {"source": smoke_source, "command": smoke_command,
                       "physical_gpu": str(visible_devices[0]), "log": str(attempt / "smoke.log"),
                       "result": smoke_result}
        write_json_new(attempt / "smoke.json", smoke_value)
        final_runtime["cached_vector_compatibility"] = {
            "comparison": smoke_result.get(
                "cached_vector_comparison",
                {"status": "missing", "reason": "smoke receipt omitted comparison result"},
            ),
            "cached_runtime_provenance": smoke_result.get(
                "cached_vector_runtime_provenance",
                {"status": "missing", "reason": "smoke receipt omitted cached runtime provenance"},
            ),
        }
        if needs_encoding:
            receipts = _launch_workers(full_plan, plan_path)
    merge_rows: list[dict[str, Any]] = []
    for entry in pending:
        assignments = {
            int(worker["worker_index"]): [int(task["shard"]) for task in worker["tasks"]
                                           if task["source"] == entry["source"]]
            for worker in full_plan["workers"]
        }
        merge_rows.append(merge_document_shards(entry, assignments,
                                                runtime=full_plan["runtime_provenance"]))
    write_json_new(attempt / "merge.json", {"sources": merge_rows, "at": now()})
    final_rows: list[dict[str, Any]] = []
    for entry in pending:
        final_rows.append(_finalize_source(full_plan, entry, runtime=final_runtime))
    result = {"status": "complete", "sources": final_rows,
              "skipped": [row["source"] for row in plan["sources"] if row.get("status") == "complete"],
              "attempt_root": str(attempt), "worker_receipts": receipts,
              "boundary": "KaLM document rankings and 200/1000 candidate pools completed; no formal benchmark inference",
              "runtime_provenance": final_runtime}
    write_json_new(attempt / "_COMPLETE.json", result)
    return result


def _print_plan(plan: Mapping[str, Any]) -> None:
    output = dict(plan)
    output["worker_commands"] = [
        {"worker_index": worker["worker_index"],
         "environment": worker["environment"],
         "command": _worker_command(Path(plan["state_root"]) / "<attempt>/plan.json",
                                     int(worker["worker_index"]))}
        for worker in plan["workers"]
    ]
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="CPU/data/env preflight and deterministic plan; no output changes")
    parser.add_argument("--merge-only", action="store_true",
                        help="recover completed workers from --plan on CPU; no model load or ranking")
    parser.add_argument("--run-root", type=Path, default=RUN)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--source", "--sources", dest="sources", action="append",
                        choices=sorted(SOURCE_SPECS),
                        help="source to process; repeat for multiple sources (default: BrowseComp+ and CEDAR)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path, help="saved full-mode plan for --merge-only")
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-source", dest="smoke_source", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_python_environment()
    if args.merge_only:
        if args.plan is None or args.worker or args.smoke or args.check_only or os.environ.get("DRY_RUN") == "1":
            raise ValueError("--merge-only requires --plan and cannot be combined with worker, smoke or dry-run mode")
        result = recover_completed_workers(args.plan)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
        return 0
    if args.worker:
        if args.plan is None or args.worker_index is None:
            raise ValueError("--worker requires --plan and --worker-index")
        _run_worker(args.plan, args.worker_index)
        return 0
    if args.smoke:
        if args.plan is None:
            raise ValueError("--smoke requires --plan")
        plan = read_json(args.plan)
        source = args.smoke_source or next(row["source"] for row in plan["sources"] if row.get("status") != "complete")
        value = _smoke_source(plan, source)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    source_names = tuple(args.sources or DEFAULT_SOURCES)
    slots = _visible_gpu_slots()
    plan = build_plan(run_root=args.run_root, source_names=source_names,
                      visible_devices=slots, state_root=args.state_root,
                      model_path=args.model_path)
    dry_run = args.check_only or os.environ.get("DRY_RUN") == "1"
    if dry_run:
        _print_plan(plan)
        return 0
    result = run_full(plan, visible_devices=slots)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, BrokenPipeError):
        raise SystemExit(130)
    except Exception as exc:
        print(f"gapfill launcher failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1)
