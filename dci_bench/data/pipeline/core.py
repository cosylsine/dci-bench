"""Immutable stages and the bridge into the existing DCI document contract.

All queries, evidence, judgments, and provenance live outside agent workspaces.
A completed corpus preparation stage does not imply that queries were generated,
labels were reviewed, embeddings were computed, or a model was evaluated.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import stat
import uuid

from . import BUNDLE_VERSION, PIPELINE_VERSION

DEFAULT_BASE = Path("/mnt/afs2/202609/embedding_models/dci-bench")
DEFAULT_RAW_ROOT = DEFAULT_BASE / "raw_data"
DEFAULT_EVAL_ROOT = DEFAULT_BASE / "eval_data"
_SLUG = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,119}$")
_FILES = {
    "documents": "documents.jsonl", "queries": "queries.jsonl",
    "evidence": "evidence.jsonl", "judgments": "judgments.jsonl",
}


def _meaningful_stage(value) -> bool:
    """Return whether a provenance stage contains an actual record."""
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    return bool(value)


def _model_stage(provenance: Mapping, stage: str):
    """Read model provenance from both the canonical and legacy spellings.

    The wrappers use ``provenance.models.generate`` and ``.judge``.  The
    model adapter predating those wrappers wrote ``query_generation`` and
    ``judge``; retaining those two aliases keeps already-created bundles
    auditable without treating arbitrary metadata as model provenance.
    """
    models = provenance.get("models")
    if isinstance(models, Mapping) and _meaningful_stage(models.get(stage)):
        return models[stage]
    dotted = provenance.get(f"models.{stage}")
    if _meaningful_stage(dotted):
        return dotted
    aliases = {
        "generate": ("query_generation", "generation", "generator"),
        "judge": ("judge",),
    }
    for key in aliases[stage]:
        value = provenance.get(key)
        if _meaningful_stage(value):
            return value
    return None


def _model_marker(value) -> bool:
    if not isinstance(value, str):
        return False
    value = value.lower()
    return any(token in value for token in ("llm", "model", "automatic", "generated", "judge"))


def _row_model_marker(row) -> bool:
    if not isinstance(row, Mapping):
        return False
    for key in ("source", "label", "label_source", "annotation_source", "judgment_source",
                "generated_by", "qrels_label"):
        if _model_marker(row.get(key)):
            return True
    metadata = row.get("metadata")
    return isinstance(metadata, Mapping) and _row_model_marker(metadata)


def _label_provenance(bundle: Mapping) -> dict:
    """Classify model-labelled data from structured provenance and rows.

    A free-form ``label_source=model`` marker is treated conservatively as an
    automatic-label claim: it cannot make a release ready without an actual
    judge stage.  It is still surfaced so export cannot silently treat it as
    human annotation.
    """
    provenance = bundle.get("provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    generate = _model_stage(provenance, "generate")
    judge = _model_stage(provenance, "judge")
    models = provenance.get("models")
    incomplete_stage_claim = (isinstance(models, Mapping) and
                              any(stage in models for stage in ("generate", "judge")))
    incomplete_stage_claim = incomplete_stage_claim or any(
        f"models.{stage}" in provenance for stage in ("generate", "judge")
    )
    marker = any(_model_marker(provenance.get(key)) for key in
                 ("label_source", "annotation_source", "judgment_source", "qrels_label"))
    marker = marker or any(_row_model_marker(row) for row in bundle.get("queries", [])
                           if isinstance(row, Mapping))
    marker = marker or any(_row_model_marker(row) for row in bundle.get("judgments", [])
                           if isinstance(row, Mapping))
    marker = marker or _meaningful_stage(provenance.get("generation")) or incomplete_stage_claim
    return {"automatic": bool(generate or judge or marker),
            "generate": _meaningful_stage(generate),
            "judge": _meaningful_stage(judge),
            "generate_record": generate, "judge_record": judge}


def _judgment_quotes(row: Mapping) -> list[str]:
    values = row.get("quotes", row.get("quote"))
    if isinstance(values, str):
        return [values]
    if isinstance(values, list):
        return [value for value in values if isinstance(value, str)]
    return []


def _model_positive_issues(bundle: Mapping) -> list[dict]:
    """Check generated/model qrels against judged, quoted positive rows.

    Human qrels on otherwise human queries are intentionally not included in
    this check.  Generated queries and qrels carrying model markers are the
    model-derived surface whose positives need a reviewed judgment.
    """
    documents = {row.get("_id"): row for row in bundle.get("documents", [])
                 if isinstance(row, Mapping)}
    queries = {row.get("_id"): row for row in bundle.get("queries", [])
               if isinstance(row, Mapping)}
    generated_queries = set()
    for query_id, row in queries.items():
        if _row_model_marker(row):
            generated_queries.add(query_id)
    judgments = {}
    model_judgment_pairs = set()
    for row in bundle.get("judgments", []):
        if not isinstance(row, Mapping):
            continue
        pair = (row.get("query_id"), row.get("doc_id"))
        judgments.setdefault(pair, []).append(row)
        if _row_model_marker(row):
            model_judgment_pairs.add(pair)
    qrels = bundle.get("qrels", {})
    issues = []
    model_queries = generated_queries | {query_id for query_id, _ in model_judgment_pairs}
    for query_id in sorted(model_queries, key=lambda value: str(value)):
        gold = qrels.get(query_id, {}) if isinstance(qrels, Mapping) else {}
        if not isinstance(gold, Mapping):
            continue
        if query_id in generated_queries:
            candidate_doc_ids = set(gold)
        else:
            # Existing human qrels on a query remain valid.  Only pairs that
            # carry a model judgment are model-derived unless the query itself
            # was generated by the model.
            candidate_doc_ids = {doc_id for pair_query, doc_id in model_judgment_pairs
                                 if pair_query == query_id and doc_id in gold}
        for doc_id in sorted(candidate_doc_ids, key=lambda value: str(value)):
            rows = judgments.get((query_id, doc_id), [])
            valid = False
            for row in rows:
                score = row.get("score")
                if (row.get("status") != "judged" or isinstance(score, bool) or
                        not isinstance(score, (int, float)) or not math.isfinite(score) or score <= 0):
                    continue
                document = documents.get(doc_id)
                text = document.get("text") if isinstance(document, Mapping) else None
                quotes = _judgment_quotes(row)
                if (isinstance(text, str) and quotes and
                        all(quote and text.count(quote) == 1 for quote in quotes)):
                    valid = True
                    break
            if not valid:
                issues.append({"query_id": query_id, "doc_id": doc_id,
                               "reason": "model_positive_missing_judged_positive_exact_quote"})
    return issues


def _release_label_state(bundle: Mapping, audit: Mapping) -> tuple[str, dict]:
    labels = _label_provenance(bundle)
    issues = _model_positive_issues(bundle) if labels["automatic"] else []
    review_queue = bundle.get("review_queue")
    has_queries = bool(bundle.get("queries"))
    qrels = bundle.get("qrels", {})
    has_qrels = bool(qrels) and all(qrels.get(q.get("_id")) for q in bundle.get("queries", []))
    if not has_queries or audit.get("status") == "needs_queries" or not has_qrels:
        state = "corpus_ready"
    elif audit.get("status") == "review_required" or review_queue:
        state = "review_required"
    elif labels["automatic"] and (not labels["judge"] or issues):
        state = "review_required"
    else:
        state = "labels_ready"
    return state, {**labels, "model_positive_issues": issues}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path):
    def unique_pairs(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"Duplicate JSON key: {key}")
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Non-finite JSON: {x}")))


def write_json(path: Path, value) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def slug(value: str) -> str:
    if not isinstance(value, str) or not _SLUG.fullmatch(value):
        raise ValueError(f"Unsafe task/source ID: {value!r}")
    return value


def check_roots(raw_root: Path, eval_root: Path) -> None:
    if (isinstance(raw_root, str) and not raw_root.strip()) or (isinstance(eval_root, str) and not eval_root.strip()):
        raise ValueError("raw_root and eval_root must not be empty")
    try:
        raw_path, clean_path = Path(raw_root), Path(eval_root)
    except (TypeError, ValueError) as exc:
        raise ValueError("raw_root and eval_root must be filesystem paths") from exc
    raw, clean = raw_path.resolve(), clean_path.resolve()
    if raw == clean or raw in clean.parents or clean in raw.parents:
        raise ValueError("raw_root and eval_root must be disjoint directories")


def load_config(path: Path) -> dict:
    config = read_json(path)
    if not isinstance(config, dict) or not isinstance(config.get("sources"), list):
        raise ValueError("Config must contain a sources list")
    seen = set()
    for source in config["sources"]:
        if not isinstance(source, dict):
            raise ValueError("Source specifications must be objects")
        name = slug(source.get("id"))
        if name in seen:
            raise ValueError(f"Duplicate source ID: {name}")
        seen.add(name)
        if source.get("input_kind") not in ("retrieval", "evidence_qa", "corpus"):
            raise ValueError(f"Invalid input_kind for {name}")
        if not isinstance(source.get("connector"), str) or not source["connector"]:
            raise ValueError(f"Missing connector for {name}")
        from .connectors import CONNECTOR_NAMES
        if source["connector"] not in CONNECTOR_NAMES:
            raise ValueError(f"Unknown connector for {name}: {source['connector']!r}")
        if "options" in source and not isinstance(source["options"], dict):
            raise ValueError(f"Source options must be an object for {name}")
    raw_root = config.get("raw_root", DEFAULT_RAW_ROOT)
    eval_root = config.get("eval_root", DEFAULT_EVAL_ROOT)
    check_roots(raw_root, eval_root)
    return config


def code_identity() -> dict:
    return {p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def select_queries(bundle: dict, *, limit: int | None = None, seed: int = 20261022) -> dict:
    """Select queries without reading qrels, scores, or document contents."""
    if limit is None:
        return copy.deepcopy(bundle)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("max_queries must be a positive integer")
    result = copy.deepcopy(bundle)
    rows = result.get("queries", [])
    selected = sorted(rows, key=lambda q: digest([seed, bundle["source"], q["_id"]]))[:limit]
    keep = {q["_id"] for q in selected}
    result["queries"] = sorted(selected, key=lambda q: q["_id"])
    result["qrels"] = {q: g for q, g in result.get("qrels", {}).items() if q in keep}
    for key in ("evidence", "judgments"):
        result[key] = [r for r in result.get(key, []) if r.get("query_id") in keep]
    result.setdefault("provenance", {})["query_selection"] = {
        "algorithm": "sha256(seed,source,original_query_id)", "seed": seed,
        "source_queries": len(rows), "selected_queries": len(selected),
        "corpus_scope_changed": False,
    }
    return result


def _release_validation_receipt(manifest: dict, *, verify_content_hashes: bool) -> dict:
    """Return a validation receipt without rewriting the release manifest."""
    result = dict(manifest)
    if verify_content_hashes:
        return result
    verified = bool(verify_content_hashes)
    result["content_hashes_verified"] = verified
    boundary = (
        "release inventory and bundle content hashes verified"
        if verified else
        "release inventory, sizes, JSONL and structural relationships validated; stored digests declared only"
    )
    result["verification_boundary"] = boundary
    result["validation_receipt"] = {"content_hashes_verified": verified, "boundary": boundary}
    return result


def _inventory(directory: Path, *, verify_content_hashes: bool = True) -> dict:
    if type(verify_content_hashes) is not bool:
        raise ValueError("verify_content_hashes must be a boolean")
    result = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"Symlink not allowed in release: {relative}")
        if stat.S_ISREG(info.st_mode):
            if info.st_nlink != 1:
                raise ValueError(f"Hardlink not allowed in release: {relative}")
            if relative != "_COMPLETE.json":
                result[relative] = {"bytes": info.st_size}
                if verify_content_hashes:
                    result[relative]["sha256"] = file_digest(path)
        elif not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Nonregular artifact: {relative}")
    return result


def _validate_release_inventory(complete_files: object, actual: dict, *, verify_content_hashes: bool) -> None:
    if not isinstance(complete_files, Mapping):
        raise ValueError("Release completion file catalog is malformed")
    if set(complete_files) != set(actual):
        raise ValueError("Release inventory/hash mismatch (changed, missing or extra files)")
    for relative, expected in complete_files.items():
        if (not isinstance(relative, str) or not isinstance(expected, Mapping)
                or isinstance(expected.get("bytes"), bool)
                or not isinstance(expected.get("bytes"), int)
                or expected["bytes"] < 0
                or not isinstance(expected.get("sha256"), str)
                or len(expected["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in expected["sha256"])):
            raise ValueError("Release completion file catalog is malformed")
        if actual[relative]["bytes"] != expected["bytes"]:
            raise ValueError(f"Release inventory/hash mismatch (size mismatch): {relative}")
        if verify_content_hashes and actual[relative] != dict(expected):
            raise ValueError(f"Release inventory/hash mismatch: {relative}")


def _validate_release_bundle_structure(manifest: Mapping, bundle: Mapping) -> None:
    """Validate release relationships without hashing document contents."""
    if manifest.get("schema_version") != PIPELINE_VERSION:
        raise ValueError("Unknown pipeline release schema")
    if not isinstance(bundle, Mapping) or bundle.get("schema_version") != BUNDLE_VERSION:
        raise ValueError("Unknown data bundle schema")
    input_kind = bundle.get("input_kind")
    if input_kind not in ("retrieval", "evidence_qa", "corpus"):
        raise ValueError("Invalid bundle input kind")
    source = bundle.get("source")
    if (not isinstance(source, Mapping)
            or not isinstance(source.get("dataset"), str) or not source["dataset"].strip()
            or not isinstance(source.get("revision"), str) or not source["revision"].strip()):
        raise ValueError("Bundle source identity is malformed")
    if manifest.get("input_kind") != input_kind or manifest.get("source") != dict(source):
        raise ValueError("Release manifest source relationship mismatch")
    if (not isinstance(manifest.get("bundle_hash"), str)
            or len(manifest["bundle_hash"]) != 64
            or any(char not in "0123456789abcdef" for char in manifest["bundle_hash"])):
        raise ValueError("Release manifest bundle identity is malformed")
    if manifest.get("state") not in {"corpus_ready", "review_required", "labels_ready"}:
        raise ValueError("Unknown release state")
    if not isinstance(bundle.get("provenance"), Mapping):
        raise ValueError("Bundle provenance must be an object")

    records: dict[str, list] = {}
    for key in ("documents", "queries", "evidence", "judgments"):
        value = bundle.get(key)
        if not isinstance(value, list):
            raise ValueError(f"Bundle {key} must be a JSONL record list")
        records[key] = value

    documents: dict[str, Mapping] = {}
    for index, row in enumerate(records["documents"]):
        if (not isinstance(row, Mapping) or not isinstance(row.get("_id"), str)
                or not row["_id"].strip() or not isinstance(row.get("text"), str)
                or not row["text"].strip()):
            raise ValueError(f"Malformed document record at index {index}")
        if row["_id"] in documents:
            raise ValueError("Duplicate document ID in release")
        documents[row["_id"]] = row
    queries: dict[str, Mapping] = {}
    for index, row in enumerate(records["queries"]):
        if (not isinstance(row, Mapping) or not isinstance(row.get("_id"), str)
                or not row["_id"].strip() or not isinstance(row.get("text"), str)
                or not row["text"].strip()):
            raise ValueError(f"Malformed query record at index {index}")
        if row["_id"] in queries:
            raise ValueError("Duplicate query ID in release")
        queries[row["_id"]] = row
    qrels = bundle.get("qrels")
    if not isinstance(qrels, Mapping):
        raise ValueError("Release qrels must be an object")
    if any(not isinstance(query_id, str) or not query_id for query_id in qrels):
        raise ValueError("Release qrels contain an invalid query ID")
    for query_id, gold in qrels.items():
        if query_id not in queries:
            raise ValueError("Release qrels contain an unknown query ID")
        if not isinstance(gold, Mapping) or (input_kind != "corpus" and not gold):
            raise ValueError(f"Release qrels for {query_id} are malformed")
        for document_id, score in gold.items():
            if (not isinstance(document_id, str) or document_id not in documents
                    or isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(float(score)) or float(score) <= 0):
                raise ValueError(f"Release qrels contain an invalid judgment for {query_id}")
    if (manifest.get("state") == "labels_ready" and input_kind != "corpus"
            and set(qrels) != set(queries)):
        raise ValueError("Release query/qrels coverage is incomplete")

    for key in ("evidence", "judgments"):
        for row in records[key]:
            if not isinstance(row, Mapping):
                raise ValueError(f"Release {key} rows must be objects")
            if "query_id" in row and row["query_id"] not in queries:
                raise ValueError(f"Release {key} refers to an unknown query")
            if "doc_id" in row and row["doc_id"] not in documents:
                raise ValueError(f"Release {key} refers to an unknown document")

    counts = {
        "num_documents": len(documents),
        "num_queries": len(queries),
        "num_positive_pairs": sum(len(gold) for gold in qrels.values()),
    }
    if any(manifest.get(key) != value for key, value in counts.items()):
        raise ValueError("Release manifest counts do not match parsed data")


def _load_verified_release(
    directory: Path,
    *,
    verify_content_hashes: bool = True,
) -> tuple[dict, dict]:
    if type(verify_content_hashes) is not bool:
        raise ValueError("verify_content_hashes must be a boolean")
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Release must be a real directory")
    marker = directory / "_COMPLETE.json"
    if marker.is_symlink() or not marker.is_file() or marker.stat().st_nlink != 1:
        raise ValueError("Missing/invalid completion marker")
    complete = read_json(marker)
    if complete.get("schema_version") != PIPELINE_VERSION:
        raise ValueError("Unknown pipeline release schema")
    actual = _inventory(directory, verify_content_hashes=verify_content_hashes)
    _validate_release_inventory(complete.get("files"), actual,
                                 verify_content_hashes=verify_content_hashes)
    manifest = read_json(directory / "manifest.json")
    bundle = load_bundle(directory, verify=False, verify_content_hashes=verify_content_hashes)
    if not verify_content_hashes:
        _validate_release_bundle_structure(manifest, bundle)
    if verify_content_hashes and manifest.get("bundle_hash") != digest(bundle):
        raise ValueError("Bundle hash mismatch")
    return _release_validation_receipt(manifest, verify_content_hashes=verify_content_hashes), bundle


def verify_release(directory: Path, *, verify_content_hashes: bool = True) -> dict:
    return _load_verified_release(directory, verify_content_hashes=verify_content_hashes)[0]


def load_bundle(
    directory: Path,
    *,
    verify: bool = True,
    verify_content_hashes: bool = True,
) -> dict:
    if type(verify_content_hashes) is not bool:
        raise ValueError("verify_content_hashes must be a boolean")
    directory = Path(directory)
    if verify:
        return _load_verified_release(directory, verify_content_hashes=verify_content_hashes)[1]
    bundle = read_json(directory / "bundle_metadata.json")
    for key, name in _FILES.items():
        # JSONL records end at physical newlines. Unicode separators such as
        # U+2028 are legal inside JSON strings and must remain part of the text.
        with (directory / name).open(encoding="utf-8") as stream:
            bundle[key] = [json.loads(line) for line in stream if line.strip()]
    bundle["qrels"] = read_json(directory / "qrels.json")
    return bundle


def _reject_symlink_path(path: Path) -> None:
    """Reject symlinks in every existing component of a destination path."""
    current = Path(path)
    while True:
        if current.is_symlink():
            raise ValueError(f"Symlink path is not allowed: {current}")
        if current.parent == current:
            break
        current = current.parent


def _reject_symlink_tree(directory: Path) -> None:
    if not directory.exists():
        return
    for candidate in directory.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"Symlink path is not allowed: {candidate}")


def _source_identity(source: Mapping) -> dict:
    return {key: source.get(key) for key in ("dataset", "revision", "connector", "resolved_revision")
            if source.get(key) is not None}


def _validate_parent_source(parent_manifest: Mapping, source: Mapping) -> None:
    parent_source = parent_manifest.get("source", {})
    if not isinstance(parent_source, Mapping) or not isinstance(source, Mapping):
        raise ValueError("Release source identity must be an object")
    parent_id = _source_identity(parent_source)
    child_id = _source_identity(source)
    for key in ("dataset", "revision"):
        if parent_id.get(key) != child_id.get(key):
            raise ValueError(f"Parent release source {key} does not match child bundle")
    if "connector" in parent_id or "connector" in child_id:
        if parent_id.get("connector") != child_id.get("connector"):
            raise ValueError("Parent release source connector does not match child bundle")
    if ("resolved_revision" in parent_id and "resolved_revision" in child_id and
            parent_id["resolved_revision"] != child_id["resolved_revision"]):
        raise ValueError("Parent release resolved revision does not match child bundle")


def _validate_raw_source(bundle: Mapping, snapshot: Path, specification: Mapping | None) -> dict:
    """Validate a connector snapshot manifest against the cleaned bundle."""
    snapshot = Path(snapshot)
    _reject_symlink_path(snapshot)
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ValueError("Raw snapshot must be a real directory")
    manifest_path = snapshot / "source_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Raw snapshot has no source_manifest.json")
    raw_manifest = read_json(manifest_path)
    if not isinstance(raw_manifest, Mapping) or raw_manifest.get("complete") is not True:
        raise ValueError("Raw source manifest is incomplete")
    if raw_manifest.get("schema_version") != "dci-source-snapshot-v1":
        raise ValueError("Unknown raw source manifest schema")
    requested = raw_manifest.get("requested_spec")
    if not isinstance(requested, Mapping):
        raise ValueError("Raw source manifest has no requested_spec")
    expected = dict(specification) if isinstance(specification, Mapping) else dict(requested)
    if specification is not None:
        try:
            from .connectors import _require_spec, _spec_fingerprint, _verify_snapshot
            normalized = _require_spec(specification)
            expected_fingerprint = _spec_fingerprint(normalized)
            _verify_snapshot(normalized, snapshot)
        except (ImportError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid source specification: {exc}") from exc
        recorded_fingerprint = raw_manifest.get("requested_spec_fingerprint",
                                                 raw_manifest.get("spec_fingerprint"))
        if recorded_fingerprint != expected_fingerprint:
            raise ValueError("Raw source manifest specification fingerprint does not match")
        expected = normalized
    for key in ("id", "connector", "input_kind"):
        if expected.get(key) is not None and raw_manifest.get(key) != expected.get(key):
            raise ValueError(f"Raw source manifest {key} does not match specification")
    if bundle.get("input_kind") != raw_manifest.get("input_kind"):
        raise ValueError("Bundle input_kind does not match raw source manifest")
    source = bundle.get("source", {})
    if not isinstance(source, Mapping):
        raise ValueError("Bundle source must be an object")
    options = expected.get("options", {}) if isinstance(expected.get("options"), Mapping) else {}
    dataset = options.get("dataset", options.get("repo", expected.get("id")))
    revision = expected.get("revision") or raw_manifest.get("resolved_revision")
    if source.get("dataset") != dataset or source.get("revision") != revision:
        raise ValueError("Bundle source dataset/revision does not match raw source manifest")
    if expected.get("connector") is not None and source.get("connector") != expected.get("connector"):
        raise ValueError("Bundle source connector does not match raw source manifest")
    resolved = source.get("resolved_revision")
    if resolved is not None and resolved != raw_manifest.get("resolved_revision"):
        raise ValueError("Bundle resolved revision does not match raw source manifest")
    provenance = bundle.get("provenance", {})
    if isinstance(provenance, Mapping) and provenance.get("source_id") is not None:
        if expected.get("id") is not None and provenance["source_id"] != expected.get("id"):
            raise ValueError("Bundle source_id does not match raw source manifest")
    return dict(raw_manifest)


def _check_export_destination(output_suite: Path, release: Path, task: str) -> None:
    _reject_symlink_path(output_suite)
    output_suite = Path(output_suite)
    release = Path(release)
    suite_real = output_suite.resolve()
    release_real = release.resolve()
    if suite_real == release_real or release_real in suite_real.parents:
        raise ValueError("DCI export must be outside the cleaned release")
    if output_suite.exists():
        if not output_suite.is_dir():
            raise ValueError("DCI export suite must be a directory")
        _reject_symlink_tree(output_suite)
    for subdirectory in ("workspaces", "metadata", "canonical"):
        parent = output_suite / subdirectory
        _reject_symlink_path(parent)
        if parent.exists() and not parent.is_dir():
            raise ValueError(f"DCI export path is not a directory: {parent}")
        target = parent / task
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing task: {task}")


def publish_bundle(bundle: dict, output: Path, *, specification: dict | None = None,
                   raw_snapshot: Path | None = None, policy: dict | None = None,
                   parent_release: Path | None = None) -> dict:
    """Validate and publish one immutable cleaned stage, including corpus-only stages."""
    from .quality import prepare_bundle, assign_splits
    output = Path(output)
    _reject_symlink_path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Use a new release directory: {output}")
    raw_manifest = None
    if raw_snapshot is not None:
        raw = Path(raw_snapshot)
        raw_real = raw.resolve()
        output_real = output.resolve()
        if output_real == raw_real or raw_real in output_real.parents:
            raise ValueError("Clean release must not be inside a raw snapshot")
        raw_manifest = _validate_raw_source(bundle, raw, specification)
    parent_manifest = None
    if parent_release is not None:
        parent_path = Path(parent_release)
        _reject_symlink_path(parent_path)
        parent = parent_path.resolve()
        output_real = output.resolve()
        if output_real == parent or parent in output_real.parents:
            raise ValueError("Derived release must be outside its parent release")
        parent_manifest = verify_release(parent)
    cleaned, audit = prepare_bundle(bundle, policy=policy)
    if audit.get("status") == "invalid" or audit.get("errors"):
        raise ValueError(f"Invalid bundle: {audit.get('errors')}")
    cleaned.setdefault("schema_version", BUNDLE_VERSION)
    for name in _FILES:
        cleaned.setdefault(name, [])
    cleaned.setdefault("qrels", {})
    if raw_manifest is not None:
        # Validate the post-policy bundle too.  The policy may remove rows,
        # but it must never change source identity.
        _validate_raw_source(cleaned, Path(raw_snapshot), specification)
    if parent_manifest is not None:
        _validate_parent_source(parent_manifest, cleaned.get("source", {}))
    state, label_info = _release_label_state(cleaned, audit)
    source = cleaned.get("source", {})
    manifest = {
        "schema_version": PIPELINE_VERSION, "state": state,
        "source": source, "input_kind": cleaned["input_kind"],
        "num_documents": len(cleaned["documents"]), "num_queries": len(cleaned["queries"]),
        "num_positive_pairs": sum(len(g) for g in cleaned["qrels"].values()),
        "bundle_hash": digest(cleaned), "specification": specification or {},
        "code_identity": code_identity(), "raw_snapshot": str(raw_snapshot) if raw_snapshot else None,
        "parent_release": str(parent_release) if parent_release else None,
        "parent_completion_sha256": file_digest(Path(parent_release) / "_COMPLETE.json") if parent_release else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validation_boundary": "data prepared and audited; no model benchmark implied",
        "automatic_labels": label_info["automatic"],
        "model_generate_provenance": label_info["generate"],
        "model_judge_provenance": label_info["judge"],
        "model_positive_issues": label_info["model_positive_issues"],
    }
    if raw_manifest is not None:
        manifest["raw_manifest_sha256"] = file_digest(Path(raw_snapshot) / "source_manifest.json")
    stage = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    try:
        metadata = {k: v for k, v in cleaned.items() if k not in _FILES and k != "qrels"}
        write_json(stage / "bundle_metadata.json", metadata)
        for key, filename in _FILES.items():
            with (stage / filename).open("x", encoding="utf-8") as stream:
                for row in cleaned[key]:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        write_json(stage / "qrels.json", cleaned["qrels"])
        write_json(stage / "audit.json", audit)
        write_json(stage / "splits.json", assign_splits(cleaned))
        write_json(stage / "manifest.json", manifest)
        write_json(stage / "_COMPLETE.json", {"schema_version": PIPELINE_VERSION, "files": _inventory(stage)})
        verify_release(stage)
        stage.rename(output)
    except BaseException as exc:
        # Preserve intermediate bytes for diagnosis; never advertise a failed stage as complete.
        if stage.exists():
            marker = stage / "_COMPLETE.json"
            if marker.exists():
                marker.rename(stage / "_UNPUBLISHED_COMPLETE.json")
            write_json(stage / "_FAILED.json", {"exception": type(exc).__name__, "stage": "publish"})
        raise
    return _release_validation_receipt(manifest, verify_content_hashes=True)


def build_source(spec: dict, raw_root: Path, output: Path, *, snapshot: Path | None = None,
                 policy: dict | None = None) -> dict:
    from .connectors import fetch_source, load_source
    slug(spec["id"])
    snapshot = Path(snapshot) if snapshot else fetch_source(spec, Path(raw_root))
    bundle = load_source(spec, snapshot)
    selection = spec.get("selection", {})
    bundle = select_queries(bundle, limit=selection.get("max_queries"), seed=selection.get("seed", 20261022))
    bundle.setdefault("provenance", {}).update({
        "domain": spec.get("domain", "unknown"), "language": spec.get("language", "unknown"),
        "genre": spec.get("genre", spec.get("options", {}).get("genre", "unknown")), "source_id": spec["id"],
    })
    return publish_bundle(bundle, output, specification=spec, raw_snapshot=snapshot, policy=policy)


def export_dci(
    release: Path,
    output_suite: Path,
    task: str,
    *,
    allow_automatic_labels: bool = False,
    verify_content_hashes: bool = True,
    io_workers: int = 4,
) -> dict:
    from dci_bench.data.expansion import build_document_task, verify_document_task
    if type(verify_content_hashes) is not bool:
        raise ValueError("verify_content_hashes must be a boolean")
    if type(io_workers) is not int or not 1 <= io_workers <= 256:
        raise ValueError("io_workers must be an integer from 1 through 256")
    task = slug(task)
    manifest, bundle = _load_verified_release(release, verify_content_hashes=verify_content_hashes)
    if manifest["state"] != "labels_ready" or not bundle["queries"]:
        raise ValueError(f"Release cannot be evaluated yet: {manifest['state']}")
    label_info = _label_provenance(bundle)
    model_positive_issues = _model_positive_issues(bundle) if label_info["automatic"] else []
    if label_info["automatic"] and not label_info["judge"]:
        raise ValueError("Automatic labels require actual models.judge provenance")
    if model_positive_issues:
        raise ValueError("Automatic labels contain unreviewed positive qrels")
    if label_info["automatic"] and not allow_automatic_labels:
        raise ValueError("Automatic labels require explicit --allow-automatic-labels; this does not imply human review")
    output_suite = Path(output_suite)
    _check_export_destination(output_suite, Path(release), task)
    provenance = bundle.get("provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    data = {
        "name": task, "dataset": bundle["source"]["dataset"],
        "revision": bundle["source"]["revision"],
        "resolved_revision": bundle["source"].get("resolved_revision", bundle["source"]["revision"]),
        "track": "document", "task_instruction": provenance.get("task_instruction", ""),
        "corpus": [{k: row[k] for k in ("_id", "title", "text")} for row in bundle["documents"]],
        "queries": [{k: row[k] for k in ("_id", "text")} for row in bundle["queries"]],
        "qrels": bundle["qrels"],
        "host_labels": {"evidence": bundle["evidence"], "judgments": bundle["judgments"],
                        "query_metadata": {q["_id"]: q.get("metadata", {}) for q in bundle["queries"]}},
        "provenance": {**provenance, "cleaned_release": str(release), "cleaned_bundle_hash": manifest["bundle_hash"],
                       "automatic_labels_allowed": allow_automatic_labels, "exclusions": provenance.get("exclusions", [])},
    }
    source = {"raw_directory": manifest.get("raw_snapshot") or str(release),
              "source_fingerprint": manifest["bundle_hash"], "pipeline_manifest": manifest}
    build_document_task(data, output_suite, source, io_workers=io_workers)
    return verify_document_task(
        output_suite,
        task,
        io_workers=io_workers,
        verify_content_hashes=verify_content_hashes,
    )


def write_dashboard(releases: list[Path], output: Path) -> dict:
    """Standalone filterable coverage and readiness view; no remote resources."""
    from .collection import collection_coverage, assign_collection_splits
    bundles, rows = [], []
    for release in releases:
        m = verify_release(release)
        b = load_bundle(release)
        bundles.append(b)
        p = b.get("provenance", {})
        rows.append({"source": p.get("source_id", b["source"]["dataset"]),
                     "domain": p.get("domain", "unknown"), "language": p.get("language", "unknown"),
                     "kind": b["input_kind"], "state": m["state"], "documents": m["num_documents"],
                     "queries": m["num_queries"], "positives": m["num_positive_pairs"],
                     "hash": m["bundle_hash"], "path": str(release)})
    coverage = collection_coverage(bundles)
    for row, counts in zip(rows, coverage["source_query_counts"]):
        row.update({key: counts[key] for key in ("native_queries", "translated_queries", "unknown_queries")})
    report = {"schema_version": PIPELINE_VERSION, "coverage": coverage, "releases": rows,
              "created_at": datetime.now(timezone.utc).isoformat(), "code_identity": code_identity(),
              "global_split_file": "collection_splits.json",
              "split_boundary": "Use collection_splits.json when combining sources; per-release splits only protect within one source"}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "coverage.json", report)
    write_json(output / "collection_splits.json", assign_collection_splits(bundles))
    encoded = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>DCI data coverage</title><style>body{font:16px system-ui;margin:3rem;background:#101923;color:#e5edf3}input{padding:.8rem;width:60%;border:1px solid #59718b;background:#182736;color:white}table{border-collapse:collapse;width:100%;margin-top:2rem}th,td{text-align:left;padding:.7rem;border-bottom:1px solid #35485b}small{color:#a8bacb}h1{font-weight:600}</style>
<h1>DCI data coverage</h1><p>Verified data artifacts. Corpus readiness, label readiness, and model evaluation are separate milestones.</p>
<input id="filter" placeholder="Filter by domain, language, source, or readiness"><p id="count"></p>
<table><thead><tr><th>Source</th><th>Domain</th><th>Language</th><th>Input</th><th>State</th><th>Docs</th><th>Queries</th><th>Native</th><th>Translated</th><th>Unknown</th><th>Positive pairs</th></tr></thead><tbody id="rows"></tbody></table>
<p><small>Language labels are source declarations unless separately audited. Parallel and translated versions are not independent native coverage. See coverage.json for the full audit.</small></p>
<script>const data=DATA;const keys=['source','domain','language','kind','state','documents','queries','native_queries','translated_queries','unknown_queries','positives'];function render(){const f=document.getElementById('filter').value.toLowerCase();const list=data.filter(r=>JSON.stringify(r).toLowerCase().includes(f));const body=document.getElementById('rows');body.replaceChildren();for(const row of list){const tr=document.createElement('tr');for(const k of keys){const td=document.createElement('td');td.textContent=String(row[k]);tr.append(td)}body.append(tr)}document.getElementById('count').textContent=list.length+' source cells; '+list.reduce((n,r)=>n+r.queries,0)+' queries (before cross-source overlap accounting)'}document.getElementById('filter').addEventListener('input',render);render();</script></html>'''.replace("DATA", encoded)
    (output / "index.html").write_text(page, encoding="utf-8")
    return report
