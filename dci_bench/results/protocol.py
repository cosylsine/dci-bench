"""Phase 4 result schemas, integrity helpers, and task aggregation.

All records returned by this module are JSON-compatible values.  The module
does not import the existing scorer so it can be used by lightweight result
post-processing jobs that do not have the benchmark's optional dependencies
installed.  Metric values are supplied by the caller; aggregation only
combines already-scored sample records.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


# Public schema identifiers.  Keep these stable: they are written into every
# record and are used by downstream audit tooling to reject mixed protocols.
SAMPLE_RESULT_SCHEMA: Final[str] = "dci-sample-result-v1"
TASK_RESULT_SCHEMA: Final[str] = "dci-task-result-v1"
RUN_MANIFEST_SCHEMA: Final[str] = "dci-run-manifest-v1"
TRAJECTORY_SCHEMA: Final[str] = "dci-trajectory-v1"
TASK_INDEX_SCHEMA: Final[str] = "dci-task-index-v1"

# A few callers use *_SCHEMA_VERSION names; expose both spellings without
# making the wire format ambiguous.
SAMPLE_RESULT_SCHEMA_VERSION: Final[str] = SAMPLE_RESULT_SCHEMA
TASK_RESULT_SCHEMA_VERSION: Final[str] = TASK_RESULT_SCHEMA
RUN_MANIFEST_SCHEMA_VERSION: Final[str] = RUN_MANIFEST_SCHEMA
TRAJECTORY_SCHEMA_VERSION: Final[str] = TRAJECTORY_SCHEMA
TASK_INDEX_SCHEMA_VERSION: Final[str] = TASK_INDEX_SCHEMA

DEFAULT_METRIC_KS: Final[tuple[int, ...]] = (1, 3, 5, 10, 20)
METRIC_FAMILIES: Final[tuple[str, ...]] = ("recall", "f1", "ndcg")
DEFAULT_MAX_TOTAL_MODEL_TOKENS: Final[int] = 128_000

# These are intentionally a superset of statuses emitted by the current Pi
# runner.  Unknown status strings remain valid for forward-compatible readers,
# while failure categorisation below still has stable buckets.
FAILURE_KINDS: Final[tuple[str, ...]] = (
    "timeout",
    "token_limit",
    "invalid_json",
    "tool_failure",
    "context_overflow",
    "agent_error",
    "infrastructure_error",
    "usage_unavailable",
    "other",
)
_STRICT_FAILURE_KINDS = frozenset(FAILURE_KINDS)
_SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "schemas" / "results"
_SCHEMA_FILES = {
    SAMPLE_RESULT_SCHEMA: "dci-sample-result-v1.json",
    TASK_RESULT_SCHEMA: "dci-task-result-v1.json",
    RUN_MANIFEST_SCHEMA: "dci-run-manifest-v1.json",
    TRAJECTORY_SCHEMA: "dci-trajectory-v1.json",
    TASK_INDEX_SCHEMA: "dci-task-index-v1.json",
}
_SCHEMA_VALIDATORS: dict[str, Draft202012Validator] = {}


class ResultValidationError(ValueError):
    """Raised when a result record is malformed or contains host-only data."""


class ResultIntegrityError(ResultValidationError):
    """Raised when an artifact hash or canonical record integrity check fails."""


def validate_json_schema(value: Any, schema_version: str) -> bool:
    """Validate a wire record against its checked-in versioned JSON Schema."""

    filename = _SCHEMA_FILES.get(schema_version)
    if filename is None:
        raise ResultValidationError(f"unsupported JSON schema version: {schema_version}")
    validator = _SCHEMA_VALIDATORS.get(schema_version)
    if validator is None:
        try:
            schema = json.loads((_SCHEMA_ROOT / filename).read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
        except (OSError, json.JSONDecodeError, SchemaError) as exc:
            raise ResultValidationError(f"unable to load JSON schema {schema_version}: {exc}") from exc
        validator = Draft202012Validator(schema)
        _SCHEMA_VALIDATORS[schema_version] = validator
    try:
        validator.validate(value)
    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "$"
        raise ResultValidationError(
            f"{schema_version} JSON Schema validation failed at {location}: {exc.message}"
        ) from exc
    return True


def canonical_json(value: Any) -> str:
    """Return deterministic compact JSON suitable for hashing.

    ``allow_nan=False`` is important for audit records: NaN and Infinity are
    not interoperable JSON numbers and would make a fingerprint ambiguous.
    """

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ResultValidationError(f"Value is not canonical JSON: {exc}") from exc


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash a file without loading it all into memory."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value using :func:`canonical_json`."""

    return sha256_bytes(canonical_json_bytes(value))


def fingerprint(value: Any) -> str:
    """Alias for the canonical SHA-256 fingerprint used by run manifests."""

    return sha256_json(value)


# Historical/explicit aliases used by different launchers.
canonical_json_sha256 = sha256_json
stable_json_sha256 = sha256_json


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory fsync after ``os.replace``.

    Some platforms do not permit opening a directory.  The file replacement
    itself remains atomic there, so this helper deliberately ignores only the
    platform-level ``OSError``.
    """

    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def write_json_atomic(
    path: str | os.PathLike[str],
    value: Any,
    *,
    mode: int = 0o644,
) -> Path:
    """Atomically write human-readable JSON and return the destination path.

    The temporary file is created in the destination directory, flushed and
    fsynced, then promoted with ``os.replace``.  Existing files are replaced
    atomically; callers that require immutable runs should reject an existing
    destination before calling this helper. On-disk audit records use sorted
    keys, two-space indentation, and one trailing newline. Fingerprints remain
    independent of presentation because :func:`sha256_json` still hashes the
    compact canonical representation.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResultValidationError(f"Value is not valid audit JSON: {exc}") from exc
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
        raise
    return destination


def read_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            raise ResultValidationError(f"Invalid JSON in {path}: {exc}") from exc


_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_PATH_FORBIDDEN_RE = re.compile(r"(?:^|[\\/_.-])(qrels?|gold(?:[-_]|$))", re.IGNORECASE)
_SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "access_token",
    "authorization",
    "password",
    "passwd",
    "secret",
    "credential",
    "private_key",
    "client_secret",
)


def _key_is_forbidden(key: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    if "qrel" in normalized or "gold" in normalized:
        return "qrels/gold fields are host-only and must not be stored"
    if any(part in normalized for part in _SECRET_KEY_PARTS):
        return "credential/secret fields must not be stored"
    return None


def validate_no_sensitive_fields(value: Any, *, _path: str = "$") -> bool:
    """Reject qrels/gold labels and credential-shaped fields recursively.

    Token *counts* are permitted (for example ``input_tokens``); only secret
    credential names are rejected.  Artifact paths are also checked so a
    result cannot accidentally point at a qrels/gold file.
    """

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise ResultValidationError(f"{_path}: object keys must be strings")
            reason = _key_is_forbidden(raw_key)
            if reason:
                raise ResultValidationError(f"{_path}.{raw_key}: {reason}")
            validate_no_sensitive_fields(child, _path=f"{_path}.{raw_key}")
        return True
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            validate_no_sensitive_fields(child, _path=f"{_path}[{index}]")
        return True
    if isinstance(value, str) and _path.lower().split(".")[-1] in {
        "path",
        "file",
        "filename",
        "artifact_path",
    }:
        if _PATH_FORBIDDEN_RE.search(value):
            raise ResultValidationError(f"{_path}: qrels/gold artifacts are not allowed")
    return True


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResultValidationError(f"{name} must be an object")
    return value


def _require_string(value: Any, name: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise ResultValidationError(f"{name} must be a non-empty string")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ResultValidationError(f"{name} must be a boolean")
    return value


def _require_number(value: Any, name: str, *, nonnegative: bool = False) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultValidationError(f"{name} must be a number")
    if not math.isfinite(float(value)):
        raise ResultValidationError(f"{name} must be finite")
    if nonnegative and value < 0:
        raise ResultValidationError(f"{name} must be non-negative")
    return value


def normalize_metric_ks(metric_ks: Sequence[int] | None = None) -> tuple[int, ...]:
    values = DEFAULT_METRIC_KS if metric_ks is None else tuple(metric_ks)
    if not values:
        raise ResultValidationError("metric_ks must contain at least one cutoff")
    if any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in values):
        raise ResultValidationError("metric_ks must contain positive integer cutoffs")
    if len(set(values)) != len(values):
        raise ResultValidationError("metric_ks must not contain duplicates")
    return tuple(values)


def metric_names(metric_ks: Sequence[int] | None = None) -> tuple[str, ...]:
    return tuple(
        f"{family}_at_{cutoff}"
        for family in METRIC_FAMILIES
        for cutoff in normalize_metric_ks(metric_ks)
    )


def _normalize_metrics(
    metrics: Mapping[str, Any] | None,
    metric_ks: Sequence[int],
    *,
    fill_missing: bool = True,
) -> dict[str, float]:
    """Normalize nested/flat metric input to flat ``family_at_k`` names."""

    source = {} if metrics is None else _require_mapping(metrics, "metrics")
    normalized: dict[str, float] = {}
    for family in METRIC_FAMILIES:
        nested = source.get(family)
        if isinstance(nested, Mapping):
            for cutoff in metric_ks:
                key = f"{family}_at_{cutoff}"
                value = nested.get(cutoff, nested.get(str(cutoff)))
                if value is None:
                    value = source.get(key)
                if value is None and fill_missing:
                    value = 0.0
                if value is not None:
                    normalized[key] = float(_require_number(value, f"metrics.{key}"))
        else:
            for cutoff in metric_ks:
                key = f"{family}_at_{cutoff}"
                value = source.get(key)
                if value is None and fill_missing:
                    value = 0.0
                if value is not None:
                    normalized[key] = float(_require_number(value, f"metrics.{key}"))
    missing = [name for name in metric_names(metric_ks) if name not in normalized]
    if missing:
        raise ResultValidationError(f"metrics missing required fields: {missing}")
    return normalized


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _iso_or_none(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, name)


def _require_utc_timestamp(value: Any, name: str) -> str:
    timestamp = _require_string(value, name)
    normalized = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ResultValidationError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ResultValidationError(f"{name} must include the UTC timezone")
    return timestamp


def normalize_usage(usage: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize provider usage while preserving null availability semantics."""

    names = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
    )
    source = {} if usage is None else _require_mapping(usage, "usage")
    available = source.get("available")
    if available is None:
        available = all(source.get(name) is not None for name in names[:3])
    _require_bool(available, "usage.available")
    result: dict[str, Any] = {"available": bool(available)}
    for name in names:
        value = source.get(name)
        if value is None:
            result[name] = None
        else:
            result[name] = int(_require_number(value, f"usage.{name}", nonnegative=True))
    by_model = source.get("by_model", source.get("routed_models", {}))
    if by_model is None:
        by_model = {}
    by_model = _require_mapping(by_model, "usage.by_model")
    normalized_models: dict[str, Any] = {}
    for model_name, model_usage in by_model.items():
        _require_string(model_name, "usage.by_model key")
        model_map = _require_mapping(model_usage, f"usage.by_model.{model_name}")
        normalized_models[model_name] = normalize_usage(model_map)
    result["by_model"] = normalized_models
    return result


def normalize_timing(timing: Mapping[str, Any] | None) -> dict[str, Any]:
    source = {} if timing is None else _require_mapping(timing, "timing")
    result: dict[str, Any] = {
        "started_at": _iso_or_none(source.get("started_at"), "timing.started_at"),
        "ended_at": _iso_or_none(source.get("ended_at"), "timing.ended_at"),
    }
    for name in ("wall_time_seconds", "working_time_seconds"):
        value = source.get(name)
        result[name] = (
            None
            if value is None
            else float(_require_number(value, f"timing.{name}", nonnegative=True))
        )
    return result


def normalize_execution(execution: Mapping[str, Any] | None) -> dict[str, Any]:
    source = {} if execution is None else _require_mapping(execution, "execution")
    result: dict[str, Any] = {}
    for name in ("agent_steps", "tool_calls", "model_calls", "repair_attempts"):
        value = source.get(name, 0)
        result[name] = int(_require_number(value, f"execution.{name}", nonnegative=True))
    reasons = source.get("repair_reasons", [])
    if not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes, bytearray)):
        raise ResultValidationError("execution.repair_reasons must be an array")
    result["repair_reasons"] = [_require_string(reason, "execution.repair_reasons item") for reason in reasons]
    return result


def normalize_failure(failure: Mapping[str, Any] | None, *, valid_output: bool) -> dict[str, str] | None:
    if failure is None:
        if valid_output:
            return None
        return {"kind": "other", "message": "sample did not produce a valid output"}
    source = _require_mapping(failure, "failure")
    kind = _require_string(source.get("kind"), "failure.kind")
    if kind not in _STRICT_FAILURE_KINDS:
        raise ResultValidationError(f"failure.kind must be one of {sorted(_STRICT_FAILURE_KINDS)}")
    message = _require_string(source.get("message"), "failure.message")
    return {"kind": kind, "message": message}


def artifact_record(
    path: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Create a relative artifact record with a SHA-256 digest."""

    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    if root is None:
        relative = artifact_path.name
    else:
        root_path = Path(root).resolve()
        resolved = artifact_path.resolve()
        try:
            relative_path = resolved.relative_to(root_path)
        except ValueError as exc:
            raise ResultValidationError(f"Artifact {path} is outside root {root}") from exc
        relative = relative_path.as_posix()
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ResultValidationError("Artifact paths must be relative")
    return {
        "path": relative,
        "sha256": sha256_file(artifact_path),
        "size_bytes": artifact_path.stat().st_size,
    }


def build_artifacts(
    paths: Mapping[str, str | os.PathLike[str]],
    *,
    root: str | os.PathLike[str] | None = None,
) -> dict[str, dict[str, Any]]:
    return {str(name): artifact_record(path, root=root) for name, path in paths.items()}


def verify_artifact_record(
    record: Mapping[str, Any],
    *,
    root: str | os.PathLike[str],
) -> bool:
    source = _require_mapping(record, "artifact")
    relative = _require_string(source.get("path"), "artifact.path")
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ResultIntegrityError("Artifact path must be relative and confined to run root")
    expected = _require_string(source.get("sha256"), "artifact.sha256")
    if not _HEX64_RE.fullmatch(expected):
        raise ResultIntegrityError("artifact.sha256 must be a lowercase SHA-256 digest")
    actual_path = Path(root) / relative
    actual = sha256_file(actual_path)
    if actual != expected:
        raise ResultIntegrityError(f"Artifact hash mismatch for {relative}")
    size = source.get("size_bytes")
    if size is not None and int(size) != actual_path.stat().st_size:
        raise ResultIntegrityError(f"Artifact size mismatch for {relative}")
    return True


def _normalize_artifacts(artifacts: Mapping[str, Any] | None) -> dict[str, Any]:
    if artifacts is None:
        return {}
    source = _require_mapping(artifacts, "artifacts")
    result: dict[str, Any] = {}
    for name, value in source.items():
        if isinstance(value, Mapping):
            record = dict(value)
            path = _require_string(record.get("path"), f"artifacts.{name}.path")
            if Path(path).is_absolute() or ".." in Path(path).parts:
                raise ResultValidationError(f"artifacts.{name}.path must be relative")
            digest = _require_string(record.get("sha256"), f"artifacts.{name}.sha256")
            if not _HEX64_RE.fullmatch(digest):
                raise ResultValidationError(f"artifacts.{name}.sha256 must be SHA-256")
            if "size_bytes" in record:
                record["size_bytes"] = int(
                    _require_number(record["size_bytes"], f"artifacts.{name}.size_bytes", nonnegative=True)
                )
            result[str(name)] = record
        else:
            raise ResultValidationError(f"artifacts.{name} must be an object")
    return result


def _normalize_security(security: Mapping[str, Any] | None) -> dict[str, Any]:
    source = {} if security is None else _require_mapping(security, "security")
    inspect_sandbox = source.get("inspect_sandbox", "local")
    tool_sandbox = source.get("tool_sandbox", "bubblewrap")
    if inspect_sandbox != "local" or tool_sandbox != "bubblewrap":
        raise ResultValidationError(
            "security attestation must be inspect_sandbox=local and tool_sandbox=bubblewrap"
        )
    return {
        "inspect_sandbox": _require_string(inspect_sandbox, "security.inspect_sandbox"),
        "tool_sandbox": _require_string(tool_sandbox, "security.tool_sandbox"),
    }


def _normalize_provenance(provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    if provenance is None:
        return {
            "metrics": "runner.scorer",
            "usage": "inspect.sample.model_usage",
            "timing": "inspect.sample.timing",
            "execution": "pi.result",
            "trace": "pi.trace",
        }
    source = _require_mapping(provenance, "provenance")
    return {str(key): value for key, value in source.items()}


def build_sample_result(
    *,
    run_id: str,
    task: str,
    query_id: str,
    query_sha256: str,
    valid_output: bool,
    ranked_doc_ids: Sequence[str] | None = None,
    metrics: Mapping[str, Any] | None = None,
    metric_ks: Sequence[int] | None = None,
    status: str = "success",
    failure: Mapping[str, Any] | None = None,
    usage: Mapping[str, Any] | None = None,
    timing: Mapping[str, Any] | None = None,
    execution: Mapping[str, Any] | None = None,
    security: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct one normalized sample result record.

    ``ranked_doc_ids`` is intentionally not required for an invalid sample;
    invalid samples carry an empty list and zero metrics so task aggregation
    has an explicit invalid-as-zero denominator.
    """

    run_id = _require_string(run_id, "run_id")
    task = _require_string(task, "task")
    query_id = _require_string(query_id, "query_id")
    query_sha256 = _require_string(query_sha256, "query_sha256")
    if not _HEX64_RE.fullmatch(query_sha256):
        raise ResultValidationError("query_sha256 must be a lowercase SHA-256 digest")
    _require_bool(valid_output, "valid_output")
    status = _require_string(status, "status")
    ks = normalize_metric_ks(metric_ks)
    doc_ids = [] if ranked_doc_ids is None else list(ranked_doc_ids)
    if any(not isinstance(doc_id, str) or not doc_id for doc_id in doc_ids):
        raise ResultValidationError("ranked_doc_ids must contain non-empty strings")
    if len(set(doc_ids)) != len(doc_ids):
        raise ResultValidationError("ranked_doc_ids must not contain duplicates")
    normalized_metrics = _normalize_metrics(metrics, ks)
    if not valid_output:
        doc_ids = []
        normalized_metrics = {name: 0.0 for name in metric_names(ks)}
    normalized_failure = normalize_failure(failure, valid_output=valid_output)
    result = {
        "schema_version": SAMPLE_RESULT_SCHEMA,
        "run_id": run_id,
        "task": task,
        "query_id": query_id,
        "query_sha256": query_sha256,
        "status": status,
        "valid_output": bool(valid_output),
        "failure": normalized_failure,
        "ranked_doc_ids": doc_ids,
        "metric_ks": list(ks),
        "metrics": normalized_metrics,
        "usage": normalize_usage(usage),
        "timing": normalize_timing(timing),
        "execution": normalize_execution(execution),
        "security": _normalize_security(security),
        "artifacts": _normalize_artifacts(artifacts),
        "provenance": _normalize_provenance(provenance),
    }
    validate_sample_result(result)
    return result


def _validate_sha256(value: Any, name: str) -> None:
    digest = _require_string(value, name)
    if not _HEX64_RE.fullmatch(digest):
        raise ResultValidationError(f"{name} must be a lowercase SHA-256 digest")


def _validate_usage(value: Any, name: str = "usage") -> None:
    source = _require_mapping(value, name)
    available = source.get("available")
    _require_bool(available, f"{name}.available")
    for field in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
    ):
        number = source.get(field)
        if number is not None:
            _require_number(number, f"{name}.{field}", nonnegative=True)
    by_model = source.get("by_model", {})
    _require_mapping(by_model, f"{name}.by_model")
    for model, model_usage in by_model.items():
        _require_string(model, f"{name}.by_model key")
        _validate_usage(model_usage, f"{name}.by_model.{model}")


def _validate_timing(value: Any) -> None:
    source = _require_mapping(value, "timing")
    for name in ("started_at", "ended_at"):
        if source.get(name) is not None:
            _require_utc_timestamp(source[name], f"timing.{name}")
    for name in ("wall_time_seconds", "working_time_seconds"):
        if source.get(name) is not None:
            _require_number(source[name], f"timing.{name}", nonnegative=True)


def _validate_artifacts(value: Any) -> None:
    source = _require_mapping(value, "artifacts")
    for name, record in source.items():
        _require_string(name, "artifact name")
        record_map = _require_mapping(record, f"artifacts.{name}")
        path = _require_string(record_map.get("path"), f"artifacts.{name}.path")
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ResultValidationError(f"artifacts.{name}.path must be relative")
        _validate_sha256(record_map.get("sha256"), f"artifacts.{name}.sha256")
        if record_map.get("size_bytes") is not None:
            _require_number(record_map["size_bytes"], f"artifacts.{name}.size_bytes", nonnegative=True)


def validate_sample_result(value: Mapping[str, Any]) -> bool:
    """Validate a sample record and return ``True`` on success."""

    source = _require_mapping(value, "sample result")
    validate_no_sensitive_fields(source)
    if source.get("schema_version") != SAMPLE_RESULT_SCHEMA:
        raise ResultValidationError(f"sample result schema_version must be {SAMPLE_RESULT_SCHEMA}")
    for name in ("run_id", "task", "query_id"):
        _require_string(source.get(name), name)
    _validate_sha256(source.get("query_sha256"), "query_sha256")
    valid = _require_bool(source.get("valid_output"), "valid_output")
    _require_string(source.get("status"), "status")
    failure = source.get("failure")
    if valid and failure is not None:
        raise ResultValidationError("valid sample cannot contain failure")
    if not valid:
        failure_map = _require_mapping(failure, "failure")
        kind = _require_string(failure_map.get("kind"), "failure.kind")
        if kind not in _STRICT_FAILURE_KINDS:
            raise ResultValidationError(f"failure.kind must be one of {sorted(_STRICT_FAILURE_KINDS)}")
        _require_string(failure_map.get("message"), "failure.message")
    ks = normalize_metric_ks(source.get("metric_ks"))
    ranked = source.get("ranked_doc_ids")
    if not isinstance(ranked, list) or any(not isinstance(item, str) or not item for item in ranked):
        raise ResultValidationError("ranked_doc_ids must be an array of non-empty strings")
    if len(set(ranked)) != len(ranked):
        raise ResultValidationError("ranked_doc_ids must not contain duplicates")
    _normalize_metrics(source.get("metrics"), ks, fill_missing=False)
    _validate_usage(source.get("usage"))
    _validate_timing(source.get("timing"))
    if valid:
        usage = _require_mapping(source.get("usage"), "usage")
        if usage.get("available") is not True:
            raise ResultValidationError("successful sample requires usage.available=true")
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            if usage.get(name) is None:
                raise ResultValidationError(f"successful sample requires usage.{name}")
        timing = _require_mapping(source.get("timing"), "timing")
        for name in ("started_at", "ended_at", "wall_time_seconds", "working_time_seconds"):
            if timing.get(name) is None:
                raise ResultValidationError(f"successful sample requires timing.{name}")
    execution = _require_mapping(source.get("execution"), "execution")
    for name in ("agent_steps", "tool_calls", "model_calls", "repair_attempts"):
        _require_number(execution.get(name), f"execution.{name}", nonnegative=True)
    reasons = execution.get("repair_reasons")
    if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
        raise ResultValidationError("execution.repair_reasons must be an array of strings")
    security = _require_mapping(source.get("security"), "security")
    if security.get("inspect_sandbox") != "local" or security.get("tool_sandbox") != "bubblewrap":
        raise ResultValidationError(
            "security attestation must be inspect_sandbox=local and tool_sandbox=bubblewrap"
        )
    _validate_artifacts(source.get("artifacts"))
    _require_mapping(source.get("provenance"), "provenance")
    validate_json_schema(source, SAMPLE_RESULT_SCHEMA)
    return True


def is_valid_sample_result(value: Any) -> bool:
    try:
        validate_sample_result(value)
    except (ResultValidationError, TypeError, ValueError):
        return False
    return True


def _r7_percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def r7_percentile(values: Iterable[float], probability: float) -> float | None:
    """Return the Hyndman-Fan R-7 linear-interpolation percentile."""

    numbers = [float(_require_number(value, "percentile value")) for value in values]
    return _r7_percentile(numbers, probability)


def r7_p50(values: Iterable[float]) -> float | None:
    return r7_percentile(values, 0.50)


def r7_p95(values: Iterable[float]) -> float | None:
    return r7_percentile(values, 0.95)


def _metric_stats(values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {
            "count": 0,
            "mean": None,
            "population_stddev": None,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "population_stddev": statistics.pstdev(numbers),
        "min": min(numbers),
        "p50": _r7_percentile(numbers, 0.50),
        "p95": _r7_percentile(numbers, 0.95),
        "max": max(numbers),
    }


def _availability_stats(values: Sequence[Any]) -> dict[str, Any]:
    available = [float(value) for value in values if value is not None]
    stats = _metric_stats(available)
    return {
        "available_count": len(available),
        "missing_count": len(values) - len(available),
        "sum": None if not available else sum(available),
        "mean": stats["mean"],
        "min": stats["min"],
        "p50": stats["p50"],
        "p95": stats["p95"],
        "max": stats["max"],
    }


def _sample_metric(sample: Mapping[str, Any], metric_name: str) -> float:
    metrics = sample.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return 0.0
    value = metrics.get(metric_name, 0.0)
    if value is None:
        return 0.0
    return float(_require_number(value, f"sample.metrics.{metric_name}"))


def _sample_failure_kind(sample: Mapping[str, Any]) -> str:
    failure = sample.get("failure")
    if isinstance(failure, Mapping):
        kind = failure.get("kind")
        if isinstance(kind, str) and kind:
            if kind in FAILURE_KINDS:
                return kind
            return "other"
    return "other"


def _normalize_sample_sequence(samples: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result = list(samples)
    for index, sample in enumerate(result):
        validate_sample_result(sample)
        if index and sample.get("run_id") != result[0].get("run_id"):
            raise ResultValidationError("all samples in a task must belong to one run_id")
    return result


def build_task_result(
    *,
    run_id: str,
    task: str,
    samples: Iterable[Mapping[str, Any]],
    query_ids: Sequence[str] | None = None,
    model: Mapping[str, Any] | None = None,
    backend: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    code: Mapping[str, Any] | None = None,
    evaluation_elapsed_seconds: float | None = None,
    launcher_preflight_seconds: float | None = None,
    inspect_log: Mapping[str, Any] | None = None,
    integrity: Mapping[str, Any] | None = None,
    started_count: int | None = None,
    completed_count: int | None = None,
) -> dict[str, Any]:
    """Aggregate sample records with a fixed invalid-as-zero denominator."""

    run_id = _require_string(run_id, "run_id")
    task = _require_string(task, "task")
    sample_list = _normalize_sample_sequence(samples)
    if any(sample.get("run_id") != run_id for sample in sample_list):
        raise ResultValidationError("sample run_id does not match task run_id")
    if any(sample.get("task") != task for sample in sample_list):
        raise ResultValidationError("sample task does not match task task")
    inferred_query_ids = [str(sample["query_id"]) for sample in sample_list]
    selected_query_ids = inferred_query_ids if query_ids is None else list(query_ids)
    if [str(item) for item in selected_query_ids] != inferred_query_ids:
        raise ResultValidationError("query_ids must match samples in stable execution order")
    if len(set(inferred_query_ids)) != len(inferred_query_ids):
        raise ResultValidationError("task samples must have unique query ids")
    planned = len(selected_query_ids)
    valid_samples = [sample for sample in sample_list if sample["valid_output"]]
    invalid_samples = [sample for sample in sample_list if not sample["valid_output"]]
    started = planned if started_count is None else int(
        _require_number(started_count, "started_count", nonnegative=True)
    )
    completed = planned if completed_count is None else int(
        _require_number(completed_count, "completed_count", nonnegative=True)
    )
    if started > planned or completed > started:
        raise ResultValidationError("started/completed counts must be ordered within planned count")
    ks = normalize_metric_ks(sample_list[0]["metric_ks"] if sample_list else None)
    if any(tuple(sample["metric_ks"]) != ks for sample in sample_list):
        raise ResultValidationError("all samples must use the same metric_ks")

    metric_macro: dict[str, Any] = {}
    metric_valid_only: dict[str, Any] = {}
    for name in metric_names(ks):
        # Invalid samples are explicitly zero regardless of malformed input
        # from an adapter.  Builders already normalize them, but retaining this
        # guard makes aggregation fail-safe when called with hand-built records.
        all_values = [_sample_metric(sample, name) if sample["valid_output"] else 0.0 for sample in sample_list]
        valid_values = [_sample_metric(sample, name) for sample in valid_samples]
        metric_macro[name] = _metric_stats(all_values)
        metric_valid_only[name] = _metric_stats(valid_values)

    failure_counts = {kind: 0 for kind in FAILURE_KINDS}
    for sample in invalid_samples:
        failure_counts[_sample_failure_kind(sample)] += 1
    failure_rate = 0.0 if planned == 0 else len(invalid_samples) / planned

    usage_fields = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
    )
    usage_stats = {
        name: _availability_stats([sample["usage"].get(name) for sample in sample_list])
        for name in usage_fields
    }
    timing_fields = ("wall_time_seconds", "working_time_seconds")
    timing_stats = {
        name: _availability_stats([sample["timing"].get(name) for sample in sample_list])
        for name in timing_fields
    }
    if evaluation_elapsed_seconds is not None:
        evaluation_elapsed_seconds = float(
            _require_number(evaluation_elapsed_seconds, "evaluation_elapsed_seconds", nonnegative=True)
        )
    if launcher_preflight_seconds is not None:
        launcher_preflight_seconds = float(
            _require_number(launcher_preflight_seconds, "launcher_preflight_seconds", nonnegative=True)
        )
    model_info = {} if model is None else dict(_require_mapping(model, "model"))
    backend_info = {} if backend is None else dict(_require_mapping(backend, "backend"))
    data_info = {} if data is None else dict(_require_mapping(data, "data"))
    code_info = {} if code is None else dict(_require_mapping(code, "code"))
    inspect_info = {} if inspect_log is None else dict(_require_mapping(inspect_log, "inspect_log"))
    integrity_info = {} if integrity is None else dict(_require_mapping(integrity, "integrity"))
    sample_index = [
        {
            "query_id": sample["query_id"],
            "status": sample["status"],
            "valid_output": sample["valid_output"],
            "result": {
                "path": f"{sample['query_id']}/result.json",
                "sha256": sha256_json(sample),
            },
            "trace": {
                "path": str(
                    sample.get("artifacts", {})
                    .get("trace", {})
                    .get("path", f"{sample['query_id']}/trace.json")
                ),
                "sha256": sample.get("artifacts", {}).get("trace", {}).get("sha256"),
            },
        }
        for sample in sample_list
    ]
    counts = {
        "planned": planned,
        "started": started,
        "completed": completed,
        "valid": len(valid_samples),
        "invalid": len(invalid_samples),
        "failure_counts": failure_counts,
    }
    result = {
        "schema_version": TASK_RESULT_SCHEMA,
        "run_id": run_id,
        "task": task,
        "query_ids": selected_query_ids,
        "metric_ks": list(ks),
        "counts": counts,
        # Keep these concise aliases at the top level for line-oriented audit
        # tools, while ``counts`` remains the canonical grouped representation.
        "planned": planned,
        "started": started,
        "completed": completed,
        "valid": len(valid_samples),
        "invalid": len(invalid_samples),
        "failure_rate": failure_rate,
        "metrics": {"macro": metric_macro, "valid_only": metric_valid_only},
        "usage": usage_stats,
        "timing": {
            **timing_stats,
            "evaluation_elapsed_seconds": evaluation_elapsed_seconds,
            "launcher_preflight_seconds": launcher_preflight_seconds,
        },
        "model": model_info,
        "backend": backend_info,
        "data": data_info,
        "code": code_info,
        "samples": sample_index,
        "inspect_log": inspect_info,
        "integrity": integrity_info,
    }
    validate_task_result(result)
    return result


def aggregate_task_results(**kwargs: Any) -> dict[str, Any]:
    """Keyword-friendly alias for :func:`build_task_result`.

    ``aggregate_task_results(sample_results=...)`` is accepted in addition to
    ``build_task_result(samples=...)`` because adapters commonly name their
    completed records ``sample_results``.
    """

    if "sample_results" in kwargs:
        if "samples" in kwargs:
            raise TypeError("pass only one of sample_results and samples")
        kwargs["samples"] = kwargs.pop("sample_results")
    return build_task_result(**kwargs)


def _validate_stat(value: Any, *, availability: bool, name: str) -> None:
    source = _require_mapping(value, name)
    if availability:
        _require_number(source.get("available_count"), f"{name}.available_count", nonnegative=True)
        _require_number(source.get("missing_count"), f"{name}.missing_count", nonnegative=True)
        if "sum" not in source:
            raise ResultValidationError(f"{name}.sum is required")
    else:
        _require_number(source.get("count"), f"{name}.count", nonnegative=True)
    for field in ("mean", "min", "p50", "p95", "max"):
        field_value = source.get(field)
        if field_value is not None:
            _require_number(field_value, f"{name}.{field}")
    if not availability:
        std = source.get("population_stddev")
        count = source.get("count")
        if std is None and count != 0:
            raise ResultValidationError(f"{name}.population_stddev is required")
        if std is not None:
            _require_number(std, f"{name}.population_stddev", nonnegative=True)


def validate_task_result(value: Mapping[str, Any]) -> bool:
    source = _require_mapping(value, "task result")
    validate_no_sensitive_fields(source)
    if source.get("schema_version") != TASK_RESULT_SCHEMA:
        raise ResultValidationError(f"task result schema_version must be {TASK_RESULT_SCHEMA}")
    _require_string(source.get("run_id"), "run_id")
    _require_string(source.get("task"), "task")
    query_ids = source.get("query_ids")
    if not isinstance(query_ids, list) or any(not isinstance(query_id, str) for query_id in query_ids):
        raise ResultValidationError("query_ids must be an array of strings")
    ks = normalize_metric_ks(source.get("metric_ks"))
    counts = _require_mapping(source.get("counts"), "counts")
    for name in ("planned", "started", "completed", "valid", "invalid"):
        _require_number(counts.get(name), f"counts.{name}", nonnegative=True)
    if int(counts["planned"]) != len(query_ids):
        raise ResultValidationError("counts.planned must equal query_ids length")
    if int(counts["valid"]) + int(counts["invalid"]) != int(counts["completed"]):
        raise ResultValidationError("counts.valid + counts.invalid must equal counts.completed")
    failure_rate = _require_number(source.get("failure_rate"), "failure_rate")
    if not 0 <= float(failure_rate) <= 1:
        raise ResultValidationError("failure_rate must be between zero and one")
    metrics = _require_mapping(source.get("metrics"), "metrics")
    for subset in ("macro", "valid_only"):
        subset_map = _require_mapping(metrics.get(subset), f"metrics.{subset}")
        for name in metric_names(ks):
            _validate_stat(subset_map.get(name), availability=False, name=f"metrics.{subset}.{name}")
    usage = _require_mapping(source.get("usage"), "usage")
    for name in ("input_tokens", "output_tokens", "total_tokens", "cache_read_tokens", "cache_write_tokens"):
        _validate_stat(usage.get(name), availability=True, name=f"usage.{name}")
    timing = _require_mapping(source.get("timing"), "timing")
    for name in ("wall_time_seconds", "working_time_seconds"):
        _validate_stat(timing.get(name), availability=True, name=f"timing.{name}")
    for name in ("evaluation_elapsed_seconds", "launcher_preflight_seconds"):
        if timing.get(name) is not None:
            _require_number(timing[name], f"timing.{name}", nonnegative=True)
    for name in ("model", "backend", "data", "code", "inspect_log", "integrity"):
        _require_mapping(source.get(name), name)
    failure_counts = _require_mapping(counts.get("failure_counts"), "counts.failure_counts")
    for kind, count in failure_counts.items():
        if kind not in _STRICT_FAILURE_KINDS:
            raise ResultValidationError(f"unknown failure_counts kind: {kind}")
        _require_number(count, f"counts.failure_counts.{kind}", nonnegative=True)
    samples = source.get("samples")
    if not isinstance(samples, list) or len(samples) != len(query_ids):
        raise ResultValidationError("samples must index every selected query")
    if [sample.get("query_id") for sample in samples if isinstance(sample, Mapping)] != query_ids:
        raise ResultValidationError("samples must use the same query order as query_ids")
    for index, sample in enumerate(samples):
        sample_map = _require_mapping(sample, f"samples[{index}]")
        _require_string(sample_map.get("query_id"), f"samples[{index}].query_id")
        _require_string(sample_map.get("status"), f"samples[{index}].status")
        _require_bool(sample_map.get("valid_output"), f"samples[{index}].valid_output")
        result_ref = _require_mapping(sample_map.get("result"), f"samples[{index}].result")
        result_path = _require_string(result_ref.get("path"), f"samples[{index}].result.path")
        if Path(result_path).is_absolute() or ".." in Path(result_path).parts:
            raise ResultValidationError(f"samples[{index}].result.path must be relative")
        _validate_sha256(result_ref.get("sha256"), f"samples[{index}].result.sha256")
        trace_ref = _require_mapping(sample_map.get("trace"), f"samples[{index}].trace")
        trace_path = _require_string(trace_ref.get("path"), f"samples[{index}].trace.path")
        if Path(trace_path).is_absolute() or ".." in Path(trace_path).parts:
            raise ResultValidationError(f"samples[{index}].trace.path must be relative")
        trace_digest = trace_ref.get("sha256")
        if trace_digest is not None:
            _validate_sha256(trace_digest, f"samples[{index}].trace.sha256")
    validate_json_schema(source, TASK_RESULT_SCHEMA)
    return True


def is_valid_task_result(value: Any) -> bool:
    try:
        validate_task_result(value)
    except (ResultValidationError, TypeError, ValueError):
        return False
    return True


def compute_run_fingerprint(
    *,
    model: Mapping[str, Any] | None = None,
    backend: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    benchmark_contract: Mapping[str, Any] | str | None = None,
    metric_ks: Sequence[int] | None = None,
    query_ids: Sequence[str] = (),
    code: Mapping[str, Any] | None = None,
) -> str:
    """Fingerprint only reproducibility inputs, excluding run timestamps."""

    payload = {
        "model": {} if model is None else dict(model),
        "backend": {} if backend is None else dict(backend),
        "data": {} if data is None else dict(data),
        "benchmark_contract": benchmark_contract,
        "metric_ks": list(normalize_metric_ks(metric_ks)),
        "query_ids": [str(query_id) for query_id in query_ids],
        "code": {} if code is None else dict(code),
    }
    validate_no_sensitive_fields(payload)
    return sha256_json(payload)


def build_run_manifest(
    *,
    run_id: str,
    model_key: str,
    task: str,
    query_ids: Sequence[str],
    model: Mapping[str, Any] | None = None,
    backend: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    benchmark_contract: Mapping[str, Any] | str | None = None,
    metric_ks: Sequence[int] | None = None,
    code: Mapping[str, Any] | None = None,
    status: str = "running",
    created_at: str | None = None,
    run_fingerprint: str | None = None,
    max_total_model_tokens: int = DEFAULT_MAX_TOTAL_MODEL_TOKENS,
) -> dict[str, Any]:
    run_id = _require_string(run_id, "run_id")
    model_key = _require_string(model_key, "model_key")
    task = _require_string(task, "task")
    selected_query_ids = [
        _require_string(query_id, "query_ids item") for query_id in query_ids
    ]
    if len(set(selected_query_ids)) != len(selected_query_ids):
        raise ResultValidationError("query_ids must be unique")
    ks = normalize_metric_ks(metric_ks)
    model_info = {} if model is None else dict(_require_mapping(model, "model"))
    backend_info = {} if backend is None else dict(_require_mapping(backend, "backend"))
    data_info = {} if data is None else dict(_require_mapping(data, "data"))
    code_info = {} if code is None else dict(_require_mapping(code, "code"))
    computed_fingerprint = compute_run_fingerprint(
        model=model_info,
        backend=backend_info,
        data=data_info,
        benchmark_contract=benchmark_contract,
        metric_ks=ks,
        query_ids=selected_query_ids,
        code=code_info,
    )
    if run_fingerprint is not None:
        _validate_sha256(run_fingerprint, "run_fingerprint")
        if run_fingerprint != computed_fingerprint:
            raise ResultValidationError("supplied run_fingerprint does not match reproducibility inputs")
    else:
        run_fingerprint = computed_fingerprint
    max_total_model_tokens = int(
        _require_number(max_total_model_tokens, "max_total_model_tokens", nonnegative=True)
    )
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "model_key": model_key,
        "task": task,
        "created_at": created_at or _now_iso(),
        "status": _require_string(status, "status"),
        "run_fingerprint": run_fingerprint,
        "query_ids": selected_query_ids,
        "metric_ks": list(ks),
        "benchmark_contract": benchmark_contract,
        "model": model_info,
        "backend": backend_info,
        "data": data_info,
        "code": code_info,
        "budgets": {"max_total_model_tokens": max_total_model_tokens},
    }
    validate_run_manifest(manifest)
    return manifest


def validate_run_manifest(value: Mapping[str, Any]) -> bool:
    source = _require_mapping(value, "run manifest")
    validate_no_sensitive_fields(source)
    if source.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise ResultValidationError(f"run manifest schema_version must be {RUN_MANIFEST_SCHEMA}")
    for name in ("run_id", "model_key", "task", "created_at", "status"):
        _require_string(source.get(name), name)
    _validate_sha256(source.get("run_fingerprint"), "run_fingerprint")
    query_ids = source.get("query_ids")
    if not isinstance(query_ids, list) or any(not isinstance(item, str) for item in query_ids):
        raise ResultValidationError("run manifest query_ids must be an array of strings")
    ks = normalize_metric_ks(source.get("metric_ks"))
    model = _require_mapping(source.get("model"), "model")
    backend = _require_mapping(source.get("backend"), "backend")
    data = _require_mapping(source.get("data"), "data")
    code = _require_mapping(source.get("code"), "code")
    expected_fingerprint = compute_run_fingerprint(
        model=model,
        backend=backend,
        data=data,
        benchmark_contract=source.get("benchmark_contract"),
        metric_ks=ks,
        query_ids=query_ids,
        code=code,
    )
    if source.get("run_fingerprint") != expected_fingerprint:
        raise ResultValidationError("run_fingerprint does not match manifest reproducibility inputs")
    budgets = _require_mapping(source.get("budgets"), "budgets")
    _require_number(budgets.get("max_total_model_tokens"), "budgets.max_total_model_tokens", nonnegative=True)
    validate_json_schema(source, RUN_MANIFEST_SCHEMA)
    return True


def is_valid_run_manifest(value: Any) -> bool:
    try:
        validate_run_manifest(value)
    except (ResultValidationError, TypeError, ValueError):
        return False
    return True


def build_task_index(
    *,
    model_key: str,
    task: str,
    runs: Iterable[Mapping[str, Any]] = (),
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Build the model/task index without embedding sensitive run content."""

    model_key = _require_string(model_key, "model_key")
    task = _require_string(task, "task")
    entries: list[dict[str, Any]] = []
    for index, raw_run in enumerate(runs):
        run = _require_mapping(raw_run, f"runs[{index}]")
        run_id = _require_string(run.get("run_id"), f"runs[{index}].run_id")
        status = _require_string(run.get("status", "unknown"), f"runs[{index}].status")
        run_fp = run.get("run_fingerprint")
        if run_fp is not None:
            _validate_sha256(run_fp, f"runs[{index}].run_fingerprint")
        result_path = run.get("task_result_path", f"runs/{run_id}/task-result.json")
        result_path = _require_string(result_path, f"runs[{index}].task_result_path")
        if Path(result_path).is_absolute() or ".." in Path(result_path).parts:
            raise ResultValidationError("task result paths must be relative")
        entries.append(
            {
                "run_id": run_id,
                "status": status,
                "run_fingerprint": run_fp,
                "task_result_path": result_path,
            }
        )
    index = {
        "schema_version": TASK_INDEX_SCHEMA,
        "model_key": model_key,
        "task": task,
        "updated_at": updated_at or _now_iso(),
        "runs": entries,
    }
    validate_task_index(index)
    return index


def validate_task_index(value: Mapping[str, Any]) -> bool:
    source = _require_mapping(value, "task index")
    validate_no_sensitive_fields(source)
    if source.get("schema_version") != TASK_INDEX_SCHEMA:
        raise ResultValidationError(f"task index schema_version must be {TASK_INDEX_SCHEMA}")
    _require_string(source.get("model_key"), "model_key")
    _require_string(source.get("task"), "task")
    _require_string(source.get("updated_at"), "updated_at")
    runs = source.get("runs")
    if not isinstance(runs, list):
        raise ResultValidationError("task index runs must be an array")
    seen: set[str] = set()
    for index, run in enumerate(runs):
        run_map = _require_mapping(run, f"runs[{index}]")
        run_id = _require_string(run_map.get("run_id"), f"runs[{index}].run_id")
        if run_id in seen:
            raise ResultValidationError(f"duplicate run_id in task index: {run_id}")
        seen.add(run_id)
        _require_string(run_map.get("status"), f"runs[{index}].status")
        if run_map.get("run_fingerprint") is not None:
            _validate_sha256(run_map["run_fingerprint"], f"runs[{index}].run_fingerprint")
        result_path = _require_string(run_map.get("task_result_path"), f"runs[{index}].task_result_path")
        if Path(result_path).is_absolute() or ".." in Path(result_path).parts:
            raise ResultValidationError("task result paths must be relative")
    validate_json_schema(source, TASK_INDEX_SCHEMA)
    return True


def is_valid_task_index(value: Any) -> bool:
    try:
        validate_task_index(value)
    except (ResultValidationError, TypeError, ValueError):
        return False
    return True


def _safe_component(value: str, name: str) -> str:
    value = _require_string(value, name)
    if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ResultValidationError(f"{name} must be a single safe path component")
    return value


def run_directory(
    results_root: str | os.PathLike[str],
    *,
    model_key: str,
    task: str,
    run_id: str,
) -> Path:
    """Return the canonical ``results/model/task/runs/run`` directory."""

    return (
        Path(results_root)
        / _safe_component(model_key, "model_key")
        / _safe_component(task, "task")
        / "runs"
        / _safe_component(run_id, "run_id")
    )


def _write_new_json(path: Path, value: Mapping[str, Any]) -> Path:
    """Write once, refusing to mutate an existing protocol artifact."""

    if path.exists():
        raise FileExistsError(f"immutable result artifact already exists: {path}")
    return write_json_atomic(path, value)


def write_run_manifest(run_root: str | os.PathLike[str], manifest: Mapping[str, Any]) -> Path:
    validate_run_manifest(manifest)
    return _write_new_json(Path(run_root) / "run-manifest.json", manifest)


def write_sample_result(
    run_root: str | os.PathLike[str],
    sample: Mapping[str, Any],
    *,
    query_directory: str | None = None,
) -> Path:
    """Atomically publish one sample result under ``Qxx/result.json``.

    The caller should stage ``final.json`` and ``trace.json`` before calling
    this helper, then include their hashes in ``sample["artifacts"]``.  The
    helper refuses overwrite so a completed sample cannot be silently changed.
    """

    validate_sample_result(sample)
    query_dir = query_directory or str(sample["query_id"])
    query_dir = _safe_component(query_dir, "query_directory")
    destination = Path(run_root) / query_dir / "result.json"
    return _write_new_json(destination, sample)


def write_task_result(run_root: str | os.PathLike[str], task_result: Mapping[str, Any]) -> Path:
    validate_task_result(task_result)
    return _write_new_json(Path(run_root) / "task-result.json", task_result)


def write_task_index(
    task_root: str | os.PathLike[str],
    index: Mapping[str, Any],
) -> Path:
    validate_task_index(index)
    return _write_new_json(Path(task_root) / "index.json", index)


def _load_task_index(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = read_json(path)
    validate_task_index(payload)
    return dict(payload)


def append_task_index_run(
    index_path: str | os.PathLike[str],
    *,
    model_key: str,
    task: str,
    run: Mapping[str, Any],
) -> Path:
    """Atomically append one run to ``index.json`` while preserving history.

    The append operation rejects an existing ``run_id``.  This is the safe
    operation for creating a new immutable run; use
    :func:`update_task_index_run` when only a pre-existing run's status/path
    needs lifecycle bookkeeping.
    """

    path = Path(index_path)
    current = _load_task_index(path)
    run_map = _require_mapping(run, "run index entry")
    run_id = _require_string(run_map.get("run_id"), "run.run_id")
    normalized_run = dict(run_map)
    normalized_run.setdefault("status", "unknown")
    normalized_run.setdefault("run_fingerprint", None)
    normalized_run.setdefault("task_result_path", f"runs/{run_id}/task-result.json")
    if current is None:
        index = build_task_index(model_key=model_key, task=task, runs=[normalized_run])
    else:
        if current["model_key"] != model_key or current["task"] != task:
            raise ResultIntegrityError("task index identity does not match append request")
        if any(entry["run_id"] == run_id for entry in current["runs"]):
            raise FileExistsError(f"run_id already exists in task index: {run_id}")
        index = dict(current)
        index["runs"] = list(current["runs"]) + [normalized_run]
        index["updated_at"] = _now_iso()
        # Validation also normalizes path/fingerprint constraints before the
        # old index is replaced.
        validate_task_index(index)
    return write_json_atomic(path, index)


def update_task_index_run(
    index_path: str | os.PathLike[str],
    *,
    run: Mapping[str, Any],
) -> Path:
    """Atomically update an existing run entry without dropping other runs."""

    path = Path(index_path)
    current = _load_task_index(path)
    if current is None:
        raise FileNotFoundError(f"task index does not exist: {path}")
    run_map = _require_mapping(run, "run index entry")
    run_id = _require_string(run_map.get("run_id"), "run.run_id")
    normalized_run = dict(run_map)
    normalized_run.setdefault("status", "unknown")
    normalized_run.setdefault("run_fingerprint", None)
    normalized_run.setdefault("task_result_path", f"runs/{run_id}/task-result.json")
    entries = list(current["runs"])
    for index, entry in enumerate(entries):
        if entry["run_id"] == run_id:
            entries[index] = normalized_run
            break
    else:
        raise KeyError(f"run_id not found in task index: {run_id}")
    updated = dict(current)
    updated["runs"] = entries
    updated["updated_at"] = _now_iso()
    validate_task_index(updated)
    return write_json_atomic(path, updated)


# Short aliases make the lifecycle API discoverable without weakening the
# explicit append/update semantics above.
append_run_to_index = append_task_index_run
update_run_in_index = update_task_index_run


def _read_sample_result_path(run_root: Path, query_id: str) -> tuple[Path, Mapping[str, Any]]:
    query_dir = _safe_component(query_id, "query_id")
    result_path = run_root / query_dir / "result.json"
    payload = read_json(result_path)
    validate_sample_result(payload)
    return result_path, payload


def finalize_run(
    run_root: str | os.PathLike[str],
    *,
    manifest: Mapping[str, Any],
    task_result: Mapping[str, Any],
    complete_marker: str = "_COMPLETE",
) -> Path:
    """Verify a run and atomically publish its immutable completion marker.

    This is intentionally strict: every selected query must have a valid
    ``result.json`` whose digest matches the task result index, and the task
    aggregate must already agree with the manifest.  Existing completion
    markers are never replaced.
    """

    validate_run_manifest(manifest)
    validate_task_result(task_result)
    if manifest["run_id"] != task_result["run_id"] or manifest["task"] != task_result["task"]:
        raise ResultIntegrityError("manifest and task result identity mismatch")
    if list(manifest["query_ids"]) != list(task_result["query_ids"]):
        raise ResultIntegrityError("manifest and task result query order mismatch")
    root = Path(run_root)
    marker = root / complete_marker
    if marker.exists():
        raise FileExistsError(f"run is already complete: {marker}")
    indexed = {entry["query_id"]: entry for entry in task_result["samples"]}
    for query_id in task_result["query_ids"]:
        result_path, sample = _read_sample_result_path(root, query_id)
        index_entry = indexed[query_id]
        result_ref = index_entry["result"]
        expected_path = f"{query_id}/result.json"
        if result_ref["path"] != expected_path:
            raise ResultIntegrityError(
                f"sample result path mismatch for {query_id}: {result_ref['path']}"
            )
        expected = result_ref["sha256"]
        actual = sha256_json(sample)
        if actual != expected:
            raise ResultIntegrityError(f"sample result digest mismatch for {query_id}")
        # The path must remain inside this run even when a caller supplies an
        # artifact record from another adapter.
        for artifact in sample.get("artifacts", {}).values():
            verify_artifact_record(artifact, root=root)
        trace_ref = index_entry["trace"]
        trace_path = root / trace_ref["path"]
        if trace_ref["sha256"] is not None:
            verify_artifact_record(trace_ref, root=root)
        elif not trace_path.is_file():
            raise ResultIntegrityError(f"missing trajectory artifact for {query_id}: {trace_ref['path']}")
        del result_path
    task_result_path = root / "task-result.json"
    if not task_result_path.is_file():
        raise ResultIntegrityError("task-result.json must be written before finalization")
    on_disk_task = read_json(task_result_path)
    validate_task_result(on_disk_task)
    if sha256_json(on_disk_task) != sha256_json(task_result):
        raise ResultIntegrityError("on-disk task-result.json differs from supplied aggregate")
    marker_payload = {
        "schema_version": "dci-run-complete-v1",
        "run_id": manifest["run_id"],
        "run_fingerprint": manifest["run_fingerprint"],
        "task_result_sha256": sha256_json(task_result),
    }
    return _write_new_json(marker, marker_payload)


def validate_trajectory(value: Mapping[str, Any]) -> bool:
    """Validate the minimum trajectory envelope while preserving event freedom."""

    source = _require_mapping(value, "trajectory")
    validate_no_sensitive_fields(source)
    if source.get("schema_version") != TRAJECTORY_SCHEMA:
        raise ResultValidationError(f"trajectory schema_version must be {TRAJECTORY_SCHEMA}")
    for name in ("run_id", "task", "query_id"):
        _require_string(source.get(name), name)
    for name in ("started_at", "ended_at"):
        if source.get(name) is not None:
            _require_utc_timestamp(source[name], name)
        elif name not in source:
            raise ResultValidationError(f"trajectory.{name} is required")
    events = source.get("events")
    if not isinstance(events, list):
        raise ResultValidationError("trajectory.events must be an array")
    validate_json_schema(source, TRAJECTORY_SCHEMA)
    return True


def build_trajectory(
    *,
    run_id: str,
    task: str,
    query_id: str,
    events: Sequence[Mapping[str, Any]],
    started_at: str | None = None,
    ended_at: str | None = None,
) -> dict[str, Any]:
    trajectory = {
        "schema_version": TRAJECTORY_SCHEMA,
        "run_id": _require_string(run_id, "run_id"),
        "task": _require_string(task, "task"),
        "query_id": _require_string(query_id, "query_id"),
        "started_at": started_at,
        "ended_at": ended_at,
        "events": list(events),
    }
    validate_trajectory(trajectory)
    return trajectory


__all__ = [
    "DEFAULT_MAX_TOTAL_MODEL_TOKENS",
    "DEFAULT_METRIC_KS",
    "FAILURE_KINDS",
    "METRIC_FAMILIES",
    "ResultIntegrityError",
    "ResultValidationError",
    "RUN_MANIFEST_SCHEMA",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "SAMPLE_RESULT_SCHEMA",
    "SAMPLE_RESULT_SCHEMA_VERSION",
    "TASK_INDEX_SCHEMA",
    "TASK_INDEX_SCHEMA_VERSION",
    "TASK_RESULT_SCHEMA",
    "TASK_RESULT_SCHEMA_VERSION",
    "TRAJECTORY_SCHEMA",
    "TRAJECTORY_SCHEMA_VERSION",
    "artifact_record",
    "append_run_to_index",
    "append_task_index_run",
    "build_artifacts",
    "build_run_manifest",
    "build_sample_result",
    "build_task_index",
    "build_task_result",
    "build_trajectory",
    "aggregate_task_results",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "compute_run_fingerprint",
    "fingerprint",
    "finalize_run",
    "is_valid_run_manifest",
    "is_valid_sample_result",
    "is_valid_task_index",
    "is_valid_task_result",
    "metric_names",
    "normalize_metric_ks",
    "normalize_usage",
    "r7_p50",
    "r7_p95",
    "r7_percentile",
    "read_json",
    "run_directory",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "stable_json_sha256",
    "validate_no_sensitive_fields",
    "validate_json_schema",
    "validate_run_manifest",
    "validate_sample_result",
    "validate_task_index",
    "validate_task_result",
    "validate_trajectory",
    "verify_artifact_record",
    "write_json_atomic",
    "write_run_manifest",
    "write_sample_result",
    "write_task_index",
    "write_task_result",
    "update_run_in_index",
    "update_task_index_run",
]
