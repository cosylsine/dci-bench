"""Create and validate backend-neutral serving audit manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


BACKEND_MANIFEST_SCHEMA_VERSION = "dci-backend-manifest-v1"
_MODEL_KEY_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def sanitize_base_url(value: str) -> str:
    """Return a credential-free HTTP(S) endpoint suitable for an audit record."""

    parsed = urlsplit(value.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("backend base URL must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("backend base URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("backend base URL must not contain a query or fragment")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _model_revision(model_path: Path) -> tuple[str | None, str | None, dict[str, Any]]:
    candidates = (model_path / ".hfd" / "repo_metadata.json", model_path / "repo_metadata.json")
    for candidate in candidates:
        if candidate.is_file():
            metadata = _read_json(candidate)
            revision = metadata.get("sha")
            source_id = metadata.get("id") or metadata.get("modelId")
            return (
                str(source_id) if source_id else None,
                str(revision) if revision else None,
                metadata,
            )
    return None, None, {}


def _weight_manifest(model_path: Path, repo_metadata: dict[str, Any]) -> dict[str, Any]:
    metadata_by_name = {
        str(item.get("rfilename")): item
        for item in repo_metadata.get("siblings", [])
        if isinstance(item, dict) and item.get("rfilename")
    }
    entries: list[dict[str, Any]] = []
    for path in sorted((*model_path.glob("*.safetensors"), *model_path.glob("*.bin"))):
        metadata = metadata_by_name.get(path.name, {})
        lfs = metadata.get("lfs") if isinstance(metadata.get("lfs"), dict) else {}
        digest = lfs.get("sha256")
        entries.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": str(digest) if digest else None,
                "digest_source": "huggingface_lfs" if digest else None,
            }
        )
    if not entries:
        raise ValueError(f"no model weight files found under {model_path}")
    return {
        "files": entries,
        "all_digests_available": all(item["sha256"] is not None for item in entries),
        "manifest_sha256": _canonical_sha256(entries),
    }


def build_sglang_manifest(
    *,
    model_key: str,
    model_path: Path,
    served_model_name: str,
    base_url: str,
    sglang_version: str,
    sglang_git_revision: str | None,
    tool_call_parser: str,
    sampling_backend: str,
    tensor_parallel_size: int,
    context_length: int,
    dtype: str,
    generation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not _MODEL_KEY_RE.fullmatch(model_key):
        raise ValueError("model_key must match [A-Za-z0-9._-]+")
    if tensor_parallel_size < 1 or context_length < 1:
        raise ValueError("tensor_parallel_size and context_length must be positive")
    model_path = model_path.resolve(strict=True)
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"missing model config: {config_path}")
    config = _read_json(config_path)
    source_id, revision, repo_metadata = _model_revision(model_path)
    source_id = source_id or (str(config.get("_name_or_path")) if config.get("_name_or_path") else None)
    payload: dict[str, Any] = {
        "schema_version": BACKEND_MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "model_key": model_key,
            "source_id": source_id,
            "source_path": str(model_path),
            "served_model_name": served_model_name,
            "revision": revision,
            "config_sha256": _sha256_bytes(config_path.read_bytes()),
            "weights": _weight_manifest(model_path, repo_metadata),
        },
        "backend": {
            "kind": "sglang",
            "version": sglang_version,
            "git_revision": sglang_git_revision,
            "openai_service": "sglang",
            "base_url": sanitize_base_url(base_url),
            "tool_call_parser": tool_call_parser,
            "sampling_backend": sampling_backend,
            "tensor_parallel_size": tensor_parallel_size,
            "context_length": context_length,
            "dtype": dtype,
            "generation_config": generation_config
            or {"temperature": 0, "max_tokens": 1024, "enable_thinking": False},
        },
        "capabilities": {
            "usage_tokens_required": True,
            "provider_cost_available": False,
            "retry_policy": None,
            "rate_limit_policy": None,
        },
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def validate_backend_manifest(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != BACKEND_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported backend manifest schema: {payload.get('schema_version')!r}")
    model = payload.get("model")
    backend = payload.get("backend")
    if not isinstance(model, dict) or not isinstance(backend, dict):
        raise ValueError("backend manifest requires model and backend objects")
    required_model = ("model_key", "source_path", "served_model_name", "config_sha256", "weights")
    required_backend = (
        "kind",
        "version",
        "openai_service",
        "base_url",
        "tool_call_parser",
        "sampling_backend",
        "tensor_parallel_size",
        "context_length",
        "dtype",
        "generation_config",
    )
    missing = [f"model.{key}" for key in required_model if key not in model]
    missing += [f"backend.{key}" for key in required_backend if key not in backend]
    if missing:
        raise ValueError(f"backend manifest missing fields: {', '.join(missing)}")
    if not _MODEL_KEY_RE.fullmatch(str(model["model_key"])):
        raise ValueError("backend manifest contains an unsafe model_key")
    sanitized = sanitize_base_url(str(backend["base_url"]))
    if sanitized != backend["base_url"]:
        raise ValueError("backend manifest base_url is not canonical")
    recorded_digest = payload.get("manifest_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if recorded_digest != _canonical_sha256(unsigned):
        raise ValueError("backend manifest digest mismatch")


def load_backend_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    validate_backend_manifest(payload)
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--sglang-version", required=True)
    parser.add_argument("--sglang-git-revision")
    parser.add_argument("--tool-call-parser", required=True)
    parser.add_argument("--sampling-backend", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--dtype", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_sglang_manifest(
        model_key=args.model_key,
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        base_url=args.base_url,
        sglang_version=args.sglang_version,
        sglang_git_revision=args.sglang_git_revision,
        tool_call_parser=args.tool_call_parser,
        sampling_backend=args.sampling_backend,
        tensor_parallel_size=args.tensor_parallel_size,
        context_length=args.context_length,
        dtype=args.dtype,
    )
    write_json_atomic(args.output, manifest)
    print(args.output)


if __name__ == "__main__":
    main()
