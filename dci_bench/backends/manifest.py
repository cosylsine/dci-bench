"""Create and validate backend-neutral serving audit manifests."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from dci_bench.protocol.contracts import GENERATION_CONFIG, MAX_MODEL_OUTPUT_TOKENS


BACKEND_MANIFEST_SCHEMA_VERSION = "dci-backend-manifest-v1"
_MODEL_KEY_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _resolved_generation_config(
    generation_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the contract generation config, rejecting backend drift.

    Inspect owns the generation settings for the model call.  A backend
    manifest may repeat the settings for audit purposes, but it cannot invent
    a second source of truth.  This check therefore runs both while building
    and while loading a manifest.
    """

    expected = deepcopy(GENERATION_CONFIG)
    if generation_config is not None:
        try:
            same = json.dumps(
                dict(generation_config), sort_keys=True, separators=(",", ":")
            ) == json.dumps(expected, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("backend generation_config is not JSON-compatible") from exc
        if not same:
            raise ValueError(
                "backend generation_config disagrees with the shared protocol contract"
            )
    return expected


def resolve_generation_config(
    backend: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve and validate generation settings used by Inspect.

    ``max_model_output_tokens`` is the contract name for the same output cap
    that Inspect receives as its canonical ``max_tokens`` generation kwarg.
    Provider-specific ``enable_thinking`` is carried through ``extra_body``;
    it is intentionally not passed as an unknown Inspect generation field.
    """

    section = {} if backend is None else backend
    if not isinstance(section, Mapping):
        raise ValueError("backend generation settings must be an object")
    generation_config = _resolved_generation_config(section.get("generation_config"))
    recorded_max = section.get("max_model_output_tokens")
    if recorded_max is not None and (
        isinstance(recorded_max, bool)
        or not isinstance(recorded_max, int)
        or recorded_max != MAX_MODEL_OUTPUT_TOKENS
    ):
        raise ValueError(
            "backend max_model_output_tokens disagrees with the shared protocol contract"
        )
    return {
        "generation_config": generation_config,
        "max_model_output_tokens": MAX_MODEL_OUTPUT_TOKENS,
    }


def inspect_generation_kwargs(
    backend: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the actual Inspect ``eval`` generation kwargs.

    Only fields understood by Inspect 0.3.260 are emitted as top-level
    kwargs.  ``enable_thinking`` is an OpenAI-compatible provider extension
    and is sent in ``extra_body.chat_template_kwargs``.
    """

    resolved = resolve_generation_config(backend)
    config = resolved["generation_config"]
    kwargs = {
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "seed": config["seed"],
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": config["enable_thinking"],
            }
        },
    }
    if backend is not None and backend.get("kind") == "remote_api":
        # chat_template_kwargs is a local serving extension, not an API field.
        kwargs.pop("extra_body")
    return kwargs


def resolved_generation_config_record(
    backend: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an audit record describing the resolved Inspect authority."""

    resolved = resolve_generation_config(backend)
    return {
        "authority": "inspect.eval",
        "generation_config": deepcopy(resolved["generation_config"]),
        "max_model_output_tokens": resolved["max_model_output_tokens"],
    }


def _load_runtime_info(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"runtime info must be a regular JSON file: {path}")
    payload = _read_json(path)
    return payload


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
        if digest is not None and (not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest)):
            raise ValueError(f"invalid Hugging Face LFS SHA-256 for {path.name}")
        recorded_size = lfs.get("size")
        if recorded_size is not None:
            if isinstance(recorded_size, bool) or not isinstance(recorded_size, int) or recorded_size < 0:
                raise ValueError(f"invalid Hugging Face LFS size for {path.name}")
            if path.stat().st_size != recorded_size:
                raise ValueError(
                    f"local model weight size differs from Hugging Face LFS metadata: {path.name}"
                )
        digest_source = "huggingface_lfs" if digest else "local_sha256"
        entries.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": str(digest) if digest else _sha256_file(path),
                "digest_source": digest_source,
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
    runtime_info_path: Path | None = None,
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
            "generation_config": _resolved_generation_config(generation_config),
            "max_model_output_tokens": MAX_MODEL_OUTPUT_TOKENS,
        },
        "capabilities": {
            "usage_tokens_required": True,
            "provider_cost_available": False,
            "retry_policy": None,
            "rate_limit_policy": None,
        },
    }
    runtime_info = _load_runtime_info(runtime_info_path)
    if runtime_info is not None:
        payload["backend"]["runtime"] = runtime_info
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def build_vllm_manifest(
    *,
    model_key: str,
    model_path: Path,
    served_model_name: str,
    base_url: str,
    vllm_version: str,
    tool_call_parser: str,
    reasoning_parser: str | None,
    language_model_only: bool,
    tensor_parallel_size: int,
    context_length: int,
    dtype: str,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    generation_config: dict[str, Any] | None = None,
    runtime_info_path: Path | None = None,
) -> dict[str, Any]:
    """Build the same audited backend contract for a vLLM OpenAI endpoint."""

    if max_num_seqs < 1 or not 0 < gpu_memory_utilization < 1:
        raise ValueError("vLLM max_num_seqs and gpu_memory_utilization are out of range")

    payload = build_sglang_manifest(
        model_key=model_key,
        model_path=model_path,
        served_model_name=served_model_name,
        base_url=base_url,
        sglang_version=vllm_version,
        sglang_git_revision=None,
        tool_call_parser=tool_call_parser,
        sampling_backend="auto",
        tensor_parallel_size=tensor_parallel_size,
        context_length=context_length,
        dtype=dtype,
        generation_config=generation_config,
        runtime_info_path=runtime_info_path,
    )
    payload["backend"].update(
        kind="vllm",
        openai_service="vllm",
        reasoning_parser=reasoning_parser,
        language_model_only=language_model_only,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    payload["manifest_sha256"] = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    return payload


def build_api_manifest(*, model_key: str, served_model_name: str, provider: str,
                       revision: str, base_url: str, context_length: int = 65536) -> dict[str, Any]:
    """Pin a remote model identity without fabricating a local weight receipt."""
    if not provider.strip() or not revision.strip():
        raise ValueError("remote API requires provider and pinned model revision")
    payload = {
        "schema_version": BACKEND_MANIFEST_SCHEMA_VERSION,
        "model": {"model_key": model_key, "served_model_name": served_model_name,
                  "identity_kind": "provider_revision", "provider": provider, "revision": revision},
        "backend": {"kind": "remote_api", "version": "openai-compatible-chat-v1",
                    "openai_service": provider, "base_url": sanitize_base_url(base_url),
                    "tool_call_parser": "provider_native", "sampling_backend": "provider",
                    "tensor_parallel_size": None, "context_length": context_length, "dtype": None,
                    "generation_config": deepcopy(GENERATION_CONFIG),
                    "max_model_output_tokens": MAX_MODEL_OUTPUT_TOKENS},
        "capabilities": {"usage_tokens_required": True, "provider_cost_available": False,
                         "retry_policy": "no_implicit_retries", "rate_limit_policy": "configured_concurrency"},
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    validate_backend_manifest(payload)
    return payload


def validate_backend_manifest(
    payload: dict[str, Any], *, generation_contract: Mapping[str, Any] | None = None,
) -> None:
    if payload.get("schema_version") != BACKEND_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported backend manifest schema: {payload.get('schema_version')!r}")
    model = payload.get("model")
    backend = payload.get("backend")
    if not isinstance(model, dict) or not isinstance(backend, dict):
        raise ValueError("backend manifest requires model and backend objects")
    remote = backend.get("kind") == "remote_api"
    required_model = (("model_key", "served_model_name", "identity_kind", "provider", "revision") if remote
                      else ("model_key", "source_path", "served_model_name", "config_sha256", "weights"))
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
        "max_model_output_tokens",
    )
    missing = [f"model.{key}" for key in required_model if key not in model]
    missing += [f"backend.{key}" for key in required_backend if key not in backend]
    if missing:
        raise ValueError(f"backend manifest missing fields: {', '.join(missing)}")
    if not _MODEL_KEY_RE.fullmatch(str(model["model_key"])):
        raise ValueError("backend manifest contains an unsafe model_key")
    if remote:
        if model["identity_kind"] != "provider_revision" or any(
            not isinstance(model[key], str) or not model[key].strip()
            for key in ("provider", "revision", "served_model_name")
        ):
            raise ValueError("remote backend requires explicit provider/model/revision identity")
    else:
        weights = model["weights"]
        if not isinstance(weights, dict) or not isinstance(weights.get("files"), list) or not weights["files"]:
            raise ValueError("backend manifest weights.files must be a non-empty array")
        for index, entry in enumerate(weights["files"]):
            if not isinstance(entry, dict):
                raise ValueError(f"backend manifest weights.files[{index}] must be an object")
            digest = entry.get("sha256")
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise ValueError(f"backend manifest weights.files[{index}].sha256 must be SHA-256")
            if entry.get("digest_source") not in {"huggingface_lfs", "local_sha256"}:
                raise ValueError(f"backend manifest weights.files[{index}].digest_source is unsupported")
        if weights.get("all_digests_available") is not True:
            raise ValueError("backend manifest must contain SHA-256 for every model weight")
        if weights.get("manifest_sha256") != _canonical_sha256(weights["files"]):
            raise ValueError("backend manifest weight digest mismatch")
    if "runtime" in backend and not isinstance(backend["runtime"], dict):
        raise ValueError("backend manifest runtime must be a JSON object")
    if generation_contract is None:
        resolve_generation_config(backend)
    else:
        # An independent track can pin its own generation authority. The
        # default document path still resolves the existing process profile.
        for key in ("generation_config", "max_model_output_tokens"):
            if json.dumps(backend[key], sort_keys=True, allow_nan=False) != json.dumps(
                generation_contract[key], sort_keys=True, allow_nan=False
            ):
                raise ValueError(f"backend {key} differs from the selected track contract")
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
    versions = parser.add_mutually_exclusive_group(required=True)
    versions.add_argument("--sglang-version")
    versions.add_argument("--vllm-version")
    parser.add_argument("--sglang-git-revision")
    parser.add_argument("--reasoning-parser")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--tool-call-parser", required=True)
    parser.add_argument("--sampling-backend", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument(
        "--runtime-info",
        type=Path,
        help="optional serving-runtime.json to embed in the backend manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    common = dict(
        model_key=args.model_key,
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        base_url=args.base_url,
        tool_call_parser=args.tool_call_parser,
        tensor_parallel_size=args.tensor_parallel_size,
        context_length=args.context_length,
        dtype=args.dtype,
        runtime_info_path=args.runtime_info,
    )
    if args.vllm_version:
        if args.max_num_seqs is None or args.gpu_memory_utilization is None:
            raise SystemExit("vLLM manifest requires --max-num-seqs and --gpu-memory-utilization")
        manifest = build_vllm_manifest(
            **common,
            vllm_version=args.vllm_version,
            reasoning_parser=args.reasoning_parser,
            language_model_only=args.language_model_only,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    else:
        manifest = build_sglang_manifest(
            **common,
            sglang_version=args.sglang_version,
            sglang_git_revision=args.sglang_git_revision,
            sampling_backend=args.sampling_backend,
        )
    write_json_atomic(args.output, manifest)
    print(args.output)


if __name__ == "__main__":
    main()
