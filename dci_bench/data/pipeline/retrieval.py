"""Offline lexical and dense retrieval helpers for DCI data bundles.

The dense path is intentionally an audit/export path: it loads a local
sentence-transformers model only when ``encode`` is called, never downloads a
model, computes query/document dot products in document chunks, and preserves
the input order for equal scores.  The saved export is compatible with the
existing :mod:`dci_bench.data.document_pools` validator.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, asdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import unicodedata
from typing import Any, Iterator


DEFAULT_EMBEDDING_MODEL_PATH = "/mnt/afs/share/embedding_models/KaLM-embedding-multilingual-mini-instruct-v2_5"
EMBEDDING_MODEL_PATH = DEFAULT_EMBEDDING_MODEL_PATH
SAVED_RETRIEVAL_SCHEMA_VERSION = "dci-saved-retrieval-v1"
DEFAULT_QUERY_PROMPT = "Instruct: Given a query, retrieve documents that answer the query \n Query: "
DEFAULT_DOCUMENT_PROMPT = ""
DEFAULT_TOKEN_MAX_LENGTH = 512


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: Any) -> str:
    if isinstance(value, bytes):
        data = value
    elif isinstance(value, str):
        data = value.encode("utf-8")
    else:
        data = _stable_json(value).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_identity(model_path: str | os.PathLike[str], *, component: str) -> str:
    """Hash lightweight model/tokenizer files used by cache identity.

    This is intentionally separate from ``model_weight_identity``: an
    explicitly pinned weight revision may avoid rehashing multi-gigabyte
    weights, while prompt/tokenizer/config changes must still invalidate a
    cached vector.
    """

    path = Path(model_path)
    if component == "tokenizer":
        names = {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
            "spiece.model",
            "tokenizer.model",
        }
        files = [child for child in path.rglob("*") if child.is_file() and child.name in names]
    elif component == "model_config":
        names = {"config.json", "config_sentence_transformers.json", "modules.json", "sentence_bert_config.json"}
        files = [
            child
            for child in path.rglob("*")
            if child.is_file() and (child.name in names or child.as_posix().endswith("1_Pooling/config.json"))
        ]
    else:
        raise ValueError("component must be tokenizer or model_config")
    payload = []
    for child in sorted(files):
        payload.append({
            "path": child.relative_to(path).as_posix(),
            "bytes": child.stat().st_size,
            "sha256": _sha256_file(child),
        })
    return _sha256({"path": str(path.resolve()), "component": component, "files": payload})


def _np():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("dense retrieval requires numpy") from exc
    return np


@dataclass(frozen=True)
class EmbeddingConfig:
    """Pinned local encoder settings included in cache and export identities."""

    model_path: str = DEFAULT_EMBEDDING_MODEL_PATH
    trust_remote_code: bool = True
    query_prompt: str = DEFAULT_QUERY_PROMPT
    document_prompt: str = DEFAULT_DOCUMENT_PROMPT
    max_length: int | None = DEFAULT_TOKEN_MAX_LENGTH
    pooling: str = "mean"
    normalize: bool = True
    truncation_unit: str = "tokens"
    character_max_length: int | None = None
    device: str = "cpu"
    batch_size: int = 32
    chunk_size: int = 4096
    weight_identity: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.model_path, os.PathLike):
            object.__setattr__(self, "model_path", os.fspath(self.model_path))
        if self.model_path != DEFAULT_EMBEDDING_MODEL_PATH:
            # The default is the project-pinned KaLM model.  Alternate local
            # paths are useful for tests and explicit audits, but must remain
            # local and are never resolved through a hub.
            if not isinstance(self.model_path, str) or not self.model_path:
                raise ValueError("model_path must be a non-empty local path")
        if not isinstance(self.trust_remote_code, bool):
            raise ValueError("trust_remote_code must be boolean")
        if self.model_path == DEFAULT_EMBEDDING_MODEL_PATH and not self.trust_remote_code:
            raise ValueError("KaLM embedding requires trust_remote_code=True")
        if self.max_length is not None and (isinstance(self.max_length, bool) or self.max_length <= 0):
            raise ValueError("max_length must be positive or None")
        if self.truncation_unit != "tokens":
            raise ValueError("max_length is a tokenizer token limit; use character_max_length for codepoint limits")
        if self.character_max_length is not None and (
            isinstance(self.character_max_length, bool) or self.character_max_length <= 0
        ):
            raise ValueError("character_max_length must be positive or None")
        if self.batch_size <= 0 or self.chunk_size <= 0:
            raise ValueError("batch_size and chunk_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_local_model(
        cls,
        model_path: str | os.PathLike[str] = DEFAULT_EMBEDDING_MODEL_PATH,
        **overrides: Any,
    ) -> "EmbeddingConfig":
        """Read prompt and tokenizer limits from a local model directory."""

        path = Path(model_path)
        values: dict[str, Any] = {"model_path": os.fspath(path)}
        prompt_path = path / "config_sentence_transformers.json"
        if prompt_path.is_file():
            try:
                payload = json.loads(prompt_path.read_text(encoding="utf-8"))
                prompts = payload.get("prompts", {})
                if isinstance(prompts, Mapping):
                    if isinstance(prompts.get("query"), str):
                        values["query_prompt"] = prompts["query"]
                    if isinstance(prompts.get("document"), str):
                        values["document_prompt"] = prompts["document"]
            except (OSError, json.JSONDecodeError):
                pass
        tokenizer_path = path / "tokenizer_config.json"
        if tokenizer_path.is_file():
            try:
                tokenizer_config = json.loads(tokenizer_path.read_text(encoding="utf-8"))
                if isinstance(tokenizer_config.get("max_length"), int) and tokenizer_config["max_length"] > 0:
                    values["max_length"] = tokenizer_config["max_length"]
            except (OSError, json.JSONDecodeError):
                pass
        pooling_path = path / "1_Pooling" / "config.json"
        if pooling_path.is_file():
            try:
                pooling = json.loads(pooling_path.read_text(encoding="utf-8"))
                if pooling.get("pooling_mode_mean_tokens") is True:
                    values["pooling"] = "mean"
                elif pooling.get("pooling_mode_cls_token") is True:
                    values["pooling"] = "cls"
            except (OSError, json.JSONDecodeError):
                pass
        values.update(overrides)
        return cls(**values)

    from_local_files = from_local_model


def model_weight_identity(model_path: str | os.PathLike[str], explicit: str | None = None) -> str:
    """Return a content identity for local weights, code, and tokenizer files.

    Weight files are hashed by content, rather than represented by an index,
    size, or mtime alone.  A caller may provide an already verified revision
    or weight hash explicitly to avoid repeating the local hash calculation.
    """

    if explicit:
        return explicit
    path = Path(model_path)
    if not path.exists():
        return _sha256({"path": str(path), "missing": True})
    files: list[dict[str, Any]] = []
    candidates = [
        child for child in path.rglob("*")
        if child.is_file()
        and (
            child.name
            in {
                "config.json",
                "config_sentence_transformers.json",
                "modules.json",
                "sentence_bert_config.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "tokenizer.json",
                "vocab.json",
                "merges.txt",
                "spiece.model",
                "tokenizer.model",
                "README.md",
            }
            or child.name.endswith(".safetensors.index.json")
            or child.name.endswith(".bin.index.json")
            or child.suffix == ".py"
        )
    ]
    for child in sorted(candidates):
        relative = child.relative_to(path).as_posix()
        payload = child.read_bytes()
        files.append({"path": relative, "bytes": len(payload), "sha256": _sha256(payload)})
    weight_files = [
        child
        for child in path.rglob("*")
        if child.is_file() and child.suffix in {".safetensors", ".bin", ".pt", ".pth"}
    ]
    index_files = [
        child
        for child in path.rglob("*")
        if child.is_file() and (child.name.endswith(".safetensors.index.json") or child.name.endswith(".bin.index.json"))
    ]
    if index_files and not weight_files:
        raise ValueError("model weight index is present without local weight shards; index is not a weight identity")
    for child in sorted(weight_files):
        files.append({
            "path": child.relative_to(path).as_posix(),
            "bytes": child.stat().st_size,
            "sha256": _sha256_file(child),
        })
    return _sha256({"path": str(path.resolve()), "files": files})


def _required_source_identity(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Validate the exact identity keys required by document-pool exports."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    required = ("queries_hash", "corpus_files_sha256", "source_doc_ids_hash")
    result = dict(value)
    for key in required:
        if not isinstance(result.get(key), str) or not result[key]:
            raise ValueError(f"{label} is missing {key}")
    return result


def _resolve_prepared_corpus_file(
    root: Path,
    root_resolved: Path,
    relative: str,
    *,
    label: str,
    parent_resolutions: dict[Path, Path] | None = None,
) -> tuple[Path, os.stat_result]:
    """Resolve one prepared corpus path while rejecting links and non-files.

    The leaf is always ``lstat``-checked so a changed or linked corpus file is
    rejected.  Parent directories are resolved once per caller invocation;
    prepared corpora commonly put hundreds of thousands of files below one
    shared directory, and resolving every full path would repeat those
    directory stats for each leaf.
    """

    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} corpus path must be a non-empty string")
    candidate = Path(relative)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        info = candidate.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"prepared source corpus file is missing: {candidate}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"prepared source corpus file must not be a symlink: {candidate}")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"prepared source corpus file must be a regular file: {candidate}")
    parent_key = candidate.parent
    resolved_parent = None
    if parent_resolutions is not None:
        resolved_parent = parent_resolutions.get(parent_key)
    if resolved_parent is None:
        try:
            resolved_parent = candidate.parent.resolve(strict=True)
            resolved_parent.relative_to(root_resolved)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"prepared source corpus file is missing: {candidate}") from exc
        except ValueError as exc:
            raise ValueError(f"prepared source corpus path escapes its root: {relative!r}") from exc
        if parent_resolutions is not None:
            parent_resolutions[parent_key] = resolved_parent
    try:
        resolved = resolved_parent / candidate.name
        resolved.relative_to(root_resolved)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"prepared source corpus file is missing: {candidate}") from exc
    except ValueError as exc:
        raise ValueError(f"prepared source corpus path escapes its root: {relative!r}") from exc
    return resolved, info


def _read_prepared_corpus_text(
    path: Path,
    *,
    expected_bytes: int | None,
    alias: str,
) -> str:
    """Read a model-visible corpus file once and validate its byte length."""

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise OSError(f"could not read prepared source corpus file for alias {alias!r}: {path}") from exc
    if expected_bytes is not None and len(raw) != expected_bytes:
        raise ValueError(
            f"prepared source corpus size mismatch for alias {alias!r}: "
            f"expected {expected_bytes}, got {len(raw)}"
        )
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"prepared source corpus file is not UTF-8 for alias {alias!r}: {path}") from exc


class SentenceTransformerEncoder:
    """Lazy local sentence-transformers encoder with explicit audit metadata."""

    def __init__(self, config: EmbeddingConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = EmbeddingConfig(**kwargs)
        elif kwargs:
            raise ValueError("pass either EmbeddingConfig or keyword settings, not both")
        self.config = config
        self._model: Any | None = None
        self._weight_identity: str | None = None
        self._tokenizer_identity: str | None = None
        self._model_config_identity: str | None = None
        self.last_metadata: dict[str, Any] = {}

    def _identities(self) -> tuple[str, str, str]:
        if self._weight_identity is None:
            if self.config.weight_identity is not None:
                self._weight_identity = self.config.weight_identity
            else:
                self._weight_identity = model_weight_identity(self.config.model_path)
        if self._tokenizer_identity is None:
            self._tokenizer_identity = _component_identity(self.config.model_path, component="tokenizer")
        if self._model_config_identity is None:
            self._model_config_identity = _component_identity(self.config.model_path, component="model_config")
        return self._weight_identity, self._tokenizer_identity, self._model_config_identity

    @property
    def model(self) -> Any:
        if self._model is None:
            path = Path(self.config.model_path)
            if not path.is_dir():
                raise FileNotFoundError(
                    f"local embedding model directory does not exist; refusing download: {path}"
                )
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("sentence-transformers is required for dense encoding") from exc
            # A local directory is passed directly.  No hub name or fallback
            # is accepted, and trust_remote_code is fixed for KaLM.
            offline_before = os.environ.get("HF_HUB_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                self._model = SentenceTransformer(
                    str(path),
                    trust_remote_code=self.config.trust_remote_code,
                    device=self.config.device,
                )
            finally:
                if offline_before is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = offline_before
        return self._model

    @staticmethod
    def _format_prompt(prompt: str, text: str) -> str:
        if "{text}" in prompt:
            return prompt.replace("{text}", text)
        if prompt:
            return prompt + text
        return text

    def _prepare(self, texts: Sequence[str], kind: str) -> tuple[list[str], int]:
        if kind not in {"query", "document"}:
            raise ValueError("kind must be query or document")
        prompt = self.config.query_prompt if kind == "query" else self.config.document_prompt
        prepared: list[str] = []
        truncated = 0
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("texts must contain strings")
            value = text
            if self.config.character_max_length is not None and len(value) > self.config.character_max_length:
                value = value[: self.config.character_max_length]
                truncated += 1
            prepared.append(self._format_prompt(prompt, value))
        return prepared, truncated

    @staticmethod
    def _token_truncation_count(model: Any, texts: Sequence[str], max_length: int | None) -> int | None:
        if max_length is None:
            return 0
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            return None
        count = 0
        for text in texts:
            try:
                encoded = tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                    return_attention_mask=False,
                )
                ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
                if isinstance(ids, list) and ids and isinstance(ids[0], list):
                    ids = ids[0]
                if isinstance(ids, list) and len(ids) > max_length:
                    count += 1
                    continue
            except Exception:
                try:
                    ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
                    if isinstance(ids, list) and len(ids) > max_length:
                        count += 1
                except Exception:
                    return None
        return count

    def encode(self, texts: Sequence[str], *, kind: str) -> Any:
        prepared, character_truncated = self._prepare(texts, kind)
        np = _np()
        if not prepared:
            return np.empty((0, 0), dtype="float32")
        model = self.model
        if self.config.max_length is not None:
            # SentenceTransformer's tokenizer performs the actual token
            # truncation.  Never replace this with a character slice.
            try:
                model.max_seq_length = self.config.max_length
            except (AttributeError, TypeError):
                pass
        token_truncated = self._token_truncation_count(model, prepared, self.config.max_length)
        requested_batch_size = self.config.batch_size
        effective_batch_size = requested_batch_size
        oom_retries = 0

        def model_encode(batch_size: int) -> Any:
            try:
                return model.encode(
                    prepared,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    # Prompts are prepared exactly once by this wrapper.
                    # Passing an explicit empty prompt prevents
                    # SentenceTransformer from applying its configured
                    # default prompt a second time.
                    prompt="",
                )
            except TypeError:
                # Tiny test doubles and older sentence-transformers versions
                # may accept only the text list.  The prompt and normalization
                # are still applied by this wrapper.  Disable an ST default
                # prompt for this compatibility path as well, when exposed.
                default_prompt_name = getattr(model, "default_prompt_name", None)
                if default_prompt_name is not None:
                    try:
                        model.default_prompt_name = None
                    except (AttributeError, TypeError):
                        pass
                try:
                    try:
                        return model.encode(
                            prepared,
                            batch_size=batch_size,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=False,
                        )
                    except TypeError:
                        return model.encode(prepared)
                finally:
                    if default_prompt_name is not None:
                        try:
                            model.default_prompt_name = default_prompt_name
                        except (AttributeError, TypeError):
                            pass

        while True:
            try:
                values = model_encode(effective_batch_size)
                break
            except RuntimeError as exc:
                message = str(exc).lower()
                is_cuda_oom = "out of memory" in message and (
                    "cuda" in message
                    or str(self.config.device).lower().startswith("cuda")
                )
                if not is_cuda_oom or effective_batch_size <= 1:
                    raise
                oom_retries += 1
                effective_batch_size = max(1, effective_batch_size // 2)
                try:
                    import torch
                    torch.cuda.empty_cache()
                except (ImportError, AttributeError, RuntimeError):
                    pass
        matrix = np.asarray(values, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[0] != len(prepared):
            raise ValueError("embedding model returned an invalid matrix")
        if not np.isfinite(matrix).all():
            raise ValueError("embedding model returned non-finite values")
        if self.config.normalize and matrix.shape[1]:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("cannot normalize a zero embedding")
            matrix = matrix / norms
        weight_identity, tokenizer_identity, model_config_identity = self._identities()
        self.last_metadata = {
            "model_path": self.config.model_path,
            "weight_identity": weight_identity,
            "tokenizer_identity": tokenizer_identity,
            "model_config_identity": model_config_identity,
            "trust_remote_code": self.config.trust_remote_code,
            "kind": kind,
            "query_prompt": self.config.query_prompt,
            "document_prompt": self.config.document_prompt,
            "pooling": self.config.pooling,
            "normalization": "l2" if self.config.normalize else "none",
            "normalized": self.config.normalize,
            "truncation": {
                "max_length": self.config.max_length,
                "unit": "tokens",
                "applied_by": "sentence_transformers_tokenizer",
                "count": token_truncated,
                "character_max_length": self.config.character_max_length,
                "character_count": character_truncated,
            },
            "device": self.config.device,
            "batch_size": self.config.batch_size,
            "requested_batch_size": requested_batch_size,
            "effective_batch_size": effective_batch_size,
            "oom_retries": oom_retries,
            "batch_audit": {
                "requested_batch_size": requested_batch_size,
                "effective_batch_size": effective_batch_size,
                "oom_retries": oom_retries,
            },
        }
        return matrix

    def encode_queries(self, texts: Sequence[str]) -> Any:
        return self.encode(texts, kind="query")

    def encode_documents(self, texts: Sequence[str]) -> Any:
        return self.encode(texts, kind="document")

    def encode_with_cache(
        self,
        texts: Sequence[str],
        *,
        kind: str,
        cache_dir: str | os.PathLike[str],
        cache_shard_size: int = 0,
    ) -> Any:
        """Encode texts using the identity-bound atomic cache when supplied."""

        np = _np()
        if isinstance(cache_shard_size, bool) or not isinstance(cache_shard_size, int) or cache_shard_size < 0:
            raise ValueError("cache_shard_size must be a non-negative integer")
        if cache_shard_size:
            return self._encode_with_shard_cache(
                texts,
                kind=kind,
                cache_dir=cache_dir,
                cache_shard_size=cache_shard_size,
            )
        weight_identity, tokenizer_identity, model_config_identity = self._identities()
        cache = EmbeddingCache(cache_dir, self.config, weight_identity=weight_identity)
        values: list[Any | None] = [cache.get(text, kind=kind) for text in texts]
        missing = [index for index, value in enumerate(values) if value is None]
        if missing:
            encoded = self.encode([texts[index] for index in missing], kind=kind)
            for local_index, source_index in enumerate(missing):
                values[source_index] = encoded[local_index]
                cache.set(texts[source_index], encoded[local_index], kind=kind)
        else:
            # Do not carry the preceding document/query call's audit over to
            # a fully cached call of the other kind.
            self.last_metadata = {
                **self.config.to_dict(), "kind": kind,
                "weight_identity": weight_identity, "tokenizer_identity": tokenizer_identity,
                "model_config_identity": model_config_identity, "inference_executed": False,
                "truncation": {"max_length": self.config.max_length, "unit": "tokens", "count": None,
                               "count_note": "not remeasured on cache hit"},
                "requested_batch_size": self.config.batch_size,
                "effective_batch_size": self.config.batch_size,
                "oom_retries": 0,
            }
        if not values:
            return np.empty((0, 0), dtype="float32")
        self.last_metadata = {**self.last_metadata, "embedding_cache": str(cache_dir), "cache_hits": len(texts) - len(missing)}
        return np.stack(values).astype("float32", copy=False)

    def _encode_with_shard_cache(
        self,
        texts: Sequence[str],
        *,
        kind: str,
        cache_dir: str | os.PathLike[str],
        cache_shard_size: int,
    ) -> Any:
        """Load or produce bounded, identity-bound embedding shards.

        A source is addressed by its stable input order.  Completed shards are
        immutable and can be reused after an interrupted run; a failed encode
        leaves all completed shards untouched.
        """

        np = _np()
        if kind not in {"query", "document"}:
            raise ValueError("kind must be query or document")
        text_list = list(texts)
        if any(not isinstance(text, str) for text in text_list):
            raise ValueError("texts must contain strings")
        weight_identity, tokenizer_identity, model_config_identity = self._identities()
        cache = EmbeddingShardCache(
            cache_dir,
            self.config,
            cache_shard_size,
            weight_identity=weight_identity,
        )
        if not text_list:
            self.last_metadata = {
                **self.config.to_dict(),
                "kind": kind,
                "weight_identity": weight_identity,
                "tokenizer_identity": tokenizer_identity,
                "model_config_identity": model_config_identity,
                "embedding_cache": str(cache_dir),
                "cache_shard_size": cache_shard_size,
                "cache_hits": 0,
                "cache_shards": 0,
                "inference_executed": False,
                "requested_batch_size": self.config.batch_size,
                "effective_batch_size": self.config.batch_size,
                "oom_retries": 0,
                "batch_audit": [],
            }
            return np.empty((0, 0), dtype="float32")

        values: list[Any] = []
        cache_hits = 0
        truncation_counts: list[int] = []
        character_truncation_counts: list[int] = []
        batch_audits: list[dict[str, int]] = []
        inference_executed = False
        shard_audit: list[dict[str, Any]] = []

        def shard_batch_audit(metadata: Mapping[str, Any] | None) -> dict[str, int]:
            raw = metadata.get("audit") if isinstance(metadata, Mapping) else None
            if not isinstance(raw, Mapping):
                return {
                    "requested_batch_size": self.config.batch_size,
                    "effective_batch_size": self.config.batch_size,
                    "oom_retries": 0,
                }
            try:
                result = {
                    "requested_batch_size": int(raw.get("requested_batch_size", self.config.batch_size)),
                    "effective_batch_size": int(raw.get("effective_batch_size", self.config.batch_size)),
                    "oom_retries": int(raw.get("oom_retries", 0)),
                }
            except (TypeError, ValueError) as exc:
                raise ValueError("embedding shard batch audit is invalid") from exc
            if (
                result["requested_batch_size"] <= 0
                or result["effective_batch_size"] <= 0
                or result["effective_batch_size"] > result["requested_batch_size"]
                or result["oom_retries"] < 0
            ):
                raise ValueError("embedding shard batch audit is invalid")
            return result

        for shard_index, start in enumerate(range(0, len(text_list), cache_shard_size)):
            stop = min(start + cache_shard_size, len(text_list))
            shard_texts = text_list[start:stop]
            cached, cached_metadata = cache.get_shard(
                shard_texts, kind=kind, shard_index=shard_index, with_metadata=True
            )
            if cached is not None:
                shard_values = cached
                cache_hits += len(shard_texts)
                shard_record = {"index": shard_index, "start": start, "stop": stop, "cache_hit": True}
                if isinstance(cached_metadata, Mapping):
                    cached_audit = cached_metadata.get("audit")
                    cached_truncation = (
                        cached_audit.get("truncation")
                        if isinstance(cached_audit, Mapping)
                        else None
                    )
                    cached_batch_audit = shard_batch_audit(cached_metadata)
                    batch_audits.append(cached_batch_audit)
                    shard_record.update(cached_batch_audit)
                    if isinstance(cached_truncation, Mapping) and isinstance(cached_truncation.get("count"), int):
                        truncation_counts.append(cached_truncation["count"])
                        if isinstance(cached_truncation.get("character_count"), int):
                            character_truncation_counts.append(cached_truncation["character_count"])
                        shard_record["truncation_count"] = cached_truncation["count"]
                shard_audit.append(shard_record)
            else:
                # No final shard exists here, so a model failure cannot
                # overwrite any prior artifact.  set_shard is atomic too.
                shard_values = self.encode(shard_texts, kind=kind)
                inference_executed = True
                truncation = self.last_metadata.get("truncation", {})
                batch_audit = {
                    "requested_batch_size": int(
                        self.last_metadata.get("requested_batch_size", self.config.batch_size)
                    ),
                    "effective_batch_size": int(
                        self.last_metadata.get("effective_batch_size", self.config.batch_size)
                    ),
                    "oom_retries": int(self.last_metadata.get("oom_retries", 0)),
                }
                batch_audits.append(batch_audit)
                if isinstance(truncation, Mapping) and isinstance(truncation.get("count"), int):
                    truncation_counts.append(truncation["count"])
                    if isinstance(truncation.get("character_count"), int):
                        character_truncation_counts.append(truncation["character_count"])
                cache.set_shard(
                    shard_texts,
                    shard_values,
                    kind=kind,
                    shard_index=shard_index,
                    audit={"truncation": truncation, **batch_audit},
                )
                shard_record = {"index": shard_index, "start": start, "stop": stop, "cache_hit": False}
                shard_record.update(batch_audit)
                if isinstance(truncation, Mapping) and isinstance(truncation.get("count"), int):
                    shard_record["truncation_count"] = truncation["count"]
                shard_audit.append(shard_record)
            values.append(np.asarray(shard_values, dtype="float32"))

        matrix = np.concatenate(values, axis=0).astype("float32", copy=False)
        self.last_metadata = {
            **self.config.to_dict(),
            "kind": kind,
            "weight_identity": weight_identity,
            "tokenizer_identity": tokenizer_identity,
            "model_config_identity": model_config_identity,
            "embedding_cache": str(cache_dir),
            "cache_shard_size": cache_shard_size,
            "cache_hits": cache_hits,
            "cache_shards": len(shard_audit),
            "cache_shard_audit": shard_audit,
            "inference_executed": inference_executed,
            "requested_batch_size": self.config.batch_size,
            "effective_batch_size": min(
                (audit["effective_batch_size"] for audit in batch_audits),
                default=self.config.batch_size,
            ),
            "oom_retries": sum(audit["oom_retries"] for audit in batch_audits),
            "batch_audit": batch_audits,
        }
        self.last_metadata["truncation"] = {
            "max_length": self.config.max_length,
            "unit": "tokens",
            "applied_by": "sentence_transformers_tokenizer",
            "count": sum(truncation_counts) if len(truncation_counts) == len(shard_audit) else None,
            "count_note": None if len(truncation_counts) == len(shard_audit) else "not recorded for an older cache shard",
            "character_max_length": self.config.character_max_length,
            "character_count": (
                sum(character_truncation_counts)
                if len(character_truncation_counts) == len(shard_audit)
                else None
            ),
        }
        return matrix


# Short aliases make the API convenient without hiding the full class name.
LocalEmbeddingEncoder = SentenceTransformerEncoder


class EmbeddingCache:
    """Atomic per-text embedding cache bound to model/config identity."""

    def __init__(self, cache_dir: str | os.PathLike[str], config: EmbeddingConfig, *, weight_identity: str | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.config = config
        pinned_weight_identity = weight_identity if weight_identity is not None else config.weight_identity
        self.weight_identity = (
            pinned_weight_identity
            if pinned_weight_identity is not None
            else model_weight_identity(config.model_path)
        )
        self.tokenizer_identity = _component_identity(config.model_path, component="tokenizer")
        self.model_config_identity = _component_identity(config.model_path, component="model_config")
        self.identity = {
            "config": config.to_dict(),
            "weight_identity": self.weight_identity,
            "model_identity": {
                "model_path": str(Path(config.model_path).resolve()),
                "weight_identity": self.weight_identity,
                "tokenizer_identity": self.tokenizer_identity,
                "model_config_identity": self.model_config_identity,
            },
        }

    def key(self, text: str, *, kind: str) -> str:
        if not isinstance(text, str) or kind not in {"query", "document"}:
            raise ValueError("text must be a string and kind must be query or document")
        return _sha256({"text": text, "kind": kind, "identity": self.identity})

    def get(self, text: str, *, kind: str) -> Any | None:
        np = _np()
        key = self.key(text, kind=kind)
        metadata_path = self.cache_dir / f"{key}.json"
        vector_path = self.cache_dir / f"{key}.npy"
        if not metadata_path.exists() and not vector_path.exists():
            return None
        if not metadata_path.exists() or not vector_path.exists():
            raise ValueError("embedding cache entry is incomplete")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("embedding cache metadata is invalid") from exc
        if metadata.get("schema_version") != "dci-embedding-cache-v1" or metadata.get("identity") != self.identity or metadata.get("key") != key:
            raise ValueError("embedding cache identity mismatch")
        vector = np.load(vector_path, allow_pickle=False)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("embedding cache vector is invalid")
        if metadata.get("vector_sha256") != _sha256(vector.tobytes()):
            raise ValueError("embedding cache vector hash mismatch")
        return vector.astype("float32", copy=False)

    def set(self, text: str, vector: Any, *, kind: str) -> None:
        np = _np()
        key = self.key(text, kind=kind)
        value = np.asarray(vector, dtype="float32")
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError("embedding cache vector must be a finite one-dimensional array")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = self.cache_dir / f"{key}.json"
        vector_path = self.cache_dir / f"{key}.npy"
        vector_tmp = self.cache_dir / f".{key}.npy.tmp"
        metadata_tmp = self.cache_dir / f".{key}.json.tmp"
        try:
            with vector_tmp.open("wb") as handle:
                np.save(handle, value, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            metadata = {
                "schema_version": "dci-embedding-cache-v1",
                "key": key,
                "identity": self.identity,
                "vector_sha256": _sha256(value.tobytes()),
                "dimension": int(value.shape[0]),
            }
            with metadata_tmp.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(vector_tmp, vector_path)
            os.replace(metadata_tmp, metadata_path)
        finally:
            for path in (vector_tmp, metadata_tmp):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


class EmbeddingShardCache:
    """Atomic cache for bounded contiguous embedding shards.

    Each shard is a single ``.npz`` artifact containing its metadata and
    vectors.  The directory is namespaced by the complete model/config
    identity, so prompt or tokenizer changes naturally select a fresh cache
    namespace.  Existing artifacts are never replaced: a concurrent writer
    either observes the completed shard or leaves its temporary file behind
    for cleanup.
    """

    schema_version = "dci-embedding-cache-shard-v1"

    def __init__(
        self,
        cache_dir: str | os.PathLike[str],
        config: EmbeddingConfig,
        shard_size: int,
        *,
        weight_identity: str | None = None,
    ) -> None:
        if isinstance(shard_size, bool) or not isinstance(shard_size, int) or shard_size <= 0:
            raise ValueError("shard_size must be a positive integer")
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.shard_size = shard_size
        pinned_weight_identity = weight_identity if weight_identity is not None else config.weight_identity
        self.weight_identity = (
            pinned_weight_identity
            if pinned_weight_identity is not None
            else model_weight_identity(config.model_path)
        )
        self.tokenizer_identity = _component_identity(config.model_path, component="tokenizer")
        self.model_config_identity = _component_identity(config.model_path, component="model_config")
        self.identity = {
            "config": config.to_dict(),
            "weight_identity": self.weight_identity,
            "model_identity": {
                "model_path": str(Path(config.model_path).resolve()),
                "weight_identity": self.weight_identity,
                "tokenizer_identity": self.tokenizer_identity,
                "model_config_identity": self.model_config_identity,
            },
            "shard_size": shard_size,
        }
        self.identity_hash = _sha256(self.identity)
        self.namespace = self.cache_dir / "shards" / self.identity_hash

    def _path(self, kind: str, shard_index: int) -> Path:
        if kind not in {"query", "document"}:
            raise ValueError("kind must be query or document")
        if isinstance(shard_index, bool) or not isinstance(shard_index, int) or shard_index < 0:
            raise ValueError("shard_index must be a non-negative integer")
        return self.namespace / f"{kind}-{shard_index:08d}.npz"

    @staticmethod
    def _text_hash(texts: Sequence[str], *, kind: str) -> str:
        return _sha256({"kind": kind, "texts": list(texts)})

    def get_shard(
        self,
        texts: Sequence[str],
        *,
        kind: str,
        shard_index: int,
        with_metadata: bool = False,
    ) -> Any | tuple[Any | None, dict[str, Any] | None]:
        np = _np()
        text_list = list(texts)
        if not text_list or any(not isinstance(text, str) for text in text_list):
            raise ValueError("texts must be a non-empty sequence of strings")
        path = self._path(kind, shard_index)
        if not path.exists():
            return (None, None) if with_metadata else None
        try:
            with np.load(path, allow_pickle=False) as payload:
                metadata_raw = payload["metadata"]
                vectors = np.asarray(payload["vectors"], dtype="float32")
        except (OSError, ValueError, KeyError, EOFError) as exc:
            raise ValueError(f"embedding shard is invalid: {path}") from exc
        try:
            if isinstance(metadata_raw, np.ndarray):
                metadata_text = metadata_raw.item()
            else:
                metadata_text = metadata_raw
            metadata = json.loads(str(metadata_text))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"embedding shard metadata is invalid: {path}") from exc
        if not isinstance(metadata, Mapping):
            raise ValueError(f"embedding shard metadata is invalid: {path}")
        expected_text_hash = self._text_hash(text_list, kind=kind)
        if (
            metadata.get("schema_version") != self.schema_version
            or metadata.get("identity") != self.identity
            or metadata.get("identity_hash") != self.identity_hash
            or metadata.get("kind") != kind
            or metadata.get("shard_index") != shard_index
            or metadata.get("text_hash") != expected_text_hash
            or metadata.get("count") != len(text_list)
        ):
            raise ValueError(f"embedding shard identity or input mismatch: {path}")
        if vectors.ndim != 2 or vectors.shape[0] != len(text_list) or not np.isfinite(vectors).all():
            raise ValueError(f"embedding shard vectors are invalid: {path}")
        if metadata.get("vector_sha256") != _sha256(vectors.tobytes()):
            raise ValueError(f"embedding shard vector hash mismatch: {path}")
        if metadata.get("shape") != [int(vectors.shape[0]), int(vectors.shape[1])]:
            raise ValueError(f"embedding shard shape mismatch: {path}")
        return (vectors, metadata) if with_metadata else vectors

    def set_shard(
        self,
        texts: Sequence[str],
        vectors: Any,
        *,
        kind: str,
        shard_index: int,
        audit: Mapping[str, Any] | None = None,
    ) -> None:
        np = _np()
        text_list = list(texts)
        if not text_list or any(not isinstance(text, str) for text in text_list):
            raise ValueError("texts must be a non-empty sequence of strings")
        value = np.asarray(vectors, dtype="float32")
        if value.ndim != 2 or value.shape[0] != len(text_list) or not np.isfinite(value).all():
            raise ValueError("embedding shard vectors must be a finite two-dimensional matrix")
        path = self._path(kind, shard_index)
        self.namespace.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": self.schema_version,
            "identity": self.identity,
            "identity_hash": self.identity_hash,
            "kind": kind,
            "shard_index": shard_index,
            "count": len(text_list),
            "text_hash": self._text_hash(text_list, kind=kind),
            "vector_sha256": _sha256(value.tobytes()),
            "shape": [int(value.shape[0]), int(value.shape[1])],
        }
        if audit is not None:
            metadata["audit"] = copy.deepcopy(dict(audit))
        if path.exists():
            # Immutable artifacts are never replaced.  Validate a pre-existing
            # shard before treating this call as a benign concurrent publish;
            # a corrupt or different vector must remain in place and surface
            # an error for the caller.
            existing = self.get_shard(text_list, kind=kind, shard_index=shard_index)
            if existing is None or not np.array_equal(existing, value):
                raise ValueError(f"embedding shard already exists with different vectors: {path}")
            return
        # np.savez writes a complete archive to a private temporary path.  A
        # hard-link publish avoids replacing an existing immutable shard if a
        # second worker races this writer.
        temp_path = self.namespace / f".{path.name}.{os.getpid()}.{id(value)}.tmp"
        try:
            with temp_path.open("wb") as handle:
                np.savez(handle, vectors=value, metadata=np.asarray(_stable_json(metadata)))
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temp_path, path)
            except FileExistsError:
                # A valid completed artifact wins; do not overwrite it.  A
                # corrupt or different artifact is an explicit failure.
                existing = self.get_shard(text_list, kind=kind, shard_index=shard_index)
                if existing is None or not np.array_equal(existing, value):
                    raise ValueError(f"embedding shard already exists with different vectors: {path}")
        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def unicode_tokens(text: str, *, char_bigrams: bool = True) -> list[str]:
    """Tokenize Unicode text with CJK character/bigram support."""

    if not isinstance(text, str):
        raise ValueError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    latin_buffer: list[str] = []
    cjk_buffer: list[str] = []

    def flush_latin() -> None:
        if latin_buffer:
            tokens.append("".join(latin_buffer))
            latin_buffer.clear()

    def flush_cjk() -> None:
        if cjk_buffer:
            tokens.extend(cjk_buffer)
            if char_bigrams:
                tokens.extend("".join(cjk_buffer[index : index + 2]) for index in range(len(cjk_buffer) - 1))
            cjk_buffer.clear()

    for char in normalized:
        codepoint = ord(char)
        is_cjk = (
            0x3040 <= codepoint <= 0x30FF
            or 0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xAC00 <= codepoint <= 0xD7AF
            or 0xF900 <= codepoint <= 0xFAFF
        )
        if is_cjk:
            flush_latin()
            cjk_buffer.append(char)
        elif char.isalnum() or char == "_":
            flush_cjk()
            latin_buffer.append(char)
        else:
            flush_latin()
            flush_cjk()
    flush_latin()
    flush_cjk()
    return tokens


class BM25Index:
    """Deterministic BM25 index with stable input-order ties."""

    def __init__(self, documents: Sequence[Mapping[str, Any]] | Mapping[str, str], *, k1: float = 1.2, b: float = 0.75, char_bigrams: bool = True) -> None:
        self.ids, texts = _documents_to_ids_texts(documents)
        if not self.ids:
            raise ValueError("documents must be non-empty")
        if k1 < 0 or b < 0 or b > 1:
            raise ValueError("BM25 k1 must be non-negative and b must be in [0, 1]")
        self.k1 = float(k1)
        self.b = float(b)
        self.char_bigrams = char_bigrams
        self.tokens = [unicode_tokens(text, char_bigrams=char_bigrams) for text in texts]
        self.lengths = [len(row) for row in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths)
        self.document_frequency: Counter[str] = Counter()
        for row in self.tokens:
            self.document_frequency.update(set(row))
        count = len(self.ids)
        self.inverse_document_frequency = {
            token: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in self.document_frequency.items()
        }

    def score(self, query: str, index: int) -> float:
        query_tokens = Counter(unicode_tokens(query, char_bigrams=self.char_bigrams))
        document = Counter(self.tokens[index])
        length = self.lengths[index]
        score = 0.0
        for token, query_frequency in query_tokens.items():
            frequency = document.get(token, 0)
            if not frequency:
                continue
            denominator = frequency + self.k1 * (1 - self.b + self.b * length / self.average_length)
            score += self.inverse_document_frequency.get(token, 0.0) * (
                frequency * (self.k1 + 1) / denominator
            ) * min(query_frequency, 1)
        return float(score)

    def search(self, query: str, *, top_k: int = 10) -> dict[str, Any]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        scores = [self.score(query, index) for index in range(len(self.ids))]
        order = sorted(range(len(self.ids)), key=lambda index: (-scores[index], index))
        selected = order[:top_k]
        return {
            "ranked_doc_ids": [self.ids[index] for index in selected],
            "scores": {self.ids[index]: scores[index] for index in selected},
            "all_scores": {doc_id: scores[index] for index, doc_id in enumerate(self.ids)},
            "scoring": "bm25-unicode-char-bigram-v1" if self.char_bigrams else "bm25-unicode-token-v1",
        }


def _documents_to_ids_texts(documents: Sequence[Mapping[str, Any]] | Mapping[str, str]) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    rows: Iterator[tuple[str, str]]
    if isinstance(documents, Mapping):
        rows = ((str(doc_id), text) for doc_id, text in documents.items())
    else:
        rows = (
            (row.get("_id"), row.get("text"))
            for row in documents
            if isinstance(row, Mapping)
        )
    for doc_id, text in rows:
        if not isinstance(doc_id, str) or not doc_id or not isinstance(text, str):
            raise ValueError("documents require non-empty IDs and text")
        if doc_id in ids:
            raise ValueError(f"duplicate document ID: {doc_id!r}")
        ids.append(doc_id)
        texts.append(text)
    return ids, texts


def bm25_search(query: str, documents: Sequence[Mapping[str, Any]] | Mapping[str, str], *, top_k: int = 10, char_bigrams: bool = True, k1: float = 1.2, b: float = 0.75) -> dict[str, Any]:
    return BM25Index(documents, k1=k1, b=b, char_bigrams=char_bigrams).search(query, top_k=top_k)


def _matrix_rows(values: Any, ids: Sequence[str] | None, *, name: str) -> tuple[list[str], Any]:
    np = _np()
    if isinstance(values, Mapping):
        row_ids = list(values)
        matrix = np.asarray([values[row_id] for row_id in row_ids], dtype="float32")
    else:
        matrix = np.asarray(values, dtype="float32")
        if ids is None:
            row_ids = [str(index) for index in range(len(matrix))]
        else:
            row_ids = list(ids)
    if matrix.ndim != 2 or len(row_ids) != matrix.shape[0] or len(set(row_ids)) != len(row_ids):
        raise ValueError(f"{name} vectors and IDs have incompatible shapes")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError(f"{name} vectors must be a non-empty finite matrix")
    return row_ids, matrix


def dense_search(
    query_vectors: Any,
    document_vectors: Any,
    document_ids: Sequence[str] | None = None,
    *,
    query_ids: Sequence[str] | None = None,
    top_k: int = 10,
    gold_doc_ids: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
    qrels: Mapping[str, Mapping[str, Any]] | None = None,
    chunk_size: int = 4096,
    normalize: bool = False,
    search_device: str = "cpu",
    search_query_batch_size: int = 32,
) -> dict[str, dict[str, Any]]:
    """Search dense vectors with bounded, stable top-k selection.

    CPU search keeps only the best ``top_k`` rows while scanning document
    chunks.  CUDA search additionally batches queries and transfers one
    document chunk at a time, so it never materializes a full query-by-corpus
    score matrix.  Ties are resolved by original corpus position.
    """

    np = _np()
    if query_ids is None:
        if qrels is not None and isinstance(qrels, Mapping):
            query_ids = list(qrels)
        elif isinstance(gold_doc_ids, Mapping):
            query_ids = list(gold_doc_ids)
    doc_ids, docs = _matrix_rows(document_vectors, document_ids, name="document")
    q_ids, queries = _matrix_rows(query_vectors, query_ids, name="query")
    if docs.shape[1] != queries.shape[1]:
        raise ValueError("query and document embedding dimensions differ")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(search_device, str) or not search_device:
        raise ValueError("search_device must be a non-empty device string")
    if (
        isinstance(search_query_batch_size, bool)
        or not isinstance(search_query_batch_size, int)
        or search_query_batch_size <= 0
    ):
        raise ValueError("search_query_batch_size must be a positive integer")
    if normalize:
        query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        doc_norms = np.linalg.norm(docs, axis=1, keepdims=True)
        if np.any(query_norms == 0) or np.any(doc_norms == 0):
            raise ValueError("cannot normalize zero vectors")
        queries = queries / query_norms
        docs = docs / doc_norms
    if gold_doc_ids is not None and qrels is not None:
        raise ValueError("pass gold_doc_ids or qrels, not both")
    if qrels is not None:
        unknown_queries = sorted(set(qrels) - set(q_ids))
        if unknown_queries:
            raise ValueError(f"qrels contain unknown query IDs: {unknown_queries[:5]}")
        gold_doc_ids = {
            query_id: [doc_id for doc_id, label in row.items() if float(label) > 0]
            for query_id, row in qrels.items()
        }
    gold_mapping: dict[str, list[str]] = {}
    if gold_doc_ids is None:
        gold_mapping = {query_id: [] for query_id in q_ids}
    elif isinstance(gold_doc_ids, Mapping):
        gold_mapping = {query_id: list(gold_doc_ids.get(query_id, [])) for query_id in q_ids}
    else:
        gold_mapping = {query_id: list(gold_doc_ids) for query_id in q_ids}
    known_docs = set(doc_ids)
    for query_id, gold in gold_mapping.items():
        if len(set(gold)) != len(gold) or any(doc_id not in known_docs for doc_id in gold):
            raise ValueError(f"gold document IDs are invalid for query {query_id!r}")

    doc_positions = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    gold_positions = {
        query_id: [doc_positions[doc_id] for doc_id in gold_mapping[query_id]]
        for query_id in q_ids
    }

    def stable_chunk_topk(scores: Any, start: int, limit: int) -> tuple[Any, Any]:
        """Return local positions sorted by score desc, corpus index asc."""

        scores = np.asarray(scores, dtype="float32")
        count = int(scores.shape[0])
        if count == 0 or limit <= 0:
            return np.empty(0, dtype="int64"), np.empty(0, dtype="float32")
        limit = min(limit, count)
        global_indices = np.arange(start, start + count, dtype="int64")
        if limit < count:
            # Keep every item at the cutoff, then select the earliest equal
            # scores.  This avoids argpartition's unspecified tie behavior.
            cutoff = np.partition(scores, count - limit)[count - limit]
            candidates = np.flatnonzero(scores >= cutoff)
        else:
            candidates = np.arange(count, dtype="int64")
        order = np.lexsort((global_indices[candidates], -scores[candidates]))[:limit]
        chosen = candidates[order]
        return global_indices[chosen], scores[chosen]

    def merge_topk(
        best_indices: Any,
        best_scores: Any,
        candidate_indices: Any,
        candidate_scores: Any,
        limit: int,
    ) -> tuple[Any, Any]:
        if len(candidate_indices) == 0:
            return best_indices, best_scores
        merged_indices = np.concatenate((best_indices, np.asarray(candidate_indices, dtype="int64")))
        merged_scores = np.concatenate((best_scores, np.asarray(candidate_scores, dtype="float32")))
        order = np.lexsort((merged_indices, -merged_scores))[:limit]
        return merged_indices[order], merged_scores[order]

    def set_gold_scores(
        score_map: dict[str, float], query_id: str, start: int, values: Any,
    ) -> None:
        stop = start + len(values)
        wanted = {position: doc_ids[position] for position in gold_positions[query_id] if start <= position < stop}
        for position, doc_id in wanted.items():
            score_map[doc_id] = float(values[position - start])

    def make_row(
        query_id: str, selected_indices: Any, selected_scores: Any, score_map: dict[str, float],
        *, effective_query_batch_size: int,
    ) -> dict[str, Any]:
        selected_ids = [doc_ids[int(index)] for index in selected_indices]
        # Ranking scores and every gold score are retained.  Golds outside
        # top-k therefore carry their true dot product, not a placeholder.
        for index, value in zip(selected_indices, selected_scores):
            score_map.setdefault(doc_ids[int(index)], float(value))
        output_scores: dict[str, float] = {doc_id: score_map[doc_id] for doc_id in selected_ids}
        for doc_id in gold_mapping[query_id]:
            if doc_id not in output_scores:
                output_scores[doc_id] = score_map[doc_id]
        return {
            "ranked_doc_ids": selected_ids,
            "scores": output_scores,
            "top_k": top_k,
            "gold_doc_ids": gold_mapping[query_id],
            "similarity": "dot_product" if not normalize else "cosine",
            "chunk_size": chunk_size,
            "search_device": search_device,
            "search_query_batch_size": search_query_batch_size,
            "search_query_batch_size_effective": effective_query_batch_size,
        }

    output: dict[str, dict[str, Any]] = {}
    if search_device == "cpu":
        for query_index, query_id in enumerate(q_ids):
            best_indices = np.empty(0, dtype="int64")
            best_scores = np.empty(0, dtype="float32")
            score_map: dict[str, float] = {}
            query = queries[query_index]
            for start in range(0, len(doc_ids), chunk_size):
                stop = min(start + chunk_size, len(doc_ids))
                scores = docs[start:stop] @ query
                candidate_indices, candidate_scores = stable_chunk_topk(scores, start, top_k)
                best_indices, best_scores = merge_topk(
                    best_indices, best_scores, candidate_indices, candidate_scores, min(top_k, len(doc_ids))
                )
                set_gold_scores(score_map, query_id, start, scores)
            output[query_id] = make_row(
                query_id, best_indices, best_scores, score_map,
                effective_query_batch_size=search_query_batch_size,
            )
        return output

    if not search_device.startswith("cuda"):
        raise ValueError("search_device must be 'cpu' or a CUDA device such as 'cuda:0'")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional GPU dependency
        raise RuntimeError("CUDA dense search requires torch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA dense search requested but unavailable: {search_device}")
    try:
        device = torch.device(search_device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid search_device: {search_device}") from exc
    if device.type != "cuda":
        raise ValueError("search_device must be 'cpu' or a CUDA device")
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    precision_getter = getattr(torch, "get_float32_matmul_precision", None)
    precision_setter = getattr(torch, "set_float32_matmul_precision", None)
    old_precision = precision_getter() if precision_getter is not None else None
    effective_query_batch_size = search_query_batch_size
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if precision_setter is not None:
            precision_setter("highest")
        with torch.inference_mode():
            query_start = 0
            while query_start < len(q_ids):
                query_stop = min(query_start + effective_query_batch_size, len(q_ids))
                batch_count = query_stop - query_start
                best_indices = [np.empty(0, dtype="int64") for _ in range(batch_count)]
                best_scores = [np.empty(0, dtype="float32") for _ in range(batch_count)]
                score_maps: list[dict[str, float]] = [{} for _ in range(batch_count)]
                batch_queries = None
                try:
                    # Keep only the active query batch on GPU; the full query
                    # matrix remains in host memory and is bounded by retryable
                    # transfer/matmul work below.
                    batch_queries = torch.as_tensor(
                        queries[query_start:query_stop], dtype=torch.float32, device=device
                    )
                    for start in range(0, len(doc_ids), chunk_size):
                        stop = min(start + chunk_size, len(doc_ids))
                        doc_chunk = None
                        try:
                            doc_chunk = torch.as_tensor(docs[start:stop], dtype=torch.float32, device=device)
                            chunk_scores = (batch_queries @ doc_chunk.transpose(0, 1)).float().cpu().numpy()
                        finally:
                            if doc_chunk is not None:
                                del doc_chunk
                        for local_index in range(batch_count):
                            candidate_indices, candidate_values = stable_chunk_topk(
                                chunk_scores[local_index], start, top_k
                            )
                            best_indices[local_index], best_scores[local_index] = merge_topk(
                                best_indices[local_index], best_scores[local_index],
                                candidate_indices, candidate_values, min(top_k, len(doc_ids)),
                            )
                            set_gold_scores(
                                score_maps[local_index], q_ids[query_start + local_index], start,
                                chunk_scores[local_index],
                            )
                    for local_index in range(batch_count):
                        query_id = q_ids[query_start + local_index]
                        output[query_id] = make_row(
                            query_id, best_indices[local_index], best_scores[local_index], score_maps[local_index],
                            effective_query_batch_size=effective_query_batch_size,
                        )
                except RuntimeError as exc:
                    if batch_count <= 1 or "out of memory" not in str(exc).lower():
                        raise
                    effective_query_batch_size = max(1, batch_count // 2)
                    if batch_queries is not None:
                        del batch_queries
                    torch.cuda.empty_cache()
                    continue
                if batch_queries is not None:
                    del batch_queries
                query_start = query_stop
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        if precision_setter is not None and old_precision is not None:
            precision_setter(old_precision)
    return output


search_dense = dense_search
search_embeddings = dense_search


def _bundle_identity_inputs(bundle: Mapping[str, Any], source_doc_ids: Mapping[str, str] | None = None, corpus_file_hashes: Mapping[str, str] | None = None) -> tuple[str, dict[str, str], dict[str, str]]:
    source = bundle["source"]
    task = str(source.get("dataset", "bundle"))
    docs = bundle["documents"]
    mapping = dict(source_doc_ids or {row["_id"]: row["_id"] for row in docs})
    if corpus_file_hashes is None:
        reverse = {original: alias for alias, original in mapping.items()}
        corpus_file_hashes = {
            alias: _sha256(next(row["text"] for row in docs if row["_id"] == original).encode("utf-8"))
            for alias, original in mapping.items()
            if original in {row["_id"] for row in docs}
        }
    identity = build_source_identity(
        task=task,
        corpus_ids=[row["_id"] for row in docs],
        query_ids=[row["_id"] for row in bundle["queries"]],
        source_doc_ids=mapping,
        qrels=bundle["qrels"],
        query_records=bundle["queries"],
        corpus_file_hashes=corpus_file_hashes,
    )
    return task, mapping, identity


def build_source_identity(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Delegate to the canonical document-pool identity builder.

    The wrapper keeps the established helper importable from this module while
    avoiding a module-level dependency on the document-pool implementation.
    """

    from dci_bench.data.document_pools import build_source_identity as canonical

    return canonical(*args, **kwargs)


def build_source_identity_for_bundle(bundle: Mapping[str, Any], *, source_doc_ids: Mapping[str, str] | None = None, corpus_file_hashes: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build the three hashes consumed by ``document_pools`` for a bundle."""

    return _bundle_identity_inputs(bundle, source_doc_ids, corpus_file_hashes)[2]


def _prepared_query_projection(
    records: Sequence[Mapping[str, Any]], *, loaded_bundle: bool = False
) -> list[dict[str, Any]]:
    """Normalize raw prepared query rows and loaded bundle rows alike.

    ``load_prepared_source`` exposes the base query fields plus a ``metadata``
    object.  The source suite hashes the original JSONL rows, so this helper
    reconstructs the same flat row shape for an integrity comparison before
    the original rows are passed to ``build_source_identity``.
    """

    projected: list[dict[str, Any]] = []
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("prepared query records must be objects")
        query_id = row.get("_id", row.get("query_id"))
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("prepared query records require non-empty IDs")
        result: dict[str, Any] = {
            "_id": query_id,
            "text": row.get("text", ""),
            "language": row.get("language", "unknown"),
        }
        metadata = row.get("metadata")
        if loaded_bundle and isinstance(metadata, Mapping):
            result.update(copy.deepcopy(dict(metadata)))
        for key, value in row.items():
            if key not in {"_id", "query_id", "text", "language"} and (
                not loaded_bundle or key != "metadata"
            ):
                result[key] = copy.deepcopy(value)
        projected.append(result)
    return sorted(projected, key=lambda row: row["_id"])


def _prepared_source_identity(
    bundle: Mapping[str, Any], *, verify_content_hashes: bool = True
) -> dict[str, Any]:
    """Recompute and validate a loaded prepared source's identity.

    The prepared suite carries the alias mapping, rendered-file hashes and
    source query rows in provenance.  Recomputing from those bytes prevents a
    stale ``source_identity.json`` from being trusted after a corpus or query
    file changes.  When content verification is explicitly disabled, the
    saved per-file digests remain the declared identity and the current file
    bytes are checked only for presence, regular-file status, and size.
    """

    if not isinstance(verify_content_hashes, bool):
        raise ValueError("verify_content_hashes must be boolean")

    provenance = bundle.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("bundle provenance must be an object")
    root_value = provenance.get("prepared_source")
    mapping_value = provenance.get("source_doc_ids")
    hashes_value = provenance.get("corpus_file_hashes")
    files_value = provenance.get("corpus_files")
    sizes_value = provenance.get("corpus_file_sizes", {})
    raw_queries = provenance.get("source_query_records")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("prepared source provenance is missing prepared_source")
    if not isinstance(mapping_value, Mapping) or not mapping_value:
        raise ValueError("prepared source provenance is missing source_doc_ids")
    if not isinstance(hashes_value, Mapping) or not hashes_value:
        raise ValueError("prepared source provenance is missing corpus_file_hashes")
    if not isinstance(files_value, Mapping) or not files_value:
        raise ValueError("prepared source provenance is missing corpus_files")
    if not isinstance(sizes_value, Mapping):
        raise ValueError("prepared source provenance corpus_file_sizes is invalid")
    if not isinstance(raw_queries, list):
        raise ValueError("prepared source provenance is missing source_query_records")

    mapping = {str(alias): str(original) for alias, original in mapping_value.items()}
    hashes = {str(alias): value for alias, value in hashes_value.items()}
    files = {str(alias): value for alias, value in files_value.items()}
    sizes = {str(alias): value for alias, value in sizes_value.items()}
    if set(hashes) != set(files) or any(not isinstance(value, str) or not value for value in hashes.values()):
        raise ValueError("prepared source corpus hashes and files do not match")
    if not set(hashes).issubset(mapping):
        raise ValueError("prepared source corpus hashes contain unknown aliases")
    if sizes and set(sizes) != set(hashes):
        raise ValueError("prepared source corpus sizes and hashes do not match")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in sizes.values()):
        raise ValueError("prepared source corpus sizes are invalid")

    root = Path(root_value)
    root_resolved = root.resolve()
    documents = {row["_id"]: row for row in bundle.get("documents", [])}
    active_originals = {mapping[alias] for alias in hashes}
    if set(documents) != active_originals:
        raise ValueError("prepared source documents do not match its materialized source aliases")
    parent_resolutions: dict[Path, Path] = {}
    for alias in sorted(hashes):
        relative = files[alias]
        candidate, stat_info = _resolve_prepared_corpus_file(
            root,
            root_resolved,
            relative,
            label="prepared source",
            parent_resolutions=parent_resolutions,
        )
        expected_bytes = sizes.get(alias)
        if verify_content_hashes:
            actual_hash = _sha256_file(candidate)
            if actual_hash != hashes[alias]:
                raise ValueError(f"prepared source corpus hash mismatch for alias {alias!r}")
            original = mapping[alias]
            text = candidate.read_text(encoding="utf-8")
            if documents[original].get("text") != text:
                raise ValueError(f"prepared source document text does not match alias {alias!r}")
        if expected_bytes is not None and stat_info.st_size != expected_bytes:
            raise ValueError(f"prepared source corpus size mismatch for alias {alias!r}")

    if _prepared_query_projection(raw_queries) != _prepared_query_projection(bundle.get("queries", []), loaded_bundle=True):
        raise ValueError("prepared source query rows do not match source_query_records")
    task = str(bundle["source"].get("dataset", "bundle"))
    computed = build_source_identity(
        task=task,
        corpus_ids=[row["_id"] for row in bundle["documents"]],
        query_ids=[row["_id"] for row in bundle["queries"]],
        source_doc_ids=mapping,
        qrels=bundle["qrels"],
        query_records=raw_queries,
        corpus_file_hashes=hashes,
    )
    recorded = _required_source_identity(provenance.get("source_identity", {}), label="prepared source_identity")
    for key in ("queries_hash", "corpus_files_sha256", "source_doc_ids_hash"):
        if recorded[key] != computed[key]:
            raise ValueError(f"prepared source_identity {key} does not match source bytes")
    for key in ("source_manifest_sha256", "source_inventory_sha256"):
        if key in recorded:
            computed[key] = recorded[key]
    return computed


def _verify_export_identity(
    bundle: Mapping[str, Any],
    *,
    source_identity: Mapping[str, str] | None,
    source_doc_ids: Mapping[str, str] | None,
    corpus_file_hashes: Mapping[str, str] | None,
) -> tuple[str, dict[str, Any]]:
    """Build the local identity and compare any supplied identity to it."""

    provenance = bundle.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    prepared = bool(provenance.get("prepared_source"))
    if prepared:
        provenance_mapping = provenance.get("source_doc_ids")
        provenance_hashes = provenance.get("corpus_file_hashes")
        if source_doc_ids is not None and dict(source_doc_ids) != dict(provenance_mapping or {}):
            raise ValueError("source_doc_ids does not match prepared source provenance")
        if corpus_file_hashes is not None and dict(corpus_file_hashes) != dict(provenance_hashes or {}):
            raise ValueError("corpus_file_hashes does not match prepared source provenance")
        built_identity = _prepared_source_identity(
            bundle,
            verify_content_hashes=provenance.get("content_hashes_verified", True),
        )
    else:
        provenance_mapping = provenance.get("source_doc_ids")
        provenance_hashes = provenance.get("corpus_file_hashes")
        mapping = source_doc_ids
        hashes = corpus_file_hashes
        if mapping is None and isinstance(provenance_mapping, Mapping):
            mapping = dict(provenance_mapping)
        if hashes is None and isinstance(provenance_hashes, Mapping):
            hashes = dict(provenance_hashes)
        task, _, built_identity = _bundle_identity_inputs(bundle, mapping, hashes)

    recorded = source_identity
    if recorded is None:
        candidate = provenance.get("source_identity")
        if isinstance(candidate, Mapping) and candidate:
            recorded = candidate
    if recorded is not None:
        checked = _required_source_identity(recorded, label="source_identity")
        for key in ("queries_hash", "corpus_files_sha256", "source_doc_ids_hash"):
            if checked[key] != built_identity[key]:
                raise ValueError(f"source_identity {key} does not match the active bundle")
        for key in ("source_manifest_sha256", "source_inventory_sha256"):
            if key in checked:
                if built_identity.get(key) != checked[key]:
                    raise ValueError(f"source_identity {key} does not match the active bundle")
        for key, value in checked.items():
            built_identity.setdefault(key, value)
    task = str(bundle["source"].get("task" if prepared else "dataset", "bundle"))
    return task, built_identity


def export_saved_retrieval(
    bundle: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
    *,
    model: Mapping[str, Any] | EmbeddingConfig | None = None,
    source_identity: Mapping[str, str] | None = None,
    source_doc_ids: Mapping[str, str] | None = None,
    corpus_file_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Export rankings and real similarity scores in saved-retrieval format."""

    if not isinstance(bundle, Mapping) or bundle.get("schema_version") != "dci-data-bundle-v1":
        raise ValueError("bundle must be a dci-data-bundle-v1 object")
    documents = {row["_id"]: row for row in bundle["documents"]}
    qrels = bundle["qrels"]
    rankings: dict[str, list[str]] = {}
    scores: dict[str, dict[str, float]] = {}
    for query in bundle["queries"]:
        query_id = query["_id"]
        if query_id not in results:
            raise ValueError(f"retrieval results are missing query {query_id!r}")
        row = results[query_id]
        ranking = row.get("ranked_doc_ids")
        row_scores = row.get("scores")
        if not isinstance(ranking, list) or len(set(ranking)) != len(ranking):
            raise ValueError(f"ranking for query {query_id!r} must be a unique list")
        if not isinstance(row_scores, Mapping):
            raise ValueError(f"scores for query {query_id!r} are required")
        for doc_id in ranking:
            if doc_id not in documents or doc_id not in row_scores:
                raise ValueError(f"ranking/scores contain an unknown or unscored document: {doc_id!r}")
        wanted = list(ranking)
        for doc_id in qrels.get(query_id, {}):
            if doc_id not in documents:
                raise ValueError(f"qrels contain unknown document {doc_id!r}")
            if doc_id not in row_scores:
                raise ValueError(f"gold document {doc_id!r} has no real similarity score")
            if doc_id not in wanted:
                wanted.append(doc_id)
        for previous, current in zip(ranking, ranking[1:]):
            if float(row_scores[previous]) < float(row_scores[current]):
                raise ValueError(f"ranking for query {query_id!r} is not sorted by descending score")
        clean_scores: dict[str, float] = {}
        for doc_id in wanted:
            value = row_scores[doc_id]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"score for {query_id!r}/{doc_id!r} must be finite numeric")
            clean_scores[doc_id] = float(value)
        rankings[query_id] = list(ranking)
        scores[query_id] = clean_scores
    task, built_identity = _verify_export_identity(
        bundle,
        source_identity=source_identity,
        source_doc_ids=source_doc_ids,
        corpus_file_hashes=corpus_file_hashes,
    )
    if isinstance(model, EmbeddingConfig):
        model_payload = model.to_dict()
        model_payload["weight_identity"] = (
            model.weight_identity
            if model.weight_identity is not None
            else model_weight_identity(model.model_path)
        )
        model_payload["tokenizer_identity"] = _component_identity(model.model_path, component="tokenizer")
        model_payload["model_config_identity"] = _component_identity(model.model_path, component="model_config")
    elif model is None:
        default = EmbeddingConfig()
        model_payload = default.to_dict()
        model_payload["weight_identity"] = model_weight_identity(default.model_path)
        model_payload["tokenizer_identity"] = _component_identity(default.model_path, component="tokenizer")
        model_payload["model_config_identity"] = _component_identity(default.model_path, component="model_config")
    elif isinstance(model, Mapping):
        model_payload = copy.deepcopy(dict(model))
        model_path = model_payload.get("model_path", model_payload.get("path"))
        if model_path and not model_payload.get("weight_identity"):
            model_payload["weight_identity"] = model_weight_identity(model_path)
        if model_path:
            model_payload.setdefault("tokenizer_identity", _component_identity(model_path, component="tokenizer"))
            model_payload.setdefault("model_config_identity", _component_identity(model_path, component="model_config"))
    else:
        raise ValueError("model must be an EmbeddingConfig, mapping, or None")
    if model_payload.get("path") is None:
        model_payload["path"] = model_payload.get("model_path", DEFAULT_EMBEDDING_MODEL_PATH)
    if "trust_remote_code" not in model_payload:
        model_payload["trust_remote_code"] = True
    if model_payload.get("trust_remote_code") is not True:
        raise ValueError("saved retrieval must record trust_remote_code=true")
    return {
        "schema_version": SAVED_RETRIEVAL_SCHEMA_VERSION,
        "task": task,
        "model": model_payload,
        "source_identity": built_identity,
        "rankings": rankings,
        "scores": scores,
    }


def load_prepared_source(
    suite_root: str | os.PathLike[str],
    task: str,
    *,
    verify_content_hashes: bool = True,
) -> dict[str, Any]:
    """Load a prepared document-pool source into the common bundle shape.

    ``verify_content_hashes=False`` is intended for a caller that already
    verified the immutable suite.  It still checks paths, regular-file type,
    saved byte sizes, IDs, labels, and UTF-8 decoding, but treats the saved
    per-file digests as declared identity without scanning every file for a
    fresh SHA-256.
    """

    if not isinstance(verify_content_hashes, bool):
        raise ValueError("verify_content_hashes must be boolean")
    root = Path(suite_root)
    metadata_dir = root / "metadata" / task
    workspace_dir = root / "workspaces" / task
    try:
        manifest = json.loads((metadata_dir / "manifest.json").read_text(encoding="utf-8"))
        mapping = json.loads((metadata_dir / "source_doc_ids.json").read_text(encoding="utf-8"))
        qrels_alias = json.loads((metadata_dir / "qrels.json").read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"prepared source is missing required file: {exc.filename}") from exc
    if not isinstance(manifest, Mapping) or manifest.get("task") != task:
        raise ValueError("prepared source manifest task mismatch")
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("prepared source document mapping is invalid")
    if isinstance(qrels_alias, Mapping) and isinstance(qrels_alias.get("qrels"), Mapping):
        qrels_alias = qrels_alias["qrels"]
    if not isinstance(qrels_alias, Mapping):
        raise ValueError("prepared source qrels are invalid")
    alias_to_original: dict[str, str] = {}
    for alias, original in mapping.items():
        if not isinstance(alias, str) or not alias or not isinstance(original, str) or not original:
            raise ValueError("prepared source document mapping contains an invalid ID")
        if alias in alias_to_original:
            raise ValueError(f"prepared source document mapping contains duplicate alias: {alias!r}")
        alias_to_original[alias] = original
    if len(set(alias_to_original.values())) != len(alias_to_original):
        raise ValueError("prepared source document mapping contains duplicate original IDs")
    original_to_alias = {original: alias for alias, original in alias_to_original.items()}
    inventory_path = metadata_dir / "file_inventory.json"
    inventory_rows: list[tuple[str, Mapping[str, Any]]] = []
    if inventory_path.is_file():
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        if not isinstance(inventory, Mapping):
            raise ValueError("prepared source file inventory is invalid")
        for relative, row in inventory.items():
            if not isinstance(relative, str) or not isinstance(row, Mapping):
                raise ValueError("prepared source file inventory contains an invalid row")
            inventory_rows.append((relative, row))
    else:
        # ``document_pools.prepare_document_pools`` writes this JSONL name;
        # accept it alongside the newer file_inventory.json representation.
        corpus_inventory_path = metadata_dir / "corpus_files.jsonl"
        if corpus_inventory_path.is_file():
            with corpus_inventory_path.open("r", encoding="utf-8", newline="") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        raise ValueError("prepared source corpus_files.jsonl rows must be objects")
                    filename = row.get("filename")
                    alias = row.get("doc_id")
                    original = row.get("source_doc_id")
                    if not all(isinstance(value, str) and value for value in (filename, alias, original)):
                        raise ValueError("prepared source corpus_files.jsonl row is missing IDs or filename")
                    inventory_rows.append((
                        f"workspaces/{task}/corpus/{filename}",
                        {
                            "alias": alias,
                            "original_id": original,
                            "filename": filename,
                            "bytes": row.get("bytes"),
                            "sha256": row.get("sha256"),
                        },
                    ))
    root_resolved = root.resolve()
    parent_resolutions: dict[Path, Path] = {}
    inventory_by_alias: dict[str, dict[str, Any]] = {}
    for relative, row in inventory_rows:
        alias = row.get("alias")
        original = row.get("original_id")
        expected_bytes = row.get("bytes")
        expected_hash = row.get("sha256")
        if not isinstance(alias, str) or not alias or not isinstance(original, str) or not original:
            raise ValueError("prepared source inventory row is missing IDs")
        if alias not in alias_to_original or alias_to_original[alias] != original:
            raise ValueError(f"prepared source inventory ID mismatch for alias {alias!r}")
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise ValueError(f"prepared source inventory is missing a valid byte size for alias {alias!r}")
        if not isinstance(expected_hash, str) or not expected_hash:
            if not verify_content_hashes:
                raise ValueError(f"prepared source inventory is missing sha256 for alias {alias!r}")
            raise ValueError(f"prepared source inventory is missing sha256 for alias {alias!r}")
        filename = row.get("filename")
        if filename is not None and (not isinstance(filename, str) or not filename):
            raise ValueError(f"prepared source inventory filename is invalid for alias {alias!r}")
        path, stat_info = _resolve_prepared_corpus_file(
            root,
            root_resolved,
            relative,
            label="prepared source inventory",
            parent_resolutions=parent_resolutions,
        )
        if filename is not None and path.name != filename:
            raise ValueError(f"prepared source inventory filename mismatch for alias {alias!r}")
        if stat_info.st_size != expected_bytes:
            raise ValueError(f"prepared source corpus size mismatch for alias {alias!r}")
        if verify_content_hashes:
            actual_hash = _sha256_file(path)
            if actual_hash != expected_hash:
                raise ValueError(f"prepared source inventory hash mismatch for alias {alias!r}")
        else:
            actual_hash = expected_hash
        if alias in inventory_by_alias:
            raise ValueError(f"prepared source inventory contains duplicate alias: {alias!r}")
        inventory_by_alias[alias] = {
            "path": path,
            "bytes": expected_bytes,
            "sha256": actual_hash,
        }
    if inventory_rows:
        missing = sorted(set(alias_to_original) - set(inventory_by_alias))
        extra = sorted(set(inventory_by_alias) - set(alias_to_original))
        if extra or (missing and not verify_content_hashes):
            raise ValueError(
                "prepared source inventory IDs differ from source_doc_ids; "
                f"missing={missing[:5]} extra={extra[:5]}"
            )

    documents: list[dict[str, Any]] = []
    corpus_dir = workspace_dir / "corpus"
    corpus_file_hashes: dict[str, str] = {}
    corpus_file_sizes: dict[str, int] = {}
    corpus_files: dict[str, str] = {}
    for alias, original in sorted(alias_to_original.items()):
        inventory_entry = inventory_by_alias.get(alias)
        if inventory_entry is None:
            path, stat_info = _resolve_prepared_corpus_file(
                root,
                root_resolved,
                os.fspath(corpus_dir / f"{alias}.md"),
                label="prepared source",
                parent_resolutions=parent_resolutions,
            )
            if not verify_content_hashes:
                raise ValueError(
                    f"prepared source has no recorded digest for alias {alias!r}; "
                    "verify_content_hashes=False cannot fabricate one"
                )
            expected_bytes = stat_info.st_size
            actual_hash = _sha256_file(path)
        else:
            path = inventory_entry["path"]
            expected_bytes = inventory_entry["bytes"]
            actual_hash = inventory_entry["sha256"]
        text = _read_prepared_corpus_text(path, expected_bytes=expected_bytes, alias=alias)
        lines = text.splitlines()
        title = lines[0].removeprefix("# ") if lines else original
        documents.append({
            "_id": original,
            "title": title or original,
            "text": text,
            "language": "unknown",
            "metadata": {"alias": alias},
        })
        corpus_file_hashes[alias] = actual_hash
        corpus_file_sizes[alias] = expected_bytes
        corpus_files[alias] = os.fspath(path.relative_to(root_resolved))
    queries: list[dict[str, Any]] = []
    source_query_records: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()
    query_file = metadata_dir / "queries.jsonl"
    # Iterate physical JSONL records.  ``str.splitlines`` also treats legal
    # JSON string characters U+2028/U+2029 as separators and would corrupt a
    # query containing one of them.
    with query_file.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError("prepared source query rows must be objects")
                query_id = row.get("_id", row.get("query_id"))
                if not isinstance(query_id, str) or not query_id:
                    raise ValueError("prepared source query rows require non-empty IDs")
                if query_id in seen_query_ids:
                    raise ValueError(f"prepared source query rows contain duplicate ID: {query_id!r}")
                seen_query_ids.add(query_id)
                if not isinstance(row.get("text", ""), str):
                    raise ValueError(f"prepared source query text is invalid for {query_id!r}")
                source_query_records.append(copy.deepcopy(dict(row)))
                queries.append({
                    "_id": query_id,
                    "text": row.get("text", ""),
                    "language": row.get("language", "unknown"),
                    "metadata": {
                        key: value for key, value in row.items()
                        if key not in {"_id", "query_id", "text", "language"}
                    },
                })
    qrels: dict[str, dict[str, float]] = {}
    for query_id, row in qrels_alias.items():
        if not isinstance(query_id, str) or not query_id or not isinstance(row, Mapping):
            raise ValueError("prepared source qrels contain an invalid query or label row")
        clean_row: dict[str, float] = {}
        for alias, score in row.items():
            if alias not in alias_to_original:
                raise ValueError(f"prepared source qrels contain unknown document alias: {alias!r}")
            try:
                clean_row[alias_to_original[alias]] = float(score)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"prepared source qrels contain an invalid score for {query_id!r}") from exc
        qrels[query_id] = clean_row
    query_ids = {query["_id"] for query in queries}
    if set(qrels) != query_ids:
        missing = sorted(query_ids - set(qrels))
        extra = sorted(set(qrels) - query_ids)
        raise ValueError(f"prepared source qrels/query IDs differ; missing={missing[:5]} extra={extra[:5]}")
    source = {
        "dataset": manifest.get("dataset", task),
        "revision": manifest.get("resolved_revision", manifest.get("revision", "unknown")),
        "task": task,
    }
    recorded_source_identity: dict[str, Any] = {}
    source_identity_path = metadata_dir / "source_identity.json"
    if source_identity_path.is_file():
        payload = json.loads(source_identity_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("prepared source source_identity.json is invalid")
        recorded_source_identity = dict(payload)
    elif isinstance(manifest.get("source_identity"), Mapping):
        recorded_source_identity = dict(manifest["source_identity"])
    computed_source_identity = build_source_identity(
        task=task,
        corpus_ids=[row["_id"] for row in documents],
        query_ids=[row["_id"] for row in queries],
        source_doc_ids=alias_to_original,
        qrels=qrels,
        query_records=source_query_records,
        corpus_file_hashes=corpus_file_hashes,
    )
    for key in ("queries_hash", "source_doc_ids_hash"):
        manifest_value = manifest.get(key)
        if manifest_value is not None and manifest_value != computed_source_identity[key]:
            raise ValueError(f"prepared source manifest {key} does not match source bytes")
    if recorded_source_identity:
        required_recorded = _required_source_identity(recorded_source_identity, label="prepared source_identity")
        for key in ("queries_hash", "corpus_files_sha256", "source_doc_ids_hash"):
            if required_recorded[key] != computed_source_identity[key]:
                raise ValueError(f"prepared source_identity {key} does not match source bytes")
        computed_source_identity.update(
            {
                key: required_recorded[key]
                for key in ("source_manifest_sha256", "source_inventory_sha256")
                if key in required_recorded
            }
        )
    prepared_bundle = {
        "schema_version": "dci-data-bundle-v1",
        "input_kind": "retrieval",
        "source": source,
        "documents": documents,
        "queries": queries,
        "qrels": qrels,
        "evidence": [],
        "judgments": [],
        "provenance": {
            "prepared_source": str(root),
            "source_identity": computed_source_identity,
            "source_doc_ids": alias_to_original,
            "original_to_alias": original_to_alias,
            "corpus_file_hashes": corpus_file_hashes,
            "corpus_file_sizes": corpus_file_sizes,
            "corpus_files": corpus_files,
            "source_query_records": source_query_records,
            "content_hashes_verified": verify_content_hashes,
            "content_hash_verification": (
                "verified" if verify_content_hashes else "declared_saved_digest"
            ),
        },
    }
    _prepared_source_identity(prepared_bundle, verify_content_hashes=verify_content_hashes)
    return prepared_bundle


load_prepared_source_bundle = load_prepared_source


__all__ = [
    "BM25Index",
    "DEFAULT_DOCUMENT_PROMPT",
    "DEFAULT_EMBEDDING_MODEL_PATH",
    "DEFAULT_QUERY_PROMPT",
    "DEFAULT_TOKEN_MAX_LENGTH",
    "EMBEDDING_MODEL_PATH",
    "EmbeddingCache",
    "EmbeddingShardCache",
    "EmbeddingConfig",
    "LocalEmbeddingEncoder",
    "SAVED_RETRIEVAL_SCHEMA_VERSION",
    "SentenceTransformerEncoder",
    "bm25_search",
    "build_source_identity",
    "build_source_identity_for_bundle",
    "dense_search",
    "export_saved_retrieval",
    "load_prepared_source",
    "load_prepared_source_bundle",
    "model_weight_identity",
    "search_dense",
    "search_embeddings",
    "unicode_tokens",
]
