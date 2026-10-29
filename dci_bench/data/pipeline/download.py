"""Small, dependency-free primitives used by the DCI source pipeline.

The pipeline deliberately treats a downloaded source as an immutable snapshot.
This module contains the parts which are easy to get subtly wrong when a source
is downloaded over an unreliable connection: direct HTTP requests, bounded
streaming, hash/size checks, path validation, and archive extraction.

The public helpers are intentionally useful on their own.  Connector-specific
selection and parsing live in :mod:`dci_bench.data.pipeline.connectors`.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shutil
import stat
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Mapping

try:
    import fcntl
except ImportError:  # pragma: no cover - the supported runtime is POSIX.
    fcntl = None


DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 3
DEFAULT_MAX_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_MAX_ARCHIVE_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_MAX_ARCHIVE_FILES = 100_000
DEFAULT_MAX_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_COMPRESSION_RATIO = 200.0
CHUNK_SIZE = 1024 * 1024

_SECRET_KEY_RE = re.compile(
    r"(?:token|password|passwd|secret|authorization|auth|credential|api[_-]?key|access[_-]?token|private[_-]?key)",
    re.IGNORECASE,
)
_TRANSIENT_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504, 522, 524}


class DownloadError(RuntimeError):
    """A network or local materialisation failure."""


class IntegrityError(DownloadError):
    """A downloaded/copyied object failed its declared integrity checks."""


class ArchiveSecurityError(DownloadError):
    """An archive contains unsafe paths, links, or excessive content."""


class SnapshotConflictError(DownloadError):
    """A supposedly immutable snapshot is missing, changed, or incompatible."""


@contextmanager
def nonblocking_file_lock(path: Path):
    """Hold a POSIX advisory lock, failing immediately when already held."""

    if fcntl is None:
        raise DownloadError("nonblocking snapshot locks require POSIX fcntl")
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.is_symlink():
        raise SnapshotConflictError(f"snapshot lock path is a symlink: {lock_path}")
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SnapshotConflictError(f"source snapshot is already being materialised: {lock_path}") from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def canonical_json(value: Any) -> str:
    """Return deterministic JSON suitable for a content/configuration hash."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _redact_url(value: str) -> str:
    """Remove credentials and secret query values without hiding the URL host."""

    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError:
        return "<redacted-url>"
    if parsed.scheme not in {"http", "https", "ftp", "file"}:
        return value
    username = parsed.username
    hostname = parsed.hostname or ""
    port = ""
    try:
        if parsed.port is not None:
            port = f":{parsed.port}"
    except ValueError:
        port = ""
    host = hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}{port}"
    params: list[tuple[str, str]] = []
    for key, val in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
        if _SECRET_KEY_RE.search(key):
            val = "<redacted>"
        params.append((key, val))
    query = urllib.parse.urlencode(params, doseq=True)
    # Usernames are not credentials in all URL schemes, but preserving a
    # username can still leak an account identifier.  Keep only the host.
    del username
    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, query, ""))


def redact_secrets(value: Any, *, _key: str = "") -> Any:
    """Return a JSON-compatible copy with obvious credentials removed."""

    if isinstance(value, Mapping):
        return {
            str(key): "<redacted>"
            if _SECRET_KEY_RE.search(str(key))
            else redact_secrets(item, _key=str(key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item, _key=_key) for item in value]
    if isinstance(value, str) and ("url" in _key.lower() or "uri" in _key.lower()):
        return _redact_url(value)
    return value


def safe_relative_path(name: str | os.PathLike[str]) -> str:
    """Validate and normalise a relative archive/snapshot member path.

    Both slash styles are accepted as input because ZIP files produced on
    Windows can contain backslashes.  The returned path always uses POSIX
    separators and never escapes its destination root.
    """

    raw = os.fspath(name)
    if "\x00" in raw:
        raise ArchiveSecurityError("path contains NUL")
    raw = raw.replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or raw.startswith("/"):
        raise ArchiveSecurityError(f"absolute path is not allowed: {name!r}")
    # PurePosixPath('') is '.', which is a directory marker rather than a file.
    parts = path.parts
    if not parts or parts == (".",):
        raise ArchiveSecurityError(f"empty path is not allowed: {name!r}")
    if any(part in {"", ".", ".."} for part in parts):
        raise ArchiveSecurityError(f"path traversal is not allowed: {name!r}")
    # Reject Windows drive paths even when running on POSIX.
    if re.match(r"^[A-Za-z]:$", parts[0]):
        raise ArchiveSecurityError(f"drive path is not allowed: {name!r}")
    return "/".join(parts)


def _destination_path(root: Path, relative: str | os.PathLike[str]) -> Path:
    safe = safe_relative_path(relative)
    destination = (root / safe).resolve()
    root_resolved = root.resolve()
    try:
        destination.relative_to(root_resolved)
    except ValueError as exc:  # pragma: no cover - defence in depth
        raise ArchiveSecurityError(f"path escapes destination: {relative!r}") from exc
    return destination


def _validate_limit(name: str, value: int | float | None, minimum: int | float = 0) -> int | float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be a number >= {minimum}")
    return value


def file_digest(path: Path, *, max_bytes: int | None = None) -> tuple[str, int]:
    """Hash a regular file while enforcing an optional byte limit."""

    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_symlink() or not path.is_file():
        raise IntegrityError(f"expected a regular file: {path}")
    limit = _validate_limit("max_bytes", max_bytes)
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            size += len(chunk)
            if limit is not None and size > limit:
                raise IntegrityError(f"file exceeds size limit ({limit} bytes): {path}")
            hasher.update(chunk)
    return hasher.hexdigest(), size


def _check_expected(path: Path, *, expected_sha256: str | None, expected_bytes: int | None) -> tuple[str, int]:
    digest, size = file_digest(path, max_bytes=expected_bytes)
    if expected_bytes is not None and size != expected_bytes:
        raise IntegrityError(
            f"size mismatch for {path.name}: expected {expected_bytes}, got {size}"
        )
    if expected_sha256 is not None:
        expected = str(expected_sha256).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise IntegrityError(f"invalid expected sha256: {expected_sha256!r}")
        if digest != expected:
            raise IntegrityError(f"sha256 mismatch for {path.name}: expected {expected}, got {digest}")
    return digest, size


def copy_file(
    source: Path,
    destination: Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    overwrite: bool = False,
) -> tuple[str, int]:
    """Copy bytes from a local path, never creating a symlink.

    A symlink source is read as bytes and written as a regular destination file;
    this is intentional for imported local data.  Symlink directories are not
    followed because they make a source tree's boundary ambiguous.
    """

    if source.is_dir() or (source.is_symlink() and source.resolve().is_dir()):
        raise IntegrityError(f"expected a file, got directory: {source}")
    if not source.exists():
        raise FileNotFoundError(source)
    max_bytes = int(_validate_limit("max_bytes", max_bytes, 0) or 0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    part = destination.with_name(destination.name + ".partial")
    if part.exists() and not overwrite:
        raise FileExistsError(part)
    hasher = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as source_handle, part.open("wb") as output:
            while True:
                chunk = source_handle.read(CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise IntegrityError(f"file exceeds size limit ({max_bytes} bytes): {source}")
                hasher.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        digest = hasher.hexdigest()
        if expected_bytes is not None and size != expected_bytes:
            raise IntegrityError(f"size mismatch for {source.name}: expected {expected_bytes}, got {size}")
        if expected_sha256 is not None and digest != str(expected_sha256).lower():
            raise IntegrityError(
                f"sha256 mismatch for {source.name}: expected {expected_sha256}, got {digest}"
            )
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        os.replace(part, destination)
        return digest, size
    except Exception:
        # Keep the incomplete bytes as a useful recovery/debugging artefact.
        raise


def _direct_opener() -> urllib.request.OpenerDirector:
    # urllib otherwise honours HTTP(S)_PROXY from the process environment.  A
    # DCI source snapshot should be reproducible and must not accidentally use
    # the legacy 27897 proxy.  This is local to this opener; no global env is
    # changed.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _TRANSIENT_HTTP_STATUS
    return isinstance(
        exc,
        (
            urllib.error.URLError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            http.client.BadStatusLine,
            ConnectionError,
            TimeoutError,
        ),
    )


def download_url(
    url: str,
    destination: Path,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    max_bytes: int = DEFAULT_MAX_BYTES,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    headers: Mapping[str, str] | None = None,
    overwrite: bool = False,
    opener: urllib.request.OpenerDirector | None = None,
) -> tuple[str, int]:
    """Download an HTTP(S) URL with bounded streaming and atomic finalisation.

    The destination's ``.partial`` sibling is retained after any failure.  A
    later call can resume it when the server supports ``Range``; if the server
    returns a full 200 response, the partial is safely restarted from byte 0.
    """

    if not isinstance(url, str) or not url:
        raise ValueError("url must be a non-empty string")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise DownloadError(f"only http(s) URLs are supported: {url!r}")
    if parsed.username or parsed.password:
        raise DownloadError("URL credentials are not accepted")
    timeout = float(_validate_limit("timeout", timeout, 0.001))
    retries = int(_validate_limit("retries", retries, 0))
    max_bytes = int(_validate_limit("max_bytes", max_bytes, 0) or 0)
    if expected_bytes is not None:
        expected_bytes = int(_validate_limit("expected_bytes", expected_bytes, 0) or 0)
        if expected_bytes > max_bytes:
            raise IntegrityError("expected_bytes exceeds max_bytes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        # A completed object is safe to reuse only when its declared digest
        # agrees.  Without a declaration, refusing to guess prevents silent
        # overwrites and makes snapshot immutability explicit.
        if expected_sha256 is not None or expected_bytes is not None:
            return _check_expected(
                destination, expected_sha256=expected_sha256, expected_bytes=expected_bytes
            )
        raise FileExistsError(destination)
    part = destination.with_name(destination.name + ".partial")
    # Keep a visible partial marker even when a server's announced length is
    # rejected before any body bytes are read.  A failed attempt must never be
    # mistaken for a completed destination, but it should remain diagnosable
    # and eligible for a later retry.
    part.touch(exist_ok=True)
    last_error: BaseException | None = None
    request_headers = {str(key): str(value) for key, value in (headers or {}).items()}
    request_opener = opener or _direct_opener()

    for attempt in range(retries + 1):
        resume_from = part.stat().st_size if part.exists() else 0
        current_headers = dict(request_headers)
        if resume_from:
            current_headers["Range"] = f"bytes={resume_from}-"
        request = urllib.request.Request(url, headers=current_headers, method="GET")
        try:
            with request_opener.open(request, timeout=timeout) as response:
                status = int(getattr(response, "status", response.getcode() or 200))
                if status not in {200, 206}:
                    raise DownloadError(f"HTTP {status} while downloading {url!r}")
                append = resume_from > 0 and status == 206
                if not append:
                    resume_from = 0
                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        announced = int(content_length)
                    except ValueError:
                        announced = None
                    if announced is not None and resume_from + announced > max_bytes:
                        raise IntegrityError(
                            f"remote object exceeds size limit ({max_bytes} bytes): {url!r}"
                        )
                    if expected_bytes is not None and resume_from + announced != expected_bytes:
                        # A server returning a stale range length is not a
                        # reason to accept a truncated snapshot.
                        raise IntegrityError(
                            f"remote size mismatch for {url!r}: expected {expected_bytes}, "
                            f"got {resume_from + announced}"
                        )
                mode = "ab" if append else "wb"
                hasher = hashlib.sha256()
                size = 0
                if append:
                    # Hashing the existing partial makes the final digest
                    # independent of whether the request was resumed.
                    with part.open("rb") as previous:
                        for chunk in iter(lambda: previous.read(CHUNK_SIZE), b""):
                            size += len(chunk)
                            if size > max_bytes:
                                raise IntegrityError("partial file exceeds size limit")
                            hasher.update(chunk)
                with part.open(mode) as output:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > max_bytes:
                            raise IntegrityError(
                                f"download exceeds size limit ({max_bytes} bytes): {url!r}"
                            )
                        hasher.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if content_length and announced is not None and size != resume_from + announced:
                    raise http.client.IncompleteRead(
                        partial=b"", expected=resume_from + announced
                    )
                if expected_bytes is not None and size != expected_bytes:
                    raise IntegrityError(
                        f"remote size mismatch for {url!r}: expected {expected_bytes}, got {size}"
                    )
                digest = hasher.hexdigest()
                if expected_sha256 is not None and digest != str(expected_sha256).lower():
                    raise IntegrityError(
                        f"sha256 mismatch for {url!r}: expected {expected_sha256}, got {digest}"
                    )
                if destination.exists() and not overwrite:
                    raise FileExistsError(destination)
                os.replace(part, destination)
                return digest, size
        except Exception as exc:
            last_error = exc
            if attempt >= retries or not _retryable_error(exc):
                raise
            # Do not sleep after the final attempt.  The short bounded backoff
            # avoids turning a transient server error into a busy loop.
            time.sleep(min(0.25 * (2**attempt), 2.0))
    assert last_error is not None  # pragma: no cover - loop always returns/raises
    raise DownloadError(str(last_error)) from last_error


def _archive_member_path(name: str) -> str:
    try:
        return safe_relative_path(name)
    except ArchiveSecurityError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise ArchiveSecurityError(f"invalid archive path: {name!r}") from exc


def _check_archive_member_budget(
    *,
    name: str,
    size: int,
    compressed_size: int,
    total_size: int,
    max_member_bytes: int,
    max_total_bytes: int,
    max_ratio: float,
) -> int:
    if size < 0 or size > max_member_bytes:
        raise ArchiveSecurityError(f"archive member exceeds size limit: {name!r}")
    if total_size + size > max_total_bytes:
        raise ArchiveSecurityError("archive expanded size exceeds limit")
    if compressed_size > 0 and size / compressed_size > max_ratio:
        raise ArchiveSecurityError(f"archive compression ratio is too high: {name!r}")
    return total_size + size


def _extract_zip(
    archive: Path,
    destination: Path,
    *,
    max_member_bytes: int,
    max_total_bytes: int,
    max_files: int,
    max_ratio: float,
) -> None:
    seen: set[str] = set()
    total = 0
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        if len(members) > max_files:
            raise ArchiveSecurityError("archive contains too many members")
        for info in members:
            name = _archive_member_path(info.filename)
            if name in seen:
                raise ArchiveSecurityError(f"duplicate archive member: {name!r}")
            seen.add(name)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ArchiveSecurityError(f"archive symlink is not allowed: {name!r}")
            is_dir = info.is_dir() or info.filename.endswith(("/", "\\"))
            if is_dir:
                _destination_path(destination, name).mkdir(parents=True, exist_ok=True)
                continue
            total = _check_archive_member_budget(
                name=name,
                size=info.file_size,
                compressed_size=info.compress_size,
                total_size=total,
                max_member_bytes=max_member_bytes,
                max_total_bytes=max_total_bytes,
                max_ratio=max_ratio,
            )
            output = _destination_path(destination, name)
            output.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(info, "r") as source, output.open("wb") as target:
                copied = 0
                while True:
                    chunk = source.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > max_member_bytes or copied > info.file_size:
                        raise ArchiveSecurityError(f"archive member size changed: {name!r}")
                    target.write(chunk)
                if copied != info.file_size:
                    raise ArchiveSecurityError(f"truncated archive member: {name!r}")


def _extract_tar(
    archive: Path,
    destination: Path,
    *,
    max_member_bytes: int,
    max_total_bytes: int,
    max_files: int,
    max_ratio: float,
) -> None:
    del max_ratio  # tar metadata has no reliable compressed member size.
    seen: set[str] = set()
    total = 0
    with tarfile.open(archive, mode="r:*") as handle:
        members = handle.getmembers()
        if len(members) > max_files:
            raise ArchiveSecurityError("archive contains too many members")
        for info in members:
            name = _archive_member_path(info.name)
            if name in seen:
                raise ArchiveSecurityError(f"duplicate archive member: {name!r}")
            seen.add(name)
            if info.issym() or info.islnk() or info.isdev() or not (info.isfile() or info.isdir()):
                raise ArchiveSecurityError(f"archive link/special file is not allowed: {name!r}")
            if info.isdir():
                _destination_path(destination, name).mkdir(parents=True, exist_ok=True)
                continue
            total = _check_archive_member_budget(
                name=name,
                size=info.size,
                compressed_size=0,
                total_size=total,
                max_member_bytes=max_member_bytes,
                max_total_bytes=max_total_bytes,
                max_ratio=float("inf"),
            )
            source = handle.extractfile(info)
            if source is None:
                raise ArchiveSecurityError(f"cannot read archive member: {name!r}")
            output = _destination_path(destination, name)
            output.parent.mkdir(parents=True, exist_ok=True)
            with source, output.open("wb") as target:
                copied = 0
                while True:
                    chunk = source.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > max_member_bytes or copied > info.size:
                        raise ArchiveSecurityError(f"archive member size changed: {name!r}")
                    target.write(chunk)
                if copied != info.size:
                    raise ArchiveSecurityError(f"truncated archive member: {name!r}")


def safe_extract_archive(
    archive: Path,
    destination: Path,
    *,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    max_total_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
    max_files: int = DEFAULT_MAX_ARCHIVE_FILES,
    max_compression_ratio: float = DEFAULT_MAX_COMPRESSION_RATIO,
    overwrite: bool = False,
) -> Path:
    """Safely extract a ZIP/TAR archive and atomically publish its directory.

    Extraction happens in ``<destination>.partial``.  Unsafe or truncated
    archives therefore cannot appear as a completed source directory, while
    the partial directory remains available for diagnosis.
    """

    archive = Path(archive)
    destination = Path(destination)
    if not archive.exists() or not archive.is_file() or archive.is_symlink():
        raise FileNotFoundError(archive)
    max_member_bytes = int(_validate_limit("max_member_bytes", max_member_bytes, 0) or 0)
    max_total_bytes = int(_validate_limit("max_total_bytes", max_total_bytes, 0) or 0)
    max_files = int(_validate_limit("max_files", max_files, 1) or 1)
    max_compression_ratio = float(_validate_limit("max_compression_ratio", max_compression_ratio, 1.0))
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists() and not overwrite:
        raise FileExistsError(partial)
    if partial.exists() and overwrite:
        if partial.is_dir() and not partial.is_symlink():
            shutil.rmtree(partial)
        else:
            partial.unlink()
    partial.mkdir(parents=True, exist_ok=False)
    try:
        try:
            if zipfile.is_zipfile(archive):
                _extract_zip(
                    archive,
                    partial,
                    max_member_bytes=max_member_bytes,
                    max_total_bytes=max_total_bytes,
                    max_files=max_files,
                    max_ratio=max_compression_ratio,
                )
            elif tarfile.is_tarfile(archive):
                _extract_tar(
                    archive,
                    partial,
                    max_member_bytes=max_member_bytes,
                    max_total_bytes=max_total_bytes,
                    max_files=max_files,
                    max_ratio=max_compression_ratio,
                )
            else:
                raise ArchiveSecurityError(f"unsupported archive format: {archive}")
        except (zipfile.BadZipFile, tarfile.TarError, OSError) as exc:
            raise ArchiveSecurityError(f"invalid or unreadable archive: {archive}") from exc
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        os.replace(partial, destination)
        return destination
    except Exception:
        # Keep ``partial`` by design.  It contains no source_manifest.json and
        # can never be mistaken for a complete snapshot.
        raise


def verify_file_inventory(root: Path, files: Iterable[Mapping[str, Any]]) -> None:
    """Verify a manifest's path, byte count, and SHA-256 entries."""

    root = Path(root)
    listed: set[str] = set()
    for entry in files:
        if not isinstance(entry, Mapping):
            raise SnapshotConflictError("source manifest file entry is not an object")
        relative = safe_relative_path(str(entry.get("path", "")))
        if relative in listed:
            raise SnapshotConflictError(f"duplicate file in source manifest: {relative}")
        listed.add(relative)
        path = _destination_path(root, relative)
        if path.is_symlink() or not path.is_file():
            raise SnapshotConflictError(f"manifest file is missing or not regular: {relative}")
        digest, size = file_digest(path)
        if str(entry.get("sha256", "")).lower() != digest:
            raise SnapshotConflictError(f"manifest sha256 mismatch: {relative}")
        if int(entry.get("bytes", -1)) != size:
            raise SnapshotConflictError(f"manifest byte count mismatch: {relative}")


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fetch_source(spec: Mapping[str, Any], raw_root: Path) -> Path:
    """Lazy compatibility wrapper for callers that import the download module.

    Connector dispatch belongs in ``connectors.py``; the lazy import keeps this
    convenience entry point cycle-free for existing callers.
    """

    from .connectors import fetch_source as _fetch_source

    return _fetch_source(spec, raw_root)


def load_source(spec: Mapping[str, Any], snapshot: Path) -> dict[str, Any]:
    """Lazy compatibility wrapper for :func:`connectors.load_source`."""

    from .connectors import load_source as _load_source

    return _load_source(spec, snapshot)


__all__ = [
    "ArchiveSecurityError",
    "DEFAULT_MAX_ARCHIVE_BYTES",
    "DEFAULT_MAX_ARCHIVE_FILES",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_COMPRESSION_RATIO",
    "DEFAULT_MAX_MEMBER_BYTES",
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT",
    "DownloadError",
    "IntegrityError",
    "SnapshotConflictError",
    "canonical_json",
    "copy_file",
    "download_url",
    "file_digest",
    "fetch_source",
    "load_source",
    "nonblocking_file_lock",
    "read_json",
    "redact_secrets",
    "safe_extract_archive",
    "safe_relative_path",
    "sha256_bytes",
    "sha256_json",
    "verify_file_inventory",
]
