"""Offline content-addressed RootFS cache manager."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import uuid
from typing import Any, Callable, Iterator

from .bundle import BundleError, extract_bundle, freeze_tree, sha256_file, tree_digest
from .config import (
    READY_SCHEMA_VERSION,
    RuntimeConfig,
    load_config,
    runtime_policy_sha256,
)


class RuntimeManagerError(RuntimeError):
    """Base class for fail-closed runtime preparation/verification failures."""


class RuntimeNotReadyError(RuntimeManagerError):
    """Raised when a valid local runtime is not available."""


class RuntimeVerificationError(RuntimeManagerError):
    """Raised when an existing cache or READY marker is corrupt."""


class RuntimePreparationError(RuntimeManagerError):
    """Raised when an offline bundle cannot be safely published."""


ToolProbe = Callable[[Path, str], str]


@dataclass(frozen=True)
class VerifiedRuntime:
    """Host-side verified runtime handle and path-safe logical identity."""

    identity: dict[str, str]
    rootfs_path: Path

    def to_payload(self) -> dict[str, Any]:
        return {"identity": dict(self.identity), "rootfs_path": str(self.rootfs_path)}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise RuntimePreparationError(f"cannot atomically write READY marker: {path}") from exc


@contextmanager
def _digest_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        raise RuntimeManagerError(f"cannot open runtime lock: {lock_path}") from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as exc:
            raise RuntimeManagerError(
                f"runtime cache filesystem locking failed at {lock_path}; "
                "use an executable node-local cache_dir and retain the bundle on persistent storage"
            ) from exc
        yield
    finally:
        # Closing the descriptor releases flock, including when acquisition
        # failed. Do not issue LOCK_UN after a failed acquisition on FUSE.
        os.close(descriptor)


def _safe_relative_tool_path(rootfs: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not relative or "\\" in relative:
        raise RuntimeVerificationError(f"invalid tool path in READY: {relative!r}")
    candidate = rootfs / path
    try:
        candidate.resolve(strict=True).relative_to(rootfs.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise RuntimeVerificationError(f"tool path escapes rootfs: {relative!r}") from exc
    if not candidate.is_file():
        raise RuntimeVerificationError(f"runtime tool is missing: {relative}")
    return candidate


class RuntimeManager:
    """Prepare and verify a single content-addressed frozen RootFS."""

    REQUIRED_TOOLS = {"bash": "usr/bin/bash", "rg": "usr/bin/rg"}

    def __init__(self, config: RuntimeConfig, *, tool_probe: ToolProbe | None = None) -> None:
        self.config = config
        self._tool_probe = tool_probe

    @classmethod
    def from_config(cls, config_path: Path | str, *, tool_probe: ToolProbe | None = None) -> "RuntimeManager":
        return cls(load_config(config_path), tool_probe=tool_probe)

    @property
    def runtime_dir(self) -> Path:
        return self.config.cache_platform_dir()

    @property
    def rootfs_path(self) -> Path:
        return self.runtime_dir / "rootfs"

    @property
    def ready_path(self) -> Path:
        return self.runtime_dir / "READY.json"

    @property
    def lock_path(self) -> Path:
        return self.config.cache_dir / "locks" / f"{self.config.image_cache_name}.lock"

    def ensure_ready(
        self,
        *,
        bundle: Path | str | None = None,
        offline: bool = True,
        preflight: bool = False,
    ) -> VerifiedRuntime:
        """Verify/reuse a cache entry or import one explicit local bundle."""

        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        with _digest_lock(self.lock_path):
            if self.runtime_dir.exists() or self.runtime_dir.is_symlink():
                # A present cache directory is authoritative.  Even if a new
                # bundle was supplied, never repair/overwrite a failed entry.
                return self._verify_published(full=True, preflight=preflight)
            if not offline:
                raise RuntimeNotReadyError("online runtime retrieval is unsupported; use an offline bundle")
            if bundle is None:
                raise RuntimeNotReadyError(
                    "runtime cache is missing and no offline bundle was supplied"
                )
            return self._import_bundle(Path(bundle), preflight=preflight)

    def prepare(
        self,
        *,
        bundle: Path | str | None = None,
        offline: bool = True,
        preflight: bool = False,
    ) -> VerifiedRuntime:
        return self.ensure_ready(bundle=bundle, offline=offline, preflight=preflight)

    def verify(self, *, full: bool = True, preflight: bool = False) -> VerifiedRuntime:
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        with _digest_lock(self.lock_path):
            return self._verify_published(full=full, preflight=preflight)

    def inspect(self) -> dict[str, Any]:
        """Return a JSON-safe cache inspection without changing the cache."""

        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        with _digest_lock(self.lock_path):
            if not self.runtime_dir.exists():
                return {
                    "config": {
                        "schema_version": self.config.schema_version,
                        "mode": self.config.mode,
                        "image_digest": self.config.image_digest,
                        "platform": self.config.platform,
                        "rootfs_tree_sha256": self.config.rootfs_tree_sha256,
                        "cache_dir": str(self.config.cache_dir),
                    },
                    "cache_path": str(self.runtime_dir),
                    "ready": False,
                }
            verified = self._verify_published(full=True)
            with self.ready_path.open("r", encoding="utf-8") as handle:
                ready = json.load(handle)
            return {
                "config": {
                    "schema_version": self.config.schema_version,
                    "mode": self.config.mode,
                    "image_digest": self.config.image_digest,
                    "platform": self.config.platform,
                    "rootfs_tree_sha256": self.config.rootfs_tree_sha256,
                    "cache_dir": str(self.config.cache_dir),
                },
                "cache_path": str(self.runtime_dir),
                "ready": ready,
                "runtime": verified.to_payload(),
            }

    def _import_bundle(self, bundle: Path, *, preflight: bool) -> VerifiedRuntime:
        if not bundle.is_file():
            raise RuntimePreparationError(f"offline bundle is missing: {bundle}")
        try:
            actual_bundle_sha256 = sha256_file(bundle)
        except BundleError as exc:
            raise RuntimePreparationError(str(exc)) from exc
        if actual_bundle_sha256 != self.config.bundle_sha256:
            raise RuntimePreparationError(
                f"bundle SHA-256 mismatch: expected {self.config.bundle_sha256}, got {actual_bundle_sha256}"
            )

        staging_parent = self.config.cache_dir / "staging"
        images_parent = self.config.cache_dir / "images"
        staging_parent.mkdir(parents=True, exist_ok=True)
        images_parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f"{self.config.image_cache_name}-", dir=staging_parent))
        try:
            rootfs = staging / "rootfs"
            try:
                extract_bundle(bundle, rootfs)
            except BundleError as exc:
                raise RuntimePreparationError(str(exc)) from exc
            try:
                tree_hash = tree_digest(rootfs)
            except BundleError as exc:
                raise RuntimePreparationError(str(exc)) from exc
            if tree_hash != self.config.rootfs_tree_sha256:
                raise RuntimePreparationError(
                    f"rootfs tree SHA-256 mismatch: expected {self.config.rootfs_tree_sha256}, got {tree_hash}"
                )
            tools = self._collect_tools(rootfs, preflight=preflight)
            ready = {
                "schema_version": READY_SCHEMA_VERSION,
                "content_identity": self.config.content_identity,
                "tools": tools,
            }
            staging_runtime = staging
            self.runtime_dir.parent.mkdir(parents=True, exist_ok=True)
            if self.runtime_dir.exists() or self.runtime_dir.is_symlink():
                raise RuntimePreparationError(f"runtime cache appeared during import: {self.runtime_dir}")
            try:
                os.replace(staging_runtime, self.runtime_dir)
            except OSError as exc:
                raise RuntimePreparationError(f"cannot atomically publish runtime: {self.runtime_dir}") from exc
            staging = None  # type: ignore[assignment]
            _write_json_atomic(self.ready_path, ready)
            os.chmod(self.runtime_dir, 0o555)
            directory_fd = os.open(self.runtime_dir.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return self._verify_published(full=True, preflight=preflight)
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)

    def _read_ready(self) -> dict[str, Any]:
        if self.ready_path.is_symlink() or not self.ready_path.is_file():
            raise RuntimeVerificationError(f"READY marker is missing or not a regular file: {self.ready_path}")
        try:
            with self.ready_path.open("r", encoding="utf-8") as handle:
                ready = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeVerificationError(f"cannot read READY marker: {self.ready_path}") from exc
        if not isinstance(ready, dict):
            raise RuntimeVerificationError("READY marker must be a JSON object")
        if ready.get("schema_version") != READY_SCHEMA_VERSION:
            raise RuntimeVerificationError("unsupported READY schema_version")
        content_identity = ready.get("content_identity")
        if not isinstance(content_identity, dict):
            raise RuntimeVerificationError("READY content_identity must be an object")
        if set(content_identity) != {"image_digest", "platform", "rootfs_tree_sha256"}:
            raise RuntimeVerificationError("READY content_identity fields are invalid")
        if content_identity != self.config.content_identity:
            raise RuntimeVerificationError("READY content identity does not match runtime config")
        tools = ready.get("tools")
        if not isinstance(tools, dict):
            raise RuntimeVerificationError("READY tools must be an object")
        for key, expected_path in self.REQUIRED_TOOLS.items():
            item = tools.get(key)
            if not isinstance(item, dict) or set(item) != {"path", "sha256", "version"}:
                raise RuntimeVerificationError(f"READY tool entry is invalid: {key}")
            if item["path"] != expected_path:
                raise RuntimeVerificationError(f"READY tool path is invalid: {key}")
            if not isinstance(item["version"], str) or not item["version"]:
                raise RuntimeVerificationError(f"READY tool version is invalid: {key}")
            digest = item["sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise RuntimeVerificationError(f"READY tool hash is invalid: {key}")
        return ready

    def _verify_published(self, *, full: bool, preflight: bool = False) -> VerifiedRuntime:
        if not self.runtime_dir.is_dir() or self.runtime_dir.is_symlink():
            raise RuntimeNotReadyError(f"runtime cache is missing: {self.runtime_dir}")
        if not self.rootfs_path.is_dir() or self.rootfs_path.is_symlink():
            raise RuntimeVerificationError(f"published rootfs is missing: {self.rootfs_path}")
        ready = self._read_ready()
        try:
            actual_tree_hash = tree_digest(self.rootfs_path) if full else None
        except BundleError as exc:
            raise RuntimeVerificationError(str(exc)) from exc
        if full and actual_tree_hash != self.config.rootfs_tree_sha256:
            raise RuntimeVerificationError(
                f"published rootfs tree changed: expected {self.config.rootfs_tree_sha256}, got {actual_tree_hash}"
            )
        self._verify_read_only(self.rootfs_path)
        for name, item in ready["tools"].items():
            if name not in self.REQUIRED_TOOLS:
                continue
            tool_path = _safe_relative_tool_path(self.rootfs_path, item["path"])
            actual_tool_hash = sha256_file(tool_path)
            if actual_tool_hash != item["sha256"]:
                raise RuntimeVerificationError(f"published tool bytes changed: {name}")
        if preflight:
            for name in self.REQUIRED_TOOLS:
                version = self._probe_tool_version(self.rootfs_path, name)
                if ready["tools"][name]["version"] == "unverified" or version != ready["tools"][name]["version"]:
                    raise RuntimeVerificationError(f"bwrap tool probe does not match READY: {name}")
        return VerifiedRuntime(identity=self.config.runtime_identity, rootfs_path=self.rootfs_path)

    @staticmethod
    def _verify_read_only(rootfs: Path) -> None:
        try:
            for directory, subdirectories, files in os.walk(rootfs, followlinks=False):
                directory_mode = os.lstat(directory).st_mode
                if stat.S_IMODE(directory_mode) != 0o555:
                    raise RuntimeVerificationError(f"published directory mode is not 0555: {directory}")
                for name in (*subdirectories, *files):
                    path = Path(directory) / name
                    info = os.lstat(path)
                    if info.st_mode & 0o7222 and not path.is_symlink():
                        raise RuntimeVerificationError(f"published rootfs entry has write or special mode bits: {path}")
        except OSError as exc:
            raise RuntimeVerificationError(f"cannot inspect published rootfs permissions: {rootfs}") from exc

    def _collect_tools(self, rootfs: Path, *, preflight: bool) -> dict[str, dict[str, str]]:
        tools: dict[str, dict[str, str]] = {}
        for name, relative_path in self.REQUIRED_TOOLS.items():
            tool_path = _safe_relative_tool_path(rootfs, relative_path)
            try:
                digest = sha256_file(tool_path)
            except BundleError as exc:
                raise RuntimePreparationError(str(exc)) from exc
            try:
                version = self._probe_tool_version(rootfs, name)
            except RuntimeManagerError:
                if preflight:
                    raise
                version = "unverified"
            tools[name] = {"path": relative_path, "sha256": digest, "version": version}
        return tools

    def _probe_tool_version(self, rootfs: Path, name: str) -> str:
        if self._tool_probe is not None:
            version = self._tool_probe(rootfs, name)
            if not isinstance(version, str) or not version.strip():
                raise RuntimePreparationError(f"tool probe returned no version: {name}")
            return version.strip().splitlines()[0]
        executable = self.REQUIRED_TOOLS[name]
        system_bwrap = Path("/usr/bin/bwrap")
        if not system_bwrap.is_file() or not os.access(system_bwrap, os.X_OK):
            raise RuntimePreparationError("/usr/bin/bwrap is unavailable for rootfs tool probe")
        bwrap = str(system_bwrap)
        environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": "/scratch",
            "TMPDIR": "/tmp",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TZ": "UTC",
            "DCI_WORKSPACE": "/workspace",
            "DCI_CORPUS": "/workspace/corpus",
            "DCI_SCRATCH": "/scratch",
        }
        arguments = [
            bwrap,
            "--ro-bind",
            str(rootfs),
            "/",
            "--dev",
            "/dev",
            "--unshare-user",
            "--uid",
            "65534",
            "--gid",
            "65534",
            "--unshare-pid",
            "--unshare-net",
            "--unshare-ipc",
            "--unshare-uts",
            "--new-session",
            "--die-with-parent",
            "--cap-drop",
            "ALL",
            "--chdir",
            "/",
            f"/{executable}",
            "--version",
        ]
        try:
            completed = subprocess.run(
                arguments,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimePreparationError(f"bwrap tool probe failed: {name}") from exc
        if completed.returncode != 0:
            output = completed.stdout.decode("utf-8", errors="replace")[-400:]
            raise RuntimePreparationError(f"bwrap tool probe failed for {name}: {output}")
        output = completed.stdout.decode("utf-8", errors="replace").strip()
        if not output:
            raise RuntimePreparationError(f"bwrap tool probe returned no version: {name}")
        return output.splitlines()[0]


def verify_runtime(config_path: Path, *, full: bool = True, preflight: bool = False) -> VerifiedRuntime:
    """Verify a prepared runtime and return the fixed Pi-facing contract."""

    return RuntimeManager.from_config(config_path).verify(full=full, preflight=preflight)


def prepare_runtime(
    config_path: Path,
    *,
    bundle: Path | None = None,
    offline: bool = True,
    preflight: bool = False,
) -> VerifiedRuntime:
    return RuntimeManager.from_config(config_path).prepare(bundle=bundle, offline=offline, preflight=preflight)


def inspect_runtime(config_path: Path) -> dict[str, Any]:
    return RuntimeManager.from_config(config_path).inspect()
