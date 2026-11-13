"""Configuration and identity contracts for the frozen agent runtime.

The runtime configuration is deliberately small.  Paths are deployment
details and are kept out of the logical identity returned to Pi.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


class RuntimeConfigError(ValueError):
    """Raised when a runtime configuration is not the supported contract."""


SCHEMA_VERSION = "dci-runtime-v1"
MODE = "frozen-rootfs"
PLATFORM = "linux/amd64"
TOOL_POLICY = "open"
FILESYSTEM_POLICY = "dci-bwrap-frozen-ro-v1"
READY_SCHEMA_VERSION = "dci-rootfs-ready-v1"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE = re.compile(r"^.+@sha256:([0-9a-f]{64})$")

_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "mode",
        "image",
        "platform",
        "bundle_sha256",
        "cache_dir",
        "network_during_eval",
        "provisioning",
        "allow_download_during_eval",
        "tool_policy",
        "filesystem_policy",
        "rootfs_tree_sha256",
    }
)


def _duplicate_free_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeConfigError(f"duplicate configuration field: {key}")
        result[key] = value
    return result


def _require_hex(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise RuntimeConfigError(f"{field} must be 64 lowercase hexadecimal characters")
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _stable_rg_config() -> str:
    path = Path(__file__).resolve().parents[1] / "protocol" / "stable-rg.conf"
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - package installation failure
        raise RuntimeConfigError(f"cannot read stable-rg.conf: {path}") from exc


# This is the policy consumed by the Pi/bwrap implementation.  Keep only
# logical mount targets and fixed values here: the verified rootfs and cache
# deployment paths must never affect the cross-node identity.
POLICY_SPEC: dict[str, Any] = {
    "schema_version": FILESYSTEM_POLICY,
    "mounts": [
        {"kind": "ro-bind", "source": "verified-rootfs", "target": "/"},
        {"kind": "ro-bind", "source": "query-corpus", "target": "/workspace/corpus"},
        {"kind": "bind", "source": "sample-scratch", "target": "/scratch"},
        {"kind": "bind", "source": "sample-scratch/tmp", "target": "/tmp"},
        {"kind": "dev", "target": "/dev"},
    ],
    "namespaces": ["user", "pid", "net", "ipc", "uts"],
    "identity": {"uid": 65534, "gid": 65534, "hostname": "dci-sample"},
    "working_directory": "/workspace",
    "proc_mount": False,
    "stable_rg_mount": {
        "condition": "protocol.search_policy",
        "kind": "ro-bind",
        "target": "/sandbox-bin/rg.conf",
        "environment": {"RIPGREP_CONFIG_PATH": "/sandbox-bin/rg.conf"},
    },
    "session": {"new_session": True, "die_with_parent": True, "cap_drop": "ALL"},
    "environment": {
        "PATH": "/usr/bin:/bin",
        "HOME": "/scratch",
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "DCI_WORKSPACE": "/workspace",
        "DCI_CORPUS": "/workspace/corpus",
        "DCI_SCRATCH": "/scratch",
    },
    "ulimits": {
        "nofile": {"soft": 256, "hard": 256},
        "nproc": {"soft": 128, "hard": 128},
        "fsize": {"soft": 2048, "hard": 2048},
        "as": {"soft": 1048576, "hard": 1048576},
        "cpu": {"soft": 60, "hard": 60},
    },
    "umask": "077",
    "stable_rg_config": None,
}


def policy_spec() -> dict[str, Any]:
    """Return the normalized bwrap policy material used for its hash."""

    policy = json.loads(json.dumps(POLICY_SPEC))
    policy["stable_rg_config"] = _stable_rg_config()
    return policy


def runtime_policy_sha256() -> str:
    return hashlib.sha256(_canonical_json(policy_spec())).hexdigest()


@dataclass(frozen=True)
class RuntimeConfig:
    schema_version: str
    mode: str
    image: str
    image_digest: str
    platform: str
    bundle_sha256: str
    cache_dir: Path
    network_during_eval: str
    provisioning: str
    allow_download_during_eval: bool
    tool_policy: str
    filesystem_policy: str
    rootfs_tree_sha256: str

    @property
    def image_cache_name(self) -> str:
        return f"sha256-{self.image_digest.removeprefix('sha256:')}"

    @property
    def platform_cache_name(self) -> str:
        return self.platform.replace("/", "-")

    @property
    def content_identity(self) -> dict[str, str]:
        return {
            "image_digest": self.image_digest,
            "platform": self.platform,
            "rootfs_tree_sha256": self.rootfs_tree_sha256,
        }

    @property
    def runtime_identity(self) -> dict[str, str]:
        return {
            "schema_version": "dci-agent-runtime-v1",
            "mode": self.mode,
            "image_digest": self.image_digest,
            "platform": self.platform,
            "rootfs_tree_sha256": self.rootfs_tree_sha256,
            "tool_policy": self.tool_policy,
            "filesystem_policy": self.filesystem_policy,
            "runtime_policy_sha256": runtime_policy_sha256(),
        }

    def cache_platform_dir(self) -> Path:
        return self.cache_dir / "images" / self.image_cache_name / self.platform_cache_name


def load_config(path: Path | str) -> RuntimeConfig:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle, object_pairs_hook=_duplicate_free_pairs)
    except RuntimeConfigError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeConfigError(f"cannot read runtime config: {config_path}") from exc
    if not isinstance(raw, dict):
        raise RuntimeConfigError("runtime config must be a JSON object")
    unknown = sorted(set(raw) - _CONFIG_FIELDS)
    missing = sorted(_CONFIG_FIELDS - set(raw))
    if unknown:
        raise RuntimeConfigError(f"unknown runtime config field(s): {', '.join(unknown)}")
    if missing:
        raise RuntimeConfigError(f"missing runtime config field(s): {', '.join(missing)}")

    schema_version = raw["schema_version"]
    mode = raw["mode"]
    platform = raw["platform"]
    if schema_version != SCHEMA_VERSION:
        raise RuntimeConfigError(f"unsupported schema_version: {schema_version!r}")
    if mode != MODE:
        raise RuntimeConfigError(f"unsupported runtime mode: {mode!r}")
    if platform != PLATFORM:
        raise RuntimeConfigError(f"unsupported platform: {platform!r}")

    image = raw["image"]
    if not isinstance(image, str) or any(character.isspace() for character in image):
        raise RuntimeConfigError("image must be a full image@sha256 digest")
    image_match = _IMAGE.fullmatch(image)
    if image_match is None:
        raise RuntimeConfigError("image must end with @sha256:<64 lowercase hex characters>")
    bundle_sha256 = _require_hex(raw["bundle_sha256"], "bundle_sha256")
    rootfs_tree_sha256 = _require_hex(raw["rootfs_tree_sha256"], "rootfs_tree_sha256")
    cache_value = raw["cache_dir"]
    if not isinstance(cache_value, str) or not cache_value or "\x00" in cache_value:
        raise RuntimeConfigError("cache_dir must be a non-empty path")
    cache_path = Path(cache_value)
    if not cache_path.is_absolute():
        # Build artifacts intentionally use ``./cache`` so they can be moved
        # as a self-contained release.  Resolve that deployment detail against
        # the config location; it is never part of RuntimeIdentity.
        cache_path = config_path.parent / cache_path
    cache_path = cache_path.resolve()
    if raw["network_during_eval"] != "disabled":
        raise RuntimeConfigError("network_during_eval must be 'disabled'")
    if raw["provisioning"] != "prepare-before-eval":
        raise RuntimeConfigError("provisioning must be 'prepare-before-eval'")
    if raw["allow_download_during_eval"] is not False:
        raise RuntimeConfigError("allow_download_during_eval must be false")
    if raw["tool_policy"] != TOOL_POLICY:
        raise RuntimeConfigError(f"tool_policy must be {TOOL_POLICY!r}")
    if raw["filesystem_policy"] != FILESYSTEM_POLICY:
        raise RuntimeConfigError(f"filesystem_policy must be {FILESYSTEM_POLICY!r}")

    return RuntimeConfig(
        schema_version=schema_version,
        mode=mode,
        image=image,
        image_digest=f"sha256:{image_match.group(1)}",
        platform=platform,
        bundle_sha256=bundle_sha256,
        cache_dir=cache_path,
        network_during_eval=raw["network_during_eval"],
        provisioning=raw["provisioning"],
        allow_download_during_eval=raw["allow_download_during_eval"],
        tool_policy=raw["tool_policy"],
        filesystem_policy=raw["filesystem_policy"],
        rootfs_tree_sha256=rootfs_tree_sha256,
    )
