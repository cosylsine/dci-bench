"""Tree identity and safe extraction for offline RootFS bundles."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import stat
import tarfile
from typing import Iterable


class BundleError(ValueError):
    """Raised for invalid RootFS archives or filesystem trees."""


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise BundleError(f"cannot hash bundle: {path}") from exc
    return digest.hexdigest()


def _relative_path(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    return relative.as_posix() if relative.parts else "."


def _normal_file_mode(mode: int) -> int:
    # Keep ordinary read/execute bits while removing every write and special
    # bit.  A setuid/setgid executable must not become a privilege boundary
    # inside the agent namespace.
    return stat.S_IMODE(mode) & 0o777 & ~0o222


def _resolved_inside(root: Path, path: Path) -> Path:
    root_resolved = root.resolve(strict=True)
    try:
        resolved = path.resolve(strict=False)
        resolved.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise BundleError(f"symlink escapes rootfs: {_relative_path(root, path)}") from exc
    return resolved


def _validate_tree_root(root: Path) -> Path:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise BundleError(f"rootfs must be a real directory: {root}")
    return root.resolve(strict=True)


def _walk_entries(root: Path) -> list[tuple[str, Path, os.stat_result]]:
    root = _validate_tree_root(root)
    entries: list[tuple[str, Path, os.stat_result]] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            children = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise BundleError(f"cannot enumerate rootfs: {directory}") from exc
        for entry in children:
            child = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise BundleError(f"cannot stat rootfs entry: {child}") from exc
            relative = _relative_path(root, child)
            kind = stat.S_IFMT(info.st_mode)
            if kind == stat.S_IFDIR:
                entries.append((relative, child, info))
                stack.append(child)
            elif kind == stat.S_IFREG:
                entries.append((relative, child, info))
            elif kind == stat.S_IFLNK:
                target = os.readlink(child)
                if not target or Path(target).is_absolute():
                    raise BundleError(f"absolute or empty symlink target: {relative}")
                _resolved_inside(root, child.parent / target)
                entries.append((relative, child, info))
            else:
                raise BundleError(f"unsupported special file in rootfs: {relative}")
    entries.sort(key=lambda item: item[0].encode("utf-8"))
    return entries


def _entry_record(root: Path, relative: str, path: Path, info: os.stat_result) -> dict[str, object]:
    kind = stat.S_IFMT(info.st_mode)
    if kind == stat.S_IFDIR:
        return {"path": relative, "type": "dir", "mode": 0o555}
    if kind == stat.S_IFREG:
        return {
            "path": relative,
            "type": "file",
            "mode": _normal_file_mode(info.st_mode),
            "sha256": sha256_file(path),
        }
    if kind == stat.S_IFLNK:
        return {
            "path": relative,
            "type": "symlink",
            "mode": 0,
            "target": os.readlink(path),
        }
    raise BundleError(f"unsupported special file in rootfs: {relative}")


def tree_digest(path: Path | str) -> str:
    """Return the deterministic digest of a normalized RootFS tree.

    The digest includes ``.`` (the root directory), each relative POSIX path,
    entry type, normalized permission bits, and either file content SHA-256 or
    the literal symlink target.  Owner, timestamps, and filesystem inode data
    are intentionally excluded.
    """

    root = _validate_tree_root(Path(path))
    digest = hashlib.sha256()
    root_record = {"path": ".", "type": "dir", "mode": 0o555}
    digest.update((json.dumps(root_record, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    for relative, child, info in _walk_entries(root):
        record = _entry_record(root, relative, child, info)
        line = json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def freeze_tree(path: Path | str) -> str:
    """Normalize a published RootFS to read-only permissions and return its digest."""

    root = _validate_tree_root(Path(path))
    # Validate the complete tree before changing any permissions.  In
    # particular this rejects devices/FIFOs and escaping symlinks atomically
    # from the caller's perspective.
    entries = _walk_entries(root)
    try:
        os.chmod(root, 0o555)
        for _, child, info in entries:
            kind = stat.S_IFMT(info.st_mode)
            if kind == stat.S_IFDIR:
                os.chmod(child, 0o555)
            elif kind == stat.S_IFREG:
                os.chmod(child, _normal_file_mode(info.st_mode))
    except OSError as exc:
        raise BundleError(f"cannot normalize rootfs permissions: {root}") from exc
    return tree_digest(root)


def _safe_member_name(raw_name: str) -> str:
    if not isinstance(raw_name, str) or not raw_name or "\x00" in raw_name:
        raise BundleError("invalid empty/NUL tar member path")
    if "\\" in raw_name:
        raise BundleError(f"backslash tar member path is not allowed: {raw_name!r}")
    if raw_name.startswith("/") or PurePosixPath(raw_name).is_absolute():
        raise BundleError(f"absolute tar member path: {raw_name!r}")
    parts = tuple(part for part in PurePosixPath(raw_name).parts if part not in ("", "."))
    if any(part == ".." for part in parts):
        raise BundleError(f"parent traversal tar member path: {raw_name!r}")
    return "/".join(parts) or "."


def _safe_link_target(member_path: str, raw_target: str) -> str:
    if not isinstance(raw_target, str) or not raw_target or "\x00" in raw_target:
        raise BundleError(f"invalid symlink target for {member_path}")
    if "\\" in raw_target or raw_target.startswith("/") or PurePosixPath(raw_target).is_absolute():
        raise BundleError(f"escaping symlink target for {member_path}: {raw_target!r}")
    parent = posixpath.dirname(member_path)
    combined = posixpath.normpath(posixpath.join(parent, raw_target))
    if combined == ".." or combined.startswith("../"):
        raise BundleError(f"escaping symlink target for {member_path}: {raw_target!r}")
    return raw_target


def _prevalidate_members(members: Iterable[tarfile.TarInfo]) -> list[tuple[tarfile.TarInfo, str, str | None]]:
    normalized: list[tuple[tarfile.TarInfo, str, str | None]] = []
    seen: dict[str, str] = {}
    for member in members:
        name = _safe_member_name(member.name)
        if name in seen:
            raise BundleError(f"duplicate tar member path: {member.name!r}")
        if member.islnk():
            target = _safe_member_name(member.linkname)
            if target == ".":
                raise BundleError(f"hard link target must be a regular file: {member.name!r}")
            kind = "hardlink"
            link_target = target
        elif member.issym():
            _safe_link_target(name, member.linkname)
            kind = "symlink"
            link_target = None
        elif member.isdir():
            kind = "dir"
            link_target = None
        elif member.isreg():
            if member.size < 0:
                raise BundleError(f"negative tar member size: {member.name!r}")
            kind = "file"
            link_target = None
        else:
            raise BundleError(f"device/FIFO/special tar member rejected: {member.name!r}")
        normalized.append((member, name, link_target))
        seen[name] = kind

    for member, name, link_target in normalized:
        if not member.islnk():
            continue
        if not isinstance(link_target, str) or seen.get(link_target) != "file":
            raise BundleError(f"hard link target must be an archive regular file: {name!r}")

    # Never let a symlink or ordinary file become an archive directory prefix.
    # This also closes the path traversal route where a later member would be
    # extracted through an earlier symlink.
    for _, name, _ in normalized:
        prefixes = name.split("/")[:-1]
        for index in range(1, len(prefixes) + 1):
            prefix = "/".join(prefixes[:index])
            if prefix in seen and seen[prefix] != "dir":
                raise BundleError(f"non-directory tar member used as parent: {name!r}")
    normalized.sort(
        key=lambda item: (
            0 if item[0].isdir() else 1 if item[0].isreg() else 2 if item[0].islnk() else 3,
            item[1].count("/"),
            item[1],
        )
    )
    return normalized


def _mkdir_fresh_or_existing(path: Path, mode: int) -> None:
    try:
        path.mkdir(mode=mode)
    except FileExistsError:
        try:
            if path.is_symlink() or not path.is_dir():
                raise BundleError(f"tar directory collides with non-directory: {path}")
        except OSError as exc:
            raise BundleError(f"cannot inspect tar directory: {path}") from exc
    except OSError as exc:
        raise BundleError(f"cannot create tar directory: {path}") from exc
    try:
        os.chmod(path, mode)
    except OSError as exc:
        raise BundleError(f"cannot set tar directory mode: {path}") from exc


def extract_bundle(bundle: Path | str, destination: Path | str) -> None:
    """Safely extract a merged RootFS tar into an empty staging directory."""

    bundle_path = Path(bundle)
    destination_path = Path(destination)
    if destination_path.exists() or destination_path.is_symlink():
        raise BundleError(f"bundle destination must not already exist: {destination_path}")
    try:
        destination_path.mkdir(parents=True, mode=0o700)
    except OSError as exc:
        raise BundleError(f"cannot create bundle destination: {destination_path}") from exc
    try:
        with tarfile.open(bundle_path, mode="r:*") as archive:
            members = _prevalidate_members(archive.getmembers())
            for member, name, hardlink_target in members:
                if name == ".":
                    if not member.isdir():
                        raise BundleError("root tar member must be a directory")
                    continue
                target = destination_path / Path(*name.split("/"))
                parent = target.parent
                relative_parent = parent.relative_to(destination_path)
                current = destination_path
                for part in relative_parent.parts:
                    current = current / part
                    _mkdir_fresh_or_existing(current, 0o755)
                if member.isdir():
                    _mkdir_fresh_or_existing(target, stat.S_IMODE(member.mode) or 0o755)
                elif member.issym():
                    if target.exists() or target.is_symlink():
                        raise BundleError(f"duplicate extracted path: {name}")
                    try:
                        os.symlink(_safe_link_target(name, member.linkname), target)
                    except OSError as exc:
                        raise BundleError(f"cannot create symlink: {name}") from exc
                elif member.islnk():
                    if target.exists() or target.is_symlink():
                        raise BundleError(f"duplicate extracted path: {name}")
                    if not isinstance(hardlink_target, str):
                        raise BundleError(f"missing hardlink target: {name}")
                    source_target = destination_path / Path(*hardlink_target.split("/"))
                    try:
                        source_info = os.lstat(source_target)
                    except OSError as exc:
                        raise BundleError(f"hardlink target was not extracted: {name}") from exc
                    if not stat.S_ISREG(source_info.st_mode):
                        raise BundleError(f"hardlink target is not a regular file: {name}")
                    try:
                        shutil.copyfile(source_target, target, follow_symlinks=False)
                        os.chmod(target, stat.S_IMODE(member.mode) or 0o600)
                    except OSError as exc:
                        raise BundleError(f"cannot copy hardlink member: {name}") from exc
                else:
                    if target.exists() or target.is_symlink():
                        raise BundleError(f"duplicate extracted path: {name}")
                    source = archive.extractfile(member)
                    if source is None:
                        raise BundleError(f"cannot read tar member: {name}")
                    try:
                        file_descriptor = os.open(
                            target,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                            stat.S_IMODE(member.mode) or 0o600,
                        )
                        with os.fdopen(file_descriptor, "wb") as output:
                            shutil.copyfileobj(source, output, length=1024 * 1024)
                    except OSError as exc:
                        raise BundleError(f"cannot extract tar member: {name}") from exc
                    finally:
                        source.close()
                    try:
                        os.chmod(target, stat.S_IMODE(member.mode) or 0o600)
                    except OSError as exc:
                        raise BundleError(f"cannot set tar file mode: {name}") from exc
    except BundleError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise BundleError(f"cannot read RootFS bundle: {bundle_path}") from exc

    # Validate links and special-file policy before handing the staging tree to
    # the manager.  freeze_tree also normalizes all permissions.
    freeze_tree(destination_path)
