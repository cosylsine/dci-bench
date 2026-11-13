"""Offline cache, locking, and safe-import tests for the frozen runtime."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import stat
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from dci_bench.runtime import (
    RuntimeManager,
    RuntimeNotReadyError,
    RuntimePreparationError,
    RuntimeVerificationError,
    freeze_tree,
    tree_digest,
)


class FrozenRuntimeFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="dci-frozen-runtime-")
        self.root = Path(self.tempdir.name)
        self.source = self.root / "source"
        (self.source / "usr/bin").mkdir(parents=True)
        (self.source / "etc").mkdir()
        (self.source / "usr/bin/bash").write_text("#!/bin/sh\necho bash fixture\n", encoding="utf-8")
        (self.source / "usr/bin/rg").write_text("#!/bin/sh\necho rg fixture\n", encoding="utf-8")
        (self.source / "etc/os-release").write_text("PRETTY_NAME=fixture\n", encoding="utf-8")
        (self.source / "usr/bin/bash").chmod(0o755)
        (self.source / "usr/bin/rg").chmod(0o755)
        self.tree_hash = freeze_tree(self.source)
        self.bundle = self.root / "rootfs.tar"
        with tarfile.open(self.bundle, "w") as archive:
            archive.add(self.source, arcname=".")
        self.config_path = self.root / "runtime.json"
        self._write_config(self.root / "cache-a")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_config(self, cache_dir: Path, **updates: object) -> None:
        config = {
            "schema_version": "dci-runtime-v1",
            "mode": "frozen-rootfs",
            "image": "registry.invalid/dci-runtime@sha256:" + "a" * 64,
            "platform": "linux/amd64",
            "bundle_sha256": hashlib.sha256(self.bundle.read_bytes()).hexdigest(),
            "cache_dir": str(cache_dir),
            "network_during_eval": "disabled",
            "provisioning": "prepare-before-eval",
            "allow_download_during_eval": False,
            "tool_policy": "open",
            "filesystem_policy": "dci-bwrap-frozen-ro-v1",
            "rootfs_tree_sha256": self.tree_hash,
        }
        config.update(updates)
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

    @staticmethod
    def _probe(_rootfs: Path, name: str) -> str:
        return {"bash": "GNU bash fixture", "rg": "ripgrep fixture"}[name]

    def _manager(self, cache_dir: Path | None = None) -> RuntimeManager:
        if cache_dir is not None:
            self._write_config(cache_dir)
        return RuntimeManager.from_config(self.config_path, tool_probe=self._probe)

    def test_first_import_then_reuses_content_addressed_cache(self) -> None:
        manager = self._manager()
        first = manager.prepare(bundle=self.bundle, offline=True)
        self.assertTrue((first.rootfs_path / "usr/bin/bash").exists())
        ready = manager.ready_path
        self.assertTrue(ready.is_file())
        ready_before = ready.read_bytes()

        second = manager.prepare(bundle=self.root / "does-not-exist.tar", offline=True)
        self.assertEqual(first.identity, second.identity)
        self.assertEqual(first.rootfs_path, second.rootfs_path)
        self.assertEqual(ready_before, ready.read_bytes())
        self.assertEqual(
            manager.runtime_dir,
            self.root / ("cache-a/images/sha256-" + "a" * 64) / "linux-amd64",
        )

    def test_concurrent_prepare_publishes_one_ready_marker(self) -> None:
        managers = [self._manager(), self._manager()]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(item.prepare, bundle=self.bundle, offline=True) for item in managers]
            results = [future.result() for future in futures]
        self.assertEqual(results[0].identity, results[1].identity)
        self.assertEqual(len(list((self.root / "cache-a/staging").iterdir())), 0)
        self.assertEqual(json.loads(managers[0].ready_path.read_text())["schema_version"], "dci-rootfs-ready-v1")

    def test_evaluation_rechecks_tools_even_with_a_valid_warm_cache(self) -> None:
        from dci_bench.results.phase4_runner import verify_runtime_config

        prepared = self._manager().prepare(bundle=self.bundle)
        with patch.object(RuntimeManager, "_probe_tool_version", side_effect=self._probe) as probe:
            verified = verify_runtime_config(self.config_path)
        self.assertEqual(verified, prepared)
        self.assertEqual([call.args for call in probe.call_args_list],
                         [(prepared.rootfs_path, "bash"), (prepared.rootfs_path, "rg")])

        def unavailable_rg(rootfs: Path, name: str) -> str:
            if name == "rg":
                raise RuntimePreparationError("bwrap tool probe failed for rg")
            return self._probe(rootfs, name)

        with patch.object(RuntimeManager, "_probe_tool_version", side_effect=unavailable_rg):
            with self.assertRaisesRegex(RuntimePreparationError, "failed for rg"):
                verify_runtime_config(self.config_path)

    def test_evaluation_rejects_missing_host_bwrap_with_a_valid_warm_cache(self) -> None:
        from dci_bench.results.phase4_runner import verify_runtime_config

        self._manager().prepare(bundle=self.bundle)
        with patch("dci_bench.runtime.manager.os.access", return_value=False):
            with self.assertRaisesRegex(RuntimePreparationError, "/usr/bin/bwrap is unavailable"):
                verify_runtime_config(self.config_path)

    def test_missing_runtime_and_corrupt_bundle_fail_closed(self) -> None:
        manager = self._manager()
        with self.assertRaises(RuntimeNotReadyError):
            manager.verify()
        corrupt = self.root / "corrupt.tar"
        corrupt.write_bytes(b"not a tar")
        with self.assertRaises(RuntimePreparationError):
            manager.prepare(bundle=corrupt, offline=True)
        self.assertFalse(manager.runtime_dir.exists())

    def test_path_attack_and_special_tar_members_are_rejected(self) -> None:
        attacks = {
            "absolute": ("/escape", False),
            "parent": ("../escape", False),
            "symlink": ("inside/link", True),
        }
        for label, (name, symlink) in attacks.items():
            bundle = self.root / f"{label}.tar"
            with tarfile.open(bundle, "w") as archive:
                info = tarfile.TarInfo(name)
                if symlink:
                    info.type = tarfile.SYMTYPE
                    info.linkname = "../../escape"
                else:
                    info.size = 1
                archive.addfile(info, None if symlink else BytesIO(b"x"))
            self._write_config(self.root / f"cache-{label}",
                               bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest())
            manager = self._manager()
            with self.assertRaises(RuntimePreparationError):
                manager.prepare(bundle=bundle, offline=True)

        fifo = self.root / "fifo.tar"
        with tarfile.open(fifo, "w") as archive:
            info = tarfile.TarInfo("pipe")
            info.type = tarfile.FIFOTYPE
            archive.addfile(info)
        self._write_config(self.root / "cache-fifo",
                           bundle_sha256=hashlib.sha256(fifo.read_bytes()).hexdigest())
        manager = self._manager()
        with self.assertRaises(RuntimePreparationError):
            manager.prepare(bundle=fifo, offline=True)

    def test_hardlink_is_copied_to_a_regular_file_inside_rootfs(self) -> None:
        bundle = self.root / "hardlink.tar"
        with tarfile.open(bundle, "w") as archive:
            directory = tarfile.TarInfo("usr")
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)
            directory = tarfile.TarInfo("usr/bin")
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)
            second = tarfile.TarInfo("usr/bin/rg")
            second.type = tarfile.LNKTYPE
            second.linkname = "usr/bin/bash"
            archive.addfile(second)
            first = tarfile.TarInfo("usr/bin/bash")
            first.size = 3
            archive.addfile(first, BytesIO(b"abc"))
        source = self.root / "hardlink-source"
        (source / "usr/bin").mkdir(parents=True)
        (source / "usr/bin/bash").write_bytes(b"abc")
        (source / "usr/bin/rg").write_bytes(b"abc")
        (source / "usr/bin/bash").chmod(0o444)
        (source / "usr/bin/rg").chmod(0o444)
        hardlink_hash = freeze_tree(source)
        self._write_config(self.root / "cache-hardlink", rootfs_tree_sha256=hardlink_hash,
                           bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest())
        result = self._manager().prepare(bundle=bundle)
        self.assertTrue(result.rootfs_path.joinpath("usr/bin/rg").is_file())
        self.assertFalse(result.rootfs_path.joinpath("usr/bin/rg").is_symlink())
        self.assertEqual((result.rootfs_path / "usr/bin/bash").read_bytes(),
                         (result.rootfs_path / "usr/bin/rg").read_bytes())

    def test_cache_path_does_not_change_logical_identity(self) -> None:
        first = self._manager(self.root / "cache-one").prepare(bundle=self.bundle)
        second = self._manager(self.root / "cache-two").prepare(bundle=self.bundle)
        self.assertEqual(first.identity, second.identity)
        self.assertNotEqual(first.rootfs_path, second.rootfs_path)
        self.assertNotIn(str(self.root), json.dumps(first.identity))

    def test_tree_change_and_ready_change_are_rejected(self) -> None:
        manager = self._manager()
        result = manager.prepare(bundle=self.bundle)
        (result.rootfs_path / "etc/os-release").write_text("changed\n", encoding="utf-8")
        with self.assertRaises(RuntimeVerificationError):
            manager.verify()

        # Restore the file and corrupt the authoritative marker.  No implicit
        # repair is allowed after a published directory exists.
        (result.rootfs_path / "etc/os-release").write_text("PRETTY_NAME=fixture\n", encoding="utf-8")
        ready = json.loads(manager.ready_path.read_text())
        ready["content_identity"]["rootfs_tree_sha256"] = "b" * 64
        manager.ready_path.chmod(0o644)
        manager.ready_path.write_text(json.dumps(ready), encoding="utf-8")
        manager.ready_path.chmod(0o444)
        with self.assertRaises(RuntimeVerificationError):
            manager.prepare(bundle=self.bundle)

    def test_published_rootfs_permissions_are_normalized(self) -> None:
        result = self._manager().prepare(bundle=self.bundle)
        for path in result.rootfs_path.rglob("*"):
            mode = stat.S_IMODE(path.lstat().st_mode)
            if path.is_symlink():
                continue
            self.assertEqual(mode & 0o222, 0, path)
        self.assertEqual(stat.S_IMODE(result.rootfs_path.stat().st_mode), 0o555)

    def test_permission_changes_fail_even_when_normalized_tree_hash_is_unchanged(self) -> None:
        manager = self._manager()
        result = manager.prepare(bundle=self.bundle)
        for relative, mode in (("usr/bin/bash", 0o4555), ("etc", 0o444), ("usr/bin/rg", 0o755)):
            with self.subTest(relative=relative, mode=mode):
                path = result.rootfs_path / relative
                previous = stat.S_IMODE(path.stat().st_mode)
                path.chmod(mode)
                with self.assertRaises(RuntimeVerificationError):
                    manager.verify()
                path.chmod(previous)


if __name__ == "__main__":
    unittest.main()
