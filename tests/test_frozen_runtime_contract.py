"""Configuration, identity, digest, and CLI contract tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from dci_bench.runtime import (
    FILESYSTEM_POLICY,
    RuntimeConfigError,
    RuntimeManager,
    VerifiedRuntime,
    freeze_tree,
    load_config,
    runtime_policy_sha256,
    tree_digest,
)


class FrozenRuntimeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="dci-frozen-contract-")
        self.root = Path(self.tempdir.name)
        self.source = self.root / "rootfs"
        (self.source / "usr/bin").mkdir(parents=True)
        (self.source / "usr/bin/bash").write_text("bash fixture", encoding="utf-8")
        (self.source / "usr/bin/rg").write_text("rg fixture", encoding="utf-8")
        self.tree_hash = freeze_tree(self.source)
        self.bundle = self.root / "rootfs.tar"
        with tarfile.open(self.bundle, "w") as archive:
            archive.add(self.source, arcname=".")
        self.config = {
            "schema_version": "dci-runtime-v1",
            "mode": "frozen-rootfs",
            "image": "example.invalid/runtime@sha256:" + "c" * 64,
            "platform": "linux/amd64",
            "bundle_sha256": hashlib.sha256(self.bundle.read_bytes()).hexdigest(),
            "cache_dir": str(self.root / "cache"),
            "network_during_eval": "disabled",
            "provisioning": "prepare-before-eval",
            "allow_download_during_eval": False,
            "tool_policy": "open",
            "filesystem_policy": FILESYSTEM_POLICY,
            "rootfs_tree_sha256": self.tree_hash,
        }
        self.config_path = self.root / "runtime.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_config_is_strict_and_only_supports_frozen_linux_amd64(self) -> None:
        config = load_config(self.config_path)
        self.assertEqual(config.image_digest, "sha256:" + "c" * 64)
        self.assertEqual(config.content_identity["platform"], "linux/amd64")
        for field, value in (
            ("unknown", True),
            ("mode", "host-tools"),
            ("platform", "linux/arm64"),
            ("filesystem_policy", "other"),
        ):
            changed = dict(self.config)
            changed[field] = value
            self.config_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(RuntimeConfigError):
                load_config(self.config_path)

    def test_identity_fields_and_payload_are_exact_and_path_free(self) -> None:
        manager = RuntimeManager.from_config(self.config_path, tool_probe=lambda _root, name: f"{name} fixture")
        result = manager.prepare(bundle=self.bundle)
        self.assertEqual(
            list(result.identity),
            [
                "schema_version",
                "mode",
                "image_digest",
                "platform",
                "rootfs_tree_sha256",
                "tool_policy",
                "filesystem_policy",
                "runtime_policy_sha256",
            ],
        )
        self.assertEqual(result.identity["schema_version"], "dci-agent-runtime-v1")
        self.assertEqual(result.identity["mode"], "frozen-rootfs")
        self.assertEqual(result.identity["filesystem_policy"], FILESYSTEM_POLICY)
        self.assertEqual(len(result.identity["runtime_policy_sha256"]), 64)
        self.assertNotIn(str(self.root), json.dumps(result.identity))
        self.assertEqual(result.to_payload(), {"identity": result.identity, "rootfs_path": str(result.rootfs_path)})

    def test_policy_hash_is_stable_and_tree_digest_ignores_mtime(self) -> None:
        first_hash = runtime_policy_sha256()
        first_tree = tree_digest(self.source)
        (self.source / "usr/bin/bash").touch()
        self.assertEqual(first_hash, runtime_policy_sha256())
        self.assertEqual(first_tree, tree_digest(self.source))
        (self.source / "usr/bin/bash").write_text("changed", encoding="utf-8")
        self.assertNotEqual(first_tree, tree_digest(self.source))

    def test_cli_prepare_verify_inspect(self) -> None:
        # The CLI has no fixture injection point, so create executable shell
        # fixture files and accept the normal prepare-time unverified marker.
        environment = {"PYTHONPATH": str(Path(__file__).resolve().parents[1])}
        prepare = subprocess.run(
            [sys.executable, "-m", "dci_bench.runtime", "prepare", "--config", str(self.config_path), "--bundle", str(self.bundle), "--offline"],
            cwd=self.config_path.parents[0],
            env={**environment},
            capture_output=True,
            text=True,
        )
        self.assertEqual(prepare.returncode, 0, prepare.stderr)
        self.assertIn('"rootfs_path"', prepare.stdout)
        verify = subprocess.run(
            [sys.executable, "-m", "dci_bench.runtime", "verify", "--config", str(self.config_path)],
            cwd=self.config_path.parents[0],
            env={**environment},
            capture_output=True,
            text=True,
        )
        self.assertEqual(verify.returncode, 0, verify.stderr)
        inspect = subprocess.run(
            [sys.executable, "-m", "dci_bench.runtime", "inspect", "--config", str(self.config_path)],
            cwd=self.config_path.parents[0],
            env={**environment},
            capture_output=True,
            text=True,
        )
        self.assertEqual(inspect.returncode, 0, inspect.stderr)
        self.assertIn('"dci-rootfs-ready-v1"', inspect.stdout)


if __name__ == "__main__":
    unittest.main()
