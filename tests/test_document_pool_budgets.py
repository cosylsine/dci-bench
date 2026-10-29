"""Contract parity and limits for dataset-neutral document-pool profiles."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from dci_bench.protocol.document_pools import get_tier, load_document_pools, pool_policy


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
PROFILES = {
    "document-pool-v1-small": ("small", 200, 96000, 24000),
    "document-pool-v1-medium": ("medium", 500, 128000, 32000),
    "document-pool-v1-large": ("large", 1000, 192000, 48000),
}
OLD_PROFILE_HASHES = {
    "v3": "1f3c693242361afa3934cb1ea37d53034e83db418d68545c8a729938fd3a6747",
    "budgeted": "e3cbd15c8014c3cd66b350237a6e9c1cdae369c5e1b6d78cf5a022d1b1a8a22d",
    "budgeted-stable": "e155daec787f39c79ce6f3a6b18383ca792f452b863ec5ef819ea346ac0d12e9",
    "budgeted-selective": "c6f4a9e46de4f0ebf9b21aca6f73c86a7b1dfd726783362bc3c476807be6419f",
}


def _python_contract(profile: str) -> dict:
    code = (
        "import json; "
        "from dci_bench.protocol.contracts import CONTRACT_SHA256, protocol_contract; "
        "print(json.dumps({'sha': CONTRACT_SHA256, 'contract': protocol_contract()}, "
        "ensure_ascii=False, sort_keys=True))"
    )
    environment = {**os.environ, "PYTHONPATH": str(ROOT), "DCI_PROTOCOL_PROFILE": profile}
    return json.loads(subprocess.check_output([PYTHON, "-c", code], cwd=ROOT, env=environment, text=True))


class DocumentPoolBudgetTest(unittest.TestCase):
    def test_tiers_and_invalid_names(self):
        for name, (tier, capacity, total, reserve) in PROFILES.items():
            policy = get_tier(name)
            self.assertEqual(policy["tier"], tier)
            self.assertEqual(policy["capacity"], capacity)
            self.assertEqual(policy["max_total_model_tokens"], total)
            self.assertEqual(policy["reserve_tokens"], reserve)
            self.assertEqual(policy["context_window"], 65536)
            self.assertEqual(policy["max_model_output_tokens"], 1024)
            self.assertEqual(policy["max_final_output_tokens"], 1024)
            self.assertEqual(policy["sample_timeout_seconds"], 600)
            self.assertEqual(policy["id_policy"]["version"], "short-document-id-v1")
            self.assertEqual(policy["id_policy"]["declared_aliases"]["d000001.md"], "d000001")
            self.assertTrue(policy["retriever"]["planning_only"])
            self.assertFalse(policy["retriever"]["load_model"])
            self.assertEqual(pool_policy(tier), policy)
        for invalid in ("document-pool-v1-unknown", "unknown", ""):
            with self.assertRaises(ValueError):
                get_tier(invalid)

    def test_negative_tier_values_are_rejected(self):
        config = load_document_pools()
        config["tiers"]["small"]["capacity"] = -1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "document_pools.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_document_pools(path)

    def test_new_python_profiles_have_expected_contract_and_prompt_limits(self):
        for profile, (tier, capacity, total, reserve) in PROFILES.items():
            resolved = _python_contract(profile)
            contract = resolved["contract"]
            self.assertEqual(resolved["sha"], contract_hash(contract))
            self.assertEqual(contract["search_policy"]["version"], "stable-rg-v1")
            self.assertEqual(contract["candidate_pool"]["tier"], tier)
            self.assertEqual(contract["candidate_pool"]["capacity"], capacity)
            self.assertEqual(contract["max_total_model_tokens"], total)
            self.assertEqual(contract["context_policy"]["max_total_model_tokens"], total)
            self.assertEqual(contract["budget_control"]["reserve_tokens"], reserve)
            self.assertIn(f"at most {total} cumulative input plus output", contract["budget_control"]["initial_prompt"])
            self.assertNotIn("forecast_bytes_per_token", contract["budget_control"])
            self.assertNotIn("forecast_request_safety_tokens", contract["budget_control"])
            self.assertIn("d000001.md -> d000001", json.dumps(contract, ensure_ascii=False))
            for legacy in ("l%27exemple", "report.md.md", "original document ID", "original corpus ID"):
                self.assertNotIn(legacy, json.dumps(contract, ensure_ascii=False))

    @unittest.skipUnless(shutil.which("node"), "Node is required for Python/Pi contract parity")
    def test_python_and_pi_contracts_match(self):
        pi_root = ROOT / "pi-dci"
        if not (pi_root / "node_modules/tsx").exists():
            self.skipTest("Pi dependencies are not installed")
        node_code = (
            "const m = await import('./scripts/dci-run.ts'); "
            "console.log(JSON.stringify({sha:m.CONTRACT_SHA256, contract:m.DEFAULT_CONTRACT}));"
        )
        for profile in PROFILES:
            py = _python_contract(profile)
            environment = {**os.environ, "DCI_PROTOCOL_PROFILE": profile}
            pi = json.loads(subprocess.check_output(
                ["node", "--import", "tsx", "--input-type=module", "-e", node_code],
                cwd=pi_root,
                env=environment,
                text=True,
            ))
            self.assertEqual(pi, py)

    def test_old_profile_hashes_are_unchanged(self):
        for profile, expected in OLD_PROFILE_HASHES.items():
            self.assertEqual(_python_contract(profile)["sha"], expected)


def contract_hash(contract: dict) -> str:
    """Match Python's canonical contract digest without importing profile state."""

    import hashlib

    payload = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()
