"""Contract and budget checks for the additive document-pool v2 protocol."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

from dci_bench.protocol.document_pools import get_tier, load_document_pools_v2


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python")
if not PYTHON.exists():
    PYTHON = Path(os.environ.get("PYTHON", "python3"))
V2_PROFILES = ("document-pool-v2-lite", "document-pool-v2-full")
V2_FIELDS = (
    "max_total_model_tokens", "reserve_tokens", "context_window", "max_model_output_tokens",
    "max_final_output_tokens", "sample_timeout_seconds", "max_search_turns",
    "max_tool_invocations", "max_final_repair_turns", "max_agent_steps", "tool_result_max_bytes",
)


def python_contract(profile: str, token_budget: str | None = None) -> dict:
    code = (
        "import json; "
        "from dci_bench.protocol.contracts import CONTRACT_SHA256, protocol_contract; "
        "print(json.dumps({'sha': CONTRACT_SHA256, 'contract': protocol_contract()}, "
        "ensure_ascii=False, sort_keys=True))"
    )
    environment = {**os.environ, "PYTHONPATH": str(ROOT), "DCI_PROTOCOL_PROFILE": profile}
    environment.pop("DCI_DOCUMENT_TOKEN_BUDGET", None)
    if token_budget is not None:
        environment["DCI_DOCUMENT_TOKEN_BUDGET"] = token_budget
    return json.loads(subprocess.check_output([str(PYTHON), "-c", code], cwd=ROOT, env=environment, text=True))


class DocumentPoolV2BudgetTests(unittest.TestCase):
    def test_lite_and_full_have_identical_controls_and_aliases(self):
        lite = get_tier("lite")
        full = get_tier("full")
        self.assertEqual(lite, get_tier("document-pool-v2-lite"))
        self.assertEqual(full, get_tier("document-pool-v2-full"))
        self.assertEqual(lite["capacity"], 200)
        self.assertEqual(full["capacity"], 1000)
        for field in V2_FIELDS:
            self.assertEqual(lite[field], full[field], field)
        self.assertEqual(lite["max_total_model_tokens"], 64000)
        self.assertEqual(lite["reserve_tokens"], 0)
        self.assertEqual(lite["context_window"], 65536)
        self.assertEqual(lite["max_model_output_tokens"], 1024)
        self.assertEqual(lite["max_final_output_tokens"], 1024)
        self.assertEqual(lite["sample_timeout_seconds"], 300)
        self.assertEqual(lite["max_search_turns"], 12)
        self.assertEqual(lite["max_tool_invocations"], 24)
        self.assertEqual(lite["max_final_repair_turns"], 1)
        self.assertEqual(lite["max_agent_steps"], 14)
        self.assertEqual(lite["tool_result_max_bytes"], 8192)

    def test_v2_config_is_sealed_and_v1_loader_remains_separate(self):
        config = load_document_pools_v2()
        self.assertEqual(config["schema_version"], "document-pool-v2")
        self.assertEqual(set(config["tiers"]), {"lite", "full"})
        self.assertEqual(config["tiers"]["lite"]["capacity"], 200)
        self.assertEqual(config["tiers"]["full"]["capacity"], 1000)

    def test_calibration_budgets_are_validated_and_fingerprinted(self):
        expected = {"32000": 32000, "64000": 64000, "96000": 96000}
        hashes = set()
        for raw, value in expected.items():
            resolved = python_contract("document-pool-v2-lite", raw)
            hashes.add(resolved["sha"])
            contract = resolved["contract"]
            self.assertEqual(contract["max_total_model_tokens"], value)
            self.assertEqual(contract["context_policy"]["max_total_model_tokens"], value)
            self.assertEqual(contract["candidate_pool"]["max_total_model_tokens"], value)
            self.assertIn(f"at most {value} cumulative input plus output", contract["budget_control"]["initial_prompt"])
        self.assertEqual(len(hashes), 3)
        for invalid in ("0", "1", "31999", "64001", "64_000", "64000 "):
            environment = {**os.environ, "PYTHONPATH": str(ROOT),
                           "DCI_PROTOCOL_PROFILE": "document-pool-v2-lite",
                           "DCI_DOCUMENT_TOKEN_BUDGET": invalid}
            process = subprocess.run([str(PYTHON), "-c", "import dci_bench.protocol.contracts"],
                                     cwd=ROOT, env=environment, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0, invalid)
        environment = {**os.environ, "PYTHONPATH": str(ROOT),
                       "DCI_PROTOCOL_PROFILE": "budgeted", "DCI_DOCUMENT_TOKEN_BUDGET": "64000"}
        process = subprocess.run([str(PYTHON), "-c", "import dci_bench.protocol.contracts"],
                                 cwd=ROOT, env=environment, capture_output=True, text=True)
        self.assertNotEqual(process.returncode, 0)

    @unittest.skipUnless(shutil.which("node") and (ROOT / "pi-dci/node_modules/tsx").exists(), "Pi dependencies are not installed")
    def test_v2_python_and_pi_contracts_match(self):
        node_code = (
            "const m = await import('./scripts/dci-run.ts'); "
            "console.log(JSON.stringify({sha:m.CONTRACT_SHA256, contract:m.DEFAULT_CONTRACT}));"
        )
        for profile in (*V2_PROFILES, "lite", "full"):
            for token_budget in (None, "32000", "64000", "96000"):
                py = python_contract(profile, token_budget)
                environment = {**os.environ, "DCI_PROTOCOL_PROFILE": profile}
                environment.pop("DCI_DOCUMENT_TOKEN_BUDGET", None)
                if token_budget is not None:
                    environment["DCI_DOCUMENT_TOKEN_BUDGET"] = token_budget
                pi = json.loads(subprocess.check_output(
                    ["node", "--import", "tsx", "--input-type=module", "-e", node_code],
                    cwd=ROOT / "pi-dci", env=environment, text=True,
                ))
                self.assertEqual(pi, py, (profile, token_budget))


if __name__ == "__main__":
    unittest.main()
