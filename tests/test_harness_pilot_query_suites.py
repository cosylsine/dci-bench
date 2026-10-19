"""CPU-only checks for frozen pilot query selection and scheduling."""

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]


class HarnessPilotQuerySuiteTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dci-pilot-query-suite-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "scripts").mkdir()
        (self.root / "data/metadata").mkdir(parents=True)
        diagnostic = json.loads((REPO / "scripts/harness_pilot_queries.json").read_text())
        (self.root / "scripts/harness_pilot_queries.json").write_text(json.dumps(diagnostic))
        for task, excluded in diagnostic.items():
            task_root = self.root / "data/metadata" / task
            task_root.mkdir()
            ids = excluded + [f"extra-{index}" for index in range(20)]
            records = [{"_id": query_id, "text": "fixture"} for query_id in ids]
            (task_root / "queries.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
        self.backend = self.root / "backend.json"
        self.backend.write_text(json.dumps({
            "model": {"model_key": "fixture"},
            "backend": {"max_num_seqs": 8},
            "manifest_sha256": "fixture",
        }))
        spec = importlib.util.spec_from_file_location(
            "pilot_query_suite_test", REPO / "scripts/run_harness_pilot.py"
        )
        self.pilot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.pilot)
        self.pilot.REPO = self.root
        self.freeze_path = self.root / "scripts/harness_pilot_holdout_queries.json"
        self.write_freeze()
        (self.root / "scripts/harness_selection_round3_plan.json").write_text(json.dumps({
            "profiles": {name: {"base_contract_sha256": "contract"}
                         for name in ("budgeted-stable", "budgeted-selective")},
        }))
        self.validation_path = self.root / "scripts/harness_pilot_validation_queries.json"
        holdout = json.loads(self.freeze_path.read_text())
        validation = self.pilot._compute_validation_freeze(self.root / "data/metadata", diagnostic, holdout)
        self.validation_path.write_text(json.dumps(validation))

    def write_freeze(self):
        diagnostic = json.loads((self.root / "scripts/harness_pilot_queries.json").read_text())
        freeze = self.pilot._compute_holdout_freeze(self.root / "data/metadata", diagnostic)
        self.freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def invoke(self, *args):
        output = io.StringIO()
        payload = json.dumps({
            "sha256": "contract",
            "contract": {"generation_config": {"temperature": 0}},
        })
        selector = ModuleType("dci_bench.tasks.mteb_llm_retrieval")
        def select_query_ids(*selector_args, **selector_kwargs):
            if "holdout" in args or "validation" in args:
                raise AssertionError("metadata-only suites must not call the qrels-backed selector")
            return selector_kwargs["query_ids"]
        selector._select_query_ids = select_query_ids
        with patch.object(self.pilot.subprocess, "check_output", return_value=payload), \
             patch.dict(sys.modules, {"dci_bench.tasks.mteb_llm_retrieval": selector}), \
             redirect_stdout(output):
            self.pilot.main(["--backend-manifest", str(self.backend), *args, "--dry-run"])
        return json.loads(output.getvalue())

    def test_holdout_is_six_times_ten_disjoint_and_hash_deterministic(self):
        plan = self.invoke("--query-suite", "holdout", "--profiles", "v3", "budgeted")
        diagnostic = json.loads((self.root / "scripts/harness_pilot_queries.json").read_text())
        self.assertEqual(set(plan["query_ids"]), set(diagnostic))
        self.assertTrue(all(len(ids) == 10 for ids in plan["query_ids"].values()))
        for task, ids in plan["query_ids"].items():
            self.assertTrue(set(ids).isdisjoint(diagnostic[task]))
            expected = sorted(
                (f"extra-{index}" for index in range(20)),
                key=lambda query_id: hashlib.sha256(
                    f"dci-budget-round2-v1:{task}:{query_id}".encode()
                ).hexdigest(),
            )[:10]
            self.assertEqual(ids, expected)

    def test_holdout_does_not_require_qrels_or_scores(self):
        # There are intentionally no qrels/scores files in this fixture.  The
        # launcher must use only queries.jsonl for holdout planning.
        self.assertFalse(list((self.root / "data/metadata").glob("*/qrels.json")))
        plan = self.invoke("--query-suite", "holdout", "--profiles", "v3", "budgeted")
        self.assertEqual(plan["query_suite"], "holdout")

    def test_alternating_order_groups_tasks_and_keeps_unique_ports(self):
        plan = self.invoke(
            "--query-suite", "holdout", "--profiles", "v3", "budgeted",
            "--profile-order", "alternating", "--max-concurrency", "2",
        )
        tasks = list(plan["query_ids"])
        self.assertEqual(plan["execution_groups"][0]["profiles"], ["budgeted", "v3"])
        self.assertEqual(plan["execution_groups"][1]["profiles"], ["v3", "budgeted"])
        self.assertEqual(
            [(item["task"], item["profile"]) for item in plan["execution_order"][:4]],
            [(tasks[0], "budgeted"), (tasks[0], "v3"),
             (tasks[1], "v3"), (tasks[1], "budgeted")],
        )
        ports = [port for profile in plan["bridge_ports"].values() for port in profile.values()]
        self.assertEqual(len(ports), len(set(ports)))
        self.assertEqual(plan["query_suite_source"]["path"], "scripts/harness_pilot_holdout_queries.json")
        self.assertIn("source_queries", plan["holdout_freeze"])

    def test_default_diagnostic_and_opt_in_calibrated_profile(self):
        plan = self.invoke()
        self.assertEqual(list(plan["profiles"]), ["v3", "v3-temp1", "budgeted", "budgeted-temp1"])
        plan = self.invoke("--profiles", "v3", "budgeted")
        self.assertEqual(plan["query_suite"], "diagnostic")
        self.assertEqual(plan["selection"], self.pilot.DIAGNOSTIC_SELECTION)
        self.assertEqual(plan["query_ids"], json.loads(
            (self.root / "scripts/harness_pilot_queries.json").read_text()
        ))
        calibrated = self.invoke("--profiles", "budgeted-calibrated")
        self.assertEqual(list(calibrated["profiles"]), ["budgeted-calibrated"])

    def test_query_limit_slices_frozen_ten_without_changing_freeze(self):
        plan = self.invoke(
            "--query-suite", "holdout", "--profiles", "v3", "budgeted", "--query-limit", "1"
        )
        self.assertTrue(all(len(ids) == 1 for ids in plan["query_ids"].values()))
        self.assertTrue(all(len(ids) == 10 for ids in plan["holdout_freeze"]["query_ids"].values()))

    def test_missing_freeze_fails_closed(self):
        self.freeze_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "required holdout freeze"):
            self.invoke("--query-suite", "holdout", "--profiles", "v3", "budgeted")

    def test_freeze_selection_mismatch_fails_closed(self):
        frozen = json.loads(self.freeze_path.read_text())
        task = next(iter(frozen["query_ids"]))
        frozen["query_ids"][task][0] = "changed"
        self.freeze_path.write_text(json.dumps(frozen))
        with self.assertRaisesRegex(ValueError, "query_ids"):
            self.invoke("--query-suite", "holdout", "--profiles", "v3", "budgeted")

    def test_source_hash_drift_fails_closed(self):
        task = next(iter(json.loads((self.root / "scripts/harness_pilot_queries.json").read_text())))
        source = self.root / "data/metadata" / task / "queries.jsonl"
        source.write_text(source.read_text().replace("fixture", "drift", 1))
        with self.assertRaisesRegex(ValueError, "source_queries"):
            self.invoke("--query-suite", "holdout", "--profiles", "v3", "budgeted")

    def test_freeze_uses_one_canonical_selection_schema(self):
        frozen = json.loads(self.freeze_path.read_text())
        for field in ("query_ids", "excluded_query_ids", "selected_sha256", "excluded_sha256", "source_queries"):
            self.assertIn(field, frozen)
        for alias in ("selected_query_ids", "selected_ids_sha256", "excluded_ids", "excluded_ids_sha256",
                      "source_queries_sha256", "source_query_counts"):
            self.assertNotIn(alias, frozen)


    def test_validation_is_paired_six_times_ten_outside_both_development_suites(self):
        plan = self.invoke("--query-suite", "validation", "--profiles", "budgeted-stable", "budgeted-selective",
                           "--profile-order", "alternating", "--max-concurrency", "4")
        diagnostic = json.loads((self.root / "scripts/harness_pilot_queries.json").read_text())
        holdout = json.loads(self.freeze_path.read_text())["query_ids"]
        self.assertEqual(len(plan["query_ids"]), 6)
        self.assertEqual(len(plan["execution_order"]), 12)
        self.assertFalse(list((self.root / "data/metadata").glob("*/qrels.json")))
        for task, ids in plan["query_ids"].items():
            self.assertEqual(len(ids), 10)
            self.assertTrue(set(ids).isdisjoint(diagnostic[task]))
            self.assertTrue(set(ids).isdisjoint(holdout[task]))
            remaining = set(f"extra-{index}" for index in range(20)) - set(holdout[task])
            expected = sorted(remaining, key=lambda query_id: hashlib.sha256(
                f"dci-selection-validation-v1:{task}:{query_id}".encode()).hexdigest())[:10]
            self.assertEqual(ids, expected)
        self.assertEqual(plan["query_suite_source"]["path"], "scripts/harness_pilot_validation_queries.json")
        self.assertEqual(plan["validation_freeze"], json.loads(self.validation_path.read_text()))

    def test_validation_smoke_keeps_full_freeze(self):
        plan = self.invoke("--query-suite", "validation", "--profiles", "budgeted-stable", "budgeted-selective",
                           "--query-limit", "1")
        self.assertTrue(all(len(ids) == 1 for ids in plan["query_ids"].values()))
        self.assertTrue(all(len(ids) == 10 for ids in plan["validation_freeze"]["query_ids"].values()))

    def test_validation_freeze_drift_and_missing_file_fail_closed(self):
        original = json.loads(self.validation_path.read_text())
        for field in ("schema_version", "suite", "seed", "take", "selection_method", "query_ids",
                      "selected_sha256", "excluded_query_ids", "excluded_sha256", "source_queries",
                      "excluded_suites", "profile_plan", "profile_contract_sha256"):
            with self.subTest(field=field):
                self.validation_path.write_text(json.dumps({**original, field: "changed"}))
                with self.assertRaisesRegex(ValueError, field):
                    self.invoke("--query-suite", "validation", "--profiles", "budgeted-stable", "budgeted-selective")
        self.validation_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "required validation freeze"):
            self.invoke("--query-suite", "validation", "--profiles", "budgeted-stable", "budgeted-selective")

    def test_validation_rejects_query_source_or_exclusion_file_drift(self):
        sources = [next((self.root / "data/metadata").glob("*/queries.jsonl")), self.freeze_path]
        for source in sources:
            with self.subTest(source=source):
                original = source.read_bytes()
                source.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, "freeze mismatch"):
                    self.invoke("--query-suite", "validation", "--profiles", "budgeted-stable", "budgeted-selective")
                source.write_bytes(original)

    def test_validation_requires_both_unchanged_profile_contracts(self):
        with self.assertRaisesRegex(ValueError, "both frozen"):
            self.invoke("--query-suite", "validation", "--profiles", "budgeted-selective")
        frozen = json.loads(self.validation_path.read_text())
        actual = {name: {"sha256": value} for name, value in frozen["profile_contract_sha256"].items()}
        actual["budgeted-selective"]["sha256"] = "edited-prompt"
        with self.assertRaisesRegex(ValueError, "hash or profile mismatch"):
            self.pilot._check_validation_profiles(actual, frozen)

    def test_validation_preflight_rejects_drift_without_backend_or_snapshot(self):
        self.validation_path.unlink()
        with patch.object(self.pilot, "snapshot_sources") as snapshot, redirect_stdout(io.StringIO()), \
             patch("sys.stderr", new_callable=io.StringIO):
            status = self.pilot.main(["--preflight-only", "--query-suite", "validation",
                                      "--profiles", "budgeted-stable", "budgeted-selective"])
        self.assertEqual(status, 2)
        snapshot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
