import unittest

from dci_bench.data.registry import TASKS, TaskSpec
from dci_bench.protocol.contracts import (
    BENCHMARK_CONTRACT_VERSION,
    INSPECT_OUTPUT_CONTRACT,
    QUERY_PROMPT_TEMPLATE,
    TASK_INSTRUCTION_TEMPLATE,
    protocol_contract,
)
from dci_bench.results.protocol import compute_run_fingerprint


class TaskInstructionContractTest(unittest.TestCase):
    def test_base_contract_stays_task_neutral(self):
        self.assertEqual(BENCHMARK_CONTRACT_VERSION, "dci-mvp-v2")
        self.assertEqual(
            INSPECT_OUTPUT_CONTRACT["retrieval_scoring_version"],
            "dci-retrieval-scoring-v2",
        )
        self.assertEqual(
            INSPECT_OUTPUT_CONTRACT["precision_denominator"],
            "metric_window_length",
        )
        self.assertNotIn("task_context", protocol_contract())
        self.assertIn("{task_instruction_block}", QUERY_PROMPT_TEMPLATE)
        self.assertIn("{task_instruction}", TASK_INSTRUCTION_TEMPLATE)

    def test_resolved_contract_records_instruction_for_fingerprint(self):
        resolved = protocol_contract(
            task_name="ToyTask",
            task_instruction="Search the domain phrase, verify, and stop.",
        )
        self.assertEqual(
            resolved["task_context"],
            {
                "task_name": "ToyTask",
                "instruction": "Search the domain phrase, verify, and stop.",
            },
        )

    def test_instruction_change_changes_run_fingerprint(self):
        first = compute_run_fingerprint(
            benchmark_contract=protocol_contract(
                task_name="ToyTask",
                task_instruction="Search terminology A.",
            ),
            query_ids=["Q1"],
        )
        second = compute_run_fingerprint(
            benchmark_contract=protocol_contract(
                task_name="ToyTask",
                task_instruction="Search terminology B.",
            ),
            query_ids=["Q1"],
        )
        self.assertNotEqual(first, second)

    def test_new_tasks_may_use_the_generic_prompt(self):
        task = TaskSpec("ToyTask", "mteb/toy", "revision", "toy")
        self.assertEqual(task.task_instruction, "")
        resolved = protocol_contract(
            task_name=task.name,
            task_instruction=task.task_instruction,
        )
        self.assertEqual(resolved["task_context"]["instruction"], "")

    def test_current_tasks_have_bounded_strategy_only_guidance(self):
        for name, task in TASKS.items():
            with self.subTest(task=name):
                self.assertTrue(task.task_instruction.strip())
                self.assertLess(len(task.task_instruction), 1200)
                lowered = task.task_instruction.lower()
                for forbidden in ("ranked_doc_ids", "qrels", "gold label"):
                    self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
