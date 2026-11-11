from __future__ import annotations

import unittest

from dci_bench.data.benchmark_v2 import (
    SamplingShortageError,
    assert_no_group_leakage,
    sample_global_query_splits,
)


def _grouped_rows(sizes: list[int], *, prefix: str = "q") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group_index, size in enumerate(sizes):
        group_id = f"{prefix}-group-{group_index}"
        rows.extend(
            {
                "query_id": f"{prefix}-{group_index}-{member_index}",
                "group_id": group_id,
            }
            for member_index in range(size)
        )
    return rows


class BenchmarkGroupQuotaTests(unittest.TestCase):
    def test_task_local_exact_fallback_is_deterministic_and_keeps_groups(self):
        # The greedy size priority takes 4+5 and gets stuck at 9 for a
        # quota of 10.  Exact alternatives are 4+6 and 5+5; the remaining
        # groups still form an exact test quota of 50.
        rows = _grouped_rows([4, 6, *([5] * 10)])
        first = sample_global_query_splits(
            {"task": rows}, dev_size=10, test_size=50, seed=42
        )
        second = sample_global_query_splits(
            {"task": list(reversed(rows))}, dev_size=10, test_size=50, seed=42
        )

        self.assertEqual(first, second)
        splits = first["task"]
        self.assertEqual(len(splits["dev"]), 10)
        self.assertEqual(len(splits["test"]), 50)
        self.assertTrue(set(splits["dev"]).isdisjoint(splits["test"]))
        self.assertEqual(
            assert_no_group_leakage(
                splits,
                rows,
                group_key="group_id",
            ),
            {},
        )

    def test_impossible_exact_group_quota_still_raises(self):
        # Every component has size four, so neither a ten-query dev split nor
        # a disjoint exact ten-query component subset can be formed.
        rows = _grouped_rows([4] * 15)
        with self.assertRaises(SamplingShortageError):
            sample_global_query_splits(
                {"task": rows}, dev_size=10, test_size=50, seed=42
            )

    def test_fallback_does_not_split_cross_task_components(self):
        def rows(task_id: str) -> list[dict[str, str]]:
            shared = [
                {
                    "query_id": f"{task_id}-shared-{index}",
                    "group_id": "shared-component",
                }
                for index in range(3)
            ]
            local = _grouped_rows(
                [2, 4, 5, 6, *([5] * 10)],
                prefix=f"{task_id}-local",
            )
            return shared + local

        splits = sample_global_query_splits(
            {"a": rows("a"), "b": rows("b")},
            dev_size=10,
            test_size=50,
            seed=42,
        )
        for split in ("dev", "test"):
            self.assertEqual(
                bool(set(splits["a"][split]) & {f"a-shared-{i}" for i in range(3)}),
                bool(set(splits["b"][split]) & {f"b-shared-{i}" for i in range(3)}),
            )

    def test_fallback_does_not_change_other_task_samples(self):
        singleton_rows = _grouped_rows([1] * 60, prefix="singleton")
        arguana_rows = _grouped_rows(
            [4] * 2 + [5] * 5 + [6] * 43 + [7] * 44 + [8] * 32 + [9] * 22 + [10] * 13,
            prefix="arguana",
        )
        standalone = sample_global_query_splits(
            {"singleton": singleton_rows}, dev_size=10, test_size=50, seed=42
        )
        combined = sample_global_query_splits(
            {"singleton": singleton_rows, "arguana": arguana_rows},
            dev_size=10,
            test_size=50,
            seed=42,
        )

        self.assertEqual(standalone["singleton"], combined["singleton"])
        self.assertEqual(len(combined["arguana"]["dev"]), 10)
        self.assertEqual(len(combined["arguana"]["test"]), 50)


if __name__ == "__main__":
    unittest.main()
