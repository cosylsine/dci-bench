# Code evaluation smoke fixture

This four-query synthetic fixture exercises the offline code scorer without a
model, external repository, or test execution. The all-zero base commit is an
explicit synthetic identifier, not a real Git revision. The candidate manifest
was generated from the UTF-8 files in `corpus/` using `file_metadata()`.

From the repository root:

```bash
python -m dci_bench.scoring --track code \
  --labels examples/code_retrieval/labels.json \
  --predictions examples/code_retrieval/predictions.json
```

- `demo-trace` selects the complete `divide` function and receives full coverage.
- `demo-missing` deliberately has no prediction and counts as a failure.
- `demo-natural` correctly abstains on an explicitly labeled no-gold query.
- `demo-counterfactual` incorrectly retrieves a file on a no-gold query.

Expected: 4 planned queries, 3 valid outputs, 1 missing prediction; all-sample
positive file recall, line coverage, function recall and functionality recall
are 0.5, while their valid-only means are 1.0. Natural no-gold abstention is 1.0;
counterfactual no-gold abstention is 0.0. These are fixture expectations, not
benchmark results.

See [the evaluation contract and formats](../../docs/code_evaluation.md).

For online acceptance with Inspect and Pi/Bubblewrap, use the same frozen corpus:

```bash
python scripts/run_code_task.py \
  --queries examples/code_retrieval/online_queries.json \
  --labels examples/code_retrieval/labels.json --query-ids demo-trace \
  --mock-script examples/code_retrieval/mock_trace.json
```

This executes source-line reading and the per-sample host scorer without a model
service. Expected: one valid output and full file/line/function/functionality
coverage for `demo-trace`. `--preflight-only` checks data and the real sandbox
without running the mock or creating results. New runs are written to unique
`results/code/` directories; `--output-dir` must name a new directory outside this
fixture. To use an existing model service, replace `--mock-script` with
`--backend-manifest /path/to/backend-manifest.json`, or `--model` and `--base-url`.

The deliberately missing prediction in the offline fixture does not affect an
online run: `online_queries.json` contains all four queries for explicit selection.
