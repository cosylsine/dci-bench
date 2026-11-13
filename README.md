# DCI-Bench

Benchmark for evaluating LLM Direct Corpus Interaction (DCI) capabilities.

Current progress (2026-11-13): [interactive data and evaluation overview](docs/project-progress-20261028/index.html),
[harness workflow](docs/project-progress-20261028/workflow.svg), and
[short collaborator update](docs/project-progress-20261028/progress.md).
The active document benchmark is **Lite / Full v2: 20 tasks across 10 domains
and native English/Chinese sources**. The frozen suite is data-ready; formal
model evaluation and budget calibration are separate acceptance stages.
Frozen Debian RootFS support is available as an **opt-in development runtime**;
Host Tools remain the default, and Code Track keeps its own contract.

## Current document benchmark: Lite / Full

The current suite is `benchmark_v2_20261103_v1`. Sampling uses seed 42,
deduplication and source/parallel-group separation. Both tiers use the same
queries, prompts and cumulative token budget. Lite contains 200 documents per
query and is nested in Full's 1,000; both retain all labelled positives.

| Split | Queries per task | Unique queries | Executions across both tiers |
| --- | ---: | ---: | ---: |
| Smoke (subset of dev) | 1 | 20 | 40 |
| Development, per configuration | 10 | 200 | 400 |
| Test | 50 | 1,000 | 2,000 |

These are prepared data and planned execution counts, not completed model runs.
The 2026-11-03 acceptance records 20/20 tasks and 40/40 data preflights with no
model calls. See the [data acceptance](docs/benchmark20_20261103.md),
[sampling configuration](configs/benchmark_v2.json), and
[execution/calibration guide](docs/benchmark_v2.md).

The profiles are `document-pool-v2-lite` and `document-pool-v2-full`:

- Default cumulative input + output budget: **64,000 tokens**, including cached
  input; request context: 65,536; maximum output per request: 1,024 tokens.
- Tool output: 8 KiB; up to 12 search/read rounds and 24 tool calls, with final
  submission and at most one repair also charged to the token budget.
- At least one nonempty `read` receipt is required. Bash reads do not create
  receipts. The current prompts allow installed commands, pipelines, loops and
  scripts without prescribing a retrieval strategy or command order.
- No default per-sample or whole-run wall-time limit. Individual Bash calls
  retain their default 10-second / maximum 60-second limits. Hardware, elapsed
  time and P95 are descriptive, not cross-hardware quality gates.

Formal test execution requires complete development calibration at 32k/64k/96k,
validated budget selection, and matching data, code, runtime and backend
identities. Smoke validates the execution chain; it does not establish a formal
ranking, throughput acceptance or the effect of changing runtime.

## Qwen3.8-27B smoke on four RTX 5090 GPUs

Run from the repository root on a node with the project Python/Node environments,
four RTX 5090 GPUs, `/mnt/afs2/202610/Qwen3.8-27B`, and the suite plus referenced
corpora under `/mnt/afs2/202611/embedding_models/dci-bench/eval_data/`.
These are deployment-local paths, not files downloaded by a Git checkout.

```bash
# Print the 40-execution plan; no model calls.
DRY_RUN=1 bash scripts/launch_benchmark20_qwen38_5090.sh

# Check data and the real sandbox/Inspect bridge; no GPU/model startup.
bash scripts/launch_benchmark20_qwen38_5090.sh --preflight-only

# Explicitly start the service and execute the smoke.
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_GPU_MEMORY_UTILIZATION=0.80 DRY_RUN=0 \
  bash scripts/launch_benchmark20_qwen38_5090.sh
```

The launcher uses BF16, TP=4, context 65,536, the GDN warmup worker and a default
query concurrency of 4. Each smoke task/tier batch contains only one query, so
this is not a four-query throughput test. `PREFLIGHT_CONCURRENCY` controls CPU
data checks separately. `--serving-smoke-only` with `DRY_RUN=0` checks the GPU
service and synthetic protocol requests without running the benchmark.

Each run gets a fresh directory under
`results/Qwen3.8-27B/benchmark-v2-smoke-runs/`; failures retain their logs and
artifacts. A completed historical Host smoke does not validate the new Frozen
runtime or revised prompts. See the
[current Frozen/Open Bash smoke plan](docs/frozen_qwen38_smoke_20261113.md).

## Optional Frozen Debian runtime

Frozen mode pins the Bash userland while Inspect, Pi, the model bridge, custom
`read` and scoring stay on the host. Bubblewrap mounts the RootFS and candidate
corpus read-only, gives each sample separate scratch space and disables network
access. Evaluation neither installs packages nor falls back to host tools.
The shipped Debian tools do not include an added Python interpreter.

To use the existing local release, set these variables before the smoke commands
above (the bundle must already be available on the target node):

```bash
export DCI_RUNTIME_CONFIG="$PWD/configs/runtime/frozen-debian13.json"
export DCI_RUNTIME_BUNDLE="$PWD/results/frozen-runtime/20261111-xaDmpt/release-v2/rootfs.tar"
bash scripts/launch_benchmark20_qwen38_5090.sh --preflight-only
```

The launcher prepares host Bubblewrap, imports or reuses the verified offline
cache at `/tmp/dci-bench-runtime`, probes the runtime, then checks data and the
Inspect bridge before loading a model. Keep the bundle on persistent storage;
the audited AFS mount did not support the required concurrent cache locking.
The local release is not a published registry image or an artifact included in
this repository. Direct runners accept `--runtime-config`; see
[Frozen runtime setup, identity and validation](docs/frozen_runtime.md).

Recorded development acceptance includes real Pi sandbox checks and 15/15
synthetic Inspect bridge samples, with zero external model calls. Real-model
Frozen smoke, an equal-prompt/equal-budget Host–Frozen comparison and promotion
to the official default remain pending. Historical results keep their original
runtime and prompt identities; do not combine them as one configuration.

## Evaluation tracks

DCI-Bench separates **document** and **code** evaluation. The legacy six-task
Inspect/Pi workflow remains available; the document scoring API is
`dci_bench.scoring.document` (the original `scoring.retrieval` API remains valid).
`dci_bench.scoring.code` adds independent repository-file metrics,
line coverage, function/functional-evidence recall, explicit no-gold abstention,
and ARB budgeted context yield. Fine-grained metrics require host annotations;
retrieval coverage does not imply test execution coverage.

Run the included CPU-only synthetic example:

```bash
python -m dci_bench.scoring --track code \
  --labels examples/code_retrieval/labels.json \
  --predictions examples/code_retrieval/predictions.json
```

The code track also runs online through `scripts/run_code_task.py`: Pi retrieves
repository context, then Inspect scores each sample on the host. It accepts a
completed ARB export or custom query/label manifests and uses source-line reads
and an independent code output contract. Existing document launchers retain their
current behavior. Run the CPU mock through the online path:

```bash
python scripts/run_code_task.py \
  --queries examples/code_retrieval/online_queries.json \
  --labels examples/code_retrieval/labels.json --query-ids demo-trace \
  --mock-script examples/code_retrieval/mock_trace.json
```

Use `--preflight-only` for data/sandbox checks without requests. See
[the separate contracts, metrics, and input formats](docs/code_evaluation.md).
